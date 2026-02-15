r"""
Triple Barrier Method — López de Prado

Replaces naive forward-return labeling with 3-barrier events:

  ┌───── upper barrier (profit target) ─────────────┐
  │                                                  │
  │   price path ~~~~/\~~~/\~~~~\___/\~~...          │
  │                                                  │
  │───── entry price ──────────────────...──── time ──│
  │                                                  │
  │   price path ~~~~\___/\~~/\~~~~\/\~~...          │
  │                                                  │
  └───── lower barrier (stop-loss) ──────────────────┘
                                      ▲
                                  vertical barrier (max hold)

Label = first barrier touched:
  +1 → upper barrier (profitable)
  -1 → lower barrier (stopped out)
   0 → timeout (time barrier hit)

Return = actual return at barrier event.

Reference: AFML Chapter 3
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import timedelta
from ..utils.timezone_utils import unify_timezone

logger = logging.getLogger(__name__)


@dataclass
class BarrierEvent:
    """Result of applying triple barrier to a single trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    ret: float               # realized return at exit
    label: int               # +1 profit, -1 stop, 0 timeout
    barrier_type: str        # "upper", "lower", "vertical"
    holding_days: int
    touch_upper: bool        # did price touch upper at any point?
    touch_lower: bool        # did price touch lower at any point?
    max_drawdown: float      # worst drawdown during holding
    max_runup: float         # best unrealized gain during holding


class TripleBarrierLabeler:
    """
    Triple Barrier labeling for insider-trade events.

    Instead of simple "did price go up 180 days later?", this answers:
    "What happened FIRST — did we hit our profit target, our stop-loss,
    or did we time out?"

    This is critical because it captures the real trading experience:
    a stock might be +50% at day 90 but -10% at day 180.  Simple forward
    returns would label that as a loss, but the triple barrier says WIN
    (upper barrier hit at day 90).

    Usage:
        labeler = TripleBarrierLabeler()
        events = labeler.apply(transactions_df, price_data_map)
    """

    def __init__(
        self,
        profit_mult: float = 2.0,
        stop_mult: float = 1.0,
        max_holding_days: int = 180,
        vol_lookback: int = 60,
        min_vol: float = 0.01,
    ):
        """
        Args:
            profit_mult:  Profit barrier = profit_mult × daily_vol
                          Default 2.0 means target = 2σ of daily returns
            stop_mult:    Stop barrier = stop_mult × daily_vol
                          Default 1.0 means stop = 1σ of daily returns
            max_holding_days: Time (vertical) barrier in calendar days
            vol_lookback: Days to look back for volatility estimation
            min_vol:      Minimum daily vol floor (prevents degenerate barriers)
        """
        self.profit_mult = profit_mult
        self.stop_mult = stop_mult
        self.max_holding_days = max_holding_days
        self.vol_lookback = vol_lookback
        self.min_vol = min_vol

    def _estimate_daily_vol(
        self,
        prices: pd.DataFrame,
        entry_date: pd.Timestamp,
    ) -> float:
        """
        Estimate daily volatility using close-to-close returns
        over the lookback window before entry_date.

        This is the 'width' of our barriers — wider for volatile stocks,
        narrower for stable ones.  Critically, it uses ONLY past data
        (no future leakage).
        """
        if prices.empty:
            return self.min_vol

        # Use Close column (yfinance returns 'Close')
        close_col = "Close" if "Close" in prices.columns else "close"
        if close_col not in prices.columns:
            return self.min_vol

        close = prices[close_col]

        # Filter to before entry date
        entry_date = unify_timezone(entry_date, close.index)
        mask = close.index <= entry_date
        pre_entry = close[mask].tail(self.vol_lookback)

        if len(pre_entry) < 10:
            return self.min_vol

        daily_returns = pre_entry.pct_change().dropna()
        vol = daily_returns.std()

        return max(float(vol), self.min_vol)

    def apply_single(
        self,
        entry_date: pd.Timestamp,
        prices: pd.DataFrame,
    ) -> Optional[BarrierEvent]:
        """
        Apply triple barrier to a single trade entry.

        Args:
            entry_date:  When the insider bought
            prices:      OHLCV DataFrame for this ticker (indexed by date)

        Returns:
            BarrierEvent or None if insufficient data
        """
        close_col = "Close" if "Close" in prices.columns else "close"
        if close_col not in prices.columns or prices.empty:
            return None

        close = prices[close_col]

        # Get entry price (first available on or after entry_date)
        entry_date = unify_timezone(entry_date, close.index)

        future = close[close.index >= entry_date]
        if len(future) < 5:
            return None

        entry_price = float(future.iloc[0])
        if entry_price <= 0:
            return None

        # Estimate volatility
        daily_vol = self._estimate_daily_vol(prices, entry_date)

        # Set barriers (as return thresholds)
        upper = self.profit_mult * daily_vol * np.sqrt(self.max_holding_days)
        lower = -self.stop_mult * daily_vol * np.sqrt(self.max_holding_days)

        # Walk forward through prices
        end_date = entry_date + timedelta(days=self.max_holding_days)
        path = future[future.index <= end_date]

        if len(path) < 2:
            return None

        # Compute returns from entry
        returns = (path / entry_price) - 1.0

        # Track extremes
        max_runup = float(returns.max())
        max_drawdown = float(returns.min())
        touch_upper = max_runup >= upper
        touch_lower = max_drawdown <= lower

        # Find first barrier touch
        barrier_type = "vertical"  # default: timeout
        exit_idx = len(returns) - 1  # default: last point

        for i in range(1, len(returns)):
            ret_i = float(returns.iloc[i])

            if ret_i >= upper:
                barrier_type = "upper"
                exit_idx = i
                break
            elif ret_i <= lower:
                barrier_type = "lower"
                exit_idx = i
                break

        exit_price = float(path.iloc[exit_idx])
        exit_date = path.index[exit_idx]
        realized_return = (exit_price / entry_price) - 1.0

        # Assign label
        if barrier_type == "upper":
            label = 1
        elif barrier_type == "lower":
            label = -1
        else:
            # Timeout: label by sign of return at exit
            label = 1 if realized_return > 0 else (-1 if realized_return < 0 else 0)

        holding_days = (exit_date - entry_date).days

        return BarrierEvent(
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            ret=float(realized_return),
            label=label,
            barrier_type=barrier_type,
            holding_days=holding_days,
            touch_upper=touch_upper,
            touch_lower=touch_lower,
            max_drawdown=float(max_drawdown),
            max_runup=float(max_runup),
        )

    def apply(
        self,
        transactions: pd.DataFrame,
        price_data_map: Dict[str, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray, List[BarrierEvent]]:
        """
        Apply triple barrier to all transactions.

        Args:
            transactions: DataFrame with 'ticker' and 'transaction_date'
            price_data_map: {ticker: OHLCV DataFrame}

        Returns:
            labels:   np.ndarray of shape (n_valid,) — +1, -1, or 0
            returns:  np.ndarray of shape (n_valid,) — realized returns
            events:   List[BarrierEvent] with full detail
        """
        labels = []
        returns = []
        events = []
        valid_mask = []

        for idx, row in transactions.iterrows():
            ticker = row.get("ticker", "")
            tx_date = pd.to_datetime(row.get("transaction_date"))

            if pd.isna(tx_date) or ticker not in price_data_map:
                valid_mask.append(False)
                continue

            prices = price_data_map[ticker]
            event = self.apply_single(tx_date, prices)

            if event is None:
                valid_mask.append(False)
                continue

            labels.append(event.label)
            returns.append(event.ret)
            events.append(event)
            valid_mask.append(True)

        labels_arr = np.array(labels, dtype=np.float32)
        returns_arr = np.array(returns, dtype=np.float32)

        # Log summary
        n = len(labels_arr)
        if n > 0:
            n_win = (labels_arr == 1).sum()
            n_loss = (labels_arr == -1).sum()
            n_timeout = (labels_arr == 0).sum()
            logger.info(
                f"Triple Barrier: {n} events — "
                f"Win: {n_win} ({n_win/n:.0%}), "
                f"Loss: {n_loss} ({n_loss/n:.0%}), "
                f"Timeout: {n_timeout} ({n_timeout/n:.0%})"
            )
            logger.info(
                f"  Avg return: {returns_arr.mean():.2%}, "
                f"  Median: {np.median(returns_arr):.2%}"
            )

            # Barrier type breakdown
            upper_count = sum(1 for e in events if e.barrier_type == "upper")
            lower_count = sum(1 for e in events if e.barrier_type == "lower")
            vert_count = sum(1 for e in events if e.barrier_type == "vertical")
            logger.info(
                f"  Barrier types: Upper={upper_count}, "
                f"Lower={lower_count}, Vertical={vert_count}"
            )

        return labels_arr, returns_arr, events
        return weights

    def get_concurrent_labels(
        self, events: List[BarrierEvent]
    ) -> np.ndarray:
        """
        Compute concurrent label count for each event using an O(n log n) sweep-line.

        Used for sample weight calculation — events that overlap
        temporally share information and should be down-weighted.
        """
        if not events:
            return np.array([], dtype=np.float32)

        # 1. Create event list: (time, type, index)
        # type: +1 for start (entry), -1 for end (exit)
        times = []
        for i, ev in enumerate(events):
            times.append((ev.entry_date, 1, i))
            times.append((ev.exit_date, -1, i))

        # 2. Sort by time
        times.sort()

        # 3. Sweep line to find max concurrency for each interval
        n = len(events)
        concurrency = np.zeros(n, dtype=np.float32)
        active_indices = set()
        
        # Store the concurrency level at which each event became active
        # This is needed because an event's concurrency might increase later
        # but we need to track the *maximum* concurrency it experienced.
        event_concurrency_at_start = {} 

        for t, type, idx in times:
            if type == 1: # Event starts
                active_indices.add(idx)
                current_active_count = len(active_indices)
                event_concurrency_at_start[idx] = current_active_count
                
                # Update the concurrency for all currently active events
                # with the new, potentially higher, concurrency level.
                for active_idx in active_indices:
                    concurrency[active_idx] = max(concurrency[active_idx], current_active_count)
            else: # Event ends
                # When an event ends, it contributes to the concurrency of other active events
                # up until its exit. Its own concurrency is determined by the max it experienced.
                if idx in active_indices: # Ensure it was active (could be already removed if exit_date == entry_date)
                    active_indices.remove(idx)

        return concurrency

    def compute_sample_weights(
        self, events: List[BarrierEvent]
    ) -> np.ndarray:
        """
        Compute uniqueness-based sample weights (López de Prado Ch 4).

        Samples that overlap with many others get lower weight.
        Unique samples get higher weight.

        Returns:
            np.ndarray of shape (n_events,) — normalized weights
        """
        concurrency = self.get_concurrent_labels(events)

        # Weight = 1 / average_concurrency
        weights = 1.0 / concurrency

        # Normalize so weights sum to len(events)
        weights = weights * len(weights) / weights.sum()

        return weights
