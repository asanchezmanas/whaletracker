"""
QDN Feature Engineer — Phase 1 (25 features)

Computes all Phase 1 features from free/public data sources.
Features are organized into 7 categories:

1. INSIDER (5)   — win_rate, frequency, consistency, holding_period, trade_size_vs_ownership
2. TRANSACTION (3) — is_purchase, filing_delay_days, value_zscore
3. TIMING (4)    — days_since_last_crash, earnings_proximity, sector_momentum_30d, sector_momentum_90d
4. COMPANY (4)   — log_market_cap, volatility_90d, short_interest_pct, volume_anomaly
5. CLUSTER (3)   — num_insiders_buying_30d, temporal_density, cluster_quality
6. MACRO (3)     — vix_normalized, yield_curve_spread, dxy_momentum
7. POLITICAL (3) — is_politician, has_committee_alignment, seniority_score

Total: 25 features

All features are designed to be:
- Computable from publicly available data only
- Look-ahead bias free (uses only data available BEFORE the transaction)
- Normalized to reasonable ranges for neural network consumption
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from .frac_diff import FractionalDifferencer
from .microstructure import MicrostructureAnalyzer

logger = logging.getLogger(__name__)


# Feature names in order (this defines the model's input indices)
FEATURE_NAMES = [
    # Insider (5)
    "insider_win_rate",
    "insider_trade_frequency",
    "insider_consistency",
    "insider_avg_holding_period",
    "trade_size_vs_ownership",
    # Transaction (3)
    "is_purchase",
    "filing_delay_days",
    "value_zscore",
    # Timing (4)
    "days_since_last_crash",
    "earnings_proximity",
    "sector_momentum_30d",
    "sector_momentum_90d",
    # Company (4)
    "log_market_cap",
    "volatility_90d",
    "short_interest_pct",
    "volume_anomaly",
    # Cluster (3)
    "num_insiders_buying_30d",
    "temporal_density",
    "cluster_quality",
    # Macro (3)
    "vix_normalized",
    "yield_curve_spread",
    "dxy_momentum",
    # Political (3)
    "is_politician",
    "has_committee_alignment",
    "seniority_score",
    # Whale Expansion (5) - Phase 1+
    "whale_momentum_rel",
    "whale_volatility_adj",
    "whale_cluster_size",
    "whale_persistence",
    "whale_impact_score",
    # Microstructure & Memory (5) - Phase 3
    "micro_hurst_exponent",
    "micro_vpin",
    "micro_tail_alpha",
    "micro_tail_beta",
    "frac_diff_momentum",
    # Alternative Data (5) - Phase 4
    "has_recent_grant",
    "patent_velocity",
    "otc_liquidity_zscore",
    "early_institutional_entry",
    "grant_vs_market_cap",
]

# Feature index mapping for adversarial training
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Sector-Committee alignment map (senators)
COMMITTEE_SECTOR_MAP = {
    "Armed Services": ["Industrials", "Aerospace & Defense"],
    "Banking": ["Financial Services", "Real Estate"],
    "Finance": ["Financial Services", "Real Estate"],
    "Commerce": ["Consumer Cyclical", "Communication Services"],
    "Energy": ["Energy", "Utilities"],
    "Health": ["Healthcare", "Biotechnology"],
    "Technology": ["Technology", "Communication Services"],
    "Agriculture": ["Consumer Defensive", "Basic Materials"],
}


class FeatureEngineer:
    """
    Computes the 25 Phase 1 features for each insider transaction.
    
    Usage:
        engineer = FeatureEngineer()
        features = engineer.compute_features(
            transaction={"ticker": "AAPL", "insider_name": "Tim Cook", ...},
            historical_transactions=past_transactions_df,
            company_info={"sector": "Technology", "market_cap": 3e12, ...},
            market_data=prices_df,
            macro_snapshot={"vix": 20, "yield_curve": 0.5, ...},
        )
        # Returns: np.array of shape (25,)
    """

    def __init__(self):
        self.fd = FractionalDifferencer()
        self.ms = MicrostructureAnalyzer()
        from .data.alternative_data import AlternativeDataConnector
        self.alt = AlternativeDataConnector()
        self.feature_names = FEATURE_NAMES
        self.n_features = len(FEATURE_NAMES)

    def compute_features(
        self,
        transaction: Dict,
        historical_transactions: pd.DataFrame,
        company_info: Dict,
        market_data: pd.DataFrame,
        macro_snapshot: Dict,
    ) -> np.ndarray:
        """
        Compute all 40 features for a single transaction.
        """
        features = np.zeros(self.n_features, dtype=np.float32)

        tx_date = pd.to_datetime(transaction.get("transaction_date"))
        ticker = transaction.get("ticker", "")
        insider = transaction.get("insider_name", "")

        # --- Insider features (5) ---
        insider_hist = historical_transactions[
            (historical_transactions["insider_name"] == insider)
            & (historical_transactions["transaction_date"] < tx_date)
        ]

        features[0] = self._insider_win_rate(insider_hist)
        features[1] = self._insider_frequency(insider_hist, tx_date)
        features[2] = self._insider_consistency(insider_hist)
        features[3] = self._insider_avg_holding(insider_hist)
        features[4] = self._trade_size_vs_ownership(transaction)

        # --- Transaction features (3) ---
        features[5] = 1.0 if transaction.get("transaction_code") == "P" else 0.0
        features[6] = self._filing_delay(transaction)
        features[7] = self._value_zscore(transaction, historical_transactions, ticker)

        # --- Timing features (4) ---
        features[8] = self._days_since_crash(market_data, tx_date)
        features[9] = self._earnings_proximity(tx_date)
        features[10] = self._sector_momentum(market_data, tx_date, 30)
        features[11] = self._sector_momentum(market_data, tx_date, 90)

        # --- Company features (4) ---
        features[12] = self._log_market_cap(company_info)
        features[13] = self._volatility_90d(market_data, tx_date)
        features[14] = self._short_interest(company_info)
        features[15] = self._volume_anomaly(market_data, tx_date)

        # --- Cluster features (3) ---
        cluster = self._compute_cluster(historical_transactions, ticker, tx_date)
        features[16] = cluster["num_buyers"]
        features[17] = cluster["temporal_density"]
        features[18] = cluster["quality"]

        # --- Macro features (3) ---
        features[19] = self._vix_normalized(macro_snapshot)
        features[20] = self._yield_curve(macro_snapshot)
        features[21] = self._dxy_momentum(macro_snapshot)

        # --- Political features (3) ---
        features[22] = self._is_politician(transaction)
        features[23] = self._committee_alignment(transaction, company_info)
        features[24] = self._seniority_score(transaction)

        # --- Phase 2 placeholders (5) ---
        # Features 25-29 are reserved for Whale features or future expansion
        # Currently keeping them 0 to match 35 features config
        for i in range(25, 30):
            features[i] = 0.0

        # --- Microstructure & Memory (5) - Phase 3 ---
        micro_results = self._compute_micro_features(market_data, tx_date)
        features[30] = micro_results["hurst"]
        features[31] = micro_results["vpin"]
        features[32] = micro_results["alpha"]
        features[33] = micro_results["beta"]
        features[34] = self._compute_frac_diff_momentum(market_data, tx_date)

        # --- Alternative Data (5) - Phase 4 ---
        alt_results = self._compute_alternative_signals(ticker, tx_date, company_info)
        features[35] = alt_results["has_grant"]
        features[36] = alt_results["patent_velocity"]
        features[37] = alt_results["otc_zscore"]
        features[38] = alt_results["early_entry"]
        features[39] = alt_results["grant_intensity"]

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

        return features

    def compute_batch(
        self,
        transactions: pd.DataFrame,
        all_historical: pd.DataFrame,
        company_info_map: Dict[str, Dict],
        market_data_map: Dict[str, pd.DataFrame],
        macro_snapshots: Dict[str, Dict],
    ) -> np.ndarray:
        """
        Compute features for a batch of transactions.
        """
        features_list = []

        for _, row in transactions.iterrows():
            ticker = row.get("ticker", "")
            tx_date_str = str(row.get("transaction_date", ""))[:10]

            features = self.compute_features(
                transaction=row.to_dict(),
                historical_transactions=all_historical,
                company_info=company_info_map.get(ticker, {}),
                market_data=market_data_map.get(ticker, pd.DataFrame()),
                macro_snapshot=macro_snapshots.get(tx_date_str, {}),
            )
            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def _compute_micro_features(self, market_data: pd.DataFrame, tx_date: pd.Timestamp) -> Dict:
        """Compute Hurst, VPIN, and Tail parameters."""
        if market_data.empty:
            return {"hurst": 0.5, "vpin": 0.5, "alpha": 2.0, "beta": 0.0}

        # Use 100 days of look-back for microstructure
        history = market_data[market_data.index < tx_date].tail(100)
        if len(history) < 30:
            return {"hurst": 0.5, "vpin": 0.5, "alpha": 2.0, "beta": 0.0}

        prices = history["Close"].values
        volumes = history["Volume"].values
        returns = history["Close"].pct_change().dropna().values

        return {
            "hurst": self.ms.compute_hurst(prices),
            "vpin": self.ms.compute_vpin(prices, volumes),
            "alpha": self.ms.estimate_tail_parameters(returns)[0],
            "beta": self.ms.estimate_tail_parameters(returns)[1],
        }

    def _compute_frac_diff_momentum(self, market_data: pd.DataFrame, tx_date: pd.Timestamp) -> float:
        """Compute momentum on a fractionally differenced price series."""
        if market_data.empty:
            return 0.0
        
        # We need a longer window for FFD to converge (e.g., 250 days)
        history = market_data[market_data.index < tx_date].tail(300)
        if len(history) < 250:
            return 0.0
            
        # Standard FFD with d=0.3
        ffd_prices = self.fd.apply_ffd(history[["Close"]], d=0.3, threshold=1e-4)
        if ffd_prices.empty or len(ffd_prices) < 20:
            return 0.0
            
        # Return last 20-day momentum of stationary series
        last_val = ffd_prices["Close"].iloc[-1]
        prev_val = ffd_prices["Close"].iloc[-20] if len(ffd_prices) > 20 else ffd_prices["Close"].iloc[0]
        
        return float(last_val - prev_val) / abs(prev_val) if abs(prev_val) > 1e-8 else 0.0

    # ─────────────────────────────────────────────
    # INSIDER FEATURES (5)
    # ─────────────────────────────────────────────

    @staticmethod
    def _insider_win_rate(history: pd.DataFrame) -> float:
        """
        Historical win rate of this insider.
        Win = positive return in 6 months after purchase.
        """
        if history.empty or len(history) < 3:
            return 0.5  # Prior: 50% win rate with no data

        purchases = history[history.get("transaction_code", pd.Series()) == "P"]
        if purchases.empty:
            return 0.5

        if "actual_return_6m" in purchases.columns:
            wins = (purchases["actual_return_6m"] > 0).sum()
            return float(wins / len(purchases))

        return 0.5

    @staticmethod
    def _insider_frequency(history: pd.DataFrame, current_date) -> float:
        """
        Trade frequency: average trades per year (log-scaled).
        Higher frequency = more data points = more reliable signal.
        """
        if history.empty or len(history) < 2:
            return 0.0

        dates = pd.to_datetime(history["transaction_date"])
        span_days = (dates.max() - dates.min()).days
        if span_days < 30:
            return 0.0

        trades_per_year = len(history) / (span_days / 365.25)
        return float(np.log1p(trades_per_year))  # Log scale

    @staticmethod
    def _insider_consistency(history: pd.DataFrame) -> float:
        """
        Consistency: what fraction of trades are in the same direction.
        High consistency (mostly buys) = conviction, but could also be
        routine accumulation. Moderate values are most interesting.
        """
        if history.empty or len(history) < 3:
            return 0.5

        codes = history.get("transaction_code", pd.Series())
        if codes.empty:
            return 0.5

        buy_frac = (codes == "P").mean()
        return float(buy_frac)

    @staticmethod
    def _insider_avg_holding(history: pd.DataFrame) -> float:
        """
        Average estimated holding period (normalized).
        Short holding = quick trade, Long = strategic conviction.
        """
        if history.empty or len(history) < 2:
            return 0.5

        # Estimate from gap between buys and sells
        buys = history[history.get("transaction_code", pd.Series()) == "P"]
        sells = history[history.get("transaction_code", pd.Series()) == "S"]

        if buys.empty or sells.empty:
            return 0.5

        # Average gap between any buy and the next sell
        buy_dates = pd.to_datetime(buys["transaction_date"]).sort_values()
        sell_dates = pd.to_datetime(sells["transaction_date"]).sort_values()

        gaps = []
        for bd in buy_dates:
            future_sells = sell_dates[sell_dates > bd]
            if not future_sells.empty:
                gap = (future_sells.iloc[0] - bd).days
                gaps.append(gap)

        if not gaps:
            return 0.5

        avg_gap = np.mean(gaps)
        # Normalize: 0-30 days = short (0), 180 days = medium (0.5), 365+ = long (1)
        return float(min(avg_gap / 365.0, 1.0))

    @staticmethod
    def _trade_size_vs_ownership(transaction: Dict) -> float:
        """
        Transaction size relative to total ownership.
        Large fraction = high conviction signal.
        """
        shares = transaction.get("shares")
        ownership = transaction.get("ownership_after")

        if not shares or not ownership or ownership <= 0:
            return 0.0

        # What fraction of total holding does this trade represent?
        ratio = abs(shares) / ownership
        return float(min(ratio, 1.0))  # Cap at 100%

    # ─────────────────────────────────────────────
    # TRANSACTION FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _filing_delay(transaction: Dict) -> float:
        """
        Days between transaction and filing.
        Short delay = routine. Long delay = possibly strategic timing.
        Normalized to [0, 1] with 30-day cap.
        """
        tx_date = pd.to_datetime(transaction.get("transaction_date"))
        filing_date = pd.to_datetime(transaction.get("filing_date"))

        if pd.isna(tx_date) or pd.isna(filing_date):
            return 0.0

        delay = (filing_date - tx_date).days
        return float(min(max(delay, 0), 30) / 30.0)

    @staticmethod
    def _value_zscore(
        transaction: Dict,
        history: pd.DataFrame,
        ticker: str,
    ) -> float:
        """
        Z-score of transaction value relative to historical transactions
        for the same ticker. Large unusual trades are more informative.
        """
        value = transaction.get("value")
        if not value:
            return 0.0

        ticker_hist = history[history["ticker"] == ticker]
        if ticker_hist.empty or "value" not in ticker_hist.columns:
            return 0.0

        hist_values = ticker_hist["value"].dropna()
        if len(hist_values) < 5 or hist_values.std() < 1e-8:
            return 0.0

        zscore = (value - hist_values.mean()) / hist_values.std()
        return float(np.clip(zscore, -3, 3))

    # ─────────────────────────────────────────────
    # TIMING FEATURES (4)
    # ─────────────────────────────────────────────

    @staticmethod
    def _days_since_crash(market_data: pd.DataFrame, tx_date) -> float:
        """
        Days since the last significant market drop (>5% drawdown).
        Buying during/after crashes is a Taleb-inspired conviction signal.
        Normalized: 0 = crash today, 1 = 365+ days since crash.
        """
        if market_data.empty:
            return 0.5

        try:
            close = market_data["Close"]
            # Find drawdowns
            rolling_max = close.expanding().max()
            drawdown = (close - rolling_max) / rolling_max

            # Significant drops (> -5%)
            crash_dates = drawdown[drawdown < -0.05].index
            crash_dates = crash_dates[crash_dates < tx_date]

            if crash_dates.empty:
                return 1.0

            days = (tx_date - crash_dates[-1]).days
            return float(min(days / 365.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _earnings_proximity(tx_date) -> float:
        """
        Proximity to typical earnings dates (Jan/Apr/Jul/Oct quarters).
        Insiders buying BEFORE earnings may signal foreknowledge.
        Returns distance in days to nearest earnings, normalized [0, 1].
        """
        # Typical earnings months: mid-Jan, mid-Apr, mid-Jul, mid-Oct
        tx = pd.to_datetime(tx_date)
        year = tx.year
        earnings_dates = [
            pd.Timestamp(year, 1, 20),
            pd.Timestamp(year, 4, 20),
            pd.Timestamp(year, 7, 20),
            pd.Timestamp(year, 10, 20),
            pd.Timestamp(year + 1, 1, 20),  # Next year Jan
        ]

        min_days = min(abs((tx - ed).days) for ed in earnings_dates)
        return float(min(min_days / 45.0, 1.0))  # 45 days = max relevant

    @staticmethod
    def _sector_momentum(
        market_data: pd.DataFrame, tx_date, lookback_days: int
    ) -> float:
        """
        Stock return over lookback period.
        Buying during sector weakness = contrarian conviction.
        """
        if market_data.empty:
            return 0.0

        try:
            close = market_data["Close"]
            start = tx_date - timedelta(days=lookback_days)
            period = close[(close.index >= start) & (close.index <= tx_date)]

            if len(period) < 5:
                return 0.0

            ret = (period.iloc[-1] / period.iloc[0]) - 1
            return float(np.clip(ret, -0.5, 0.5))
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────
    # COMPANY FEATURES (4)
    # ─────────────────────────────────────────────

    @staticmethod
    def _log_market_cap(company_info: Dict) -> float:
        """
        Log10 of market cap. Small caps have more information asymmetry.
        Normalized: log10(1M) = 6, log10(1T) = 12 → [6, 12] → [0, 1]
        """
        mc = company_info.get("market_cap")
        if not mc or mc <= 0:
            return 0.5

        log_mc = np.log10(mc)
        return float(np.clip((log_mc - 6) / 6, 0, 1))

    @staticmethod
    def _volatility_90d(market_data: pd.DataFrame, tx_date) -> float:
        """
        90-day historical volatility (annualized).
        High vol = more uncertainty = more room for information edge.
        Normalized to [0, 1] with 100% vol as cap.
        """
        if market_data.empty:
            return 0.3

        try:
            close = market_data["Close"]
            start = tx_date - timedelta(days=100)
            period = close[(close.index >= start) & (close.index <= tx_date)]

            if len(period) < 20:
                return 0.3

            returns = period.pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            return float(min(vol, 1.0))
        except Exception:
            return 0.3

    @staticmethod
    def _short_interest(company_info: Dict) -> float:
        """
        Short interest as % of float.
        High short interest + insider buying = potential short squeeze setup.
        Normalized to [0, 1] with 50% as cap.
        """
        si = company_info.get("short_percent_of_float")
        if not si:
            return 0.0
        return float(min(si, 0.5) / 0.5)

    @staticmethod
    def _volume_anomaly(market_data: pd.DataFrame, tx_date) -> float:
        """
        Volume on transaction day vs 30-day average.
        Unusual volume near insider trades may indicate broader awareness.
        """
        if market_data.empty or "Volume" not in market_data.columns:
            return 0.0

        try:
            vol = market_data["Volume"]
            end = tx_date
            start = tx_date - timedelta(days=35)

            recent = vol[(vol.index >= start) & (vol.index <= end)]
            if len(recent) < 10:
                return 0.0

            day_vol = recent.iloc[-1]
            avg_vol = recent.iloc[:-1].mean()

            if avg_vol < 1e-8:
                return 0.0

            ratio = day_vol / avg_vol
            return float(np.clip(np.log(ratio), -2, 2) / 2)  # Log ratio, capped
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────
    # CLUSTER FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _compute_cluster(
        history: pd.DataFrame, ticker: str, tx_date
    ) -> Dict[str, float]:
        """
        Detect insider cluster buying: multiple insiders buying the same
        stock in a short window signals coordinated conviction.
        
        Returns:
            num_buyers: number of distinct insiders buying in 30-day window
            temporal_density: how concentrated the buys are in time
            quality: cluster quality score (buyer diversity × size)
        """
        if history.empty:
            return {"num_buyers": 0, "temporal_density": 0, "quality": 0}

        # 30-day window before transaction
        window_start = tx_date - timedelta(days=30)
        window = history[
            (history["ticker"] == ticker)
            & (history["transaction_date"] >= window_start)
            & (history["transaction_date"] <= tx_date)
            & (history.get("transaction_code", pd.Series()) == "P")
        ]

        if window.empty:
            return {"num_buyers": 0, "temporal_density": 0, "quality": 0}

        # Distinct buyers
        unique_buyers = window["insider_name"].nunique()
        num_buyers = min(unique_buyers / 5.0, 1.0)  # Normalize: 5 buyers = max

        # Temporal density (how bunched in time)
        dates = pd.to_datetime(window["transaction_date"])
        if len(dates) > 1:
            span = (dates.max() - dates.min()).days
            density = 1.0 - min(span / 30.0, 1.0)  # Tight = high
        else:
            density = 0.5

        # Cluster quality (buyers × density)
        quality = num_buyers * (0.5 + 0.5 * density)

        return {
            "num_buyers": float(num_buyers),
            "temporal_density": float(density),
            "quality": float(quality),
        }

    # ─────────────────────────────────────────────
    # MACRO FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _vix_normalized(macro: Dict) -> float:
        """
        VIX normalized to [0, 1]. Historical range ~10-80.
        High VIX = fear = potential buying opportunity if insiders are buying.
        """
        vix = macro.get("vix")
        if vix is None:
            return 0.3
        return float(np.clip((vix - 10) / 70, 0, 1))

    @staticmethod
    def _yield_curve(macro: Dict) -> float:
        """
        10Y-2Y yield spread. Negative = inverted = recession signal.
        Normalized: -1% to +3% → [0, 1]
        """
        spread = macro.get("yield_curve")
        if spread is None:
            return 0.5
        return float(np.clip((spread + 1) / 4, 0, 1))

    @staticmethod
    def _dxy_momentum(macro: Dict) -> float:
        """
        Dollar index level normalized.
        Strong dollar affects different sectors differently.
        Range ~80-120 → [0, 1]
        """
        dxy = macro.get("dxy")
        if dxy is None:
            return 0.5
        return float(np.clip((dxy - 80) / 40, 0, 1))

    # ─────────────────────────────────────────────
    # POLITICAL FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _is_politician(transaction: Dict) -> float:
        """Binary: is this a politician (senator/representative)?"""
        source = transaction.get("source", "")
        title = transaction.get("insider_title", "")

        if source in ("senate", "house"):
            return 1.0
        if title and any(
            t in title.lower()
            for t in ["senator", "representative", "congressman"]
        ):
            return 1.0
        return 0.0

    @staticmethod
    def _committee_alignment(transaction: Dict, company_info: Dict) -> float:
        """
        Does the insider's committee oversee the company's sector?
        
        High alignment = potential information advantage.
        Only applicable to politicians.
        """
        committees = transaction.get("committees", [])
        sector = company_info.get("sector", "")

        if not committees or not sector:
            return 0.0

        for committee in committees:
            for comm_name, sectors in COMMITTEE_SECTOR_MAP.items():
                if comm_name.lower() in committee.lower():
                    if sector in sectors:
                        return 1.0

        return 0.0

    @staticmethod
    def _seniority_score(transaction: Dict) -> float:
        """
        Seniority of the politician (years in office, normalized).
        More senior = more connections = more information access.
        """
        years = transaction.get("seniority_years")
        if not years:
            return 0.0
        return float(min(years / 30.0, 1.0))  # 30 years = max

    # ─────────────────────────────────────────────
    # PHASE 3 & 4 HELPERS
    # ─────────────────────────────────────────────

    def _compute_alternative_signals(self, ticker: str, tx_date: datetime, company_info: Dict) -> Dict:
        """
        Compute Phase 4 alternative signals (grants, patents, OTC).
        """
        # 1. Grants (SBIR/STTR)
        company_name = company_info.get("name", ticker)
        grants = self.alt.fetch_sbir_grants(company_name)
        has_grant = 0.0
        grant_intensity = 0.0
        
        if not grants.empty:
            recent_grants = grants[grants["grant_date"] < tx_date]
            if not recent_grants.empty:
                has_grant = 1.0
                mc = company_info.get("market_cap", 1e8)
                total_val = recent_grants["award_amount"].astype(float).sum() if "award_amount" in recent_grants.columns else 0
                grant_intensity = np.log1p(total_val / mc)

        # 2. Patents (USPTO)
        patents = self.alt.fetch_uspto_patents(company_name)
        vel = 0.0
        if not patents.empty:
            cutoff = tx_date - timedelta(days=730)
            recent_p = patents[(pd.to_datetime(patents["date"]) < tx_date) & (pd.to_datetime(patents["date"]) > cutoff)]
            vel = len(recent_p) / 2.0  # Patents per year

        # 3. OTC & Institutional
        extra = self.alt.get_otc_activity(ticker)
        otc_z = 1.0 if extra["is_otc"] else 0.0
        
        is_micro = company_info.get("market_cap", 1e9) < 5e8 # < 500M
        early_entry = 1.0 if is_micro else 0.0

        return {
            "has_grant": has_grant,
            "patent_velocity": vel,
            "otc_zscore": otc_z,
            "early_entry": early_entry,
            "grant_intensity": grant_intensity
        }

    def _compute_frac_diff_momentum(self, market_data: pd.DataFrame, tx_date: datetime) -> float:
        """Compute momentum on fractionally differentiated series."""
        try:
            prices = market_data.loc[:tx_date]["Close"].tail(500)
            if len(prices) < 50: return 0.0
            ffd_df = self.fd.apply_ffd(pd.DataFrame(prices), d=0.4)
            if ffd_df.empty: return 0.0
            return float(ffd_df.iloc[-1, 0])
        except Exception:
            return 0.0

    def _compute_micro_features(self, market_data: pd.DataFrame, tx_date: datetime) -> Dict:
        """Wrapper for microstructure features."""
        try:
            window = market_data.loc[:tx_date].tail(100)
            if len(window) < 50: 
                return {"hurst": 0.5, "vpin": 0.5, "alpha": 1.5, "beta": 0.0}
            h = self.ms.hurst_exponent(window["Close"].values)
            v = self.ms.vpin(window["Close"].values, window.get("Volume", np.ones(len(window))).values)
            alpha, beta = self.ms.levy_stable_params(window["Close"].pct_change().dropna().values)
            return {"hurst": h, "vpin": v, "alpha": alpha, "beta": beta}
        except Exception:
            return {"hurst": 0.5, "vpin": 0.5, "alpha": 1.5, "beta": 0.0}

