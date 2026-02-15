"""
QDN Inference API

Scores new opportunities and generates exit signals.

Antifragile approach:
- Entry: Score insider/institutional purchases → BUY high-conviction ones
- Exit: Monitor insider sales for held positions → EXIT when insiders sell

Usage:
    api = QDNInference("checkpoints/best_model.pth")

    # Score a new purchase signal
    result = api.score_opportunity(transaction, market_data)
    # → {score: 72, action: "BUY", conviction: "HIGH", ticker: "XYZ"}

    # Check if you should exit held positions
    exits = api.check_exit_signals(["AAPL", "MSFT", "XYZ"])
    # → [{ticker: "XYZ", action: "EXIT", insider_sales_30d: 5, urgency: "HIGH"}]
"""

import torch
import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .config import QDNConfig
from .dense_network import DenseNetwork
from .features.engineer import FeatureEngineer
from .data.whale_connector import WhaleConnector
from .data.market_connector import MarketConnector

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """A scored entry (buy) signal."""
    ticker: str
    score: float
    action: str          # "BUY" | "WATCH" | "SKIP"
    conviction: str      # "HIGH" | "MEDIUM" | "LOW"
    whale_type: str      # insider_purchase | cluster_buy | activist_13d | institutional_13f
    insider_name: str
    price: Optional[float]
    value: Optional[float]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExitSignal:
    """An exit (sell) signal for a held position."""
    ticker: str
    action: str          # "EXIT" | "HOLD" | "WATCH"
    urgency: str         # "HIGH" | "MEDIUM" | "LOW"
    insider_sales_30d: int
    insider_sales_value: float
    unique_sellers: int
    latest_sale_date: Optional[str]
    reason: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class QDNInference:
    """
    Production inference API for the QDN model.

    Combines:
    1. Model scoring (purchase conviction quality)
    2. Exit signal detection (insider sales monitoring)
    3. Signal logging for audit trail
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[QDNConfig] = None,
        signal_log_path: str = "signals.json",
    ):
        self.config = config or QDNConfig()
        self.device = torch.device(self.config.resolve_device())

        # Model
        self.model = DenseNetwork(self.config.model).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from {checkpoint_path}")
        self.model.eval()

        # Components
        self.engineer = FeatureEngineer()
        self.whale = WhaleConnector()
        self.market = MarketConnector()

        # Signal log
        self.signal_log_path = signal_log_path
        self._signals: List[dict] = []
        self._load_signal_log()

    # ═══════════════════════════════════════════════════════
    # ENTRY SIGNALS — Score purchase opportunities
    # ═══════════════════════════════════════════════════════

    def score_opportunity(
        self,
        transaction: dict,
        market_data: Optional[pd.DataFrame] = None,
        company_info: Optional[dict] = None,
        historical_transactions: Optional[pd.DataFrame] = None,
    ) -> EntrySignal:
        """
        Score a single purchase opportunity.

        Args:
            transaction: Dict with ticker, transaction_date, insider_name,
                        transaction_code, price, shares, value, etc.
            market_data: OHLCV DataFrame (fetched automatically if None)
            company_info: Company metadata (fetched automatically if None)
            historical_transactions: Past transactions for context

        Returns:
            EntrySignal with score, action, and conviction level
        """
        ticker = transaction.get("ticker", "")

        # Auto-fetch if not provided
        if market_data is None:
            market_data = self.market.get_stock_prices(ticker, period="1y")
        if company_info is None:
            company_info = self.market.get_company_info(ticker)
        if historical_transactions is None:
            historical_transactions = pd.DataFrame()

        # Compute features
        features = self.engineer.compute_features(
            transaction=transaction,
            historical_transactions=historical_transactions,
            company_info=company_info,
            market_data=market_data,
            macro_snapshot={"vix": 20, "yield_curve": 0.5, "dxy": 103},
        )

        # Score via model
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32).to(self.device)
            output = self.model(x)
            score = float(output["convexity_score"].cpu().item())

        # Determine action and conviction
        action, conviction = self._classify_score(score)

        signal = EntrySignal(
            ticker=ticker,
            score=round(score, 1),
            action=action,
            conviction=conviction,
            whale_type=transaction.get("whale_type", "insider_purchase"),
            insider_name=transaction.get("insider_name", "Unknown"),
            price=transaction.get("price"),
            value=transaction.get("value"),
            timestamp=datetime.now().isoformat(),
        )

        # Log
        self._log_signal("ENTRY", signal.to_dict())

        return signal

    def score_batch(
        self, transactions: pd.DataFrame
    ) -> List[EntrySignal]:
        """Score a batch of transactions. Returns sorted by score (highest first)."""
        signals = []
        for _, row in transactions.iterrows():
            try:
                signal = self.score_opportunity(row.to_dict())
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Score error for {row.get('ticker')}: {e}")

        return sorted(signals, key=lambda s: s.score, reverse=True)

    def scan_market(self, days_back: int = 30) -> List[EntrySignal]:
        """
        Full market scan: fetch all whale signals, score them, return top picks.
        
        This is the main "what should I buy?" function.
        """
        logger.info("Scanning market for whale signals...")
        
        # Fetch all whale signals
        signals_df = self.whale.fetch_all_whale_signals(days_back=days_back)
        
        if signals_df.empty:
            logger.info("No whale signals found")
            return []
        
        # Filter to purchases only
        signals_df = signals_df[signals_df["transaction_code"] == "P"]
        
        # Score each
        scored = self.score_batch(signals_df)
        
        # Return only BUY and WATCH signals
        actionable = [s for s in scored if s.action in ("BUY", "WATCH")]
        
        logger.info(f"Market scan: {len(actionable)} actionable signals")
        return actionable

    # ═══════════════════════════════════════════════════════
    # EXIT SIGNALS — Monitor held positions
    # ═══════════════════════════════════════════════════════

    def check_exit_signals(
        self,
        held_tickers: List[str],
        lookback_days: int = 30,
    ) -> List[ExitSignal]:
        """
        Check for exit signals on held positions.

        Monitors insider SALES for each held ticker.
        Multiple insiders selling = stronger exit signal.

        Args:
            held_tickers: List of tickers you currently hold
            lookback_days: How far back to check for sales

        Returns:
            List of ExitSignals, sorted by urgency (HIGH first)
        """
        logger.info(f"Checking exit signals for {len(held_tickers)} positions...")
        exit_signals = []

        for ticker in held_tickers:
            try:
                sales = self.whale.fetch_insider_sales(ticker, days_back=lookback_days)

                if sales.empty:
                    exit_signals.append(ExitSignal(
                        ticker=ticker,
                        action="HOLD",
                        urgency="LOW",
                        insider_sales_30d=0,
                        insider_sales_value=0,
                        unique_sellers=0,
                        latest_sale_date=None,
                        reason="No insider sales detected",
                        timestamp=datetime.now().isoformat(),
                    ))
                    continue

                # Analyze sales
                n_sales = len(sales)
                total_value = float(sales["value"].sum()) if "value" in sales.columns else 0
                unique_sellers = sales["insider_name"].nunique() if "insider_name" in sales.columns else 0
                latest_date = str(sales["transaction_date"].max()) if "transaction_date" in sales.columns else None

                # Determine urgency
                action, urgency, reason = self._classify_exit(
                    n_sales, total_value, unique_sellers
                )

                signal = ExitSignal(
                    ticker=ticker,
                    action=action,
                    urgency=urgency,
                    insider_sales_30d=n_sales,
                    insider_sales_value=total_value,
                    unique_sellers=unique_sellers,
                    latest_sale_date=latest_date,
                    reason=reason,
                    timestamp=datetime.now().isoformat(),
                )
                exit_signals.append(signal)

                # Log
                self._log_signal("EXIT", signal.to_dict())

            except Exception as e:
                logger.warning(f"Exit check error for {ticker}: {e}")

        # Sort by urgency (EXIT > WATCH > HOLD)
        priority = {"EXIT": 0, "WATCH": 1, "HOLD": 2}
        exit_signals.sort(key=lambda s: priority.get(s.action, 3))

        return exit_signals

    # ═══════════════════════════════════════════════════════
    # FULL PORTFOLIO SIGNALS
    # ═══════════════════════════════════════════════════════

    def get_portfolio_signals(
        self,
        held_tickers: Optional[List[str]] = None,
        scan_days: int = 30,
    ) -> dict:
        """
        Complete portfolio analysis: new entries + exit checks.

        Returns:
            {
                "entries": [EntrySignal, ...],  # Buy signals, sorted by score
                "exits": [ExitSignal, ...],      # Exit signals, sorted by urgency
                "timestamp": "...",
            }
        """
        entries = self.scan_market(days_back=scan_days)
        exits = self.check_exit_signals(held_tickers or [])

        result = {
            "entries": [e.to_dict() for e in entries],
            "exits": [e.to_dict() for e in exits],
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_buy_signals": sum(1 for e in entries if e.action == "BUY"),
                "total_watch": sum(1 for e in entries if e.action == "WATCH"),
                "total_exit": sum(1 for e in exits if e.action == "EXIT"),
                "total_hold": sum(1 for e in exits if e.action == "HOLD"),
            },
        }

        return result

    # ═══════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════

    def _classify_score(self, score: float) -> tuple:
        """Classify model score into action and conviction."""
        if score >= 70:
            return "BUY", "HIGH"
        elif score >= 50:
            return "BUY", "MEDIUM"
        elif score >= 35:
            return "WATCH", "LOW"
        else:
            return "SKIP", "LOW"

    def _classify_exit(
        self, n_sales: int, total_value: float, unique_sellers: int
    ) -> tuple:
        """Classify exit signal urgency."""
        # Multiple insiders selling = strongest exit signal
        if unique_sellers >= 3 or (n_sales >= 5 and total_value > 1_000_000):
            return "EXIT", "HIGH", f"{unique_sellers} insiders sold ${total_value:,.0f}"
        elif unique_sellers >= 2 or n_sales >= 3:
            return "EXIT", "MEDIUM", f"{unique_sellers} insiders, {n_sales} sales"
        elif n_sales >= 1:
            return "WATCH", "LOW", f"{n_sales} sale(s), monitoring"
        else:
            return "HOLD", "LOW", "No insider sales detected"

    def _log_signal(self, signal_type: str, signal_data: dict):
        """Log signal for audit trail."""
        entry = {
            "type": signal_type,
            "data": signal_data,
            "logged_at": datetime.now().isoformat(),
        }
        self._signals.append(entry)

        # Auto-save every 10 signals
        if len(self._signals) % 10 == 0:
            self._save_signal_log()

    def _load_signal_log(self):
        """Load existing signal log."""
        if os.path.exists(self.signal_log_path):
            try:
                with open(self.signal_log_path, "r") as f:
                    self._signals = json.load(f)
            except Exception:
                self._signals = []

    def _save_signal_log(self):
        """Save signal log to disk."""
        try:
            with open(self.signal_log_path, "w") as f:
                json.dump(self._signals[-1000:], f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Signal log save error: {e}")
