"""
QDN Signal Monitor

Background monitoring system that:
1. Polls OpenInsider every 6h for new purchases + sales
2. Polls SEC EDGAR daily for new 13D filings
3. Scores new opportunities through the model
4. Checks exit signals for held positions
5. Logs all signals to signals.json + console

Usage:
    monitor = QDNMonitor(
        checkpoint_path="checkpoints/best_model.pth",
        held_positions=["AAPL", "MSFT", "XYZ"],
    )
    monitor.run_once()         # Single scan
    monitor.run_continuous()   # Loop every 6h
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from .inference import QDNInference, EntrySignal, ExitSignal
from .config import QDNConfig

logger = logging.getLogger(__name__)


class QDNMonitor:
    """
    Production monitoring for the QDN whale tracking system.

    Continuously scans for:
    - New whale purchase signals (entry)
    - Insider sales in held positions (exit)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        held_positions: Optional[List[str]] = None,
        config: Optional[QDNConfig] = None,
        signal_log_path: str = "signals.json",
        poll_interval_hours: float = 6.0,
        min_score_alert: float = 50.0,
    ):
        self.held_positions = held_positions or []
        self.poll_interval = poll_interval_hours * 3600
        self.min_score_alert = min_score_alert
        self._last_scan = None

        self.inference = QDNInference(
            checkpoint_path=checkpoint_path,
            config=config,
            signal_log_path=signal_log_path,
        )

        logger.info(
            f"QDN Monitor initialized | "
            f"Positions: {len(self.held_positions)} | "
            f"Poll: {poll_interval_hours}h | "
            f"Min score: {min_score_alert}"
        )

    def add_position(self, ticker: str):
        """Add a ticker to the held positions watchlist."""
        ticker = ticker.upper()
        if ticker not in self.held_positions:
            self.held_positions.append(ticker)
            logger.info(f"Added {ticker} to watchlist ({len(self.held_positions)} total)")

    def remove_position(self, ticker: str):
        """Remove a ticker from the held positions watchlist."""
        ticker = ticker.upper()
        if ticker in self.held_positions:
            self.held_positions.remove(ticker)
            logger.info(f"Removed {ticker} from watchlist ({len(self.held_positions)} total)")

    def run_once(self, scan_days: int = 30) -> dict:
        """
        Run a single scan cycle:
        1. Scan market for new buy signals
        2. Check exit signals for held positions

        Returns:
            Full portfolio signals dict
        """
        logger.info("=" * 60)
        logger.info(f"QDN SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        logger.info("=" * 60)

        result = self.inference.get_portfolio_signals(
            held_tickers=self.held_positions,
            scan_days=scan_days,
        )

        self._last_scan = datetime.now()

        # Print summary
        self._print_results(result)

        return result

    def run_continuous(self, scan_days: int = 30):
        """
        Continuous monitoring loop. Scans every poll_interval hours.
        
        Press Ctrl+C to stop.
        """
        logger.info(f"Starting continuous monitoring (every {self.poll_interval/3600:.1f}h)")
        logger.info(f"Held positions: {self.held_positions}")
        logger.info("Press Ctrl+C to stop.\n")

        try:
            while True:
                self.run_once(scan_days=scan_days)

                next_scan = datetime.now() + timedelta(seconds=self.poll_interval)
                logger.info(f"\nNext scan: {next_scan.strftime('%Y-%m-%d %H:%M')}")
                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("\nMonitor stopped.")

    def _print_results(self, result: dict):
        """Print formatted scan results."""
        summary = result.get("summary", {})
        entries = result.get("entries", [])
        exits = result.get("exits", [])

        # Entry signals
        buy_signals = [e for e in entries if e.get("action") == "BUY"]
        watch_signals = [e for e in entries if e.get("action") == "WATCH"]

        if buy_signals:
            print(f"\n  🟢 BUY SIGNALS ({len(buy_signals)})")
            print(f"  {'Ticker':>8} {'Score':>6} {'Conv':>8} {'Type':>20} {'Insider':>20}")
            print(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*20} {'─'*20}")
            for s in buy_signals[:15]:
                print(
                    f"  {s['ticker']:>8} {s['score']:>6.1f} "
                    f"{s['conviction']:>8} {s['whale_type']:>20} "
                    f"{s['insider_name'][:20]:>20}"
                )
        else:
            print("\n  🔵 No buy signals")

        if watch_signals:
            print(f"\n  🟡 WATCH ({len(watch_signals)})")
            for s in watch_signals[:5]:
                print(f"    {s['ticker']} — score {s['score']:.1f} ({s['whale_type']})")

        # Exit signals
        exit_alerts = [e for e in exits if e.get("action") == "EXIT"]
        watch_exits = [e for e in exits if e.get("action") == "WATCH"]

        if exit_alerts:
            print(f"\n  🔴 EXIT SIGNALS ({len(exit_alerts)})")
            for s in exit_alerts:
                print(
                    f"    {s['ticker']} — {s['urgency']} urgency | "
                    f"{s['insider_sales_30d']} sales, "
                    f"{s['unique_sellers']} sellers | "
                    f"{s['reason']}"
                )
        elif self.held_positions:
            print(f"\n  🟢 All {len(self.held_positions)} positions HOLD")

        if watch_exits:
            print(f"  ⚠️  Watch: {', '.join(s['ticker'] for s in watch_exits)}")

        # Summary
        print(f"\n  Summary: {summary.get('total_buy_signals', 0)} buys, "
              f"{summary.get('total_watch', 0)} watch, "
              f"{summary.get('total_exit', 0)} exits, "
              f"{summary.get('total_hold', 0)} holds")
