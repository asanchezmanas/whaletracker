"""
QDN Pipeline Orchestrator

Coordinates the full data flow:
1. Fetch insider transactions (SEC + Senate)
2. Fetch market data for each ticker
3. Compute features for each transaction
4. Label with Triple Barrier or forward returns (for training)
5. Store in Supabase

Designed to run as:
- Daily batch job (fetch recent data, compute features, predict)
- Historical backfill (process all past data for initial training)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List

from .data.sec_connector import SECConnector
from .data.senate_connector import SenateConnector
from .data.market_connector import MarketConnector
from .features.engineer import FeatureEngineer
from .labeling.triple_barrier import TripleBarrierLabeler

logger = logging.getLogger(__name__)


class QDNPipeline:
    """
    End-to-end data pipeline from raw sources to model-ready features.
    
    Usage:
        pipeline = QDNPipeline()
        features, labels, dates = pipeline.run_historical_backfill()
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        sec_user_agent: str = "WhaleTracker QDN research@whaletracker.dev",
    ):
        self.sec = SECConnector(user_agent=sec_user_agent)
        self.senate = SenateConnector(include_house=True)
        self.market = MarketConnector(fred_api_key=fred_api_key)
        self.engineer = FeatureEngineer()

    def fetch_all_transactions(
        self,
        sec_days_back: int = 365,
        include_senate: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch and merge transactions from all sources.
        
        Returns:
            Unified DataFrame with columns: ticker, insider_name,
            insider_title, transaction_date, filing_date,
            transaction_code, shares, price, value, ownership_after, source
        """
        frames = []

        # SEC Form 4
        logger.info("Fetching SEC Form 4 filings...")
        sec_df = self.sec.fetch_recent_filings(days_back=sec_days_back)
        if not sec_df.empty:
            frames.append(sec_df)
            logger.info(f"SEC: {len(sec_df)} transactions")

        # Congressional trades
        if include_senate:
            logger.info("Fetching congressional trading data...")
            senate_df = self.senate.fetch_all_transactions()
            if not senate_df.empty:
                frames.append(senate_df)
                logger.info(f"Congress: {len(senate_df)} transactions")

        if not frames:
            logger.error("No data fetched from any source")
            return pd.DataFrame()

        # Unify
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("transaction_date").reset_index(drop=True)

        logger.info(f"Total transactions: {len(combined)}")
        return combined

    def compute_features_and_labels(
        self,
        transactions: pd.DataFrame,
        return_horizon_days: int = 180,
        use_triple_barrier: bool = False,
    ) -> Dict:
        """
        Compute features and forward-return labels for all transactions.
        
        Args:
            transactions: Unified transaction DataFrame
            return_horizon_days: Days for label computation (default: 6 months)
        
        Returns:
            Dict with features (np.ndarray), labels (np.ndarray),
            dates (np.ndarray), tickers (list), metadata (DataFrame)
        """
        logger.info("Computing features and labels...")

        # Unique tickers
        tickers = transactions["ticker"].unique()
        logger.info(f"Processing {len(tickers)} unique tickers")

        # Fetch company info for all tickers
        company_info_map = {}
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"Fetching company info: {i}/{len(tickers)}")
            company_info_map[ticker] = self.market.get_company_info(ticker)

        # Fetch market data
        market_data_map = {}
        earliest = transactions["transaction_date"].min()
        start = (earliest - timedelta(days=120)).strftime("%Y-%m-%d")

        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"Fetching market data: {i}/{len(tickers)}")
            market_data_map[ticker] = self.market.get_stock_prices(
                ticker, start
            )

        # Compute features
        features_list = []
        labels_list = []
        dates_list = []
        valid_indices = []

        for idx, row in transactions.iterrows():
            ticker = row.get("ticker", "")
            tx_date = pd.to_datetime(row.get("transaction_date"))

            if pd.isna(tx_date):
                continue

            # Compute features
            try:
                features = self.engineer.compute_features(
                    transaction=row.to_dict(),
                    historical_transactions=transactions[
                        transactions.index < idx
                    ],
                    company_info=company_info_map.get(ticker, {}),
                    market_data=market_data_map.get(ticker, pd.DataFrame()),
                    macro_snapshot={},  # FRED needs API key
                )
            except Exception as e:
                logger.debug(f"Feature error for {ticker} on {tx_date}: {e}")
                continue

            # Compute label (forward return — unless triple barrier)
            if not use_triple_barrier:
                returns = self.market.get_returns(
                    ticker,
                    tx_date.strftime("%Y-%m-%d"),
                    horizons=[return_horizon_days],
                )
                label = returns.get(f"return_{return_horizon_days}d")

                if label is None:
                    continue

            features_list.append(features)
            labels_list.append(label if not use_triple_barrier else 0.0)
            dates_list.append(tx_date)
            valid_indices.append(idx)

        if not features_list:
            logger.error("No valid features computed")
            return {"features": np.array([]), "labels": np.array([])}

        features_array = np.array(features_list, dtype=np.float32)
        dates_array = np.array(dates_list)
        valid_tx = transactions.loc[valid_indices].reset_index(drop=True)

        # --- Triple Barrier labeling (batch) ---
        if use_triple_barrier:
            logger.info("Applying Triple Barrier labeling...")
            labeler = TripleBarrierLabeler(
                max_holding_days=return_horizon_days,
            )
            tb_labels, tb_returns, tb_events = labeler.apply(
                valid_tx, market_data_map
            )

            # Filter to successfully labeled samples
            n_labeled = len(tb_labels)
            if n_labeled < len(features_array):
                features_array = features_array[:n_labeled]
                dates_array = dates_array[:n_labeled]
                valid_tx = valid_tx.iloc[:n_labeled]

            labels_array = tb_returns  # Realized returns at barrier
            sample_weights = labeler.compute_sample_weights(tb_events)
            event_ends = np.array([e.exit_date for e in tb_events])
        else:
            labels_array = np.array(labels_list, dtype=np.float32)
            sample_weights = None
            event_ends = None

        logger.info(
            f"Pipeline complete: {len(features_array)} samples, "
            f"{features_array.shape[1]} features"
        )

        result = {
            "features": features_array,
            "labels": labels_array,
            "dates": dates_array,
            "tickers": valid_tx["ticker"].tolist(),
            "metadata": valid_tx,
        }

        if use_triple_barrier:
            result["sample_weights"] = sample_weights
            result["event_ends"] = event_ends
            result["barrier_events"] = tb_events

        return result

    def run_historical_backfill(
        self,
        sec_days_back: int = 365 * 3,
    ) -> Dict:
        """
        Full historical backfill: fetch → features → labels.
        
        This is the main entry point for initial data preparation.
        """
        transactions = self.fetch_all_transactions(
            sec_days_back=sec_days_back,
        )

        if transactions.empty:
            return {"features": np.array([]), "labels": np.array([])}

        return self.compute_features_and_labels(transactions)
