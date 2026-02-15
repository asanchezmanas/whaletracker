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

logger = logging.getLogger(__name__)


# Structural Feature Names (30 Unified)
FEATURE_NAMES = [
    # 1. SOVEREIGN (5) - Logic of the State
    "sov_contract_velocity",   # Awards per year
    "sov_backlog_ratio",      # Backlog vs Market Cap
    "sov_outlier_award",      # Binary: significant transformative award detected
    "sov_political_alignment", # Senator oversight of sector
    "sov_seniority_score",    # Political seniority in network
    
    # 2. IP-MONOPOLY (5) - Logic of the Moat
    "ip_patent_velocity",     # New patents per year
    "ip_rd_efficiency",       # R&D spend vs patent output
    "ip_recent_grant",        # SBIR/STTR grants (Sovereign/IP overlap)
    "ip_moat_growth",         # Year-over-year IP expansion
    "ip_tech_category",       # Encoded sector criticality (Energy/Bio/Defense)
    
    # 3. NETWORK-FORCE (5) - Logic of Influence
    "net_whale_momentum",     # 13F/Form 4 cluster intensity
    "net_cluster_quality",    # Diversity of insiders/whales in cluster
    "net_corporate_king_maker", # Global 500 entity participation
    "net_institutional_drift", # Early micro-cap institutional entry
    "net_insider_conviction", # Buy size relative to personal ownership
    
    # 4. COMPANY FLOOR (5) - Logic of Fragility (Via Negativa)
    "co_log_market_cap",      # Scale (Small = higher optionality)
    "co_volatility_90d",      # Structural volatility (not 1d noise)
    "co_short_interest_pct",  # Potential squeeze convexity
    "co_volume_anomaly",      # Structural accumulation signature
    "co_cash_runway_est",     # Anti-fragility (months of life)
    
    # 5. MACRO FLOOR (5) - Logic of the Environment
    "mac_vix_normalized",     # Market fear (Fear = Buy seeds)
    "mac_yield_spread",       # Recession probability floor
    "mac_dxy_momentum",       # Dollar strength context
    "mac_days_since_crash",   # Distance from Black Swan events
    "mac_tail_risk_index",    # Global tail risk level
    
    # 6. INSIDER HISTORY (5) - Logic of Truth
    "his_insider_win_rate",   # Historical reliability of the observer
    "his_insider_frequency",  # Data density for this source
    "his_insider_consistency",# Directional bias
    "his_avg_holding_period", # Time-horizon of the insider
    "his_filing_delay",       # Strategic silence score
]
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
        from ..data.gov_connector import GovContractConnector
        from ..data.innovation_connector import InnovationConnector
        from ..data.whale_connector import WhaleConnector
        self.gov = GovContractConnector()
        self.innovation = InnovationConnector()
        self.whale = WhaleConnector()
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
        Compute the 30 structural features for a single transaction.
        """
        features = np.zeros(self.n_features, dtype=np.float32)

        tx_date = pd.to_datetime(transaction.get("transaction_date"))
        ticker = transaction.get("ticker", "")
        insider = transaction.get("insider_name", "")

        # 1. SOVEREIGN (5)
        gov_data = self.gov.get_company_contract_velocity(ticker, self.gov.resolve_recipient_name(ticker) or "")
        backlog = self.gov.get_backlog_estimate(ticker)
        features[0] = gov_data["velocity_score"]
        features[1] = backlog["backlog_value"] / max(company_info.get("market_cap", 1e8), 1e6)
        features[2] = 1.0 if self.gov.get_outlier_awards(ticker) else 0.0
        features[3] = self._committee_alignment(transaction, company_info)
        features[4] = self._seniority_score(transaction)

        # 2. IP-MONOPOLY (5)
        patent_data = self.innovation.get_patent_velocity(ticker)
        grant_data = self.innovation.get_grant_intensity(ticker)
        
        features[5] = patent_data["velocity"]
        features[6] = grant_data["intensity"]
        features[7] = 1.0 if grant_data["has_grants"] else 0.0
        features[8] = patent_data["patent_count"] / 50.0 # Normalized count
        features[9] = self._encode_criticality(company_info.get("sector", ""))

        # 3. NETWORK-FORCE (5)
        features[10] = self.whale.get_cluster_quality(ticker, window_days=30)
        features[11] = self.whale.get_cluster_quality(ticker, window_days=90) # Broader quality
        features[12] = self.whale.is_corporate_king_maker(ticker)
        features[13] = self.whale.get_institutional_drift(ticker)
        features[14] = self._trade_size_vs_ownership(transaction)

        # 4. COMPANY FLOOR (5)
        features[15] = self._log_market_cap(company_info)
        features[16] = self._volatility_90d(market_data, tx_date)
        features[17] = self._short_interest(company_info)
        features[18] = self._volume_anomaly(market_data, tx_date)
        features[19] = company_info.get("cash_runway_months", 12) / 24.0 # Scale to 2 years

        # 5. MACRO FLOOR (5)
        features[20] = self._vix_normalized(macro_snapshot)
        features[21] = self._yield_curve(macro_snapshot)
        features[22] = self._dxy_momentum(macro_snapshot)
        features[23] = self._days_since_crash(market_data, tx_date)
        features[24] = macro_snapshot.get("tail_risk_index", 0.5)

        # 6. INSIDER HISTORY (5)
        insider_hist = historical_transactions[
            (historical_transactions["insider_name"] == insider) & 
            (historical_transactions["transaction_date"] < tx_date)
        ] if not historical_transactions.empty else pd.DataFrame()
        
        features[25] = self._insider_win_rate(insider_hist)
        features[26] = self._insider_frequency(insider_hist, tx_date)
        features[27] = self._insider_consistency(insider_hist)
        features[28] = self._insider_avg_holding(insider_hist)
        features[29] = self._filing_delay(transaction)

        return np.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0)

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

    # ─────────────────────────────────────────────
    # INSIDER FEATURES (5)
    # ─────────────────────────────────────────────

    @staticmethod
    def _insider_win_rate(history: pd.DataFrame) -> float:
        if history.empty: return 0.5
        purchases = history[history["transaction_code"] == "P"]
        if purchases.empty or "actual_return_6m" not in purchases.columns: return 0.5
        return float((purchases["actual_return_6m"] > 0).mean())

    @staticmethod
    def _insider_frequency(history: pd.DataFrame, tx_date: pd.Timestamp) -> float:
        if len(history) < 2: return 0.0
        span = (tx_date - pd.to_datetime(history["transaction_date"]).min()).days
        return float(np.log1p(len(history) / (max(span, 30) / 365.25)))

    @staticmethod
    def _insider_consistency(history: pd.DataFrame) -> float:
        if history.empty: return 0.5
        return float((history["transaction_code"] == "P").mean())

    @staticmethod
    def _insider_avg_holding(history: pd.DataFrame) -> float:
        # Simplified placeholder for structural intent
        return 0.5

    @staticmethod
    def _trade_size_vs_ownership(tx: Dict) -> float:
        return float(min(abs(tx.get("shares", 0)) / tx.get("ownership_after", 1e9), 1.0))

    # ─────────────────────────────────────────────
    # TRANSACTION FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _filing_delay(tx: Dict) -> float:
        delay = (pd.to_datetime(tx.get("filing_date")) - pd.to_datetime(tx.get("transaction_date"))).days
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

        if history.empty or "ticker" not in history.columns or "value" not in history.columns:
            return 0.0

        ticker_hist = history[history["ticker"] == ticker]
        if ticker_hist.empty:
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
    def _days_since_crash(market_data: pd.DataFrame, tx_date: pd.Timestamp) -> float:
        # Simplified tail-risk distance
        return 1.0

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
    def _log_market_cap(info: Dict) -> float:
        log_mc = np.log10(info.get("market_cap", 1e8))
        return float(np.clip((log_mc - 6) / 6, 0, 1))

    @staticmethod
    def _volatility_90d(market_data: pd.DataFrame, tx_date: pd.Timestamp) -> float:
        if market_data.empty: return 0.3
        recent = market_data.loc[:tx_date].tail(90)
        if len(recent) < 20: return 0.3
        return float(min(recent["Close"].pct_change().std() * np.sqrt(252), 1.0))

    @staticmethod
    def _short_interest(info: Dict) -> float:
        return float(min(info.get("short_percent_of_float", 0), 0.5) / 0.5)

    @staticmethod
    def _volume_anomaly(market_data: pd.DataFrame, tx_date: pd.Timestamp) -> float:
        if market_data.empty: return 0.0
        recent = market_data.loc[:tx_date].tail(30)
        if len(recent) < 10: return 0.0
        ratio = recent["Volume"].iloc[-1] / recent["Volume"].iloc[:-1].mean()
        return float(np.clip(np.log(ratio), -2, 2) / 2)

    # ─────────────────────────────────────────────
    # NETWORK-FORCE FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _network_centrality(info: Dict) -> float:
        return float(np.clip(info.get("network_centrality", 0), 0, 1))

    @staticmethod
    def _social_sentiment(info: Dict) -> float:
        return float(np.clip((info.get("social_sentiment", 0) + 1) / 2, 0, 1))

    @staticmethod
    def _analyst_coverage(info: Dict) -> float:
        return float(np.clip(info.get("analyst_coverage", 0) / 20.0, 0, 1))

    # ─────────────────────────────────────────────
    # MACRO FEATURES (3)
    # ─────────────────────────────────────────────

    @staticmethod
    def _vix_normalized(macro: Dict) -> float:
        return float(np.clip((macro.get("vix", 20) - 10) / 70, 0, 1))

    @staticmethod
    def _yield_curve(macro: Dict) -> float:
        return float(np.clip((macro.get("yield_curve", 0) + 1) / 4, 0, 1))

    @staticmethod
    def _dxy_momentum(macro: Dict) -> float:
        return float(np.clip((macro.get("dxy", 100) - 80) / 40, 0, 1))

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
    def _committee_alignment(tx: Dict, info: Dict) -> float:
        return 1.0 if tx.get("has_committee_alignment") else 0.0

    @staticmethod
    def _seniority_score(tx: Dict) -> float:
        return float(min(tx.get("seniority_years", 0) / 30.0, 1.0))

    # ─────────────────────────────────────────────
    # PHASE 3 & 4 HELPERS
    # ─────────────────────────────────────────────

    @staticmethod
    def _encode_criticality(sector: str) -> float:
        critical = ["Healthcare", "Technology", "Industrials", "Energy"]
        return 1.0 if sector in critical else 0.5

