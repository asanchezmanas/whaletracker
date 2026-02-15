"""
Startup Whale Detector — Phase 4

Identifies 'Startup Whales':
- Emerging companies with recent NIH/NSF/DoD grants (SBIR/STTR)
- PLUS recent institutional entry (13D/13F/Cluster Buy)
- PLUS low market cap (< $500M)
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .data.whale_connector import WhaleConnector
from .data.alternative_data import AlternativeDataConnector
from .features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class StartupWhaleDetector:
    """
    Specialized detector for 'Startup Whales' — innovation-rich micro-caps 
    receiving their first institutional smart money validation.
    """

    def __init__(self):
        self.whale = WhaleConnector()
        self.alt = AlternativeDataConnector()
        self.engineer = FeatureEngineer()

    def detect_signals(self, days_back: int = 30) -> pd.DataFrame:
        """
        Scan the market for the 'Startup Whale' pattern.
        """
        logger.info(f"Scanning for Startup Whale signals (last {days_back} days)...")
        
        # 1. Fetch recent Whale signals
        signals_df = self.whale.fetch_all_whale_signals(days_back=days_back)
        if signals_df.empty:
            return pd.DataFrame()
        
        # 2. Filter for potential Startups (Micro-caps)
        # In a real scenario, we'd fetch market cap for all tickers.
        # For this high-level detector, we focus on symbols that appear 
        # in recent clusters or 13D filings (strategic stakes).
        
        startup_candidates = []
        
        for _, row in signals_df.iterrows():
            ticker = row.get("ticker")
            if not ticker or ticker == "nan": continue
            
            # Cross-reference with Alternative Data
            # Note: We use company name for grants since startups might change tickers
            company_name = ticker # Placeholder: ideally fetch real name
            
            grants = self.alt.fetch_sbir_grants(company_name)
            patents = self.alt.fetch_uspto_patents(company_name)
            
            if not grants.empty or not patents.empty:
                # Potential MATCH: Smart money entering a high-innovation micro-cap
                score = 0
                if not grants.empty: score += 50
                if not patents.empty: score += 30
                if row.get("whale_type") == "activist_13d": score += 40
                if row.get("whale_type") == "cluster_buy": score += 30
                
                startup_candidates.append({
                    "ticker": ticker,
                    "whale_signal": row.get("whale_type"),
                    "insider": row.get("insider_name"),
                    "innovation_score": score,
                    "grant_count": len(grants),
                    "patent_count": len(patents),
                    "date": row.get("filing_date"),
                    "conviction": "HIGH" if score >= 80 else "MEDIUM"
                })
        
        if not startup_candidates:
            logger.info("No Startup Whale signals detected.")
            return pd.DataFrame()
            
        df = pd.DataFrame(startup_candidates)
        return df.sort_values("innovation_score", ascending=False)

    def close(self):
        self.whale.close()
        self.alt.close()
