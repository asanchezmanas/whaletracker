"""
Alternative Data Connectors — Phase 4

Fetches non-SEC alpha signals:
1. SBIR/STTR Grants (api.sbir.gov)
2. USPTO Patents (api.uspto.gov)
3. OTC Market Data
"""

import pandas as pd
import numpy as np
import httpx
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class AlternativeDataConnector:
    """
    Connects to grant and patent databases to find 'innovation alpha'.
    """

    def __init__(self):
        self.client = httpx.Client(timeout=30.0)

    def fetch_sbir_grants(self, company_name: str, years_back: int = 5) -> pd.DataFrame:
        """
        Fetch SBIR/STTR grants for a specific company name.
        """
        # Search by company name
        url = "https://www.sbir.gov/api/awards.json"
        
        # We try to find the company in the SBIR database
        # This API is broad, so we filter by awarded year for relevance
        cutoff_year = datetime.now().year - years_back
        
        params = {
            "keyword": company_name,
            "rows": 50
        }

        try:
            r = self.client.get(url, params=params)
            if r.status_code != 200:
                return pd.DataFrame()
            
            data = r.json()
            if not data or not isinstance(data, list):
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # Filter and cleanup
            if "award_year" in df.columns:
                df["award_year"] = pd.to_numeric(df["award_year"], errors="coerce")
                df = df[df["award_year"] >= cutoff_year]
                
            if "award_amount" in df.columns:
                df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce").fillna(0)
            
            # Map award_year to a pseudo-date if needed
            if "award_year" in df.columns:
                df["grant_date"] = pd.to_datetime(df["award_year"].astype(str) + "-01-01")
            
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_uspto_patents(self, company_name: str, days_back: int = 1095) -> pd.DataFrame:
        """
        Fetch recent patent grants via USPTO Open Data Portal or PatentsView.
        """
        # Note: PatentsView is more stable for general searches
        url = "https://api.patentsview.org/patents/query"
        
        date_cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Query: patents assigned to company_name granted after date_cutoff
        query = {
            "q": {
                "_and": [
                    {"_contains": {"assignee_organization": company_name}},
                    {"_gte": {"patent_date": date_cutoff}}
                ]
            },
            "f": ["patent_number", "patent_date", "patent_title"]
        }

        try:
            r = self.client.post(url, json=query)
            if r.status_code != 200:
                # Fallback: Check if common names or tickers exist in my local cache/mocks
                return pd.DataFrame()
            
            data = r.json()
            patents = data.get("patents", [])
            if not patents:
                return pd.DataFrame()
                
            df = pd.DataFrame(patents)
            if "patent_date" in df.columns:
                df["date"] = pd.to_datetime(df["patent_date"])
            
            return df
        except Exception:
            return pd.DataFrame()

    def get_otc_activity(self, ticker: str) -> Dict:
        """
        Specific activity metrics for OTC markets.
        """
        # Placeholder for low-liquidity market specific logic
        return {
            "is_otc": ticker.endswith(".PK") or ticker.endswith(".OB"),
            "liquidity_threat": False
        }

    def close(self):
        self.client.close()
