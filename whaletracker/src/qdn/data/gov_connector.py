"""
USASpending.gov Connector (Sovereign Brain)

Fetches federal contract awards for companies.
Provides the "State Floor" signal for the Sovereign Brain QDN.

API Reference: https://api.usaspending.gov/docs/endpoints
"""
import httpx
import pandas as pd
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from ..utils.rate_limiter import RateLimiter
from .market_connector import MarketConnector

logger = logging.getLogger(__name__)

# USASpending API Base
USASPENDING_API = "https://api.usaspending.gov/api/v2"

class GovContractConnector:
    """
    Interfaces with USASpending.gov to track "Father State" backing.
    """

    def __init__(self, market_connector: Optional[MarketConnector] = None):
        self._limiter = RateLimiter.get_limiter("usaspending", 5.0) # Conservative 5 req/s
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)
        self.market = market_connector or MarketConnector()
        self._name_map = {} # Cache for ticker -> recipient_name

    def search_recipients_via_awards(self, query: str) -> List[str]:
        """
        Search for recipient names by looking at actual award data.
        More reliable than the dedicated search endpoint.
        """
        self._limiter.wait()
        url = f"{USASPENDING_API}/search/spending_by_award/"
        payload = {
            "filters": {"recipient_search_text": [query], "time_period": [{"start_date": "2020-01-01", "end_date": datetime.now().strftime("%Y-%m-%d")}]},
            "fields": ["Recipient Name"],
            "limit": 5
        }
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            return list(set(r.get("Recipient Name") for r in results if r.get("Recipient Name")))
        except Exception as e:
            logger.warning(f"USASpending award-search failed for {query}: {e}")
            return []

    def resolve_recipient_name(self, ticker: str) -> Optional[str]:
        """
        Attempt to resolve a ticker to a USASpending recipient name.
        """
        if ticker in self._name_map:
            return self._name_map[ticker]

        info = self.market.get_company_info(ticker)
        full_name = info.get("name")
        if not full_name:
            return None

        # Clean name: remove (INC, CORP, LTD, etc.)
        clean_name = re.sub(r'\b(INC|CORP|LTD|LLC|CORPORATION|INCORPORATED|PLC|S\.A\.|NV)\b\.?', '', full_name, flags=re.IGNORECASE).strip()
        
        # Search via awards
        names = self.search_recipients_via_awards(clean_name)
        if not names:
            short_name = " ".join(clean_name.split()[:2])
            names = self.search_recipients_via_awards(short_name)

        if names:
            # Pick the most frequent or first match
            best_match = names[0]
            self._name_map[ticker] = best_match
            logger.info(f"Resolved {ticker} -> {best_match}")
            return best_match

        return None

    def fetch_awards_by_ticker(self, ticker: str, days_back: int = 365) -> pd.DataFrame:
        """Fetch awards using a ticker symbol instead of recipient name."""
        recipient = self.resolve_recipient_name(ticker)
        if not recipient:
            return pd.DataFrame()
        return self.fetch_awards_by_recipient(recipient, days_back=days_back)

    def fetch_awards_by_recipient(
        self, 
        recipient_name: str, 
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch contract awards for a specific recipient name.
        Note: Recipient names in USASpending are often UPPERCASE 
        and may include inc, corp, etc.
        """
        self._limiter.wait()
        
        url = f"{USASPENDING_API}/search/spending_by_award/"
        
        # Look for awards starting from X days ago
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        payload = {
            "filters": {
                "recipient_search_text": [recipient_name],
                "time_period": [{"start_date": start_date, "end_date": datetime.now().strftime("%Y-%m-%d")}],
                "award_type_codes": ["A", "B", "C", "D"] # Basic contracts
            },
            "fields": [
                "Award ID", 
                "Recipient Name", 
                "Start Date", 
                "End Date", 
                "Award Amount", 
                "Awarding Agency",
                "Description"
            ],
            "limit": 100,
            "page": 1
        }

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return pd.DataFrame()
                
            df = pd.DataFrame(results)
            return df
        except Exception as e:
            logger.error(f"USASpending fetch failed for {recipient_name}: {e}")
            return pd.DataFrame()

    def get_company_contract_velocity(self, ticker: str, recipient_name: str) -> Dict[str, Any]:
        """
        Computes the 'Contract Velocity' feature for a company.
        High velocity = Sovereign Brain signal.
        """
        df = self.fetch_awards_by_recipient(recipient_name)
        if df.empty:
            return {
                "ticker": ticker,
                "total_awards_1y": 0,
                "total_value_1y": 0,
                "velocity_score": 0.0
            }

        # Calculate metrics
        total_value = df["Award Amount"].sum()
        total_awards = len(df)
        
        # Simple velocity: value per award or frequency
        return {
            "ticker": ticker,
            "recipient": recipient_name,
            "total_awards_1y": total_awards,
            "total_value_1y": float(total_value),
            "velocity_score": float(total_value / 1e6) # Normalized to millions
        }

    def get_outlier_awards(self, ticker: str, threshold_mult: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detects 'Structural Gem' signals: Awards that are significantly 
        larger than the historical average for this company.
        """
        df = self.fetch_awards_by_ticker(ticker, days_back=730) # 2y lookback
        if df.empty or len(df) < 3:
            return []
            
        avg_award = df["Award Amount"].mean()
        outliers = df[df["Award Amount"] > avg_award * threshold_mult]
        
        return outliers.to_dict("records")

    def get_backlog_estimate(self, ticker: str) -> Dict[str, Any]:
        """
        Estimates the 'Sovereign Floor' by looking at awards with 
        future end dates (Backlog).
        """
        df = self.fetch_awards_by_ticker(ticker, days_back=1000)
        if df.empty:
            return {"backlog_value": 0, "active_awards": 0}
            
        now = datetime.now().strftime("%Y-%m-%d")
        active = df[df["End Date"] > now]
        
        return {
            "backlog_value": float(active["Award Amount"].sum()),
            "active_awards": len(active),
            "ticker": ticker
        }
