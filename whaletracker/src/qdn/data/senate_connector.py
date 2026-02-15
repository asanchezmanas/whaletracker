"""
US Senate & House Stock Trade Connector

Fetches congressional stock trading data from public APIs.
Senate: Senate Stock Watcher (https://senatestockwatcher.com)
House: House Stock Watcher (https://housestockwatcher.com)

These are public datasets derived from STOCK Act disclosures.
"""

import httpx
import pandas as pd
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


SENATE_API_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
HOUSE_API_URL = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"


class SenateConnector:
    """
    Fetches US congressional stock trading data.
    
    Combines Senate and House disclosures into a unified format
    compatible with SEC Form 4 data.
    """

    def __init__(self, include_house: bool = True):
        self.include_house = include_house

    def fetch_all_transactions(self) -> pd.DataFrame:
        """
        Download all available congressional trading data.
        
        Returns:
            DataFrame with standardized columns matching SEC format:
            ticker, insider_name, insider_title, transaction_date,
            filing_date, transaction_code, shares, price, value,
            ownership_after, source
        """
        frames = []

        # Senate data
        senate_df = self._fetch_source(SENATE_API_URL, "senate")
        if not senate_df.empty:
            frames.append(senate_df)
            logger.info(f"Senate: {len(senate_df)} transactions")

        # House data
        if self.include_house:
            house_df = self._fetch_source(HOUSE_API_URL, "house")
            if not house_df.empty:
                frames.append(house_df)
                logger.info(f"House: {len(house_df)} transactions")

        if not frames:
            logger.warning("No congressional data fetched")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._clean_data(combined)
        logger.info(f"Total congressional transactions: {len(combined)}")
        return combined

    def _fetch_source(self, url: str, source: str) -> pd.DataFrame:
        """Fetch from a single data source."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            df = pd.DataFrame(data)

            # Standardize column names
            column_map = {
                # Senate format
                "senator": "insider_name",
                "ticker": "ticker",
                "transaction_date": "transaction_date",
                "disclosure_date": "filing_date",
                "type": "transaction_type_raw",
                "amount": "amount_range",
                "asset_description": "asset_description",
                # House format
                "representative": "insider_name",
                "disclosure_year": "disclosure_year",
            }

            df = df.rename(columns={
                k: v for k, v in column_map.items() if k in df.columns
            })

            # Set source
            df["source"] = source
            df["insider_title"] = "Senator" if source == "senate" else "Representative"

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {source} data: {e}")
            return pd.DataFrame()

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize congressional trading data."""
        if df.empty:
            return df

        # Parse dates
        for col in ["transaction_date", "filing_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows without essential fields
        df = df.dropna(subset=["ticker", "insider_name"])

        # Clean ticker
        df["ticker"] = df["ticker"].str.upper().str.strip()
        # Remove rows with invalid tickers
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]

        # Map transaction type to standard codes
        if "transaction_type_raw" in df.columns:
            type_map = {
                "Purchase": "P",
                "Sale": "S",
                "Sale (Full)": "S",
                "Sale (Partial)": "S",
                "Exchange": "M",
            }
            df["transaction_code"] = df["transaction_type_raw"].map(type_map)
            df = df.dropna(subset=["transaction_code"])

        # Estimate value from amount range
        # Senate discloses ranges like "$1,001 - $15,000"
        if "amount_range" in df.columns:
            df["value"] = df["amount_range"].apply(_parse_amount_range)

        # Estimated shares (value / assumed avg price = placeholder)
        df["shares"] = None
        df["price"] = None
        df["ownership_after"] = None

        # Sort
        df = df.sort_values("transaction_date").reset_index(drop=True)

        return df


def _parse_amount_range(amount_str: str) -> Optional[float]:
    """
    Parse Senate amount range to midpoint estimate.
    
    Example: "$1,001 - $15,000" → 8000.5
    """
    if not isinstance(amount_str, str):
        return None

    try:
        # Remove $ and commas
        clean = amount_str.replace("$", "").replace(",", "").strip()
        
        if " - " in clean:
            parts = clean.split(" - ")
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        
        # Single value
        return float(clean)
    except (ValueError, IndexError):
        return None
