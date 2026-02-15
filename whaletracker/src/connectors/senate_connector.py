import httpx
import pandas as pd
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SenateConnector:
    """Connector to fetch US Senate stock trade data from SenateStockWatcher."""
    
    BASE_URL = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"

    def fetch_all_transactions(self) -> pd.DataFrame:
        """
        Download all historical Senate transactions from the aggregate source.
        Returns a cleaned DataFrame.
        """
        logger.info(f"Downloading historical records from {self.BASE_URL}")
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.BASE_URL)
                response.raise_for_status()
                data = response.json()
                
                df = pd.DataFrame(data)
                logger.info(f"Downloaded {len(df)} records.")
                return self._clean_data(df)
        except Exception as e:
            logger.error(f"Failed to fetch Senate data: {e}")
            return pd.DataFrame()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning of the raw JSON data for the backtest engine.
        Required columns: [insider, ticker, type, trade_date, report_date]
        """
        # Mapping columns to our internal standard
        # SenateStockWatcher typically uses: 'senator', 'ticker', 'type', 'transaction_date', 'disclosure_date'
        col_map = {
            'senator': 'insider',
            'ticker': 'ticker',
            'type': 'type',
            'transaction_date': 'trade_date',
            'disclosure_date': 'report_date'
        }
        
        # Check available columns and rename
        df = df.rename(columns=col_map)
        
        # Keep only relevant columns and remove null tickers
        cols_to_keep = ['insider', 'ticker', 'type', 'trade_date', 'report_date']
        df = df[df['ticker'].notna() & (df['ticker'] != '--')][cols_to_keep]
        
        # Convert types
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        
        # Drop invalid dates
        df = df.dropna(subset=['trade_date', 'report_date'])
        
        return df

if __name__ == "__main__":
    connector = SenateConnector()
    df = connector.fetch_all_transactions()
    if not df.empty:
        print(df.head())
        print(f"\nSectors summary:\n{df['type'].value_counts()}")
