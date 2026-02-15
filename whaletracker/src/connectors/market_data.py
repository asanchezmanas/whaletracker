import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataConnector:
    """Connector to fetch historical market data using yfinance."""
    
    def get_historical_prices(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical daily prices for a given ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'NVDA').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format (default is today).
            
        Returns:
            DataFrame with Open, High, Low, Close, Volume.
        """
        if end_date is None:
            end_date = datetime.today().strftime('%YYYY-%MM-%DD')
            
        logger.info(f"Fetching prices for {ticker} from {start_date} to {end_date}")
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test for NVDA (The Pelosi Case)
    connector = MarketDataConnector()
    df = connector.get_historical_prices("NVDA", "2023-11-01", "2024-02-01")
    print(df.head())
