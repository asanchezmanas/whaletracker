"""
Market Data Connector

Fetches market data from free sources:
1. yfinance — stock prices, volume, company info
2. FRED (Federal Reserve) — macro indicators (VIX, yield curve, DXY)

All data sources are free and public.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import httpx
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# FRED API (free, requires API key from https://fred.stlouisfed.org/docs/api/)
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series IDs for macro indicators
FRED_SERIES = {
    "vix": "VIXCLS",             # CBOE VIX
    "yield_curve": "T10Y2Y",     # 10Y-2Y Treasury spread
    "dxy": "DTWEXBGS",           # Trade-weighted USD index
    "fed_funds": "FEDFUNDS",     # Federal funds rate
    "unemployment": "UNRATE",     # Unemployment rate
    "cpi": "CPIAUCSL",           # Consumer Price Index
}


class MarketConnector:
    """
    Fetches market data and macro indicators.
    
    Stock data via yfinance (free).
    Macro data via FRED API (free with API key).
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key

    # ─────────────────────────────────────────────
    # Stock Data (yfinance)
    # ─────────────────────────────────────────────

    def get_stock_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a ticker.
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
            )
            if data.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return data
        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            return pd.DataFrame()

    def get_company_info(self, ticker: str) -> Dict:
        """
        Fetch company fundamental data.
        
        Returns:
            Dict with: sector, industry, market_cap, float_shares,
            short_ratio, avg_volume, beta, etc.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "float_shares": info.get("floatShares"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "avg_volume": info.get("averageVolume"),
                "avg_volume_10d": info.get("averageDailyVolume10Day"),
                "beta": info.get("beta"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": info.get("dividendYield"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "exchange": info.get("exchange"),
            }
        except Exception as e:
            logger.error(f"Failed to fetch info for {ticker}: {e}")
            return {"ticker": ticker}

    def get_returns(
        self,
        ticker: str,
        transaction_date: str,
        horizons: List[int] = [30, 90, 180, 365],
    ) -> Dict[str, Optional[float]]:
        """
        Calculate forward returns at multiple horizons from a transaction date.
        
        Args:
            ticker: Stock ticker
            transaction_date: Date to measure from
            horizons: List of day horizons (e.g., [30, 90, 180, 365])
        
        Returns:
            Dict like {"return_30d": 0.05, "return_90d": 0.12, ...}
        """
        start = pd.to_datetime(transaction_date)
        end = start + timedelta(days=max(horizons) + 10)

        prices = self.get_stock_prices(
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )

        if prices.empty:
            return {f"return_{h}d": None for h in horizons}

        # Get price at transaction date (or nearest available)
        try:
            close = prices["Close"]
            base_price = close.iloc[0]  # First available price

            results = {}
            for h in horizons:
                target_date = start + timedelta(days=h)
                # Find nearest available date
                future = close[close.index >= target_date]
                if not future.empty:
                    future_price = future.iloc[0]
                    results[f"return_{h}d"] = float(
                        (future_price - base_price) / base_price
                    )
                else:
                    results[f"return_{h}d"] = None

            return results
        except (IndexError, KeyError) as e:
            logger.debug(f"Return calc error for {ticker}: {e}")
            return {f"return_{h}d": None for h in horizons}

    def get_volatility(
        self, ticker: str, date: str, lookback_days: int = 90
    ) -> Optional[float]:
        """
        Compute historical annualized volatility.
        
        Returns:
            Annualized volatility as decimal (e.g., 0.35 = 35%)
        """
        end = pd.to_datetime(date)
        start = end - timedelta(days=lookback_days + 10)

        prices = self.get_stock_prices(
            ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

        if prices.empty or len(prices) < 20:
            return None

        try:
            returns = prices["Close"].pct_change().dropna()
            return float(returns.std() * np.sqrt(252))
        except Exception:
            return None

    # ─────────────────────────────────────────────
    # Macro Data (FRED)
    # ─────────────────────────────────────────────

    def get_macro_indicator(
        self,
        indicator: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetch a macro indicator from FRED.
        
        Args:
            indicator: One of: vix, yield_curve, dxy, fed_funds, unemployment, cpi
            start_date: Start date string
            end_date: End date string (default: today)
        
        Returns:
            Series indexed by date
        """
        if not self.fred_api_key:
            logger.warning("No FRED API key set. Macro data unavailable.")
            return pd.Series(dtype=float)

        series_id = FRED_SERIES.get(indicator)
        if not series_id:
            logger.error(f"Unknown indicator: {indicator}")
            return pd.Series(dtype=float)

        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        try:
            params = {
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.get(FRED_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

            observations = data.get("observations", [])
            dates = []
            values = []
            for obs in observations:
                try:
                    dates.append(pd.to_datetime(obs["date"]))
                    val = obs["value"]
                    values.append(float(val) if val != "." else np.nan)
                except (ValueError, KeyError):
                    continue

            series = pd.Series(values, index=dates, name=indicator)
            return series.dropna()

        except Exception as e:
            logger.error(f"FRED fetch failed for {indicator}: {e}")
            return pd.Series(dtype=float)

    def get_macro_snapshot(self, date: str) -> Dict[str, Optional[float]]:
        """
        Get a snapshot of all macro indicators at a given date.
        
        Returns the most recent available value for each indicator
        on or before the specified date.
        """
        start = (pd.to_datetime(date) - timedelta(days=30)).strftime("%Y-%m-%d")
        snapshot = {}

        for name in FRED_SERIES:
            series = self.get_macro_indicator(name, start, date)
            if not series.empty:
                snapshot[name] = float(series.iloc[-1])
            else:
                snapshot[name] = None

        return snapshot
