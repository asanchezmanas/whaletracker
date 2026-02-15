import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from connectors.senate_connector import SenateConnector
from connectors.market_data import MarketDataConnector
from engines.backtest_engine import BacktestEngine

def run_test():
    print("--- WhaleTracker: Starting The Mother Test (Pelosi Case) ---")
    
    # 1. Fetch Senate Trades
    senate = SenateConnector()
    all_trades = senate.fetch_all_transactions()
    
    # Filter for Nancy Pelosi (or common variations)
    pelosi_trades = all_trades[all_trades['insider'].str.contains("Pelosi", case=False, na=False)]
    print(f"Found {len(pelosi_trades)} trades for Pelosi.")
    
    # 2. Setup Market Data
    market = MarketDataConnector()
    
    # Helper to get price with caching/logic
    def price_fetcher(ticker, date):
        # We fetch a small window of prices to be safe
        start = date.strftime('%Y-%m-%d')
        end = (date + timedelta(days=5)).strftime('%Y-%m-%d')
        df = market.get_historical_prices(ticker, start, end)
        if not df.empty:
            return float(df.iloc[0]['Close'])
        return None

    # 3. Setup and Run Engine
    # Starting with 0 capital, 1000€ budget for the test simulation points
    engine = BacktestEngine(monthly_contribution=150.0)
    
    # To keep the test fast, let's just do the last 5 trades as a POC
    poc_trades = pelosi_trades.sort_values('report_date').tail(5)
    
    summary = engine.run_simulation(poc_trades, price_fetcher)
    
    print("\n--- TEST RESULTS (POC) ---")
    for event in summary['audit_log']:
        print(f"[{event['execution_date']}] {event['action']} {event['ticker']} at {event['price']:.2f}")
        if event['gain']:
            print(f"    Gain/Loss: {event['gain']:.2f}€")

if __name__ == "__main__":
    run_test()
