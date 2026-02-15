"""
verification script for Phase 2: ML Rigor

Tests:
1. Triple Barrier Labeling
2. Sample Weight (Uniqueness) calculation
3. Purged K-Fold CV
4. CPCV (Combinatorial Purged CV)
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from whaletracker.src.qdn.pipeline import QDNPipeline
from whaletracker.src.qdn.config import QDNConfig
from whaletracker.src.qdn.walk_forward import PurgedKFoldBacktester

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_tickers=10, n_tx=1000):
    """Generate synthetic transactions and price data for offline testing."""
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    tx_list = []
    price_map = {}
    
    start_date = pd.Timestamp("2023-01-01")
    
    for ticker in tickers:
        # Generate 2 years of daily prices
        dates = pd.date_range(start_date, periods=730, freq='D')
        
        # Random walk with some drifts
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 730)))
        # Add volume for VPIN
        volumes = np.random.lognormal(10, 1, 730)
        df = pd.DataFrame({'Close': prices, 'Volume': volumes}, index=dates)
        price_map[ticker] = df
        
        # Generate random transactions
        for _ in range(n_tx // n_tickers):
            tx_date = start_date + pd.Timedelta(days=np.random.randint(30, 500))
            tx_list.append({
                'ticker': ticker,
                'transaction_date': tx_date,
                'is_purchase': True,
                'amount': np.random.randint(1000, 100000),
                'owner_name': 'Synthetic Whale',
                'owner_title': 'Director'
            })
            
    return pd.DataFrame(tx_list), price_map

def run_phase2_verification():
    # 1. Setup Config
    config = QDNConfig()
    config.training.max_epochs = 5
    config.training.cv_folds = 3
    
    pipeline = QDNPipeline()
    
    # 2. Try real data, fallback to synthetic
    logger.info("Attempting to fetch transaction data...")
    try:
        transactions = pipeline.fetch_all_transactions(sec_days_back=180)
        if transactions.empty:
            raise ValueError("No real transactions found")
        
        # Filter and get market data
        top_tickers = transactions['ticker'].value_counts().head(5).index.tolist()
        transactions = transactions[transactions['ticker'].isin(top_tickers)].head(50)
        market_data_map = {}
        for t in top_tickers:
            market_data_map[t] = pipeline.market.get_prices(t, days_back=365)
            
    except Exception as e:
        logger.warning(f"Real data fetch failed ({e}). Using synthetic data.")
        transactions, market_data_map = generate_synthetic_data()

    logger.info(f"Processing {len(transactions)} transactions for verification...")

    # 3. Setup Mocks (Market + Engineer)
    # Inject our price map directly to avoid hitting APIs
    class MockMarket:
        def __init__(self, data_map): 
            self.data_map = data_map
        def get_stock_prices(self, ticker, start_date=None): 
            return self.data_map.get(ticker, pd.DataFrame())
        def get_returns(self, *args, **kwargs): 
            return {"return_90d": 0.05}
        def get_company_info(self, ticker): 
            return {"sector": "Technology", "industry": "Software", "market_cap": 1e9}
        def get_sector_performance(self): 
            return pd.DataFrame({"Sector": ["Technology"], "Change": [0.01]}).set_index("Sector")
    
    pipeline.market = MockMarket(market_data_map)

    def mock_compute_features(*args, **kwargs):
        return np.random.normal(0, 1, 40)
    
    pipeline.engineer.compute_features = mock_compute_features

    logger.info("Step 3: Computing features and Triple Barrier labels (Offline)...")
    data = pipeline.compute_features_and_labels(
        transactions, 
        return_horizon_days=90,
        use_triple_barrier=True
    )
    
    if 'barrier_events' not in data:
        logger.error("Triple Barrier labeling failed.")
        return

    features = data['features']
    labels = data['labels']
    dates = data['dates']
    event_ends = data['event_ends']
    weights = data['sample_weights']
    events = data['barrier_events']

    logger.info(f"Labeled {len(labels)} events.")
    logger.info(f"Sample weights stats: mean={weights.mean():.2f}, min={weights.min():.2f}")
    
    # 4. Analyze Barrier Outcomes
    outcomes = pd.Series([e.barrier_type for e in events]).value_counts()
    logger.info(f"Barrier Outcomes:\n{outcomes}")

    # 5. Run Purged K-Fold Backtest
    logger.info("\nStep 5: Running Purged K-Fold CV...")
    backtester = PurgedKFoldBacktester(config, combinatorial=False)
    pkf_result = backtester.run(
        features=features,
        labels=labels,
        dates=dates,
        event_ends=event_ends,
        sample_weights=weights
    )
    
    if pkf_result['aggregate']:
        logger.info(f"PKF Aggregate Sortino: {pkf_result['aggregate'].sortino:.2f}")
    
    # 6. Run CPCV (Combinatorial Purged CV)
    logger.info("\nStep 6: Running CPCV...")
    cpcv_backtester = PurgedKFoldBacktester(config, combinatorial=True)
    cpcv_result = cpcv_backtester.run(
        features=features,
        labels=labels,
        dates=dates,
        event_ends=event_ends,
        sample_weights=weights
    )
    
    if cpcv_result['aggregate']:
        logger.info(f"CPCV Aggregate Sortino: {cpcv_result['aggregate'].sortino:.2f}")

    logger.info("\nPhase 2 Verification Complete.")


if __name__ == "__main__":
    run_phase2_verification()
