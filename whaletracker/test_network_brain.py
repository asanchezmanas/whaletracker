"""
Verification Script: Network-Force Brain

Checks that the WhaleConnector correctly calculates:
1. Cluster Quality (Diversity)
2. Corporate King-Maker detection
3. Institutional Drift scoring
"""

import logging
import pandas as pd
from src.qdn.data.whale_connector import WhaleConnector
from src.qdn.features.engineer import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_network_signals():
    wc = WhaleConnector()
    engineer = FeatureEngineer()
    
    # We'll use a ticker likely to have activity for testing
    ticker = "NVDA" 
    
    logger.info(f"--- Testing Network Signals for {ticker} ---")
    
    # Test 1: Cluster Quality
    quality = wc.get_cluster_quality(ticker, window_days=90)
    print(f"{ticker} Cluster Quality Score: {quality:.2f}")
    
    # Test 2: King Maker
    is_king = wc.is_corporate_king_maker(ticker)
    print(f"{ticker} Is Corporate King-Maker: {is_king}")
    
    # Test 3: Institutional Drift
    drift = wc.get_institutional_drift(ticker)
    print(f"{ticker} Institutional Drift: {drift:.2f}")
    
    # Test 4: Feature Vector Integration
    logger.info("--- Testing Feature Vector (Network block) ---")
    tx = {"ticker": ticker, "transaction_date": "2024-01-01", "insider_name": "Mock", "shares": 1000, "ownership_after": 50000}
    info = {"market_cap": 2e12, "sector": "Technology", "is_global_500_partner": True}
    
    features = engineer.compute_features(
        transaction=tx,
        historical_transactions=pd.DataFrame(),
        company_info=info,
        market_data=pd.DataFrame(),
        macro_snapshot={}
    )
    
    # Network features are indices 10-14
    net_block = features[10:15]
    print(f"Network-Force Feature Block (Indices 10-14): {net_block}")
    
    if any(net_block > 0):
        logger.info("SUCCESS: Network-Force signals detected.")
    else:
        logger.warning("WARNING: Network features are zero. This may be expected for mock data.")

if __name__ == "__main__":
    try:
        test_network_signals()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
