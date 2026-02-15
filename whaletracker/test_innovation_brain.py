"""
Verification Script: IP-Monopoly Brain

Checks that the InnovationConnector returns valid signals for:
1. Patent Velocity (LMT)
2. Grant Intensity (IONQ)
"""

import logging
import pandas as pd
from src.qdn.data.innovation_connector import InnovationConnector
from src.qdn.features.engineer import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_innovation_signals():
    innov = InnovationConnector()
    engineer = FeatureEngineer()
    
    # Test 1: Patent Velocity (Structural Moat)
    logger.info("--- Testing Patent Velocity (LMT) ---")
    p_data = innov.get_patent_velocity("LMT")
    print(f"LMT Patents: {p_data}")
    
    # Test 2: Grant Intensity (State-Validated Seed)
    logger.info("--- Testing Grant Intensity (IONQ) ---")
    g_data = innov.get_grant_intensity("IONQ")
    print(f"IONQ Grants: {g_data}")
    
    # Test 3: Feature Engineering Mock
    logger.info("--- Testing Feature Vector (IP block) ---")
    tx = {"ticker": "IONQ", "transaction_date": "2024-01-01", "insider_name": "Mock"}
    info = {"market_cap": 2e9, "sector": "Technology"}
    
    features = engineer.compute_features(
        transaction=tx,
        historical_transactions=pd.DataFrame(),
        company_info=info,
        market_data=pd.DataFrame(),
        macro_snapshot={}
    )
    
    # IP features are indices 5-9
    ip_block = features[5:10]
    print(f"IP-Monopoly Feature Block: {ip_block}")
    
    # Success check
    if any(ip_block > 0):
        logger.info("SUCCESS: Structural IP signals detected.")
    else:
        logger.warning("WARNING: Features are zero. Check API availability or resolution.")

if __name__ == "__main__":
    try:
        test_innovation_signals()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
