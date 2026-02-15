"""
Verification Script: Antifragile Orchestrator
"""

import logging
import pandas as pd
import numpy as np
from src.qdn.config import QDNConfig
from src.qdn.portfolio.orchestrator import Orchestrator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_orchestrator():
    config = QDNConfig()
    chef = Orchestrator(config)
    
    # 1. Monthly Fuel
    chef.add_monthly_fuel()
    
    # 2. Mock Candidates
    # Features[19] = cash runway (0.5 = 1 year, 0.05 = 1.2 months)
    candidates = [
        # A Structural Gem (High Score, Pass Veto)
        {'ticker': 'GEM1', 'score': 85.0, 'price': 0.10, 'features': [0]*19 + [0.5] + [0]*10}, 
        # A Fragile entity (High Score, FAIL Veto - low runway)
        {'ticker': 'FRAG1', 'score': 95.0, 'price': 0.05, 'features': [0]*19 + [0.05] + [0]*10},
        # Another Gem for Fattening (Score > 90)
        {'ticker': 'GEM2', 'score': 92.0, 'price': 1.0, 'features': [0]*19 + [0.5] + [0]*10},
        # Normal Gem
        {'ticker': 'GEM3', 'score': 76.0, 'price': 2.0, 'features': [0]*19 + [0.5] + [0]*10}
    ]
    
    logger.info("--- Phase 1: Monthly Sowing ---")
    chef.allocate_seeds(candidates, pd.Timestamp('2024-01-01'))
    
    summary = chef.get_portfolio_summary()
    print(f"Portfolio Summary (Jan): {summary}")
    
    # Assertions
    if 'FRAG1' in chef.portfolio:
        logger.error("FAIL: FRAG1 should have been vetoed.")
    else:
        logger.info("PASS: FRAG1 was correctly vetoed.")
        
    if chef.portfolio.get('GEM2', {}).get('total_invested') == 5.0:
        logger.info("PASS: GEM2 was correctly 'Fattened' (Max Seed).")
    
    if chef.portfolio.get('GEM1', {}).get('total_invested') == 1.0:
        logger.info("PASS: GEM1 was correctly 'Sowed' (Min Seed).")
        
    # Phase 2: Monthly accumulation
    logger.info("--- Phase 2: Next Month Accumulation ---")
    chef.add_monthly_fuel()
    
    # Update score for GEM1 -> Now high confidence, should be fattened
    candidates[0]['score'] = 91.0
    chef.allocate_seeds(candidates, pd.Timestamp('2024-02-01'))
    
    summary = chef.get_portfolio_summary()
    print(f"Final Summary (Feb): {summary}")
    
    if chef.portfolio.get('GEM1', {}).get('total_invested') == 6.0: # 1 (Jan) + 5 (Feb)
        logger.info("PASS: GEM1 was correctly fattened in Phase 2.")

if __name__ == "__main__":
    try:
        test_orchestrator()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
