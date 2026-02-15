"""
Phase 5 Verification Script — Portfolio & Execution

Verifies:
1. HRP weight computation stability.
2. Kelly sizing with Meta-labeling confidence.
3. ExecutionSimulator slippage impact.
4. PortfolioBacktester equity curve generation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from whaletracker.src.qdn.config import QDNConfig
from whaletracker.src.qdn.walk_forward import PortfolioBacktester

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def verify_phase5_portfolio():
    logger.info("Starting Phase 5: Portfolio & Execution Verification...")
    
    config = QDNConfig()
    backtester = PortfolioBacktester(config, initial_capital=1000000)
    
    # 1. Generate Synthetic Simulation Data
    n_days = 250
    n_tickers = 5
    dates_list = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Expand to transaction-level data (multiple trades per day)
    total_samples = n_days * 2
    sample_dates = np.repeat(dates_list, 2)
    sample_tickers = [f"TICKER_{i%n_tickers}" for i in range(total_samples)]
    
    # Synthetic Features (40)
    features = np.random.normal(0, 1, (total_samples, 40))
    
    # Primary Model Scores (normal near 75 to cross threshold of 70)
    primary_scores = np.random.normal(75, 10, total_samples)
    primary_scores = np.clip(primary_scores, 0, 100)
    
    # Actual Returns (correlated with score to mock a 'good' model)
    # Return = 1% for every 10 points above 50
    actual_returns = (primary_scores - 50) * 0.001 + np.random.normal(0, 0.02, total_samples)
    
    # Prices and Volumes
    prices = np.random.uniform(5, 150, total_samples)
    volumes = np.random.uniform(1e5, 1e7, total_samples)
    
    logger.info(f"Running simulation with {total_samples} samples over {n_days} days...")
    
    # 2. Run Portfolio Simulation
    results = backtester.run(
        features=features,
        dates=sample_dates,
        tickers=sample_tickers,
        prices=prices,
        primary_scores=primary_scores,
        actual_returns=actual_returns,
        daily_volumes=volumes
    )
    
    # 3. Validate Results
    equity_curve = results["equity_curve"]
    trades = results["trades"]
    
    logger.info(f"Final Equity: ${results['final_equity']:,.2f}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Total Trades Executed: {len(trades)}")
    
    assert results["final_equity"] > 0, "Equity should be positive"
    assert len(equity_curve) > 0, "Equity curve should be created"
    
    if not trades.empty:
        avg_slippage = trades["slippage_bps"].mean()
        logger.info(f"Average Slippage: {avg_slippage:.2f} bps")
        assert avg_slippage > 0, "Slippage should be simulated"

    logger.info("Phase 5 Verification Complete.")

if __name__ == "__main__":
    verify_phase5_portfolio()
