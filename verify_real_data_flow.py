"""
Verification Script — Real Data Flow (Public Only)

Fetches a small sample of REAL data from SEC and Senate (no API keys needed).
Verifies that the FeatureEngineer and TripleBarrierLabeler can handle real-world inputs.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from whaletracker.src.qdn.pipeline import QDNPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def verify_real_data():
    logger.info("Initializing QDNPipeline for EXHAUSTIVE real data verification...")
    
    pipeline = QDNPipeline()
    
    # 1. Fetch diverse batch (90 days back to get more variety)
    try:
        transactions = pipeline.fetch_all_transactions(
            sec_days_back=90, 
            include_senate=True,
            include_whale=True,
            include_clusters=True,
            include_13f=True,
            include_otc=True
        )
            
        if transactions.empty:
            logger.error("No transactions found. Check SEC/internet connectivity.")
            return

        # Filtering
        transactions = transactions.dropna(subset=["ticker"])
        transactions = transactions[transactions["ticker"].str.len() > 0]
        
        if transactions.empty:
            logger.warning("No transactions with valid tickers found.")
            return

        # Summary of different types found
        type_counts = transactions["whale_type"].value_counts()
        logger.info("\n=== WHALE SIGNALS INVENTORY (90 DAYS) ===")
        for w_type, count in type_counts.items():
            logger.info(f"  {w_type:.<20} {count} signals")

        # 2. Select Diverse Sample (Top 5 per type)
        diverse_df = transactions.groupby("whale_type").head(5).reset_index(drop=True)
        logger.info(f"\nProcessing Diverse Sample ({len(diverse_df)} events)...")

        # 3. Compute Features and Labels
        result = pipeline.compute_features_and_labels(
            diverse_df,
            return_horizon_days=30,
            use_triple_barrier=True
        )

        tickers_processed = list(set(result.get('tickers', [])))
        if len(tickers_processed) > 0:
            logger.info(f"\nSUCCESS: Features computed for {len(tickers_processed)} diverse companies.")
            logger.info(f"Verified Tickers: {', '.join(tickers_processed)}")
            
            # Show feature vector shape
            logger.info(f"Final Feature Matrix: {result['features'].shape}")
            logger.info("\nPROTOTYPE VERIFICATION COMPLETE (REAL DATA)")
        else:
            logger.error("FAILED: Could not compute features for any real signals (likely market data 404s).")

    except Exception as e:
        logger.exception(f"An error occurred during real data verification: {e}")

if __name__ == "__main__":
    verify_real_data()
