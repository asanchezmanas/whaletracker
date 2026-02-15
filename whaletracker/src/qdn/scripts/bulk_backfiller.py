"""
Bulk Backfiller — 10-Year Digital DNA Extraction
Orchestrates SEC, USASpending, and Innovation connectors to build historical dataset.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from ..data.sec_connector import SECConnector
from ..data.gov_connector import GovContractConnector
from ..data.innovation_connector import InnovationConnector
from ..pipeline import QDNPipeline
from ..utils.rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BulkBackfiller:
    def __init__(self, output_dir: str = "data/backfill"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.sec = SECConnector()
        self.gov = GovContractConnector()
        self.innovation = InnovationConnector()
        self.pipeline = QDNPipeline()
        
    def run_backfill(self, start_year: int = 2015, end_year: int = 2025):
        """
        Iterates through quarters and builds the historical dataset.
        """
        all_transactions = []
        
        for year in range(start_year, end_year + 1):
            for qtr in range(1, 5):
                # Stop if in the future
                if year == datetime.now().year and qtr > (datetime.now().month - 1) // 3 + 1:
                    break
                    
                logger.info(f"=== BACKFILLING {year} Q{qtr} ===")
                
                # 1. Fetch SEC Index (Form 4 - Insider Purchases)
                try:
                    df_qtr = self.sec.fetch_historical_index(year, qtr, form_type="4")
                    if not df_qtr.empty:
                        # Focus on PURCHASES ('P') for the Sower logic
                        df_qtr = df_qtr[df_qtr['transaction_code'] == 'P']
                        
                        # Sample to avoid exploding data size for POC
                        sample_size = min(len(df_qtr), 500)
                        df_qtr = df_qtr.sample(n=sample_size, random_state=42)
                        all_transactions.append(df_qtr)
                        
                        # Save checkpoint
                        df_qtr.to_csv(f"{self.output_dir}/sec_idx_{year}_Q{qtr}.csv", index=False)
                except Exception as e:
                    logger.error(f"Failed Q{qtr} {year}: {e}")

        if not all_transactions:
            logger.error("No transactions found.")
            return

        combined = pd.concat(all_transactions, ignore_index=True)
        logger.info(f"Total historical transactions collected: {len(combined)}")
        
        # 2. Extract Features (Digital DNA) for each transaction
        self.assemble_dna(combined)

    def assemble_dna(self, transactions: pd.DataFrame):
        """
        Runs the FeatureEngineer on historical transactions to build training set.
        """
        dna_records = []
        logger.info("Assembling Digital DNA (Point-in-Time Features)...")
        
        # In a real run, we'd process all. For POC, we take a diverse slice.
        for i, (_, txn) in enumerate(transactions.iterrows()):
            if i % 100 == 0:
                logger.info(f"Processing transaction {i}/{len(transactions)}")
            
            ticker = txn['ticker']
            tx_date = txn['transaction_date'].strftime("%Y-%m-%d")
            
            # 1. Fetch Point-in-Time Fundamentals
            info = self.sec.get_company_info(ticker, as_of_date=tx_date)
            
            # 2. Fetch Point-in-Time Structural Signals
            gov_backlog = self.gov.get_backlog_estimate(ticker) # USASpending search uses date filter now
            innovation = self.innovation.get_innovation_score(ticker) # Patent/Grant logic updated
            
            # 3. Assemble the 30-feature DNA vector
            # (Simplified for POC, mapping to the Brains)
            feature_vector = np.zeros(30)
            feature_vector[0] = gov_backlog.get('backlog_value', 0) / 1e6 # Sovereign Alpha
            feature_vector[5] = innovation # IP Alpha
            feature_vector[15] = info.get('market_cap', 1e9) / 1e12 # Size (Small = better)
            
            # 4. Calculate Forward Returns (The Label)
            returns = self.pipeline.market.get_returns(ticker, tx_date, horizons=[180])
            f_return = returns.get('return_180d')
            
            if f_return is not None:
                record = {
                    "ticker": ticker,
                    "date": tx_date,
                    "forward_return": f_return,
                    **{f"f{j}": feature_vector[j] for j in range(30)}
                }
                dna_records.append(record)
        
        dna_df = pd.DataFrame(dna_records)
        dna_df.to_csv(f"{self.output_dir}/historical_dna_10y.csv", index=False)
        logger.info(f"DONE. 10-year DNA saved to {self.output_dir}/historical_dna_10y.csv")
        
        dna_df = pd.DataFrame(dna_records)
        dna_df.to_csv(f"{self.output_dir}/historical_dna_10y.csv", index=False)
        logger.info(f"DONE. 10-year DNA saved to {self.output_dir}/historical_dna_10y.csv")

if __name__ == "__main__":
    backfiller = BulkBackfiller()
    # For verification, we run a very small subset: 2017 Q3
    backfiller.run_backfill(2017, 2017) 
    print("Test extraction complete. Check 'data/backfill' for results.")
