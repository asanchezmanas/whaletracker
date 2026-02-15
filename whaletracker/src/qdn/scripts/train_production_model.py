"""
WhaleTracker Production Training Script

Orchestrates the full 2-stage training process:
1. Primary QDN Model (Incremental Walk-Forward)
2. Meta-Labeler Sizing Model (Fitted on OOS Errors)

Usage:
    python scripts/train_production_model.py --days 1000
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from whaletracker.src.qdn.pipeline import QDNPipeline
from whaletracker.src.qdn.config import QDNConfig
from whaletracker.src.qdn.walk_forward import PurgedKFoldCV
from whaletracker.src.qdn.trainer.incremental_trainer import IncrementalTrainer
from whaletracker.src.qdn.labeling.meta_labeling import MetaModelManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_production_training(days_back: int = 1000):
    logger.info(f"Starting Production Training Shift ({days_back} days historical data)")
    
    config = QDNConfig()
    pipeline = QDNPipeline()
    
    # 1. DATA ACQUISITION
    logger.info("Fetching all transaction sources (incremental backfill)...")
    # Using 90 days increments or similar logic inside the pipeline if needed
    # For PoC, we assume the pipeline handles the full fetch
    transactions = pipeline.fetch_all_transactions(
        sec_days_back=days_back,
        include_senate=True,
        include_whale=True,
        include_clusters=True,
        include_otc=True
    )
    
    if transactions.empty:
        logger.error("No transactions fetched. Aborting.")
        return

    # 2. FEATURE ENGINEERING
    logger.info("Computing features and labels (Real Market Feed)...")
    data = pipeline.compute_features_and_labels(
        transactions,
        return_horizon_days=config.backtest.max_holding_days,
        use_triple_barrier=True
    )
    
    features = data["features"]
    labels = data["labels"]
    dates = data["dates"]
    event_ends = data["event_ends"]
    tickers = data["tickers"]
    sample_weights = data["sample_weights"]
    metadata = data["metadata"]
    
    # Traceability fields
    filing_dates = metadata["filing_date"].values if "filing_date" in metadata.columns else None
    whale_types = metadata["whale_type"].values if "whale_type" in metadata.columns else None
    prices = metadata["price"].values if "price" in metadata.columns else np.zeros(len(features))
    
    if len(features) < 20:
        logger.error(f"Insufficient data for PoC training ({len(features)} samples).")
        return

    # 3. INCREMENTAL PRIMARY TRAINING
    logger.info("Starting Primary QDN Training (Fold-by-Fold Checkpoints)...")
    inc_trainer = IncrementalTrainer(config)
    
    # We use a smaller CV for PoC if needed, but keeping PurgedKFold is better
    cv = PurgedKFoldCV(n_folds=config.training.cv_folds, embargo_pct=config.training.embargo_pct)
    
    for fold in cv.split(dates, dates, event_ends):
        inc_trainer.train_fold(
            fold.fold_id,
            features[fold.train_indices], labels[fold.train_indices],
            features[fold.test_indices], labels[fold.test_indices],
            sample_weights=sample_weights[fold.train_indices]
        )

    # 4. META-LABELER TRAINING (Stage 2)
    logger.info("Starting Stage 2: Meta-Labeler (Sizing Model) Training...")
    all_oos_scores, all_oos_labels, all_oos_indices = inc_trainer.get_aggregate_oos()
    
    if len(all_oos_indices) == 0:
        logger.warning("No OOS predictions found. Check training logs.")
    else:
        # Align OOS features and weights
        oos_features = features[all_oos_indices]
        oos_weights = sample_weights[all_oos_indices]
        
        # Binary Meta-labels: 1 if return > 0 (profitable), else 0
        meta_labels = (all_oos_labels > 0).astype(np.float32)
        
        meta_manager = MetaModelManager(input_dim=features.shape[1] + 1)
        meta_manager.train(
            oos_features,
            all_oos_scores,
            meta_labels,
            sample_weights=oos_weights,
            epochs=20
        )
        meta_manager.save(os.path.join(inc_trainer.checkpoint_dir, "meta_labeler.pth"))
    
    # 5. FINAL P&L AUDIT & TRACEABILITY
    logger.info("Generating Final PoC Performance Report...")
    from whaletracker.src.qdn.walk_forward import PortfolioBacktester
    
    backtester = PortfolioBacktester(config)
    # Re-run simulation on OOS set (to see "live-like" performance)
    results = backtester.run(
        features=features[all_oos_indices],
        dates=dates[all_oos_indices],
        tickers=np.array(tickers)[all_oos_indices].tolist(),
        prices=prices[all_oos_indices],
        primary_scores=all_oos_scores,
        actual_returns=all_oos_labels,
        daily_volumes=None, # Simplified for PoC
        filing_dates=filing_dates[all_oos_indices],
        whale_types=whale_types[all_oos_indices]
    )
    
    print("\n" + "="*60)
    print("PRODUCTION POC PERFORMANCE SUMMARY")
    print("="*60)
    print(backtester.report())
    
    # Save Trade Log CSV for external audit
    audit_file = os.path.join(inc_trainer.checkpoint_dir, "audit_log.csv")
    results["trades"].to_csv(audit_file, index=False)
    logger.info(f"Audit Log saved to {audit_file}")
    
    if results["total_return"] > 0:
        logger.info("\nSUCCESS: Strategy demonstrated positive alpha on real-world Smart Money data.")
    else:
        logger.warning("\nCAUTION: Strategy yielded suboptimal returns. Refine scoring/thresholds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="Number of days for historical backfill")
    parser.add_argument("--poc", action="store_true", help="Run quick PoC with limited data")
    args = parser.parse_args()
    
    if args.poc:
        logger.info("Running in PoC MODE (Short window, fast check)")
        run_production_training(days_back=90)
    else:
        run_production_training(args.days)
