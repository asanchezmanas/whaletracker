"""
Walk-Forward Backtesting Engine

Implements time-series cross-validation with:
- Expanding or rolling training window
- Purge gap between train and test (avoid look-ahead bias)
- Embargo period after test (avoid data leakage)

This is the ONLY valid way to backtest financial ML models.
Standard k-fold cross-validation is INVALID for time-series because
it leaks future information into training.

Based on Marcos López de Prado's "Advances in Financial Machine Learning".
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .config import QDNConfig
from .trainer import QDNTrainer
from .evaluation import EvaluationResult, evaluate_predictions
from .labeling.purged_kfold import PurgedKFoldCV, CombinatorialPurgedCV


@dataclass
class WalkForwardFold:
    """Single fold of walk-forward validation."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    result: Optional[EvaluationResult] = None


class WalkForwardBacktester:
    """
    Walk-Forward validation engine.
    
    Timeline:
    ┌──────────────────┬─purge─┬──────────┬─embargo─┬─────────
    │    TRAIN         │  gap  │   TEST   │  gap    │  next...
    │  24 months       │ 7 days│ 6 months │ 30 days │
    └──────────────────┴───────┴──────────┴─────────┴─────────
    
    Then slide forward by step_months and repeat.
    
    Purge: Remove samples from end of train that could leak
           to the beginning of test (e.g., overlapping return windows).
    
    Embargo: Remove samples from beginning of next training window
             to avoid test information leaking into future training.
    """

    def __init__(self, config: QDNConfig):
        self.config = config
        self.folds: List[WalkForwardFold] = []

    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds from date range.
        
        Args:
            start_date: First available data date
            end_date: Last available data date
        
        Returns:
            List of WalkForwardFold objects
        """
        cfg = self.config.training
        folds = []
        fold_id = 0

        train_start = start_date

        while True:
            # Training window
            train_end = train_start + timedelta(
                days=cfg.walk_forward_train_months * 30
            )

            # Purge gap
            test_start = train_end + timedelta(days=cfg.purge_days)

            # Test window
            test_end = test_start + timedelta(
                days=cfg.walk_forward_test_months * 30
            )

            # Check if test window exceeds available data
            if test_end > end_date:
                break

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    n_train=0,  # Filled during execution
                    n_test=0,
                )
            )

            # Slide forward
            train_start += timedelta(
                days=cfg.walk_forward_step_months * 30
            )
            fold_id += 1

        self.folds = folds
        return folds

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,  # Array of datetime for each sample
    ) -> Dict:
        """
        Run complete walk-forward backtest.
        
        Args:
            features: [n_samples, n_features]
            labels: [n_samples] actual returns
            dates: [n_samples] transaction dates
        
        Returns:
            Dict with fold results and aggregate metrics
        """
        if len(self.folds) == 0:
            raise ValueError(
                "No folds generated. Call generate_folds() first."
            )

        all_test_scores = []
        all_test_labels = []
        fold_results = []

        print(f"Running walk-forward backtest with {len(self.folds)} folds")
        print("=" * 60)

        for fold in self.folds:
            print(
                f"\nFold {fold.fold_id + 1}/{len(self.folds)}: "
                f"Train [{fold.train_start.date()} → {fold.train_end.date()}] | "
                f"Test [{fold.test_start.date()} → {fold.test_end.date()}]"
            )

            # Split data by date
            train_mask = (dates >= fold.train_start) & (dates < fold.train_end)
            test_mask = (dates >= fold.test_start) & (dates < fold.test_end)

            train_features = features[train_mask]
            train_labels = labels[train_mask]
            test_features = features[test_mask]
            test_labels = labels[test_mask]

            fold.n_train = len(train_features)
            fold.n_test = len(test_features)

            if fold.n_train < 100:
                print(f"  ⚠ Skipping: only {fold.n_train} train samples")
                continue

            if fold.n_test < 10:
                print(f"  ⚠ Skipping: only {fold.n_test} test samples")
                continue

            print(
                f"  Train: {fold.n_train} samples | "
                f"Test: {fold.n_test} samples"
            )

            # Train a fresh model for this fold
            trainer = QDNTrainer(self.config)
            fold_result = trainer.train(
                train_features, train_labels,
                test_features, test_labels,
            )

            fold.result = fold_result
            fold_results.append(fold_result)

            # Collect test predictions for aggregate evaluation
            trainer.model.eval()
            import torch

            with torch.no_grad():
                test_tensor = torch.tensor(
                    test_features, dtype=torch.float32
                ).to(trainer.device)
                output = trainer.model(test_tensor)
                scores = output["convexity_score"].squeeze(-1).cpu().numpy()

            all_test_scores.append(scores)
            all_test_labels.append(test_labels)

            print(f"  Result: {fold_result.summary()}")

        # Aggregate evaluation across all folds
        if all_test_scores:
            agg_scores = np.concatenate(all_test_scores)
            agg_labels = np.concatenate(all_test_labels)

            aggregate_result = evaluate_predictions(
                agg_scores,
                agg_labels,
                threshold=self.config.backtest.buy_score_threshold,
            )
        else:
            aggregate_result = None

        # Summary
        print("\n" + "=" * 60)
        print("WALK-FORWARD BACKTEST COMPLETE")
        print("=" * 60)

        if aggregate_result:
            print(aggregate_result.summary())
            print(
                f"Acceptable: "
                f"{'✅ YES' if aggregate_result.is_acceptable() else '❌ NO'}"
            )

        return {
            "folds": self.folds,
            "fold_results": fold_results,
            "aggregate": aggregate_result,
            "n_folds": len(self.folds),
            "n_valid_folds": len(fold_results),
        }


    def report(self) -> str:
        """Generate summary report of all folds."""
        lines = ["Walk-Forward Backtest Report", "=" * 40]

        for fold in self.folds:
            status = "[OK]" if fold.result else "[SKIP] skipped"
            sortino = f"{fold.result.sortino:.2f}" if fold.result else "N/A"
            win_rate = f"{fold.result.win_rate:.1%}" if fold.result else "N/A"

            lines.append(
                f"Fold {fold.fold_id+1}: "
                f"{fold.train_start.date()} → {fold.test_end.date()} | "
                f"Sortino: {sortino} | Win: {win_rate} | {status}"
            )

        return "\n".join(lines)


class PurgedKFoldBacktester:
    """
    Purged K-Fold Backtester.
    
    Implements López de Prado's Purged K-Fold and optionally 
    Combinatorial Purged CV (CPCV).
    
    This replaces/augments the WalkForwardBacktester with 
    statistically sound cross-validation.
    """

    def __init__(self, config: QDNConfig, combinatorial: bool = False):
        self.config = config
        self.combinatorial = combinatorial
        if combinatorial:
            self.cv = CombinatorialPurgedCV(
                n_folds=config.training.cv_folds,
                n_test_groups=2,
                embargo_pct=config.training.embargo_pct,
            )
        else:
            self.cv = PurgedKFoldCV(
                n_folds=config.training.cv_folds,
                embargo_pct=config.training.embargo_pct,
            )
        self.results = []

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,
        event_ends: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run Purged K-Fold backtest.
        
        Args:
            features: [n_samples, n_features]
            labels: [n_samples] actual returns
            dates: [n_samples] transaction (start) dates
            event_ends: [n_samples] barrier exit dates
            sample_weights: [n_samples] uniqueness weights
            
        Returns:
            Dict with results summary
        """
        print(f"Running {'CPCV' if self.combinatorial else 'Purged K-Fold'} CV")
        print("=" * 60)

        all_test_scores = []
        all_test_labels = []
        fold_summaries = []

        # split expects sorted dates
        sort_idx = np.argsort(dates)
        features = features[sort_idx]
        labels = labels[sort_idx]
        dates = dates[sort_idx]
        event_ends = event_ends[sort_idx]
        if sample_weights is not None:
            sample_weights = sample_weights[sort_idx]

        for fold in self.cv.split(dates, dates, event_ends):
            print(
                f"\nFold {fold.fold_id + 1}: "
                f"Train: {fold.n_train} | Test: {fold.n_test} | "
                f"Purged: {fold.n_purged}"
            )

            # Extract data for this fold
            train_features = features[fold.train_indices]
            train_labels = labels[fold.train_indices]
            test_features = features[fold.test_indices]
            test_labels = labels[fold.test_indices]
            
            train_weights = (
                sample_weights[fold.train_indices] 
                if sample_weights is not None else None
            )
            test_weights = (
                sample_weights[fold.test_indices] 
                if sample_weights is not None else None
            )

            if len(train_features) < 100:
                print("  [WARNING] Skipping: too few train samples")
                continue

            # Train
            trainer = QDNTrainer(self.config)
            result = trainer.train(
                train_features, train_labels,
                test_features, test_labels,
                train_weights=train_weights,
                val_weights=test_weights,
            )

            if result:
                fold_summaries.append(result)
                
                # Get test scores for OOS evaluation
                trainer.model.eval()
                import torch
                with torch.no_grad():
                    test_tensor = torch.tensor(
                        test_features, dtype=torch.float32
                    ).to(trainer.device)
                    output = trainer.model(test_tensor)
                    scores = output["convexity_score"].squeeze(-1).cpu().numpy()
                
                all_test_scores.append(scores)
                all_test_labels.append(test_labels)
                
                print(f"  Result: {result.summary()}")

        # Aggregate Result
        if all_test_scores:
            agg_scores = np.concatenate(all_test_scores)
            agg_labels = np.concatenate(all_test_labels)
            
            aggregate_result = evaluate_predictions(
                agg_scores,
                agg_labels,
                threshold=self.config.backtest.buy_score_threshold,
            )
        else:
            aggregate_result = None

        return {
            "fold_results": fold_summaries,
            "aggregate": aggregate_result,
            "combinatorial": self.combinatorial,
        }

