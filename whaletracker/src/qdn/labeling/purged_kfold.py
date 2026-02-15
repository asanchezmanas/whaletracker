"""
Purged K-Fold Cross-Validation — López de Prado

Standard K-Fold is INVALID for time-series financial data because:
1. Train data can contain information that leaks into test
2. Overlapping return windows create spurious autocorrelation
3. Standard CV massively overstates performance

Purged K-Fold fixes this with:
- PURGING: Remove from train any samples whose label spans overlap
  with the test period.  If a training sample's return window
  (entry → exit) overlaps with any test sample's entry, remove it.
- EMBARGO: After each test fold, add a buffer of `embargo_pct`
  of the total samples before allowing reuse in training.

This is strict but necessary.  If performance is real,
it survives purged CV.  If it doesn't, it was leakage.

Reference: AFML Chapter 7

Timeline visualization:
  ┌────────────────────────────────────────────────┐
  │ fold 1: ████████████ [purge] ████TEST████ [emb] │
  │ fold 2: ████████████ [purge] ████████████ TEST │
  │ fold 3: TEST [emb] ██████████████████████████  │
  └────────────────────────────────────────────────┘
  Purged samples (█ crossed out) are removed from train
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PurgedFold:
    """Single fold result from Purged K-Fold."""

    fold_id: int
    n_train: int
    n_test: int
    n_purged: int        # samples removed by purging
    n_embargoed: int     # samples removed by embargo
    train_indices: np.ndarray
    test_indices: np.ndarray


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation.

    For each fold:
      1. Select test indices (contiguous in time)
      2. Identify train candidates (everything else)
      3. PURGE: remove train samples whose labels overlap test
      4. EMBARGO: remove additional buffer after test window
      5. Return clean train/test indices

    Usage:
        cv = PurgedKFoldCV(n_folds=5, embargo_pct=0.01)

        for fold in cv.split(dates, event_starts, event_ends):
            X_train = features[fold.train_indices]
            X_test  = features[fold.test_indices]
            # train and evaluate
    """

    def __init__(
        self,
        n_folds: int = 5,
        embargo_pct: float = 0.01,
    ):
        """
        Args:
            n_folds:     Number of CV folds (default 5)
            embargo_pct: Fraction of total samples to embargo
                         after each test fold (default 1%)
        """
        self.n_folds = n_folds
        self.embargo_pct = embargo_pct

    def split(
        self,
        dates: np.ndarray,
        event_starts: np.ndarray,
        event_ends: np.ndarray,
    ) -> Generator[PurgedFold, None, None]:
        """
        Generate purged K-Fold splits.

        Args:
            dates:        Array of datetime — sample timestamps (sorted)
            event_starts: Array of datetime — start of each label window
                          (typically = transaction_date)
            event_ends:   Array of datetime — end of each label window
                          (from TripleBarrier exit_date or tx_date + horizon)

        Yields:
            PurgedFold for each fold
        """
        n_samples = len(dates)
        if n_samples < self.n_folds * 5:
            raise ValueError(
                f"Too few samples ({n_samples}) for {self.n_folds} folds"
            )

        # Sort by date (should already be sorted, but ensure)
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        event_starts = event_starts[sort_idx]
        event_ends = event_ends[sort_idx]

        # Embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Fold boundaries (contiguous chunks in time)
        fold_size = n_samples // self.n_folds
        fold_boundaries = [
            (i * fold_size, min((i + 1) * fold_size, n_samples))
            for i in range(self.n_folds)
        ]
        # Last fold absorbs remainder
        fold_boundaries[-1] = (fold_boundaries[-1][0], n_samples)

        for fold_id in range(self.n_folds):
            test_start_idx, test_end_idx = fold_boundaries[fold_id]
            test_indices = sort_idx[test_start_idx:test_end_idx]

            # Test period boundaries
            test_period_start = dates[test_start_idx]
            test_period_end = dates[test_end_idx - 1]

            # Train candidates: everything NOT in test
            train_candidates = np.concatenate([
                sort_idx[:test_start_idx],
                sort_idx[test_end_idx:],
            ])

            # --- PURGING ---
            # Remove train samples whose event window overlaps test period
            purged_mask = np.ones(len(train_candidates), dtype=bool)

            for i, train_idx in enumerate(train_candidates):
                ev_start = event_starts[np.searchsorted(sort_idx, train_idx)]
                ev_end = event_ends[np.searchsorted(sort_idx, train_idx)]

                # Overlap check: event window overlaps test period
                overlaps = (
                    ev_start <= test_period_end
                    and ev_end >= test_period_start
                )
                if overlaps:
                    purged_mask[i] = False

            n_purged = (~purged_mask).sum()
            train_after_purge = train_candidates[purged_mask]

            # --- EMBARGO ---
            # Remove samples right after test window
            embargo_start = test_end_idx
            embargo_end = min(test_end_idx + embargo_size, n_samples)
            embargo_indices = set(sort_idx[embargo_start:embargo_end].tolist())

            embargo_mask = np.array([
                idx not in embargo_indices for idx in train_after_purge
            ], dtype=bool)
            n_embargoed = (~embargo_mask).sum()
            final_train = train_after_purge[embargo_mask]

            fold = PurgedFold(
                fold_id=fold_id,
                n_train=len(final_train),
                n_test=len(test_indices),
                n_purged=int(n_purged),
                n_embargoed=int(n_embargoed),
                train_indices=final_train,
                test_indices=test_indices,
            )

            logger.info(
                f"Fold {fold_id + 1}/{self.n_folds}: "
                f"train={fold.n_train}, test={fold.n_test}, "
                f"purged={fold.n_purged}, embargoed={fold.n_embargoed}"
            )

            yield fold

    def validate(
        self,
        dates: np.ndarray,
        event_starts: np.ndarray,
        event_ends: np.ndarray,
    ) -> dict:
        """
        Run a dry-run to report fold statistics without training.

        Returns:
            Dict with fold stats and aggregate info
        """
        stats = []
        for fold in self.split(dates, event_starts, event_ends):
            stats.append({
                "fold": fold.fold_id + 1,
                "train": fold.n_train,
                "test": fold.n_test,
                "purged": fold.n_purged,
                "embargoed": fold.n_embargoed,
                "purge_pct": fold.n_purged / (fold.n_train + fold.n_purged),
            })

        total_purged = sum(s["purged"] for s in stats)
        total_samples = sum(s["train"] + s["test"] for s in stats) // (self.n_folds - 1)

        return {
            "n_folds": self.n_folds,
            "embargo_pct": self.embargo_pct,
            "folds": stats,
            "avg_purge_pct": np.mean([s["purge_pct"] for s in stats]),
            "total_purged": total_purged,
        }


class CombinatorialPurgedCV(PurgedKFoldCV):
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Instead of K individual folds, generates C(K, K-n_test_groups)
    combinations of folds.  This gives many more paths for backtesting,
    which lets us build a distribution of backtest results rather
    than relying on a single walk-forward.

    For K=6 folds with 2 test groups:
    C(6,2) = 15 unique train/test combinations.

    This is the gold standard per López de Prado.
    """

    def __init__(
        self,
        n_folds: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        super().__init__(n_folds=n_folds, embargo_pct=embargo_pct)
        self.n_test_groups = n_test_groups

    def split(
        self,
        dates: np.ndarray,
        event_starts: np.ndarray,
        event_ends: np.ndarray,
    ) -> Generator[PurgedFold, None, None]:
        """
        Generate CPCV splits — all C(n_folds, n_test_groups) combinations.
        """
        from itertools import combinations

        n_samples = len(dates)
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        event_starts = event_starts[sort_idx]
        event_ends = event_ends[sort_idx]

        embargo_size = int(n_samples * self.embargo_pct)

        # Create fold groups
        fold_size = n_samples // self.n_folds
        groups = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = min((i + 1) * fold_size, n_samples)
            if i == self.n_folds - 1:
                end = n_samples
            groups.append(sort_idx[start:end])

        # All combinations of test groups
        combos = list(combinations(range(self.n_folds), self.n_test_groups))

        logger.info(
            f"CPCV: {len(combos)} combinations from "
            f"{self.n_folds} folds × {self.n_test_groups} test groups"
        )

        for combo_id, test_group_ids in enumerate(combos):
            # Test indices = union of selected groups
            test_indices = np.concatenate([groups[g] for g in test_group_ids])

            # Test period boundaries
            test_dates = dates[np.isin(sort_idx, test_indices)]
            test_period_start = test_dates.min()
            test_period_end = test_dates.max()

            # Train candidates
            train_group_ids = [
                g for g in range(self.n_folds) if g not in test_group_ids
            ]
            train_candidates = np.concatenate(
                [groups[g] for g in train_group_ids]
            )

            # Purge
            purged_mask = np.ones(len(train_candidates), dtype=bool)
            for i, train_idx in enumerate(train_candidates):
                pos = np.searchsorted(sort_idx, train_idx)
                ev_end = event_ends[pos]
                ev_start = event_starts[pos]

                overlaps = (
                    ev_start <= test_period_end
                    and ev_end >= test_period_start
                )
                if overlaps:
                    purged_mask[i] = False

            n_purged = (~purged_mask).sum()
            train_after_purge = train_candidates[purged_mask]

            # Embargo (after each test group block)
            embargo_indices = set()
            for g in test_group_ids:
                group_end = groups[g][-1]
                pos = np.searchsorted(sort_idx, group_end)
                for j in range(pos + 1, min(pos + 1 + embargo_size, n_samples)):
                    embargo_indices.add(sort_idx[j])

            embargo_mask = np.array([
                idx not in embargo_indices for idx in train_after_purge
            ], dtype=bool)
            n_embargoed = (~embargo_mask).sum()
            final_train = train_after_purge[embargo_mask]

            yield PurgedFold(
                fold_id=combo_id,
                n_train=len(final_train),
                n_test=len(test_indices),
                n_purged=int(n_purged),
                n_embargoed=int(n_embargoed),
                train_indices=final_train,
                test_indices=test_indices,
            )
