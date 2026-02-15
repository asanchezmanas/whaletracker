"""
Incremental Trainer — Robust Training Orchestration

Manages the primary model training across multiple walk-forward folds
with checkpointing and resume capabilities.
"""

import torch
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..trainer import QDNTrainer
from ..config import QDNConfig

logger = logging.getLogger(__name__)

class IncrementalTrainer:
    """
    Orchestrates training across folds, allowing for persistence and resume.
    """

    def __init__(self, config: QDNConfig, checkpoint_dir: str = "checkpoints/incremental"):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(self.checkpoint_dir, "training_state.json")
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return {"completed_folds": [], "oos_predictions": []}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def train_fold(
        self,
        fold_id: int,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train a single fold, or skip if already completed.
        """
        if fold_id in self.state["completed_folds"]:
            logger.info(f"Fold {fold_id} already completed. Loading OOS predictions...")
            # Load OOS predictions from disk if needed
            return {"status": "skipped"}

        logger.info(f"Starting Fold {fold_id}...")
        
        # Simple trainer instance for this fold
        trainer = QDNTrainer(self.config)
        
        # Train
        result = trainer.train(
            train_features, train_labels,
            test_features, test_labels,
            train_weights=sample_weights if sample_weights is not None else None
        )

        # Generate OOS predictions
        trainer.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(test_features, dtype=torch.float32).to(trainer.device)
            output = trainer.model(X_test)
            scores = output["convexity_score"].squeeze(-1).cpu().numpy()

        # Save model for this fold
        fold_model_path = os.path.join(self.checkpoint_dir, f"model_fold_{fold_id}.pth")
        torch.save(trainer.model.state_dict(), fold_model_path)

        # Update state
        self.state["completed_folds"].append(fold_id)
        # We store scores as lists for JSON serialization
        fold_results = {
            "fold_id": fold_id,
            "test_indices": fold.test_indices.tolist() if hasattr(fold, 'test_indices') else [],
            "scores": scores.tolist(),
            "labels": test_labels.tolist(),
            "metrics": {
                "sortino": float(result.sortino),
                "win_rate": float(result.win_rate)
            }
        }
        self.state["oos_predictions"].append(fold_results)
        self._save_state()

        return {"status": "completed", "scores": scores, "metrics": result}

    def get_aggregate_oos(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate all OOS predictions and their original indices.
        Returns: (scores, labels, indices)
        """
        all_scores = []
        all_labels = []
        all_indices = []
        for fold in self.state["oos_predictions"]:
            all_scores.extend(fold["scores"])
            all_labels.extend(fold["labels"])
            all_indices.extend(fold.get("test_indices", []))
        return np.array(all_scores), np.array(all_labels), np.array(all_indices)
