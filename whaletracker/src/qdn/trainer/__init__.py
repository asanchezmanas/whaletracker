"""
QDN Training Loop

Orchestrates the full training process:
1. Walk-forward data splitting
2. Adversarial batch augmentation
3. Antifragile loss optimization
4. Early stopping on Sortino ratio
5. Checkpoint management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from ..config import QDNConfig, TrainingConfig
from ..dense_network import DenseNetwork
from ..loss_functions import AntifragileLoss
from ..adversarial import AdversarialTrainer
from ..evaluation import evaluate_predictions, EvaluationResult


class QDNTrainer:
    """
    Complete training pipeline for the Dense Network.
    
    Training flow:
    ┌─────────────────────┐
    │  Walk-Forward Split  │  Train/Val with purging & embargo
    └──────┬──────────────┘
           │
    ┌──────▼──────────────┐
    │  For each fold:      │
    │  ┌────────────────┐  │
    │  │ Adversarial    │  │  50% clean, 20% FGSM, 15% fake, 15% crisis
    │  │ Augmentation   │  │
    │  └──────┬─────────┘  │
    │  ┌──────▼─────────┐  │
    │  │ Forward Pass   │  │
    │  │ + Loss Compute │  │  Antifragile Loss (prediction + calibration)
    │  └──────┬─────────┘  │
    │  ┌──────▼─────────┐  │
    │  │ Validate       │  │  Sortino-based early stopping
    │  └────────────────┘  │
    └──────────────────────┘
    """

    def __init__(self, config: QDNConfig):
        self.config = config
        self.device = torch.device(config.resolve_device())

        # Model
        self.model = DenseNetwork(config.model).to(self.device)

        # Loss
        self.criterion = AntifragileLoss(config.training)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Adversarial trainer
        self.adversarial = AdversarialTrainer(
            model=self.model,
            config=config.training,
        )

        # Training state
        self.best_sortino = -np.inf
        self.patience_counter = 0
        self.training_history: List[Dict] = []

        # Ensure checkpoint dir exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        print(self.model.summary())
        print(f"Device: {self.device}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        cfg = self.config.training
        if cfg.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5
        )

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        train_weights: Optional[np.ndarray] = None,
        val_weights: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        Train the model on a single train/val split.
        
        Args:
            train_features: [n_train, n_features]
            train_labels: [n_train] actual returns
            val_features: [n_val, n_features]
            val_labels: [n_val] actual returns
            train_weights: [n_train] optional sample weights
            val_weights: [n_val] optional sample weights
        
        Returns:
            Best validation EvaluationResult
        """
        cfg = self.config.training

        # Create data loaders
        train_loader = self._make_loader(
            train_features, train_labels, train_weights, shuffle=True
        )
        val_loader = self._make_loader(
            val_features, val_labels, val_weights, shuffle=False
        )

        best_val_result = None

        for epoch in range(cfg.max_epochs):
            # --- Train epoch ---
            train_metrics = self._train_epoch(train_loader, epoch)

            # --- Validate ---
            val_result = self._validate(val_loader)

            # --- Logging ---
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["total_loss"],
                "train_pred_loss": train_metrics["prediction_loss"],
                "train_cal_loss": train_metrics["calibration_loss"],
                "val_sortino": val_result.sortino,
                "val_win_rate": val_result.win_rate,
                "val_max_dd": val_result.max_drawdown,
                "val_tail_ratio": val_result.tail_ratio,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.training_history.append(epoch_log)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:>3d} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Val Sortino: {val_result.sortino:.2f} | "
                    f"Win: {val_result.win_rate:.1%} | "
                    f"DD: {val_result.max_drawdown:.1%}"
                )

            # --- Early stopping on Sortino ---
            if val_result.sortino > self.best_sortino:
                self.best_sortino = val_result.sortino
                best_val_result = val_result
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_result)
            else:
                self.patience_counter += 1

            if self.patience_counter >= cfg.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch+1}. "
                    f"Best Sortino: {self.best_sortino:.2f}"
                )
                break

            # Step scheduler
            self.scheduler.step()

        # Load best checkpoint
        self._load_best_checkpoint()

        return best_val_result

    def _train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Run one training epoch with adversarial augmentation."""
        self.model.train()
        cfg = self.config.training

        epoch_losses = {
            "total_loss": 0,
            "prediction_loss": 0,
            "calibration_loss": 0,
        }
        n_batches = 0

        for batch in train_loader:
            if len(batch) == 3:
                features, labels, weights = batch
                weights = weights.to(self.device).unsqueeze(-1)
            else:
                features, labels = batch
                weights = None

            features = features.to(self.device)
            labels = labels.to(self.device).unsqueeze(-1)

            # Adversarial augmentation
            aug_features, aug_labels, aug_weights, adv_type = (
                self.adversarial.augment_batch(features, labels, weights)
            )
            adv_weight = self.adversarial.get_adversarial_weight(adv_type)

            # Forward
            output = self.model(aug_features)

            # Loss
            loss_dict = self.criterion(output, aug_labels, sample_weights=aug_weights)
            loss = loss_dict["total_loss"] * adv_weight

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )

            self.optimizer.step()

            # Accumulate
            epoch_losses["total_loss"] += loss_dict["total_loss"].item()
            epoch_losses["prediction_loss"] += loss_dict["prediction_loss"].item()
            epoch_losses["calibration_loss"] += loss_dict["calibration_loss"].item()
            n_batches += 1

        # Average
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def _validate(self, val_loader: DataLoader) -> EvaluationResult:
        """Run validation and compute antifragile metrics."""
        self.model.eval()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    features, labels, weights = batch
                else:
                    features, labels = batch

                features = features.to(self.device)
                output = self.model(features)

                scores = output["convexity_score"].squeeze(-1).cpu().numpy()
                all_scores.append(scores)
                all_labels.append(labels.numpy())

        predicted_scores = np.concatenate(all_scores)
        actual_returns = np.concatenate(all_labels)

        # Adaptive threshold: use 60th percentile of scores
        # This ensures we always evaluate the top ~40% of predictions,
        # even during early training when scores cluster narrowly.
        adaptive_threshold = float(np.percentile(predicted_scores, 60))

        return evaluate_predictions(
            predicted_scores,
            actual_returns,
            threshold=adaptive_threshold,
        )

    def _make_loader(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        tensors = [
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        ]
        if weights is not None:
            tensors.append(torch.tensor(weights, dtype=torch.float32))

        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _save_checkpoint(self, epoch: int, metrics: EvaluationResult):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "sortino": metrics.sortino,
                "win_rate": metrics.win_rate,
                "max_drawdown": metrics.max_drawdown,
                "model_version": self.config.model_version,
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )

    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best checkpoint (epoch {checkpoint['epoch']+1}, "
                f"Sortino {checkpoint['sortino']:.2f})"
            )

    def save_training_history(self, path: str = "training_history.json"):
        """Save training history for analysis."""
        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)
