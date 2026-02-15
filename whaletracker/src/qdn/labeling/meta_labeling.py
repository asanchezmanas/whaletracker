"""
Meta-Labeling Model — Phase 5

A secondary binary classifier that learns to 'filter' the primary model.
Primary Model: Detects the signal.
Meta-Model: Learns if the Primary Model's detection is likely to be a winner.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MetaLabeler(nn.Module):
    """
    A lightweight dense network for secondary sizing/filtration.
    Inputs: 40 QDN features + 1 Primary Model Score = 41 features.
    Output: Probability of success.
    """

    def __init__(self, input_dim: int = 41):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MetaModelManager:
    """
    Handles training and inference for the Meta-Labeler.
    """

    def __init__(self, input_dim: int = 41):
        self.model = MetaLabeler(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def train(
        self, 
        features: np.ndarray, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        sample_weights: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Full multi-epoch training loop for the Meta-Labeler.
        """
        if sample_weights is None:
            sample_weights = np.ones(len(labels))
            
        # Combine QDN features with primary scores
        X = np.hstack([features, scores.reshape(-1, 1)])
        y = labels.reshape(-1, 1)
        w = sample_weights.reshape(-1, 1)
        
        # Train/Val Split (simple 80/20)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        w_train, w_val = w[:split], w[split:]
        
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        w_train_t = torch.FloatTensor(w_train)
        
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        logger.info(f"Training Meta-Labeler on {len(X_train)} samples...")
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train_t)
            loss = (self.criterion(outputs, y_train_t) * w_train_t).mean()
            loss.backward()
            self.optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = self.criterion(val_outputs, y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best state internally
                self._best_state = self.model.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        if hasattr(self, '_best_state'):
            self.model.load_state_dict(self._best_state)
            
        logger.info(f"Meta-Labeler training complete. Best Val Loss: {best_val_loss:.4f}")

    def train_step(self, features: np.ndarray, scores: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray):
        """
        features: (N, 40)
        scores: (N, 1) - primary model outputs
        labels: (N, 1) - actual success(1)/fail(0) based on Triple Barrier
        """
        self.model.train()
        
        # Combine QDN features with scores
        X = np.hstack([features, scores.reshape(-1, 1)])
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(labels.reshape(-1, 1))
        w_tensor = torch.FloatTensor(sample_weights.reshape(-1, 1))
        
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        
        # Weighted loss (higher weight for more unique samples)
        loss = (self.criterion(outputs, y_tensor) * w_tensor).mean()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def predict_confidence(self, features: np.ndarray, primary_score: float) -> float:
        """
        Predict probability of success for sizing.
        """
        self.model.eval()
        with torch.no_grad():
            X = np.append(features, [primary_score]).reshape(1, -1)
            X_tensor = torch.FloatTensor(X)
            conf = self.model(X_tensor).item()
        return float(conf)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info("Meta-Labeler weights loaded.")
        except Exception as e:
            logger.warning(f"Failed to load Meta-Labeler: {e}")
