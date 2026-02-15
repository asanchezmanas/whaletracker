"""
QDN Configuration

Central configuration for the Dense Network model,
training, and inference parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Dense Network architecture configuration."""

    # --- Feature dimensions ---
    n_features: int = 40  # Unified 40-feature set (Phase 4)
    
    # --- Network architecture ---
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3  # Aggressive regularization for low data regime
    activation: str = "leaky_relu"
    
    # --- VDL (Variational Decomposition Layer) ---
    vdl_hidden_dim: int = 32
    # Output heads: expected_return, downside_risk, upside_potential, tail_probability
    n_decomposition_heads: int = 4
    
    # --- Convexity score output ---
    score_range: tuple = (0, 100)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # --- Optimization ---
    learning_rate: float = 1e-3
    weight_decay: float = 0.01  # L2 regularization
    batch_size: int = 64
    max_epochs: int = 200
    
    # --- Early stopping ---
    early_stopping_patience: int = 20
    early_stopping_metric: str = "sortino"  # Optimize for Sortino, not MSE
    
    # --- Antifragile loss weights ---
    downside_penalty_multiplier: float = 10.0  # 10x penalty for underestimating risk
    calibration_loss_weight: float = 0.3
    tail_sigma_threshold: float = 2.0  # Events > 2 sigma considered tail events
    
    # --- Adversarial training ---
    adversarial_epsilon: float = 0.01  # FGSM perturbation magnitude
    adversarial_ratio: float = 0.30  # 30% of batches use adversarial examples
    fake_insider_ratio: float = 0.15
    synthetic_crisis_ratio: float = 0.15
    
    # --- Walk-forward validation ---
    walk_forward_train_months: int = 24  # 2 years training window
    walk_forward_test_months: int = 6   # 6 months test window
    walk_forward_step_months: int = 3   # Slide by 3 months
    purge_days: int = 7   # 7-day purge between train/test
    embargo_days: int = 30  # 30-day embargo after test set
    
    # --- Purged K-Fold CV (Phase 2) ---
    cv_folds: int = 5
    embargo_pct: float = 0.01  # Buffer as % of total data
    
    # --- Gradient clipping ---
    max_grad_norm: float = 1.0
    
    # --- Scheduler ---
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    initial_capital: float = 0.0  # Start with 0, add monthly
    monthly_contribution: float = 150.0  # EUR
    commission_rate: float = 0.001  # 0.1%
    slippage_bps: float = 5  # 5 basis points
    
    # --- Position sizing ---
    max_positions: int = 3
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    max_single_position_pct: float = 0.40  # Max 40% in one asset
    
    # --- Thresholds ---
    buy_score_threshold: float = 70.0  # Score > 70 = actionable
    strong_buy_threshold: float = 85.0
    
    # --- Exit rules ---
    stop_loss_pct: float = 0.15  # -15% hard stop
    take_profit_pct: float = 1.0  # +100% take profit (let winners run)
    max_holding_days: int = 365  # 1 year max hold


@dataclass
class QDNConfig:
    """Master configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    model_version: str = "v0.1.0"
    
    # --- Device ---
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def resolve_device(self) -> str:
        """Resolve device string to actual device."""
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device
