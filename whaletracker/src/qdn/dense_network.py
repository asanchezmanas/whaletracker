"""
Dense Network with Variational Decomposition Layer (VDL)

Core QDN model architecture:
- Multi-layer perceptron backbone with residual connections
- VDL: decomposes output into (expected_return, downside_risk, upside_potential, tail_probability)
- Convexity scoring head: 0-100 score
- EVT-inspired tail modeling

Based on Taleb's principles:
- Asymmetric payoff detection (convexity)
- Fat tail awareness (EVT)
- Antifragile signal prioritization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .config import ModelConfig


class ResidualBlock(nn.Module):
    """
    Residual block with dropout and layer normalization.
    
    Residual connections help with gradient flow in deeper networks
    and act as implicit regularization.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class VariationalDecompositionLayer(nn.Module):
    """
    Variational Decomposition Layer (VDL)
    
    Decomposes the latent representation into four interpretable components:
    1. expected_return: Predicted directional return (unbounded)
    2. downside_risk: Probability of significant loss [0, 1]
    3. upside_potential: Magnitude of potential gain [0, ∞)
    4. tail_probability: Probability of extreme event [0, 1]
    
    Each head has its own sub-network with appropriate activation functions.
    This separation enables independent calibration and the separated loss
    function (prediction loss + calibration loss) that avoids circular
    dependency.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()

        # Head 1: Expected return (linear output, unbounded)
        self.return_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Head 2: Downside risk (sigmoid, probability [0,1])
        self.risk_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Head 3: Upside potential (softplus, positive magnitude [0, ∞))
        self.upside_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Head 4: Tail probability (sigmoid, probability [0,1])
        self.tail_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "expected_return": self.return_head(z),
            "downside_risk": self.risk_head(z),
            "upside_potential": self.upside_head(z),
            "tail_probability": self.tail_head(z),
        }


class ConvexityScorer(nn.Module):
    """
    Convexity scoring head.
    
    Combines VDL decomposition into a single 0-100 convexity score.
    Score semantics:
      85-100: STRONG BUY (high convexity, low risk, tail potential)
      70-84:  BUY (positive asymmetry detected)
      50-69:  WATCH (marginal signal)
      0-49:   PASS (no asymmetric edge)
    
    The scorer learns a non-linear combination of the VDL heads,
    with the latent representation providing additional context.
    """

    def __init__(self, latent_dim: int, vdl_output_dim: int = 4):
        super().__init__()
        combined_dim = latent_dim + vdl_output_dim

        self.scorer = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output [0, 1], then scale to [0, 100]
        )

    def forward(
        self, z: torch.Tensor, vdl_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Concatenate latent repr with VDL heads.
        # Use reduced gradient flow (10%) instead of full detach —
        # lets the scorer learn from VDL decomposition while keeping
        # the calibration loss mostly independent.
        vdl_concat = torch.cat(
            [
                vdl_output["expected_return"],
                vdl_output["downside_risk"],
                vdl_output["upside_potential"],
                vdl_output["tail_probability"],
            ],
            dim=-1,
        )
        # 30% gradient flow: enough to learn quality ranking
        vdl_reduced = vdl_concat * 0.3 + vdl_concat.detach() * 0.7
        combined = torch.cat([z, vdl_reduced], dim=-1)
        return self.scorer(combined) * 100.0  # Scale to [0, 100]


class DenseNetwork(nn.Module):
    """
    QDN Dense Network — core model.
    
    Architecture:
    ┌──────────────────────┐
    │   Input Features     │  (n_features)
    └──────┬───────────────┘
           │
    ┌──────▼───────────────┐
    │   Projection Layer   │  (n_features → hidden_dims[0])
    └──────┬───────────────┘
           │
    ┌──────▼───────────────┐
    │   Residual Blocks    │  (hidden_dims[0] → ... → hidden_dims[-1])
    │   + LayerNorm        │
    │   + Dropout          │
    └──────┬───────────────┘
           │
    ┌──────▼───────────────┐
    │   VDL Decomposition  │  → expected_return
    │   (4 independent     │  → downside_risk
    │    heads)            │  → upside_potential
    └──────┬───────────────┘  → tail_probability
           │
    ┌──────▼───────────────┐
    │   Convexity Scorer   │  → score [0-100]
    └──────────────────────┘
    
    Forward returns a dict with all outputs for use by the loss function.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Input projection ---
        dims = [config.n_features] + config.hidden_dims
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(config.dropout_rate))
        
        self.backbone = nn.Sequential(*layers)

        # --- Residual refinement at the bottleneck ---
        bottleneck_dim = config.hidden_dims[-1]
        self.residual = ResidualBlock(bottleneck_dim, config.dropout_rate)

        # --- VDL ---
        self.vdl = VariationalDecompositionLayer(
            input_dim=bottleneck_dim,
            hidden_dim=config.vdl_hidden_dim,
        )

        # --- Convexity scorer ---
        self.scorer = ConvexityScorer(
            latent_dim=bottleneck_dim,
            vdl_output_dim=4,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Widen scorer's final layer weights for broader initial score spread.
        # Default Xavier produces sigmoid outputs near 0.5 → scores ~50.
        # 3x scaling spreads initial scores across [20, 80] range.
        scorer_layers = list(self.scorer.scorer.children())
        for layer in scorer_layers:
            if isinstance(layer, nn.Linear):
                final_linear = layer  # Last Linear found
        final_linear.weight.data *= 3.0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input feature tensor [batch_size, n_features]
            
        Returns:
            Dict with keys:
                - convexity_score: [batch_size, 1] in [0, 100]
                - decomposition: Dict with expected_return, downside_risk,
                                 upside_potential, tail_probability
                - latent: [batch_size, bottleneck_dim] latent representation
        """
        # Backbone
        z = self.backbone(x)
        z = self.residual(z)

        # VDL decomposition
        decomposition = self.vdl(z)

        # Convexity score
        convexity_score = self.scorer(z, decomposition)

        return {
            "convexity_score": convexity_score,
            "decomposition": decomposition,
            "latent": z,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Model summary string."""
        n_params = self.count_parameters()
        return (
            f"DenseNetwork(\n"
            f"  features={self.config.n_features},\n"
            f"  hidden_dims={self.config.hidden_dims},\n"
            f"  dropout={self.config.dropout_rate},\n"
            f"  vdl_hidden={self.config.vdl_hidden_dim},\n"
            f"  total_params={n_params:,}\n"
            f")"
        )
