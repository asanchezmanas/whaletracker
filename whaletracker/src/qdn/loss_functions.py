"""
Antifragile Loss Functions

Loss design principles (from Taleb):
1. Asymmetric penalty: underestimating risk is 10x worse than missing upside
2. Tail awareness: reward correct identification of extreme positive events
3. Ruin avoidance: catastrophic penalty for undetected large losses
4. Separated calibration: VDL heads trained independently from convexity scorer

The loss is split into two independent components:
- Prediction Loss: operates on convexity_score vs actual_return
- Calibration Loss: operates on VDL decomposition heads vs derived labels

This separation prevents the circular dependency where the model could
learn to manipulate its own decomposition outputs to minimize loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .config import TrainingConfig


class AntifragileLoss(nn.Module):
    """
    Combined antifragile loss function.
    
    Total Loss = Prediction Loss + λ * Calibration Loss
    
    Where:
      Prediction Loss = Asymmetric MSE + Ruin Penalty
      Calibration Loss = Return MSE + Risk BCE + Tail BCE
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.prediction_loss = PredictionLoss(config)
        self.calibration_loss = CalibrationLoss(config)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        actual_returns: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            model_output: Dict from DenseNetwork.forward()
            actual_returns: Actual realized returns [batch_size, 1]
                           (fractional, e.g., 0.15 = +15%)
            sample_weights: Optional weights for each sample [batch_size, 1]
        
        Returns:
            Dict with total_loss, prediction_loss, calibration_loss,
            and component breakdowns for monitoring.
        """
        # Prediction loss (convexity_score vs actual)
        pred_loss, pred_details = self.prediction_loss(
            model_output["convexity_score"],
            actual_returns,
            sample_weights=sample_weights,
        )

        # Calibration loss (VDL heads vs derived labels)
        cal_loss, cal_details = self.calibration_loss(
            model_output["decomposition"],
            actual_returns,
            sample_weights=sample_weights,
        )

        # Combined
        total_loss = pred_loss + self.config.calibration_loss_weight * cal_loss

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss.detach(),
            "calibration_loss": cal_loss.detach(),
            **{f"pred/{k}": v for k, v in pred_details.items()},
            **{f"cal/{k}": v for k, v in cal_details.items()},
        }


class PredictionLoss(nn.Module):
    """
    Asymmetric prediction loss on convexity score.
    
    Maps actual returns to quality targets in [0, 100] scale:
      >+20%  → target 95 (exceptional trade)
      +10%   → target 80 (great trade)
      +5%    → target 65 (good trade)
      0%     → target 50 (neutral)
      -5%    → target 30 (poor trade)
      -15%   → target 10 (terrible trade)
      <-20%  → target 5  (catastrophic)
    
    Asymmetric weighting:
      Overestimating quality (high score + loss) → 10x penalty
      Underestimating quality (low score + gain) → 1x penalty
    
    Ruin penalty:
      If actual return < -15% and our score was > 50 → massive penalty
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.downside_mult = config.downside_penalty_multiplier

    @staticmethod
    def _return_to_target(returns: torch.Tensor) -> torch.Tensor:
        """
        Convert actual returns to quality targets in [0, 100].
        
        Uses a sigmoid-like mapping centered at 0 return:
        - Positive returns map to 50-100 (asymptotic at ~95)
        - Negative returns map to 0-50 (asymptotic at ~5)
        - Steeper penalty for losses (asymmetric)
        """
        # Scale returns: 10% return → sigmoid input ~2.0
        # This gives good spread across the 0-100 range
        scaled = returns * 15.0  # 10% → 1.5, 20% → 3.0, -15% → -2.25
        target = torch.sigmoid(scaled) * 90.0 + 5.0  # [5, 95] range
        return target

    def forward(
        self,
        convexity_score: torch.Tensor,
        actual_returns: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            convexity_score: [batch_size, 1] in [0, 100]
            actual_returns: [batch_size, 1] fractional returns
            sample_weights: Optional weights for each sample [batch_size, 1]
        """
        # Convert returns to quality targets on the same [0, 100] scale
        target_score = self._return_to_target(actual_returns)
        
        # Errors in score space (both in [0, 100])
        errors = target_score - convexity_score

        # Asymmetric weighting:
        # Overestimate (high score + actual was bad) → 10x penalty
        is_overestimate = (convexity_score > 50.0) & (actual_returns < 0)
        weights = torch.where(
            is_overestimate,
            torch.tensor(self.downside_mult, device=errors.device),
            torch.tensor(1.0, device=errors.device),
        )
        
        # Normalize by 100² to keep loss magnitude manageable
        asymmetric_mse_elements = weights * (errors / 100.0).pow(2)
        
        # Apply sample weights if provided
        if sample_weights is not None:
            asymmetric_mse_elements = asymmetric_mse_elements * sample_weights

        asymmetric_mse = asymmetric_mse_elements.mean()

        # Ruin penalty: large loss when we said "buy"
        ruin_mask = (actual_returns < -0.15) & (convexity_score > 50.0)
        ruin_penalty_elements = torch.where(
            ruin_mask,
            (convexity_score / 100.0 * 5.0).pow(2),  # Quadratic penalty
            torch.zeros_like(convexity_score),
        )

        if sample_weights is not None:
            ruin_penalty_elements = ruin_penalty_elements * sample_weights

        ruin_penalty = ruin_penalty_elements.mean()

        total = asymmetric_mse + ruin_penalty

        return total, {
            "asymmetric_mse": asymmetric_mse.detach(),
            "ruin_penalty": ruin_penalty.detach(),
            "target_score_mean": target_score.mean().detach(),
        }


class CalibrationLoss(nn.Module):
    """
    Calibration loss for VDL decomposition heads.
    
    Trains each VDL head against INDEPENDENTLY DERIVED labels
    (not from the model's own outputs — this prevents circular dependency).
    
    Labels derived from actual_returns:
      - expected_return label: actual_returns directly
      - downside_risk label: P(return < -5%) derived as binary indicator
      - tail_probability label: P(|return| > 2σ) derived as binary indicator
      - upside_potential: max(actual_return, 0) as magnitude target
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.tail_threshold_sigma = config.tail_sigma_threshold
        self.downside_threshold = -0.05  # -5% is "significant loss"

    def forward(
        self,
        decomposition: Dict[str, torch.Tensor],
        actual_returns: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            decomposition: VDL output dict
            actual_returns: [batch_size, 1] fractional returns
            sample_weights: Optional weights for each sample [batch_size, 1]
        """
        # Derived labels
        risk_label = (actual_returns < self.downside_threshold).float()
        upside_label = F.relu(actual_returns)
        
        returns_std = actual_returns.std() + 1e-8
        returns_mean = actual_returns.mean()
        threshold = returns_mean + self.tail_threshold_sigma * returns_std
        tail_label = (actual_returns.abs() > threshold.abs()).float()

        # Total and Apply Weights
        if sample_weights is not None:
            return_loss = (F.mse_loss(decomposition["expected_return"], actual_returns, reduction='none') * sample_weights).mean()
            risk_loss = (F.binary_cross_entropy(decomposition["downside_risk"], risk_label, reduction='none') * sample_weights).mean()
            upside_loss = (F.mse_loss(decomposition["upside_potential"], upside_label, reduction='none') * sample_weights).mean()
            tail_loss = (F.binary_cross_entropy(decomposition["tail_probability"], tail_label, reduction='none') * sample_weights).mean()
        else:
            return_loss = F.mse_loss(decomposition["expected_return"], actual_returns)
            risk_loss = F.binary_cross_entropy(decomposition["downside_risk"], risk_label)
            upside_loss = F.mse_loss(decomposition["upside_potential"], upside_label)
            tail_loss = F.binary_cross_entropy(decomposition["tail_probability"], tail_label)

        total = return_loss + risk_loss + upside_loss + tail_loss

        return total, {
            "return_mse": return_loss.detach(),
            "risk_bce": risk_loss.detach(),
            "upside_mse": upside_loss.detach(),
            "tail_bce": tail_loss.detach(),
        }
