"""
Adversarial Training System

Three adversarial strategies to build robustness:
1. FGSM noise injection — perturbation that maximizes loss
2. Fake insider generation — synthetic non-informative trades 
3. Synthetic crisis injection — stress market conditions

The model must learn to:
- Be stable under input perturbation (FGSM)
- Reject noise traders and assign them low scores (fake insiders)
- Maintain performance during extreme market conditions (crisis)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

from .config import TrainingConfig
from .dense_network import DenseNetwork


class AdversarialTrainer:
    """
    Wraps the training loop with adversarial augmentation.
    
    Training schedule per batch:
      50% clean data
      20% FGSM adversarial noise
      15% mixed with fake insiders
      15% synthetic crisis conditions
    """

    def __init__(
        self,
        model: DenseNetwork,
        config: TrainingConfig,
        feature_indices: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.config = config
        self.feature_indices = feature_indices or {}

        # Adversarial components
        self.noise_generator = FGSMGenerator(config.adversarial_epsilon)
        self.fake_generator = FakeInsiderGenerator(
            n_features=model.config.n_features
        )
        self.crisis_simulator = CrisisSimulator(feature_indices)

    def augment_batch(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        """
        Apply adversarial augmentation to a batch.
        
        Returns augmented (features, labels, weights) and the augmentation type.
        """
        roll = np.random.random()

        if roll < 0.50:
            # Clean — no augmentation
            return features, labels, weights, "clean"

        elif roll < 0.70:
            # FGSM noise
            adv_features = self.noise_generator.generate(
                self.model, features, labels
            )
            return adv_features, labels, weights, "fgsm"

        elif roll < 0.85:
            # Mix with fake insiders
            adv_features, adv_labels, adv_weights = self.fake_generator.mix_with_fakes(
                features, labels, weights
            )
            return adv_features, adv_labels, adv_weights, "fake_mixed"

        else:
            # Crisis simulation
            crisis_features = self.crisis_simulator.inject_crisis(features)
            return crisis_features, labels, weights, "crisis"

    def get_adversarial_weight(self, adv_type: str) -> float:
        """
        Adversarial examples get lower weight in the loss
        to prevent them from dominating training.
        """
        weights = {
            "clean": 1.0,
            "fgsm": 0.5,
            "fake_mixed": 0.5,
            "crisis": 0.7,
        }
        return weights.get(adv_type, 1.0)


class FGSMGenerator:
    """
    Fast Gradient Sign Method adversarial noise.
    
    Adds minimal perturbation in the direction that maximizes loss.
    This forces the model to be robust to small input variations.
    """

    def __init__(self, epsilon: float = 0.01):
        self.epsilon = epsilon

    def generate(
        self,
        model: DenseNetwork,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            model: The model to attack
            features: Clean input features
            labels: True labels (for computing loss gradient)
        
        Returns:
            Perturbed features
        """
        # Enable gradient tracking on input
        adv_features = features.clone().detach().requires_grad_(True)

        # Forward pass
        model.eval()
        output = model(adv_features)

        # Compute gradient of score w.r.t. input
        score = output["convexity_score"].mean()
        score.backward()

        # Perturbation in direction of gradient sign
        grad_sign = adv_features.grad.sign()
        perturbation = self.epsilon * grad_sign

        # Apply perturbation
        perturbed = adv_features.detach() + perturbation

        # Clamp to reasonable feature range (assumes normalized features)
        perturbed = torch.clamp(perturbed, -5.0, 5.0)

        model.train()
        return perturbed


class FakeInsiderGenerator:
    """
    Generates synthetic non-informative insider trades.
    
    These are plausible-looking feature vectors that represent
    random/noise traders (not real insider signal). The model
    must learn to assign them low convexity scores.
    
    Characteristics of fake insiders:
    - Low historical win rate
    - No cluster activity
    - Low committee power
    - Random timing (no crisis-buying pattern)
    """

    def __init__(self, n_features: int):
        self.n_features = n_features

    def generate(self, batch_size: int) -> torch.Tensor:
        """Generate batch of fake insider feature vectors."""
        # Start with random noise in normal feature range
        fakes = torch.randn(batch_size, self.n_features) * 0.5

        # Force characteristics of non-informative traders:
        # These indices correspond to the feature engineering spec
        # and will be mapped properly when features are defined.
        # For now, we zero out key signal features to create
        # "no edge" profiles.
        
        # Zero out the first few features (typically power/influence)
        if self.n_features > 5:
            fakes[:, :3] = torch.rand(batch_size, 3) * 0.2  # Low power
        
        # Low win rate region
        if self.n_features > 10:
            fakes[:, 8:10] = torch.rand(batch_size, 2) * 0.3  # Low quality

        return fakes

    def mix_with_fakes(
        self,
        real_features: torch.Tensor,
        real_labels: torch.Tensor,
        real_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Mix real batch with fake insiders (25% fakes).
        
        Fake insiders get label = 0 (zero return expected).
        """
        batch_size = len(real_features)
        n_fakes = max(batch_size // 4, 1)

        # Generate fakes with zero-return labels
        fake_features = self.generate(n_fakes).to(real_features.device)
        fake_labels = torch.zeros(n_fakes, 1, device=real_labels.device)

        # Concatenate
        mixed_features = torch.cat([real_features, fake_features], dim=0)
        mixed_labels = torch.cat([real_labels, fake_labels], dim=0)

        # Handle weights
        mixed_weights = None
        if real_weights is not None:
            fake_weights = torch.ones(n_fakes, 1, device=real_weights.device)
            mixed_weights = torch.cat([real_weights, fake_weights], dim=0)

        # Shuffle
        perm = torch.randperm(len(mixed_features))
        mixed_features = mixed_features[perm]
        mixed_labels = mixed_labels[perm]
        if mixed_weights is not None:
            mixed_weights = mixed_weights[perm]

        return mixed_features, mixed_labels, mixed_weights


class CrisisSimulator:
    """
    Injects synthetic crisis conditions into feature vectors.
    
    Simulates extreme market conditions based on historical crises:
    - 2008 Financial Crisis: VIX 3x, liquidity 0.3x
    - 2020 COVID: VIX 4x, liquidity 0.2x, correlation spike
    - 2022 Inflation: VIX 2x, liquidity 0.6x
    
    The model must maintain discriminative ability even during stress.
    """

    CRISIS_PATTERNS = {
        "2008_financial": {
            "vix_multiplier": 3.0,
            "volatility_multiplier": 2.5,
            "liquidity_multiplier": 0.3,
        },
        "2020_covid": {
            "vix_multiplier": 4.0,
            "volatility_multiplier": 3.0,
            "liquidity_multiplier": 0.2,
        },
        "2022_inflation": {
            "vix_multiplier": 2.0,
            "volatility_multiplier": 1.5,
            "liquidity_multiplier": 0.6,
        },
    }

    def __init__(self, feature_indices: Optional[Dict[str, int]] = None):
        self.feature_indices = feature_indices or {}

    def inject_crisis(
        self, features: torch.Tensor, crisis_type: str = "random"
    ) -> torch.Tensor:
        """
        Transform features to simulate crisis conditions.
        
        If feature_indices are not mapped yet, applies general
        noise increase as a proxy for crisis conditions.
        """
        crisis_features = features.clone()

        if crisis_type == "random":
            crisis_type = np.random.choice(list(self.CRISIS_PATTERNS.keys()))

        pattern = self.CRISIS_PATTERNS[crisis_type]

        if "vix_index" in self.feature_indices:
            idx = self.feature_indices["vix_index"]
            crisis_features[:, idx] = torch.clamp(
                crisis_features[:, idx] * pattern["vix_multiplier"],
                0, 3.0,
            )

        if "volatility_percentile" in self.feature_indices:
            idx = self.feature_indices["volatility_percentile"]
            crisis_features[:, idx] = torch.clamp(
                crisis_features[:, idx] * pattern["volatility_multiplier"],
                0, 1.0,
            )

        if "float_percentage" in self.feature_indices:
            idx = self.feature_indices["float_percentage"]
            crisis_features[:, idx] = torch.clamp(
                crisis_features[:, idx] * pattern["liquidity_multiplier"],
                0, 1.0,
            )

        # If indices not mapped, add general noise to simulate stress
        if not self.feature_indices:
            noise = torch.randn_like(crisis_features) * 0.3
            crisis_features = crisis_features + noise
            crisis_features = torch.clamp(crisis_features, -5.0, 5.0)

        return crisis_features
