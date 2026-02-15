"""
Market Microstructure Features — Phase 3

Implements:
1. Hurst Exponent (Trend Persistence)
2. VPIN (Volume-synchronized Probability of Informed Trading)
3. Levy Distribution (Stable) tail parameters
"""

import numpy as np
import pandas as pd
from scipy.stats import levy_stable
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class MicrostructureAnalyzer:
    """
    Computes advanced microstructure signals for price and volume series.
    """

    @staticmethod
    def compute_hurst(series: np.ndarray, max_lag: int = 50) -> float:
        """
        Calculate Hurst Exponent using R/S Analysis.
        H < 0.5: Anti-persistent (mean-reverting)
        H = 0.5: Random walk
        H > 0.5: Persistent (trending)
        """
        if len(series) < max_lag * 2:
            return 0.5

        lags = range(2, max_lag)
        # Rescaled Range (R/S) 
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        
        # log(R/S) = H * log(tau)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst is the slope (with some scaling adjustment for standard R/S)
        # Simplified version for features
        return float(poly[0] * 2.0)

    @staticmethod
    def compute_vpin(
        prices: np.ndarray, 
        volumes: np.ndarray, 
        n_buckets: int = 10
    ) -> float:
        """
        Volume-synchronized Probability of Informed Trading (VPIN).
        Buckets data into 'n_buckets' of equal volume.
        Measures the average imbalance across these buckets.
        """
        if len(prices) < n_buckets * 2:
            return 0.5

        # Calculate returns for classification
        returns = np.diff(prices) / prices[:-1]
        vols = volumes[1:]
        
        # Total volume per bucket
        total_vol = np.sum(vols)
        if total_vol == 0:
            return 0.5
            
        bucket_vol = total_vol / n_buckets
        
        imbalances = []
        current_buy = 0
        current_sell = 0
        current_bucket_vol = 0
        
        for r, v in zip(returns, vols):
            # Classify volume (Lee-Ready simplified)
            if r > 0:
                current_buy += v
            else:
                current_sell += v
            
            current_bucket_vol += v
            
            # If bucket is full, calculate imbalance and reset
            if current_bucket_vol >= bucket_vol:
                imbalances.append(abs(current_buy - current_sell))
                current_buy = 0
                current_sell = 0
                current_bucket_vol = 0
        
        if not imbalances:
            return 0.5
            
        # VPIN is the average imbalance normalized by bucket volume
        return float(np.mean(imbalances) / bucket_vol)

    @staticmethod
    def estimate_tail_parameters(returns: np.ndarray) -> Tuple[float, float]:
        """
        Estimate alpha and beta parameters of the Levy Stable distribution.
        alpha (stability): captures the fatness of tails (1.0 = Cauchy, 2.0 = Gaussian)
        beta (skewness): captures asymmetry in tails
        """
        if len(returns) < 30:
            return 2.0, 0.0 # Default to Gaussian, symmetric

        try:
            # Fit stable distribution via quantile method (faster than MLE)
            params = levy_stable.fit(returns)
            # alpha is index 0, beta is index 1
            return float(params[0]), float(params[1])
        except Exception as e:
            logger.debug(f"Stable fit error: {e}")
            return 2.0, 0.0
