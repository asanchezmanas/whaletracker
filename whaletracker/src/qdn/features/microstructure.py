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
        if len(series) < 50:
            return 0.5

        # Standard R/S Analysis
        lags = range(10, min(len(series) // 2, 500))
        rs_list = []
        for lag in lags:
            # Rescale the series into chunks of size 'lag'
            n_chunks = len(series) // lag
            if n_chunks == 0: continue
            
            chunk_rs = []
            for i in range(n_chunks):
                chunk = series[i*lag : (i+1)*lag]
                if len(chunk) < 2: continue
                # Real R/S logic: Range of cumulative deviations / Std
                m = np.mean(chunk)
                z = np.cumsum(chunk - m)
                r = np.max(z) - np.min(z)
                s = np.std(chunk, ddof=1)
                if s > 1e-12:
                    chunk_rs.append(r / s)
            
            if chunk_rs:
                rs_list.append(np.mean(chunk_rs))
            else:
                lags = [l for l in lags if l != lag] # adjust lags if no valid chunks

        if len(rs_list) < 5:
            return 0.5

        # log(R/S) = H * log(n) + c
        poly = np.polyfit(np.log(lags[:len(rs_list)]), np.log(rs_list), 1)
        return float(np.clip(poly[0], 0.0, 1.0))

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
