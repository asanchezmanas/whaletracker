"""
Fractional Differentiation (FFD) — López de Prado

Implements Fixed-Width Window Fractional Differentiation.
Standard integer differentiation (d=1, d=2) removes too much memory.
Fractional differentiation (e.g., d=0.4) achieves stationarity while 
preserving correlation with original price series.

Reference: AFML Chapter 5
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FractionalDifferencer:
    """
    Fixed-Width Window FracDiff (FFD) engine.
    """

    @staticmethod
    def get_weights(d: float, size: int) -> np.ndarray:
        """
        Compute weights for fractional differentiation using binomial expansion.
        w_k = -w_{k-1} * (d - k + 1) / k
        """
        w = [1.0]
        for k in range(1, size):
            w_k = -w[-1] * (d - k + 1) / k
            w.append(w_k)
        return np.array(w[::-1]).reshape(-1, 1)

    @staticmethod
    def get_weights_ffd(d: float, threshold: float) -> np.ndarray:
        """
        Compute weights until the absolute value of the weight falls below threshold.
        This provides the fixed-width window size.
        """
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < threshold:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def apply_ffd(self, series: pd.DataFrame, d: float, threshold: float = 1e-5) -> pd.DataFrame:
        """
        Apply Fixed-Width Window Fractional Differentiation to a series.
        Uses convolution for performance.
        """
        weights = self.get_weights_ffd(d, threshold).flatten()
        width = len(weights)
        
        df = series.copy()
        output = {}
        
        for name in df.columns:
            # Fill NaNs before applying
            vals = df[name].ffill().bfill().values
            
            if len(vals) < width:
                logger.warning(f"Series {name} is shorter ({len(vals)}) than required FFD window ({width})")
                output[name] = np.array([])
                continue

            # Convolution (valid mode gives us the same as the sliding window dot product)
            res = np.convolve(vals, weights, mode='valid')
            output[name] = res
            
        # Ensure all columns have data before creating DF
        if any(len(v) == 0 for v in output.values()):
            return pd.DataFrame()

        return pd.DataFrame(output, index=df.index[width - 1:])

    def find_min_d(
        self, 
        series: pd.Series, 
        d_range: Tuple[float, float, int] = (0.0, 1.0, 11),
        threshold: float = 1e-4
    ) -> float:
        """
        Search for the minimum 'd' that makes the series stationary (ADF test).
        The goal is the smallest 'd' that achieves stationarity to preserve memory.
        """
        from statsmodels.tsa.stattools import adfuller
        
        d_values = np.linspace(d_range[0], d_range[1], d_range[2])
        best_d = 1.0
        
        for d in d_values:
            if d == 0: continue
            
            ffd = self.apply_ffd(pd.DataFrame(series), d, threshold)
            if len(ffd) < 20: continue # Need enough samples for ADF
            
            # ADF test
            res = adfuller(ffd.iloc[:, 0], maxlag=1, regression='c', autolag=None)
            p_val = res[1]
            
            if p_val < 0.05:
                logger.debug(f"d={d:.2f} achieved stationarity (p-val={p_val:.4f})")
                return d
                
        return best_d
