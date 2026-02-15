"""
Verification script for Fractional Differentiation (FFD)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from whaletracker.src.qdn.features.frac_diff import FractionalDifferencer

def verify_frac_diff():
    # 1. Generate non-stationary series (Random Walk with Drift)
    np.random.seed(42)
    n = 5000
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n)))
    series = pd.Series(prices, name='BasePrice')
    
    print(f"Original series length: {len(series)}")
    
    # Check original stationarity (should fail/high p-val)
    adf_orig = adfuller(series, maxlag=1, regression='c', autolag=None)
    print(f"Original ADF p-value: {adf_orig[1]:.4f}")
    
    # 2. Apply FracDiff
    fd = FractionalDifferencer()
    
    # Search for min d
    min_d = fd.find_min_d(series, d_range=(0.2, 0.9, 8), threshold=1e-4)
    print(f"Minimum d for stationarity: {min_d:.2f}")
    
    # Apply with optimized d
    ffd_df = fd.apply_ffd(pd.DataFrame(series), d=min_d, threshold=1e-4)
    
    if ffd_df.empty:
        print("[ERROR] FFD result is empty. Window size might be too large for data.")
        return

    # Check transformed stationarity
    adf_ffd = adfuller(ffd_df.iloc[:, 0], maxlag=1, regression='c', autolag=None)
    print(f"FFD (d={min_d:.2f}) ADF p-value: {adf_ffd[1]:.4f}")
    
    # Check memory retention (correlation with original)
    original_sliced = series.loc[ffd_df.index]
    correlation = np.corrcoef(original_sliced, ffd_df.iloc[:, 0])[0, 1]
    print(f"FFD Memory Retention (Corr with Original): {correlation:.4f}")
    
    # Compare with standard diff (d=1)
    d1_series = series.diff().dropna()
    d1_orig = series.loc[d1_series.index]
    d1_corr = np.corrcoef(d1_orig, d1_series)[0, 1]
    print(f"Standard Diff (d=1) Memory Retention: {d1_corr:.4f}")
    
    # 3. Final Check
    if adf_ffd[1] < 0.05:
        print(f"\n[VERIFIED] FFD (d={min_d:.2f}) is stationary (p={adf_ffd[1]:.4f})")
        if correlation > d1_corr:
            print(f"FFD wins! Preserved {correlation/d1_corr:.1f}x more memory than d=1.")
    else:
        print("\n[WARNING] Quality thresholds not met. Check logic or parameters.")

if __name__ == "__main__":
    verify_frac_diff()
