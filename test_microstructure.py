"""
Verification script for Microstructure Features (Hurst & VPIN)
"""

import numpy as np
import pandas as pd
from whaletracker.src.qdn.features.microstructure import MicrostructureAnalyzer

def verify_microstructure():
    analyzer = MicrostructureAnalyzer()
    
    # 1. Verify Hurst Exponent
    # Trending series -> H > 0.5
    trending = np.cumsum(np.random.normal(0.5, 0.1, 100))
    # Mean-reverting -> H < 0.5
    reverting = np.random.normal(0, 1, 100)
    
    h_trend = analyzer.compute_hurst(trending)
    h_rev = analyzer.compute_hurst(reverting)
    
    print(f"Hurst (Trending): {h_trend:.4f}")
    print(f"Hurst (Mean-Reverting): {h_rev:.4f}")
    
    # 2. Verify VPIN
    # High toxic volume (all buys/sells in one direction)
    prices_toxic = np.linspace(100, 110, 50)
    volumes_high = np.ones(50) * 1000
    
    vpin_toxic = analyzer.compute_vpin(prices_toxic, volumes_high)
    print(f"VPIN (Toxic/One-Way): {vpin_toxic:.4f}")
    
    # Random volume
    prices_rand = 100 + np.random.normal(0, 1, 50)
    vpin_rand = analyzer.compute_vpin(prices_rand, volumes_high)
    print(f"VPIN (Random): {vpin_rand:.4f}")
    
    # 3. Verify Tail Stability (Levy)
    # Gaussian (alpha=2, beta=0)
    returns_gauss = np.random.normal(0, 0.01, 200)
    alpha_g, beta_g = analyzer.estimate_tail_parameters(returns_gauss)
    print(f"Stable Parameters (Gaussian): Alpha={alpha_g:.2f}, Beta={beta_g:.2f}")
    
    # Cauchy-like (fat tails)
    returns_fat = np.random.standard_cauchy(200) * 0.01
    alpha_f, beta_f = analyzer.estimate_tail_parameters(returns_fat)
    print(f"Stable Parameters (Fat Tails): Alpha={alpha_f:.2f}, Beta={beta_f:.2f}")

    # Final Check
    if h_trend > h_rev and vpin_toxic > 0.8 and alpha_f < 1.8:
        print("\n[VERIFIED] Microstructure features are capturing signal correctly.")
    else:
        print("\n[WARNING] Feature behavior unexpected.")

if __name__ == "__main__":
    verify_microstructure()
