"""
Portfolio Manager — Phase 5

Implements:
1. Hierarchical Risk Parity (HRP) Allocation
2. Dynamic Kelly Sizing
3. Portfolio Constraint Engine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform

class PortfolioManager:
    """
    Manages capital allocation using HRP and Kelly sizing.
    Designed for high-volatility/low-liquidity whale signals.
    """

    def __init__(self, cash: float = 1000000, max_pos_size: float = 0.1, sector_limit: float = 0.3):
        self.cash = cash
        self.max_pos_size = max_pos_size  # Max 10% per signal
        self.sector_limit = sector_limit # Max 30% per sector
        self.positions = {} # ticker -> {shares, entry_price, type}

    def compute_hrp_weights(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Computes Hierarchical Risk Parity weights.
        Ref: López de Prado (2016)
        """
        if returns_df.empty or returns_df.shape[1] < 2:
            return pd.Series(1.0, index=returns_df.columns) if not returns_df.empty else pd.Series()

        # 1. Correlation and Distance Matrix
        corr = returns_df.corr().fillna(0)
        dist = np.sqrt(0.5 * (1 - corr))
        
        # 2. Quasi-Diagonalization
        link = sch.linkage(pdist(dist), method='single')
        sort_ix = sch.leaves_list(link)
        sorted_items = returns_df.columns[sort_ix].tolist()
        
        # 3. Recursive Bisection
        weights = pd.Series(1.0, index=sorted_items)
        items = [sorted_items]
        
        while len(items) > 0:
            items = [i[j:k] for i in items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(items), 2):
                c_items_1 = items[i]
                c_items_2 = items[i+1]
                
                c_cov_1 = returns_df[c_items_1].cov().values
                c_cov_2 = returns_df[c_items_2].cov().values
                
                # Inverse-variance allocation between the two clusters
                var_1 = self._get_cluster_var(c_cov_1)
                var_2 = self._get_cluster_var(c_cov_2)
                
                alpha_1 = 1 - var_1 / (var_1 + var_2)
                alpha_2 = 1 - alpha_1
                
                weights[c_items_1] *= alpha_1
                weights[c_items_2] *= alpha_2
                
        return weights

    def _get_cluster_var(self, cov):
        """Compute variance of a cluster using inverse-variance weights."""
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return np.dot(np.dot(ivp, cov), ivp)

    def get_kelly_size(self, p: float, win_rate: float = 0.55, risk_mult: float = 0.5) -> float:
        """
        Computes fractional Kelly size.
        p: model prediction (probability of success)
        win_rate: historical system win rate
        risk_mult: 0.5 for Half-Kelly (conservative)
        """
        # Simplified Kelly: f = (p * b - (1-p)) / b where b is odds.
        # Here we use the model's score 'p' as the probability.
        # f = 2p - 1 is the simplest form for 1:1 odds.
        
        f_star = max(0, 2 * p - 1)
        return float(f_star * risk_mult)

    def optimize_allocation(
        self, 
        signals: List[Dict], 
        returns_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Combines HRP and Kelly to produce final trade sizes.
        """
        if not signals:
            return {}

        df_signals = pd.DataFrame(signals)
        tickers = df_signals['ticker'].tolist()
        
        # 1. Base sizes from Kelly/Confidence
        base_sizes = {row['ticker']: self.get_kelly_size(row.get('score', 0.5)) for _, row in df_signals.iterrows()}
        
        # 2. Risk refinement with HRP (if history provided)
        if returns_history is not None and len(tickers) > 1:
            hrp_weights = self.compute_hrp_weights(returns_history[tickers])
            # Scale Kelly by HRP relative importance
            hrp_scaling = hrp_weights / hrp_weights.mean()
            for t in tickers:
                base_sizes[t] *= hrp_scaling[t]

        # 3. Constraints (Max Position)
        for t in base_sizes:
            base_sizes[t] = min(base_sizes[t], self.max_pos_size)
            
        return base_sizes
