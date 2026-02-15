"""
Execution Simulator — Phase 5

Simulates real-world trading constraints:
1. Slippage (Spread + Volatility)
2. Market Impact (Volume-based)
3. Latency (Execution delay)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ExecutionSimulator:
    """
    Simulates how orders are filled in the real market.
    Essential for Penny Stocks and OTC trades.
    """

    def __init__(self, pct_of_volume_cap: float = 0.05, base_slippage_bps: int = 10):
        self.pct_of_volume_cap = pct_of_volume_cap  # Can't trade more than 5% of daily volume
        self.base_slippage_bps = base_slippage_bps

    def simulate_fill(
        self, 
        ticker: str, 
        requested_shares: int, 
        price: float, 
        daily_vol: float,
        is_otc: bool = False
    ) -> Dict:
        """
        Calculates execution price and filled shares.
        """
        if daily_vol <= 0:
            return {"shares": 0, "price": price, "slippage": 0}

        # 1. Volume Constraint (Liquidity Cap)
        max_shares = int(daily_vol * self.pct_of_volume_cap)
        filled_shares = min(abs(requested_shares), max_shares)
        
        if requested_shares < 0:
            filled_shares = -filled_shares

        # 2. Market Impact Model (Square Root Law)
        # Impact ~ sigma * sqrt(order_size / daily_vol)
        # For this sim, we'll use a simplified version:
        vol_ratio = abs(filled_shares) / daily_vol
        impact = 0.1 * np.sqrt(vol_ratio) # 10% sigma proxy
        
        # 3. Base Slippage (Spread)
        # Penny stocks and OTC have much higher spreads
        spread_bps = 100 if is_otc or price < 5.0 else self.base_slippage_bps
        slippage_pct = (spread_bps / 10000.0) + impact
        
        # Apply slippage
        # If buying, price goes UP. If selling, price goes DOWN.
        if filled_shares > 0:
            fill_price = price * (1 + slippage_pct)
        else:
            fill_price = price * (1 - slippage_pct)
            
        slippage_cost = abs(fill_price - price) * abs(filled_shares)

        return {
            "shares": filled_shares,
            "price": float(fill_price),
            "slippage_bps": float(slippage_pct * 10000),
            "slippage_cost": float(slippage_cost),
            "rejected_due_to_liquidity": abs(filled_shares) < abs(requested_shares)
        }

    def get_market_impact_summary(self, fill_report: Dict) -> str:
        if fill_report["rejected_due_to_liquidity"]:
            return f"LIQUIDITY REJECT: Filled only {fill_report['shares']} shares. Slippage: {fill_report['slippage_bps']:.1f} bps"
        return f"Fill OK: {fill_report['shares']} @ {fill_report['price']:.2f}. Slippage: {fill_report['slippage_bps']:.1f} bps"
