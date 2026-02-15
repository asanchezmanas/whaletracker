"""
Antifragile H-DQN Orchestrator — The 'Chef'
Implemented for Phase 5 of the WhaleTracker pivot.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from ..config import QDNConfig

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Manages capital allocation using the 'Micro-Lottery' (Seed-and-Burn) strategy.
    
    Responsibilities:
    1. Recurrent Capital Inflow (50€/month)
    2. Via Negativa Vetoing (Fragility Detection)
    3. Structural Gem Seeding (1€-5€ per ticket)
    4. Cluster Seeding (Diversity across pockets)
    5. The Fattening (Scaling winners)
    """

    def __init__(self, config: QDNConfig):
        self.config = config
        self.portfolio: Dict[str, Dict[str, Any]] = {} 
        self.cash = 0.0
        self.history: List[Dict[str, Any]] = [] 

    def add_monthly_fuel(self):
        """Injects the monthly contribution into the cash pool."""
        contribution = self.config.backtest.monthly_contribution
        self.cash += contribution
        logger.info(f"Orchestrator: Injected {contribution}€ monthly fuel. Total Cash: {self.cash:.2f}€")

    def allocate_seeds(self, candidates: List[Dict[str, Any]], current_date: pd.Timestamp):
        """
        Allocates available cash across structural gems.
        
        candidates: List of Dicts with {'ticker', 'score', 'features', 'price'}
        """
        # 1. Via Negativa: Remove candidates with 'Fragility' markers
        active_gems = [c for c in candidates if self._pass_veto_filter(c)]
        
        if not active_gems:
            return

        # 2. Sort by score (Structural Confidence)
        active_gems = sorted(active_gems, key=lambda x: x['score'], reverse=True)
        
        # 3. Micro-Lottery Sowing
        for gem in active_gems:
            ticker = gem['ticker']
            score = gem['score']
            
            # Check if we have enough cash for at least a minimum seed
            min_seed = self.config.backtest.seed_size_range[0]
            if self.cash < min_seed:
                break
                
            # Determine seed size (1€-5€)
            # If score is extremely high (Fattening), we go for the max seed
            max_seed = self.config.backtest.seed_size_range[1]
            seed_size = max_seed if score >= self.config.backtest.fatten_threshold else min_seed
            
            # Clamp to available cash
            seed_size = min(seed_size, self.cash)
            
            # Execute Seed
            self._execute_seed(ticker, seed_size, current_date, score, gem.get('price', 1.0))
            self.cash -= seed_size

    def _pass_veto_filter(self, candidate: Dict[str, Any]) -> bool:
        """
        Implements the 'No' logic of Via Negativa.
        Blocks trades that exhibit structural fragility.
        """
        features = candidate.get('features')
        if features is None or len(features) < 20: 
            return True # Fallback if features missing or incomplete
        
        # Rule 1: Cash Runway Veto (Feature index 19)
        # If runway < 3 months (score < 0.125), veto.
        if features[19] < 0.125:
            logger.info(f"VETO: {candidate['ticker']} blocked due to Low Cash Runway (Value: {features[19]:.2f}).")
            return False
            
        # Rule 2: Extreme Short Interest (Feature index 17)
        # If short interest > 40% (score > 0.8), veto (too fragile/crowded).
        if features[17] > 0.8:
            logger.info(f"VETO: {candidate['ticker']} blocked due to high Crowding/Short interest (Value: {features[17]:.2f}).")
            return False

        return True

    def _execute_seed(self, ticker: str, amount: float, date: pd.Timestamp, score: float, current_price: float):
        """Updates portfolio state with a new or scaled seed."""
        shares_to_buy = amount / current_price
        
        if ticker in self.portfolio:
            self.portfolio[ticker]['total_invested'] += amount
            self.portfolio[ticker]['shares'] += shares_to_buy
            logger.info(f"FATTENING: Increased stake in {ticker} by {amount}€ (New Total: {self.portfolio[ticker]['total_invested']:.2f}€)")
        else:
            self.portfolio[ticker] = {
                'total_invested': amount,
                'shares': shares_to_buy,
                'entry_date': date,
                'entry_score': score,
                'ticker': ticker
            }
            logger.info(f"SOWING: Seeded {ticker} with {amount}€ (Structural Alpha Score: {score:.1f})")
            
        self.history.append({
            'date': date,
            'ticker': ticker,
            'amount': amount,
            'shares': shares_to_buy,
            'type': 'SEED',
            'score': score
        })

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns statistics about the structural farm."""
        return {
            'n_positions': len(self.portfolio),
            'total_invested': sum(p['total_invested'] for p in self.portfolio.values()),
            'remaining_cash': self.cash,
            'total_value': sum(p['total_invested'] for p in self.portfolio.values()) + self.cash # Simplified
        }
