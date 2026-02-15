from typing import Dict, Any
import pandas as pd

class ScoringEngine:
    """
    The 'Taleb Filter' Engine.
    Assigns a score to transactions based on convexity, skin in the game, and strategic alignment.
    """
    
    STRATEGIC_SECTORS = [
        "Defensa", "Aeroespacial", "Tecnología", "IA", 
        "Semiconductores", "Energía", "Biomedicina", "Salud"
    ]

    def __init__(self, historical_performance: Dict[str, float] = None):
        """
        Args:
            historical_performance: Dict mapping insider names to their historical success rate (0-1).
        """
        self.historical_performance = historical_performance or {}

    def calculate_score(self, trade: Dict[str, Any], company_info: Dict[str, Any]) -> float:
        """
        Calculates the Antifragility Score (0-100).
        
        Args:
            trade: {insider, type, amount, committee, etc.}
            company_info: {sector, market_cap, ticker}
        """
        score = 0
        
        # 1. Sector Alignment (+25 points)
        # Score higher if sector is strategic
        if any(s.lower() in str(company_info.get('sector', '')).lower() for s in self.STRATEGIC_SECTORS):
            score += 25
            
        # 2. Skin in the Game (+25 points)
        # Only favor real purchases. Ignore options/awards.
        if trade.get('type', '').upper() == 'PURCHASE':
            score += 25
        else:
            # Sales or other types are less interesting for the 'accelerator' signal
            return 0  # We only bet on positive convexity (purchases)
            
        # 3. Convexity / Market Cap (+30 points)
        # Small caps allow for multipliers (x5, x8)
        mcap = company_info.get('market_cap', 0)
        if 0 < mcap < 2e9:  # < 2 Billion (Small Cap)
            score += 30
        elif 2e9 <= mcap < 10e9:  # 2B - 10B (Mid Cap)
            score += 15
        elif mcap >= 10e9:  # Large Cap
            score += 5
            
        # 4. Authority / Cargo (+20 points)
        # If the politician is in a committee related to the sector
        committee = str(trade.get('committee', '')).lower()
        sector = str(company_info.get('sector', '')).lower()
        
        if self._is_committee_related(committee, sector):
            score += 20
        
        # 5. Historical Success Bonus (+10 points)
        performance = self.historical_performance.get(trade['insider'], 0.5)
        if performance > 0.7:  # High success rate top 30%
            score += 10
            
        return min(score, 100)

    def _is_committee_related(self, committee: str, sector: str) -> bool:
        """Heuristic to check if a committee matches a sector."""
        relations = {
            "armed services": ["defense", "aerospace", "defensa"],
            "health": ["biomedicine", "health", "pharmaceutical", "biomedicina"],
            "energy": ["energy", "oil", "gas", "utilities"],
            "commerce": ["technology", "chips", "semiconductors"]
        }
        for comm_key, sectors in relations.items():
            if comm_key in committee:
                return any(s in sector for s in sectors)
        return False

if __name__ == "__main__":
    # Test cases
    engine = ScoringEngine(historical_performance={"Mark Green": 0.9})
    
    # Case 1: Strategic Small Cap Purchase by High Performance Insider
    trade1 = {"insider": "Mark Green", "type": "PURCHASE", "committee": "Armed Services"}
    info1 = {"sector": "Defense Technology", "market_cap": 500e6, "ticker": "SMALL"}
    
    # Expected: 25 (sector) + 25 (purchase) + 30 (small cap) + 20 (committee) + 10 (perf) = 110 -> 100
    print(f"Score 1: {engine.calculate_score(trade1, info1)}") 
    
    # Case 2: Large Cap Generic Purchase
    trade2 = {"insider": "Generic Senator", "type": "PURCHASE", "committee": "Ethics"}
    info2 = {"sector": "Retail", "market_cap": 200e9, "ticker": "WMT"}
    
    # Expected: 0 (sector) + 25 (purchase) + 5 (large cap) + 0 (committee) + 0 (perf) = 30
    print(f"Score 2: {engine.calculate_score(trade2, info2)}")
