import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Callable
from engines.scoring import ScoringEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    The 'Oracle' Engine.
    Simulates the strategy using historical data, selecting trades via the ScoringEngine.
    """
    
    def __init__(self, scoring_engine: ScoringEngine, monthly_contribution: float = 150.0):
        self.scoring_engine = scoring_engine
        self.monthly_contribution = monthly_contribution
        self.cash = 0.0
        self.portfolio = {}  # {ticker: {'shares': float, 'entry_price': float, 'entry_date': datetime, 'score': float}}
        self.audit_log = []
        
    def run_simulation(self, trades_df: pd.DataFrame, price_fetcher: Callable[[str, datetime], float], company_info_provider: Callable[[str], Dict]):
        """
        Runs the simulation month by month.
        """
        trades_df['report_date'] = pd.to_datetime(trades_df['report_date'])
        trades_df = trades_df.sort_values('report_date')
        
        start_date = trades_df['report_date'].min().replace(day=1)
        end_date = trades_df['report_date'].max() + timedelta(days=32)
        
        current_date = start_date
        while current_date < end_date:
            # 1. Monthly contribution (The 'Cubo' logic)
            self.cash += self.monthly_contribution
            
            # 2. Identify trades reported this month
            next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
            monthly_trades = trades_df[(trades_df['report_date'] >= current_date) & (trades_df['report_date'] < next_month)]
            
            # 3. Handle signals
            potential_buys = []
            for _, trade in monthly_trades.iterrows():
                ticker = trade['ticker']
                
                # Check for Mirror Exit (Selling when the insider sells)
                if trade['type'].upper() == 'SALE' and ticker in self.portfolio:
                    execution_date = trade['report_date'] + timedelta(days=2)
                    price = price_fetcher(ticker, execution_date)
                    self._execute_sell(ticker, execution_date, price, reason=f"Mirror Exit ({trade['insider']} sold)")
                    continue
                
                # Analyze for Purchase
                if trade['type'].upper() == 'PURCHASE':
                    company_info = company_info_provider(ticker)
                    score = self.scoring_engine.calculate_score(trade.to_dict(), company_info)
                    
                    if score >= 85: # Threshold of High Conviction
                        potential_buys.append({
                            'trade': trade,
                            'score': score,
                            'company_info': company_info
                        })
            
            # 4. Execute Top Buys (Dividing the monthly capital)
            if potential_buys:
                # Deduplicate and pick top 3
                potential_buys = sorted(potential_buys, key=lambda x: x['score'], reverse=True)[:3]
                budget_per_trade = self.cash / len(potential_buys)
                
                for signal in potential_buys:
                    trade = signal['trade']
                    ticker = trade['ticker']
                    execution_date = trade['report_date'] + timedelta(days=2)
                    price = price_fetcher(ticker, execution_date)
                    
                    if price:
                        self._execute_buy(trade, execution_date, price, budget_per_trade, signal['score'])
            
            # Move to next month
            current_date = next_month
            
        return self.get_summary()

    def _execute_buy(self, trade, date, price, amount, score):
        shares = amount / price
        self.cash -= amount
        ticker = trade['ticker']
        
        # In this POC, we update or add to position
        if ticker in self.portfolio:
            prev = self.portfolio[ticker]
            new_shares = prev['shares'] + shares
            self.portfolio[ticker] = {
                'shares': new_shares,
                'entry_price': (prev['entry_price'] * prev['shares'] + price * shares) / new_shares,
                'entry_date': date,
                'score': score
            }
        else:
            self.portfolio[ticker] = {
                'shares': shares,
                'entry_price': price,
                'entry_date': date,
                'score': score
            }
            
        self._log_event("BUY", ticker, date, price, reason=f"Score {score} signal from {trade['insider']}")

    def _execute_sell(self, ticker, date, price, reason):
        if ticker not in self.portfolio:
            return
            
        pos = self.portfolio[ticker]
        sale_value = pos['shares'] * price
        self.cash += sale_value
        gain = sale_value - (pos['shares'] * pos['entry_price'])
        
        self._log_event("SELL", ticker, date, price, gain=gain, reason=reason)
        del self.portfolio[ticker]

    def _log_event(self, action, ticker, date, price, gain=None, reason=""):
        log_entry = {
            'action': action,
            'ticker': ticker,
            'date': date.strftime('%Y-%m-%d'),
            'price': price,
            'gain': gain,
            'reason': reason
        }
        self.audit_log.append(log_entry)
        logger.info(f"[{date.date()}] {action} {ticker} at {price:.2f}. Reason: {reason}")

    def get_summary(self):
        # Calculate current valuation of portfolio
        # (This would need real-time prices at the end of simulation)
        return {
            'final_cash': self.cash,
            'audit_log': self.audit_log
        }

if __name__ == "__main__":
    from engines.scoring import ScoringEngine
    # Quick sanity test logic within the class
    pass
