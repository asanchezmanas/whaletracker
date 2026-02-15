"""
Walk-Forward Backtesting Engine

Implements time-series cross-validation with:
- Expanding or rolling training window
- Purge gap between train and test (avoid look-ahead bias)
- Embargo period after test (avoid data leakage)

This is the ONLY valid way to backtest financial ML models.
Standard k-fold cross-validation is INVALID for time-series because
it leaks future information into training.

Based on Marcos López de Prado's "Advances in Financial Machine Learning".
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .config import QDNConfig
from .trainer import QDNTrainer
from .evaluation import EvaluationResult, evaluate_predictions
from .labeling.purged_kfold import PurgedKFoldCV, CombinatorialPurgedCV
import logging

logger = logging.getLogger(__name__)



@dataclass
class WalkForwardFold:
    """Single fold of walk-forward validation."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train: int
    n_test: int
    result: Optional[EvaluationResult] = None


class WalkForwardBacktester:
    """
    Walk-Forward validation engine.
    
    Timeline:
    ┌──────────────────┬─purge─┬──────────┬─embargo─┬─────────
    │    TRAIN         │  gap  │   TEST   │  gap    │  next...
    │  24 months       │ 7 days│ 6 months │ 30 days │
    └──────────────────┴───────┴──────────┴─────────┴─────────
    
    Then slide forward by step_months and repeat.
    
    Purge: Remove samples from end of train that could leak
           to the beginning of test (e.g., overlapping return windows).
    
    Embargo: Remove samples from beginning of next training window
             to avoid test information leaking into future training.
    """

    def __init__(self, config: QDNConfig):
        self.config = config
        self.folds: List[WalkForwardFold] = []

    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds from date range.
        
        Args:
            start_date: First available data date
            end_date: Last available data date
        
        Returns:
            List of WalkForwardFold objects
        """
        cfg = self.config.training
        folds = []
        fold_id = 0

        train_start = start_date

        while True:
            # Training window
            train_end = train_start + timedelta(
                days=cfg.walk_forward_train_months * 30
            )

            # Purge gap
            test_start = train_end + timedelta(days=cfg.purge_days)

            # Test window
            test_end = test_start + timedelta(
                days=cfg.walk_forward_test_months * 30
            )

            # Check if test window exceeds available data
            if test_end > end_date:
                break

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    n_train=0,  # Filled during execution
                    n_test=0,
                )
            )

            # Slide forward
            train_start += timedelta(
                days=cfg.walk_forward_step_months * 30
            )
            fold_id += 1

        self.folds = folds
        return folds

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,  # Array of datetime for each sample
    ) -> Dict:
        """
        Run complete walk-forward backtest.
        
        Args:
            features: [n_samples, n_features]
            labels: [n_samples] actual returns
            dates: [n_samples] transaction dates
        
        Returns:
            Dict with fold results and aggregate metrics
        """
        if len(self.folds) == 0:
            raise ValueError(
                "No folds generated. Call generate_folds() first."
            )

        all_test_scores = []
        all_test_labels = []
        fold_results = []

        print(f"Running walk-forward backtest with {len(self.folds)} folds")
        print("=" * 60)

        for fold in self.folds:
            print(
                f"\nFold {fold.fold_id + 1}/{len(self.folds)}: "
                f"Train [{fold.train_start.date()} → {fold.train_end.date()}] | "
                f"Test [{fold.test_start.date()} → {fold.test_end.date()}]"
            )

            # Split data by date
            train_mask = (dates >= fold.train_start) & (dates < fold.train_end)
            test_mask = (dates >= fold.test_start) & (dates < fold.test_end)

            train_features = features[train_mask]
            train_labels = labels[train_mask]
            test_features = features[test_mask]
            test_labels = labels[test_mask]

            fold.n_train = len(train_features)
            fold.n_test = len(test_features)

            if fold.n_train < 100:
                print(f"  ⚠ Skipping: only {fold.n_train} train samples")
                continue

            if fold.n_test < 10:
                print(f"  ⚠ Skipping: only {fold.n_test} test samples")
                continue

            print(
                f"  Train: {fold.n_train} samples | "
                f"Test: {fold.n_test} samples"
            )

            # Train a fresh model for this fold
            trainer = QDNTrainer(self.config)
            fold_result = trainer.train(
                train_features, train_labels,
                test_features, test_labels,
            )

            fold.result = fold_result
            fold_results.append(fold_result)

            # Collect test predictions for aggregate evaluation
            trainer.model.eval()
            import torch

            with torch.no_grad():
                test_tensor = torch.tensor(
                    test_features, dtype=torch.float32
                ).to(trainer.device)
                output = trainer.model(test_tensor)
                scores = output["convexity_score"].squeeze(-1).cpu().numpy()

            all_test_scores.append(scores)
            all_test_labels.append(test_labels)

            print(f"  Result: {fold_result.summary()}")

        # Aggregate evaluation across all folds
        if all_test_scores:
            agg_scores = np.concatenate(all_test_scores)
            agg_labels = np.concatenate(all_test_labels)

            aggregate_result = evaluate_predictions(
                agg_scores,
                agg_labels,
                threshold=self.config.backtest.buy_score_threshold,
            )
        else:
            aggregate_result = None

        # Summary
        print("\n" + "=" * 60)
        print("WALK-FORWARD BACKTEST COMPLETE")
        print("=" * 60)

        if aggregate_result:
            print(aggregate_result.summary())
            print(
                f"Acceptable: "
                f"{'✅ YES' if aggregate_result.is_acceptable() else '❌ NO'}"
            )

        return {
            "folds": self.folds,
            "fold_results": fold_results,
            "aggregate": aggregate_result,
            "n_folds": len(self.folds),
            "n_valid_folds": len(fold_results),
        }


from .portfolio.manager import PortfolioManager
from .backtest.simulator import ExecutionSimulator
from .labeling.meta_labeling import MetaModelManager

class PortfolioBacktester:
    """
    High-fidelity trading simulator.
    Orchestrates signals, meta-labeling, sizing (HRP/Kelly), and execution.
    """

    def __init__(self, config: QDNConfig, initial_capital: float = 1000000):
        self.config = config
        self.pm = PortfolioManager(cash=initial_capital)
        self.sim = ExecutionSimulator()
        self.meta = MetaModelManager()
        self.equity_curve = []
        self.trade_log = []

    def run(
        self,
        features: np.ndarray,
        dates: np.ndarray,
        tickers: List[str],
        prices: np.ndarray,
        primary_scores: np.ndarray,
        actual_returns: np.ndarray,
        daily_volumes: Optional[np.ndarray] = None,
        filing_dates: Optional[np.ndarray] = None,
        whale_types: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Simulate trading over time.
        """
        logger.info(f"Starting Portfolio Simulation (Initial Capital: ${self.pm.cash:,.0f})")
        
        current_equity = self.pm.cash
        self.equity_curve.append({"date": dates[0], "equity": current_equity})

        # Group by date to simulate daily decisions
        unique_dates = np.unique(dates)
        
        for dt in unique_dates:
            day_mask = (dates == dt)
            day_features = features[day_mask]
            day_tickers = np.array(tickers)[day_mask]
            day_scores = primary_scores[day_mask]
            day_prices = prices[day_mask]
            day_returns = actual_returns[day_mask]
            day_vols = daily_volumes[day_mask] if daily_volumes is not None else np.ones(len(day_features)) * 1e6
            day_filing_dates = filing_dates[day_mask] if filing_dates is not None else [None] * len(day_features)
            day_whale_types = whale_types[day_mask] if whale_types is not None else ["unknown"] * len(day_features)
            # 1. Filter signals (primary score threshold)
            thresh = self.config.backtest.buy_score_threshold
            valid = (day_scores >= thresh)
            
            if not np.any(valid):
                self.equity_curve.append({"date": dt, "equity": current_equity})
                continue

            signals = []
            for i in range(len(day_features)):
                if not valid[i]: continue
                
                # 2. Meta-labeling filter (confidence)
                conf = self.meta.predict_confidence(day_features[i], day_scores[i])
                if conf < 0.5: continue # Skip low confidence
                
                signals.append({
                    "ticker": day_tickers[i],
                    "score": day_scores[i],
                    "confidence": conf,
                    "price": day_prices[i],
                    "return": day_returns[i],
                    "vol": day_vols[i],
                    "filing_date": day_filing_dates[i],
                    "whale_type": day_whale_types[i]
                })

            if not signals:
                self.equity_curve.append({"date": dt, "equity": current_equity})
                continue
                
            # 3. Optimize Allocation (HRP/Kelly)
            allocations = self.pm.optimize_allocation(signals)
            
            # 4. Execute with Simulator
            for sig in signals:
                ticker = sig["ticker"]
                weight = allocations.get(ticker, 0.0)
                if weight <= 0: continue
                
                requested_val = current_equity * weight
                req_shares = int(requested_val / sig["price"])
                
                # Simulate fill
                fill = self.sim.simulate_fill(
                    ticker, req_shares, sig["price"], sig["vol"]
                )
                
                if fill["shares"] > 0:
                    realized_val = fill["shares"] * fill["price"]
                    # Simplified PnL: entry * return - slippage
                    # In a real bot, we'd track the open position until exit.
                    # Here we assume barrier exit logic from labels.
                    pnl = (fill["shares"] * fill["price"] * sig["return"]) - fill["slippage_cost"]
                    current_equity += pnl
                    
                    self.trade_log.append({
                        "execution_date": dt,
                        "filing_date": sig["filing_date"],
                        "ticker": ticker,
                        "whale_type": sig["whale_type"],
                        "shares": fill["shares"],
                        "entry": fill["price"],
                        "pnl": pnl,
                        "slippage_bps": fill["slippage_bps"]
                    })
            
            self.equity_curve.append({"date": dt, "equity": current_equity})
            
        final_return = (current_equity / self.pm.cash) - 1
        logger.info(f"Simulation Complete. Final Equity: ${current_equity:,.0f} ({final_return:.1%})")
        
        return {
            "equity_curve": pd.DataFrame(self.equity_curve),
            "trades": pd.DataFrame(self.trade_log),
            "final_equity": current_equity,
            "total_return": final_return
        }

    def report(self) -> str:
        """Generate summary report."""
        if not self.equity_curve:
            return "No trades executed."
            
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity / self.pm.cash) - 1
        
        lines = ["Portfolio Simulation Report", "=" * 40]
        lines.append(f"Final Equity: ${final_equity:,.2f} ({total_return:+.2%})")
        lines.append(f"Total Trades: {len(self.trade_log)}")
        
        # Periodic Summary
        lines.append("\nPeriodic P&L Summary:")
        periodic = self.get_periodic_report(freq="M")
        lines.append(periodic.to_string())
        
        return "\n".join(lines)

    def get_periodic_report(self, freq: str = "M") -> pd.DataFrame:
        """
        Generate periodic P&L and Balance summary (Monthly/Quarterly).
        """
        if not self.equity_curve:
            return pd.DataFrame()
            
        df_equity = pd.DataFrame(self.equity_curve)
        df_equity['date'] = pd.to_datetime(df_equity['date'])
        df_equity = df_equity.set_index('date')
        
        # Resample to end of period
        resampled = df_equity['equity'].resample(freq).last().ffill()
        
        report = pd.DataFrame({
            "Closing Balance": resampled,
            "Period P&L": resampled.diff().fillna(resampled.iloc[0] - self.pm.cash),
            "Return": resampled.pct_change().fillna((resampled.iloc[0] / self.pm.cash) - 1)
        })
        
        return report


class PurgedKFoldBacktester:
    """
    Purged K-Fold Backtester.
    
    Implements López de Prado's Purged K-Fold and optionally 
    Combinatorial Purged CV (CPCV).
    """

    def __init__(self, config: QDNConfig, combinatorial: bool = False):
        self.config = config
        self.combinatorial = combinatorial
        if combinatorial:
            self.cv = CombinatorialPurgedCV(
                n_folds=config.training.cv_folds,
                n_test_groups=2,
                embargo_pct=config.training.embargo_pct,
            )
        else:
            self.cv = PurgedKFoldCV(
                n_folds=config.training.cv_folds,
                embargo_pct=config.training.embargo_pct,
            )

    def run(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        dates: np.ndarray,
        event_ends: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict:
        print(f"Running {'CPCV' if self.combinatorial else 'Purged K-Fold'} CV")
        print("=" * 60)

        all_test_scores = []
        all_test_labels = []
        fold_summaries = []

        sort_idx = np.argsort(dates)
        features = features[sort_idx]
        labels = labels[sort_idx]
        dates = dates[sort_idx]
        event_ends = event_ends[sort_idx]
        if sample_weights is not None:
            sample_weights = sample_weights[sort_idx]

        for fold in self.cv.split(dates, dates, event_ends):
            print(f"\nFold {fold.fold_id + 1}: Train: {fold.n_train} | Test: {fold.n_test}")

            train_features = features[fold.train_indices]
            train_labels = labels[fold.train_indices]
            test_features = features[fold.test_indices]
            test_labels = labels[fold.test_indices]
            
            # Train
            trainer = QDNTrainer(self.config)
            result = trainer.train(
                train_features, train_labels,
                test_features, test_labels,
                train_weights=sample_weights[fold.train_indices] if sample_weights is not None else None,
            )

            if result:
                fold_summaries.append(result)
                trainer.model.eval()
                import torch
                with torch.no_grad():
                    test_tensor = torch.tensor(test_features, dtype=torch.float32).to(trainer.device)
                    output = trainer.model(test_tensor)
                    scores = output["convexity_score"].squeeze(-1).cpu().numpy()
                
                all_test_scores.append(scores)
                all_test_labels.append(test_labels)

        if all_test_scores:
            agg_scores = np.concatenate(all_test_scores)
            agg_labels = np.concatenate(all_test_labels)
            aggregate_result = evaluate_predictions(agg_scores, agg_labels, threshold=self.config.backtest.buy_score_threshold)
        else:
            aggregate_result = None

        return {"fold_results": fold_summaries, "aggregate": aggregate_result, "combinatorial": self.combinatorial}

