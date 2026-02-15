"""
Evaluation Metrics for Antifragile Trading

Taleb-aligned metrics that measure what actually matters:
- Sortino ratio (penalizes only downside volatility)
- Tail ratio (upside tails vs downside tails)
- Win rate (% of profitable trades)
- Max drawdown (worst peak-to-trough decline)
- Calmar ratio (return / max drawdown)
- Convexity ratio (average win / average loss)

Standard ML metrics (accuracy, F1) are intentionally NOT used because
they don't capture the asymmetric nature of financial returns.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    sortino: float
    tail_ratio: float
    win_rate: float
    max_drawdown: float
    calmar: float
    convexity_ratio: float
    mean_return: float
    total_return: float
    num_trades: int
    num_wins: int
    num_losses: int
    avg_win: float
    avg_loss: float

    def summary(self) -> str:
        return (
            f"=== Evaluation Results ===\n"
            f"  Sortino Ratio:    {self.sortino:>8.2f}\n"
            f"  Tail Ratio:       {self.tail_ratio:>8.2f}\n"
            f"  Win Rate:         {self.win_rate:>7.1%}\n"
            f"  Mean Return:      {self.mean_return:>7.2%}\n"
            f"  Total Return:     {self.total_return:>7.2%}\n"
            f"  Max Drawdown:     {self.max_drawdown:>7.2%}\n"
            f"  Calmar Ratio:     {self.calmar:>8.2f}\n"
            f"  Convexity Ratio:  {self.convexity_ratio:>8.2f}\n"
            f"  Trades: {self.num_trades} (W:{self.num_wins} / L:{self.num_losses})\n"
            f"  Avg Win:          {self.avg_win:>7.2%}\n"
            f"  Avg Loss:         {self.avg_loss:>7.2%}\n"
        )

    def is_acceptable(self, min_sortino: float = 2.0) -> bool:
        """Check if results meet minimum quality threshold."""
        return (
            self.sortino >= min_sortino
            and self.win_rate >= 0.55
            and self.max_drawdown > -0.25
            and self.convexity_ratio >= 1.5
        )


def compute_sortino(
    returns: np.ndarray, risk_free_rate: float = 0.04, annualize: bool = True
) -> float:
    """
    Sortino ratio: excess return / downside deviation.
    
    Unlike Sharpe, Sortino only penalizes downside volatility.
    This aligns with Taleb's principle that upside variance is GOOD.
    
    Args:
        returns: Array of period returns (fractional)
        risk_free_rate: Annual risk-free rate
        annualize: Whether to annualize (assumes monthly returns)
    """
    if len(returns) == 0:
        return 0.0

    # Per-period risk-free rate (monthly)
    periods_per_year = 12 if annualize else 1
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_period
    mean_excess = np.mean(excess_returns)

    # Downside deviation: std of negative excess returns only
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return 10.0  # Cap at 10 if no downside (perfect)

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-8:
        return 10.0

    sortino = mean_excess / downside_std

    if annualize:
        sortino *= np.sqrt(periods_per_year)

    return float(sortino)


def compute_tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
    """
    Tail ratio: |P95 return| / |P5 return|.
    
    Measures asymmetry of the return distribution.
    > 1.0: positive skew (bigger wins than losses) — GOOD
    = 1.0: symmetric
    < 1.0: negative skew (bigger losses than wins) — BAD
    """
    if len(returns) < 20:
        return 1.0

    upper = np.percentile(returns, 100 - percentile)
    lower = np.percentile(returns, percentile)

    if abs(lower) < 1e-8:
        return 10.0 if upper > 0 else 0.0

    return float(abs(upper / lower))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum drawdown: worst peak-to-trough decline.
    
    Returns negative value (e.g., -0.25 = -25% drawdown).
    Uses log-space computation to avoid overflow with many returns.
    """
    if len(returns) == 0:
        return 0.0

    # Log-space to prevent overflow with many large returns
    # Use a small epsilon to handle -1.0 (total loss) without -inf
    log_returns = np.log(np.clip(1 + returns, 1e-7, None))
    cumulative_log = np.cumsum(log_returns)
    running_max_log = np.maximum.accumulate(cumulative_log)
    
    # Drawdown in log space, convert back
    dd_log = cumulative_log - running_max_log  # Always <= 0
    max_dd_log = np.min(dd_log)
    
    return float(np.exp(max_dd_log) - 1.0)  # Convert back from log space


def compute_calmar(
    returns: np.ndarray, risk_free_rate: float = 0.04
) -> float:
    """
    Calmar ratio: annualized return / |max drawdown|.
    
    Measures return per unit of drawdown risk.
    """
    if len(returns) == 0:
        return 0.0

    # Annualized return (assuming monthly)
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / 12
    if n_years < 0.1:
        return 0.0

    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    max_dd = abs(compute_max_drawdown(returns))

    if max_dd < 1e-8:
        return 10.0

    return float(annualized_return / max_dd)


def evaluate_predictions(
    predicted_scores: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = 70.0,
) -> EvaluationResult:
    """
    Evaluate model predictions end-to-end.
    
    Simulates: "invest in everything the model scores > threshold"
    and computes all antifragile metrics on the resulting returns.
    
    Args:
        predicted_scores: Model convexity scores [0-100]
        actual_returns: Realized returns (fractional)
        threshold: Score threshold for "invest" decision
    """
    # Filter to trades we would have taken
    mask = predicted_scores >= threshold
    selected_returns = actual_returns[mask]

    if len(selected_returns) == 0:
        return EvaluationResult(
            sortino=0, tail_ratio=0, win_rate=0, max_drawdown=0,
            calmar=0, convexity_ratio=0, mean_return=0, total_return=0,
            num_trades=0, num_wins=0, num_losses=0, avg_win=0, avg_loss=0,
        )

    # Basic stats
    wins = selected_returns[selected_returns > 0]
    losses = selected_returns[selected_returns <= 0]

    win_rate = len(wins) / len(selected_returns)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    convexity_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-8 else 10.0

    return EvaluationResult(
        sortino=compute_sortino(selected_returns),
        tail_ratio=compute_tail_ratio(selected_returns),
        win_rate=win_rate,
        max_drawdown=compute_max_drawdown(selected_returns),
        calmar=compute_calmar(selected_returns),
        convexity_ratio=convexity_ratio,
        mean_return=float(np.mean(selected_returns)),
        total_return=float(np.expm1(np.sum(np.log1p(np.clip(selected_returns, -0.99, None))))),
        num_trades=len(selected_returns),
        num_wins=len(wins),
        num_losses=len(losses),
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
