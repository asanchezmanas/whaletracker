# Backtest Specification: Structural Infiltration (Phase 7)

Traditional backtests measure "Max Drawdown" and "Sharpe Ratio" on a fixed principal. Our backtest measures **Explosive Optionality** on a recurrent inflow.

## 📈 The Mechanics: "The 10-Year Structural Farm"

This backtest covers **10 years of data (2015-2025)** to capture multiple "Black Swan" cycles (COVID, Energy Crises, War Cycles).

1.  **Monthly Pulse**: Every T+30 days, 50€ is added to `cash_pool`.
2.  **Structural Infiltration**: The 3 Brains scan for the **Archetypes** defined in `STRUCTURAL_ARCHETYPES.md`.
3.  **Training Target**: The model is trained to maximize the "Convexity Capture" — finding the early entries in 10-year success stories like Palantir (Sovereign) or Biotech breakthrough candidates.
3.  **Sowing Phase**:
    *   The `Orchestrator` selects 5-15 entities.
    *   Each gets a "seed" (1€-5€).
    *   This is the "Infiltration": we are now "in" dozens of potential explosions.
4.  **Survival Phase**:
    *   The simulator tracks the price of every seed.
    *   **Burn**: If a company goes bust or delists, we lose 1€. No hard feelings.
    *   **Carry**: Most seeds stay between -30% and +30%. They are just "living" in the farm.
5.  **Fattening Phase**:
    *   If a seed spikes +100% OR its `Structural Score` crosses 90 (e.g., a massive new contract is awarded), the next 50€ injection is prioritized to **Fatten** this specific winner.
6.  **The Black Swan Capture**:
    *   We exit ONLY when a "Structural Peak" is detected (e.g., Insider exit cluster + Contract completion) or after +1000% return.

---

## 📊 Success Metrics (Antifragile KPIs)

| Metric | Traditional Backtest | **Structural Backtest** |
| :--- | :--- | :--- |
| **Principal** | 10,000€ (Fixed) | 50€ / Month (Accumulated) |
| **Risk** | Portfolio Volatility | **Cost of Carry** (1€ per seed) |
| **Goal** | Limit Drawdown | **Capture Convexity** (Power laws) |
| **Primary KPI** | Sharpe Ratio | **Tail Ratio** (Upside / Downside) |
| **Win Rate** | Must be > 50% | Can be **10%** (if winners are 100x) |

---

## 🛠️ Implementation Strategy

1.  **Time-Series Iteration**: We don't vectorise. We iterate month-by-month to respect the 50€ cash constraint.
2.  **Portfolio Persistence**: We need a `PortfolioState` that survives across years.
3.  **Entity Resolution**: We must ensure tickers from 2021 are correctly mapped even if they changed.

**This is not a "trading test." This is a "Wealth Accumulation Simulation" via Structural Asymmetry.**
