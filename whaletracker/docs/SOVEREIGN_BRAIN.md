# Sovereign Brain Specification (Logic of the State)

The Sovereign Brain (Contract Expert) is designed to detect the **"State Floor"**—the absolute limit of a company's downside provided by government backing.

## 🏛️ Core Thesis
In times of crisis or market volatility, companies with deep federal roots are the most antifragile. They don't just survive; they are often the primary recipients of emergency funding and mission-critical "must-win" contracts.

## 📡 Data Sources
*   **USASpending.gov API**: Real-time federal award tracking.
*   **FPDS (Federal Procurement Data System)**: Historical contract depth.
*   **SAM.gov**: Entity status and registry details.

## 🧪 Structural Features
1.  **Annual Contract Velocity**: Sum of all awards in the last 12 months. High velocity indicates a growing dependency of the state on the entity.
2.  **Backlog Ratio**: Estimated future contract value / Market Cap. If this ratio is > 0.5, the company is fundamentally "undervalued" by its guaranteed revenue.
3.  **Outlier Award Detection**: Detects "Structural Gems" where a single award is > 300% of the company's average historical award size.
4.  **Political Alignment**: Matches company sectors with the legislative committee oversight of the insiders buying the stock.
5.  **Seniority/Network Force**: Weights the "backing" by the political seniority of the network supporting the sector.

## 🤖 Decision Policy
*   **The Veto**: If the Sovereign Brain sees zero contract activity in a sector that requires state backing (e.g., Defense), it can veto Speculative seeds.
*   **The Floor**: High sovereign scores translate to lower "Fragility Coefficients" in the QDN reward function.
