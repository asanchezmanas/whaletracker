# Structural Gem Archetypes: The 10-Year Ground Truth

To train the QDN effectively over a 10-year horizon, we define "Success" not by price alone, but by the alignment of **Structural Pillars**. Here are the archetypes from all sectors that the system must detect:

---

### 🏛️ 1. The Sovereign Gem (State Floor)
**Success Marker**: A company whose market cap is dwarfed by a single, critical government contract.
*   **Archtype 1 (Defense)**: *Anduril / Palantir (Early years)*. Massive IDIQ contracts for AI/Autonomy before they were mainstream.
*   **Archtype 2 (Space)**: *AST SpaceMobile*. Tiny company securing SDA prototype rights for space-based broadband.
*   **Archtype 3 (Infrastructure)**: Small contractors getting $400M+ ceilings for missile defense test data management (e.g., *i3*).

### 🧪 2. The IP-Monopoly Gem (Innovation Floor)
**Success Marker**: A patent or grant that grants a legal monopoly on an unavoidable future technology.
*   **Archtype 1 (Biotech)**: *IO Biotech / Spero*. Breakthrough FDA designations for cancer vaccines/infectious diseases while market cap is <$100M. The FDA designation is the "Structural Anchor."
*   **Archtype 2 (Materials)**: *Atomera / Meta Materials*. Holding critical patents in semiconductor scaling or nanotech electrification.
*   **Archtype 3 (Robotics)**: *Arbe Robotics*. 4D radar patents that become a standard for NVDA-partnered autonomous systems.

### 🐋 3. The Network-Force Gem (Information Floor)
**Success Marker**: A cluster of "Smart Money" entering a neglected ticker simultaneously.
*   **Archtype 1 (The Activist)**: *AstroNova (Askeladden Capital)*. External activist takes a 9.2% stake in a micro-cap with weak governance.
*   **Archtype 2 (The CEO Conviction)**: *Energy Vault (CEO Buy)*. Massive insider purchase in a pivot phase, followed by a 300% outperformance.
*   **Archtype 3 (Corporate King-Maker)**: *Toyota / Lockheed* taking a strategic stake in a tiny tech supplier.

---

### 🧊 4. Dynamic Discovery: The "Boring Gems" (Unsupervised)
While the archetypes above are our "Narrative Priors," the system is designed to find what we *don't* know.

**The Elbow Discovery Process**:
1.  **Mining**: The system takes 10 years of 30-feature vectors (the "Digital DNA") of every ticker.
2.  **Clustering**: Using **K-Means**, it groups these tickers into "Structural Species."
3.  **The Elbow Method**: We use the *Elbow* and *Silhouette* metrics to mathematically decide how many species exist. We don't guess; the data dictates the phylogeny.
4.  **Boring Win Detection**: A species might be "Boring" (low contracts, low patents, low volatility) but consistently yield a 5x return. This becomes an **Emergent Archetype**.

---

## 🎯 Training Reward Logic: "The Target Signal"

The model is trained to minimize the **"Detection Lag"**. We reward the QDN when it flags a company **BEFORE** the 1000% jump, triggered by:
1.  **Sovereign**: Contract award date (T).
2.  **IP**: SBIR Phase II / Patent grant date (T).
3.  **Network**: Third buyer in a cluster (T).

## 🔍 How to Audit the Logs

When the backtest runs, we look for **"The Convergence"**:
*   **Log Entry**: `[INFO] GEM DETECTED: $IOBT. Reason: IP_Moat(Designation: Breakthrough) + Network(Whale: 10% Owner Buy). Total Score: 92.2.`
*   **What to search**: `grep "GEM DETECTED" backtest.log`.
*   **Validation**: Check if $IOBT had its major move *after* that log entry.

**We are not looking for Ubers. We are looking for the 'Invisible Pillars' that make a company unavoidable.**
