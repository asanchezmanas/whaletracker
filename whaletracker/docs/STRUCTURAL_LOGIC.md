# Technical Specification: Structural Archetype Logic

This document defines how the QDN and Orchestrator translate the 30-feature vector into **Structural Archetypes**. These are the "hard-coded priors" that guide the neural network toward high-convexity opportunities.

---

### 🏛️ Archetype A: The "State-Backed Leverage" (Sovereign)
**Logic**: A small entity becoming a critical organ of the State.
*   **Primary Trigger**: `(features[0] > 0.7)` AND `(features[1] > 0.3)`
    *   *Velocity Score* (0) is high + *Backlog/Cap* (1) is significant.
*   **Secondary Anchor**: `(features[15] < 0.5)`
    *   *Market Cap* (15) is low (Micro/Small-cap).
*   **Veto Logic**: If `features[19] < 0.1` (Cash Runway < 2 months), block seeding unless `features[2]` (Outlier Award) exists.

### 🧪 Archetype B: The "Innovation Monopoly" (IP)
**Logic**: Legal ownership of a critical piece of the future.
*   **Primary Trigger**: `(features[5] > 0.6)` OR `(features[7] == 1.0)`
    *   *Patent Velocity* (5) is surging OR *Has Grants* (7) is true.
*   **Sector Multiplier**: `features[9]` (Sector Criticality).
    *   If Sector is "Biotech" (Designation-driven) or "Energy" (Grid-driven), weight score by +20%.
*   **Verification**: Check `features[8]` (Patent Count) to distinguish between a "One-Hit Wonder" and a "Scientific Fortress."

### 🐋 Archetype C: The "Information Floor" (Network)
**Logic**: High-intent actors with skin-in-the-game.
*   **Primary Trigger**: `(features[10] > 0.8)` AND `(features[12] == 1.0)`
*   **The Drift**: `features[13]` > 0.5 (Institutional inflow start).

### 🕸️ Archetype D: The "Critical Node" (Ecosystem)
**Logic**: A small company that becomes an unavoidable bottleneck for a giant.
*   **Primary Trigger**: `(features[11] > 0.7)` (Partnership Intensity).

### 🧊 Archetype E: Emergent Clusters (The Discovery Phase)
**Logic**: Using unsupervised learning to identify high-convexity patterns that defy simple categorization.

**The Mathematical Engine**:
*   **Elbow Method (Inertia)**: Minimizes intra-cluster variance. We select the `K` value where adding another cluster yields diminishing returns in "structure resolution."
*   **Silhouette Analysis**: Measures how well-defined the arquetypes are. A high silhouette score (>0.5) confirms that we've found a distinct "Structural Species" rather than just noise.
*   **Audit Metric**: We rank each emergent cluster by their **Convexity Ratio** (Mean Reward / Downside Volatility).

---

## 🎨 Philosophy: Semi-Supervised Discovery

We will not just search for what we know. We will use **Clustering** to find what we *don't* know:

1.  **Phase 1 (History Mining)**: Take the 10-year dataset and cluster all tickers into 10-15 "Structural Species" using the feature vector.
2.  **Phase 2 (Performance Audit)**: For each cluster, measure the **Convexity** (Expected return of a 1€ seed).
3.  **Phase 3 (Reward Injection)**: If a cluster shows high returns (even if it's "boring"), the QDN is rewarded for entering it.

**Conclusion**: We provide the "Known Gems" as a starting point, but the model is expected to find the "Hidden Rocks" that turn into diamonds.

## 🎨 Philosophy: Defined vs. Loose?

The architecture uses a **Hybrid Approach**:

1.  **Core Priors (Very Defined)**: We hard-code the "Veto" logic and the "Discovery Anchors" (the 4 archetypes above). This ensures the model doesn't gamble on noise. They are the **Safety Floor**.
2.  **Emergent Gems (Loosely Defined)**: The Reward Function allows the DQN to discover **combinations** we haven't named. 
    *   Example: A company with *moderate* patents (IP) but *extreme* insider clusters (Network) in a *Government-heavy* sector (Sovereign). 
    *   The model will "invent" its own sub-archetypes during the 10-year training.

**The Rule**: We define the *Grammar* (Features), but the Model writes the *Poetry* (The Prediction).

## 📈 Reward Function (The 10-Year Lesson)

The QDN "learns" by comparing these archetypes to future **Convexity**.

```python
def compute_structural_reward(feature_vector, future_returns):
    # 1. Base Logic: High Future Returns
    reward = max(0, future_returns) 
    
    # 2. Structural Bonus: Was the 'Gem' correctly identified as an Archetype?
    if is_sovereign_archetype(feature_vector):
        reward *= 1.5 # 50% bonus for structural alignment
        
    if is_ip_monopoly_archetype(feature_vector):
        reward *= 1.5
        
    # 3. Penalize False Positives (Fragility)
    if is_fragile(feature_vector):
        reward = -10.0 # Heavy penalty for 'blowing up' a seed
        
    return reward
```

## 🛠️ Implementation in `Orchestrator`

The Orchestrator uses these definitions to determine the **Seed Size**:
*   **Normal Entry**: 1€ (Generic structural signal).
*   **Archetype Hit**: 5€ (Matches a defined "Gem" logic).
*   **The Fattening**: +10€ monthly (If Archetype score > 90 and performance is positive).
