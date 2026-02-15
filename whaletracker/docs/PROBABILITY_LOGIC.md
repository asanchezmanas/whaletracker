# Logic: Probability of Success & Convexity Threshold

The QDN tells us the **Expected Value (Q)**. But to be truly Antifragile, we need to distinguish between a "High Probability/Low Return" play and a "Low Probability/Explosive Return" play.

---

### 🎲 1. The Probability Brain (Classifier Layer)
We are adding an auxiliary head to the QDN (Multitask Learning):
*   **Head 1 (DQN)**: Estimates the $Q(s, a)$ (Cumulative Future Returns).
*   **Head 2 (Classifier)**: Estimates $P(Explosion)$, where internal label is `return_180d > 5.0` (500% spike).

**Why?**
The "Seeds" (1€) only need a very small $P(Explosion)$ to be profitable. 
*Example*: If a "Gem" has a 1% chance of +10,000% (Enphase style) and a 99% chance of -100%:
$EV = (0.01 * 100) + (0.99 * -1) = 1.0 - 0.99 = 0.01$ (**Profitable!**)

### 📈 2. The Convexity Threshold (The "Profit Point")
The Orchestrator will now use the **Kelly-Lite** formula to decide if it buys:

$Score = P(Success) \times ExpectedTailReturn$

| P(Success) | Expected Multiplier | Score | Action |
| :--- | :--- | :--- | :--- |
| 0.5% | 100x (Enphase) | 0.5 | **SEED (1€)** |
| 5.0% | 10x (Standard Gem) | 0.5 | **SEED (1€)** |
| 2.0% | 2x (Boring) | 0.04 | **VETO** (Too fragile/not enough payoff) |

**Conclusion**: We only buy when the **Convexity Score > 0.1**. If a play is "Boring" and only has a 2% chance of winning, but the win is small, it's not Antifragile—it's just a bad bet.

---

### 🧠 3. Does the QDN already do this?
**Partially.**
The Q-Value is a mix of probability and reward. However, the QDN can get "tricked" by high-frequency small gains. 
By adding the **Probability Layer**, we force the model to explicitly think: *"Is this just a normal stock, or does it have the structural DNA of a 100x winner?"*
