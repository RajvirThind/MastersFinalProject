# Multi-Market BESS Co-Optimisation 
**Reinforcement Learning vs. Mixed-Integer Linear Programming (MILP)**

This repository contains a simulation environment and a Reinforcement Learning (RL) agent designed to maximise the Net Profit of a Battery Energy Storage System (BESS) by "Value Stacking" across 8 simultaneous UK energy markets.

---

## Performance Summary
* **Agent Efficiency:** Captured over **80%** of the theoretical maximum profit available.
* **Benchmark:** Validated against a **PuLP MILP** solver with perfect foresight.
* **Operational Realism:** Eliminated high-frequency cycling via a custom **Reversal Penalty**, ensuring hardware-safe operation.

---

## The Models

### 1. Proximal Policy Optimisation (PPO) Agent
The primary production model. It utilises an Actor-Critic architecture to navigate a continuous action space of 8 simultaneous markets
* **Architecture:** Deep MLP with `[512, 512]` hidden layers.
* **Observation Space:** 391 dimensions, including State-of-Charge (SOC), EFA Block timers, and a 48-step price lookahead across all 8 markets.
* **Training:** 1,000,000 steps with tuned entropy (`ent_coef=0.01`) to balance exploration of arbitrage spikes with steady ancillary income.

### MILP Benchmark
Used for performance validation and generating the "Profit Ceiling."
* **Method:** Rolling Horizon (24-hour windows).
* **Objective:** Maximise revenue minus degradation and stress costs while obeying linear energy constraints.

---

## Market Stack (8 Simultaneous Markets)
The agent co-optimises across the following markets:
* **Wholesale Arbitrage:** Day-Ahead, Intraday, Balancing Mechanism (BM), and Imbalance.
* **Ancillary Services:**
    * **DCDMHigh:** Frequency response via **Charging** (triggered by high grid frequency).
    * **DCDMLow:** Frequency response via **Discharging** (triggered by low grid frequency).
    * **DRHigh / DRLow:** Dynamic Regulation services.

---

## Physical Guardrails
To ensure realism, the environment enforces:
* **Directional Lock:** Strictly prevents simultaneous charging and discharging, respecting single-inverter physics.
* **EFA Block Constraints:** Ancillary commitments are locked in 4-hour EFA (Electricity Forward Agreement) blocks.
* **50% Ancillary Cap:** Limits ancillary reservations to 50% of `p_max`, ensuring capacity remains available for high-value arbitrage spikes.
* **80% BM Skip Rate:** Models the historical rejection risk/non-acceptance rate of the Balancing Mechanism.

---

## Tech Stack
* **Engine:** `Gymnasium` (Custom Environment)
* **RL Framework:** `Stable-Baselines3`
* **Analysis:** `Pandas`, `NumPy`, `Matplotlib`, `Plotly`
* **Benchmark Solver:** `PuLP` / `CBC`
* **Testing:** `Pytest` (Verifies physical integrity and SOC limits)

---

## 📊 Repository Structure
```text
├── src/
│   ├── rl/               # scripts for PPO, SAC, and DQN agents
│   ├── milp/             # MILP optimisation scripts
│   ├── comparisons/      # model evaluation scripts
│   └── utils.py          # Shared data processing and helper functions
├── data/                 # Price datasets & Appraisal summaries
├── README.md             # Project documentation
└── pyproject.toml        # Build system and dependencies