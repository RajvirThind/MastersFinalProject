# Multi-Market BESS Co-Optimisation 
**Reinforcement Learning vs. Mixed-Integer Linear Programming (MILP)**

This repository contains MILP Optimisation model and a Reinforcement Learning (RL) agent designed to maximise the Net Profit of a Battery Energy Storage System (BESS) by "Value Stacking" across 8 simultaneous UK energy markets.


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