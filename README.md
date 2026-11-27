## Reinforcement Learning on LunarLander-v3

A comparative study of three reinforcement learning algorithms—**Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC)**—on the Gymnasium `LunarLander-v3` environment.

---
## 1. Problem Statement

Deep reinforcement learning algorithms exhibit sharply different trade-offs in **sample efficiency**, **wall-clock training speed**, **stability across random seeds**, **hyperparameter sensitivity**, and **robustness to dynamics perturbations**. Even on low-dimensional continuous-control and discrete-control benchmarks, agents frequently suffer from reward-shaping exploitation (such as hovering to avoid crash penalties), post-convergence policy collapse, and poor generalization to unseen initial conditions or external disturbances.

Using **LunarLander-v3** as a controlled testbed, this project investigates how three representative deep RL algorithm families behave across discrete and continuous action spaces, how key architectural and hyperparameter choices govern learning dynamics, and how well learned landing policies generalize under wind and turbulence disturbances.

---
## 2. Project Objectives
### Primary Objective
Train reinforcement learning agents that reliably solve **LunarLander-v3**, defined by the standard benchmark threshold of achieving a **mean episode return $\ge 200$ over 100 unseen evaluation episodes** (safe, fuel-efficient landing on the target pad).
### Secondary Objectives
1. **Fair Algorithm Comparison:** Compare **DQN** (discrete), **PPO** (discrete and continuous), and **SAC** (continuous) against non-learning reference baseline (uniformly random policy) across both **equal environment-step budgets** and **equal wall-clock time**.
2. **Hyperparameter Analysis:** Quantify the impact of critical design choices—including Double/Dueling DQN variants, replay buffer capacity, exploration schedules, reward-shaping modifications, structured hyperparameter tuning (`Optuna`).
3. **Generalization & Robustness Evaluation:** Measure policy degradation on unseen random seeds and under progressive external disturbances (`enable_wind=True`, varying `wind_power` and `turbulence_power`), and evaluate whether randomized-wind training improves robustness.
4. **Qualitative & Failure-Mode Analysis:** Inspect learned control behavior via trajectory plots (position, velocity, thrust usage) and video recordings, categorizing failure modes such as high-speed crashes, excessive tilt, pad drift, hovering, and fuel waste.

---

## 4. Repository Structure

```text
Reinforcement_Learning_on_LunarLander-v3/
├── environments.py               # Reusable LunarLander-v3 environment factory (EnvBuilder)
├── algorithm runners/            # Algorithm training and evaluation runners
├── initial experiments/          # Checkpoint notebooks, baseline runs, and initial logs
├── testing/                      # Unit and smoke tests for environment setup & logging
├── requirements.txt              # Pinned project dependencies
└── README.md                     # Project overview, problem definition, and usage
```

---
