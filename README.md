# Deep Reinforcement Learning for Multi-Echelon Inventory Optimization

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

![](./problem_description.png)

Inventory Optimization is a critical problem in supply chain systems, where traditional heuristics such as [(s, S)](https://hub.spreadsheetweb.com/templates/view/s-s-inventory-model#:~:text=The%20s%2DS%20inventory%20policy%20analysis,stockouts%20and%20minimizing%20inventory%20expenses) policies struggle under stochastic demand, lead time variability, and multi-echelon dependencies.

This project demonstrates how **Deep Reinforcement Learning (DRL)** can learn adaptive invetory control policies that optimize long-term cost and service level trade-offs.

## Problem Formulation

We model the system as a **Markov Decision Process (MDP)**:
- **State (s)**: Inventory levels, pipeline stock, demand signals
- **Action (a)**: Replenishment quantities
- **Reward (r)**: Negative total cost
- **Transition**: Inventory dynamics + stochastic demand

### Objective

Minimize expected cumulative cost: 
$$J(\pi) = \mathbb{E} \left[ \sum_{t=0}^{T} \gamma^t C(s_t, a_t) \right]$$

## Algorithms Implemented

### Asynchronous Advantage Actor-Critic ([A3C](https://medium.com/sciforce/reinforcement-learning-and-asynchronous-actor-critic-agent-a3c-algorithm-explained-f0f3146a14ab))

- Parallel actor-learners
- Advantage-based updates
- Stable and efficient training

### Proximal Policy Optimization ([PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization))

- Clipped objective for stability
- Generalized Advantage Estimation (GAE)
- Strong emperical performance

### Baseline: (s, S) policy

Classical inventory heuristic used as benchmark

## Results Summary

Each algorithm is trained and evaluated on two environments: the original
stationary formulation (**Env-1**) and a harder non-stationary variant
with seasonal demand, correlated retailer demand, demand shocks,
heavy-tailed lead times, and stochastic capacity caps (**Env-2**).

| Algorithm | Env-1 (stationary) | Env-2 (non-stationary) |
| --------- | ------------------ | ---------------------- |
| PPO vs. fixed base-stock heuristic | **+36.76%** cost reduction | **+94.20%** cost reduction |
| A3C vs. retuned (s,S) policy       | **+1.64%** cost reduction  | **-21.79%** (heuristic wins) |

The negative A3C/Env-2 result is intentional and discussed in the report:
single-environment evaluation of deep-RL inventory policies is not
reliable evidence of real robustness.

**Full writeup with figures, methodology, and threats to validity**:
see [`docs/report.md`](docs/report.md).

# Project Structure

```
.multi-echelon-rl-inventory
├──actor-critic/
|   ├── configs/
│   |   ├── config.yaml
│   |   └── meisConfig.yaml
|   ├── src/
│   |   ├── __init__.py
│   |   ├── a3c_agent.py
│   |   ├── meis_env.py
│   |   ├── s_s_policy.py
│   |   └── trainer.py
|   ├── utils/
│   |   ├── __init__.py
│   |   ├── evaluation.py
│   |   ├── helpers.py
│   |   └── visualisation.py
|   ├── results/
│   |   ├── checkpoints/
│   |   ├── logs/
│   |   └── plots/
│   ├── main.py
|   └── README.md
├──ppo/
|   ├── configs/
│   |   └── config.yaml
|   ├── models/
│   |   ├── __init__.py
│   |   ├── actor_critic.py
│   |   ├── baseline.py
│   |   ├── env.py
│   |   ├── replay_buffer.py
│   |   └── ppo.py
|   ├── src/
│   |   ├── __init__.py
│   |   ├── train.py
│   |   ├── evaluate.py
│   |   └── visualise.py
|   ├── utils/
│   |   ├── metrics.py
│   |   ├── logger.py
│   |   └── helpers.py
|   ├── results/
│   |   ├── checkpoints/
│   |   ├── logs/
│   |   ├── plots/
│   |   ├── evaluation_results.json
│   |   └── training_stats.json
│   ├── main.py
|   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

# Setup and Execution

```bash
python3 -m venv .venv           # initialise virtual environment
source .venv/bin/activate       # activate virtual environment
pip install -r requirements.txt # install all required dependencies
```

Each algorithm accepts `--env` to pick the environment variant. Artifacts
from Env-1 and Env-2 are written to separate directories
(`results/` vs `results_complex/`) so the two experiments never clobber
each other.

## Run A3C

``` bash
cd actor-critic
python3 main.py --mode train --env meis      # train on Env-1 (original MEIS)
python3 main.py --mode train --env complex   # train on Env-2 (non-stationary MEIS)
python3 main.py --mode eval  --env meis      # evaluate on Env-1
python3 main.py --mode eval  --env complex   # evaluate on Env-2
```

## Run PPO

``` bash
cd ppo
python3 main.py --mode train --env divergent # train on Env-1 (divergent supply chain)
python3 main.py --mode train --env complex   # train on Env-2 (non-stationary)
python3 main.py --mode eval  --env divergent # evaluate on Env-1
python3 main.py --mode eval  --env complex   # evaluate on Env-2
```

## Smoke test + report

From the repo root:

``` bash
python scripts/smoke_env.py     # end-to-end sanity check for all 4 (algo, env) combos
python scripts/make_report.py   # rebuild docs/report.md + figures from saved JSONs
```

# Platform Independence

This project runs on multiple operating systems and hardware backends with no code changes required.

## Operating System

| OS | Supported | Notes |
| -- | --------- | ----- |
| macOS | ✅ | Intel and Apple Silicon (M1 / M2 / M3) |
| Windows | ✅ | Windows 10 / 11 |
| Linux | ✅ | Recommended for multi-process A3C training |

### Virtual Environment Setup

**macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt)**:
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

> On Windows, if script execution is blocked run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Hardware / GPU

PyTorch auto-detects the best available device at runtime — no manual configuration needed.

| Backend | Hardware | Auto-selected when |
| ------- | -------- | ------------------ |
| CUDA | NVIDIA GPU | `torch.cuda.is_available()` returns `True` |
| MPS | Apple Silicon (M1 / M2 / M3) | `torch.backends.mps.is_available()` returns `True` |
| CPU | Any machine | Fallback if neither CUDA nor MPS is available |

> **CUDA users**: install the CUDA-enabled PyTorch wheel that matches your driver *before* running `pip install -r requirements.txt`. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct install command.

## Python Version

Requires **Python 3.8+**. Recommended: **Python 3.10** or **3.11**.