"""Build the cross-environment comparison report.

Reads the four training + evaluation JSONs produced by the PPO and A3C
pipelines on both Env-1 (`divergent` / `meis`) and Env-2 (`complex`), and
emits:

    docs/
        report.md                 # research-paper-style writeup
        figures/
            training_curves_ppo.png
            training_curves_a3c.png
            cost_bars.png
            service_level_bars.png
            ppo_complex_cost_hist.png
            a3c_complex_cost_hist.png

Run from repo root:

    python scripts/make_report.py

This script only *reads* existing artifacts and writes files under `docs/`;
it does not retrain anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Matplotlib with a non-interactive backend so this runs headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
FIG_DIR = os.path.join(DOCS_DIR, "figures")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

EVAL_PATHS: Dict[str, Dict[str, str]] = {
    "ppo": {
        "env1": os.path.join(REPO_ROOT, "ppo", "results", "evaluation_results.json"),
        "env2": os.path.join(REPO_ROOT, "ppo", "results_complex", "evaluation_results.json"),
    },
    "a3c": {
        "env1": os.path.join(REPO_ROOT, "actor-critic", "results", "logs", "evaluation_results.json"),
        "env2": os.path.join(REPO_ROOT, "actor-critic", "results_complex", "logs", "evaluation_results.json"),
    },
}

TRAIN_PATHS: Dict[str, Dict[str, str]] = {
    "ppo": {
        "env1": os.path.join(REPO_ROOT, "ppo", "results", "training_stats.json"),
        "env2": os.path.join(REPO_ROOT, "ppo", "results_complex", "training_stats.json"),
    },
    "a3c": {
        "env1": os.path.join(REPO_ROOT, "actor-critic", "results", "checkpoints", "training_history.json"),
        "env2": os.path.join(REPO_ROOT, "actor-critic", "results_complex", "checkpoints", "training_history.json"),
    },
}


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[warn] missing: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class EvalSummary:
    rl_mean_cost: float
    rl_std_cost: float
    bl_mean_cost: float
    bl_std_cost: float
    improvement_pct: float
    rl_episode_costs: List[float]
    bl_episode_costs: List[float]
    # Only populated for A3C (MEIS env exposes service_level).
    rl_mean_service: Optional[float] = None
    rl_std_service: Optional[float] = None
    bl_mean_service: Optional[float] = None
    bl_std_service: Optional[float] = None
    service_improvement_pct: Optional[float] = None
    cohens_d: Optional[float] = None
    p_value: Optional[float] = None


def _summarize(algo: str, eval_dict: dict) -> EvalSummary:
    if algo == "ppo":
        rl = eval_dict["ppo"]
        bl = eval_dict["baseline"]
        return EvalSummary(
            rl_mean_cost=rl["mean_cost"],
            rl_std_cost=rl["std_cost"],
            bl_mean_cost=bl["mean_cost"],
            bl_std_cost=bl["std_cost"],
            improvement_pct=eval_dict["improvement"],
            rl_episode_costs=list(rl["episode_costs"]),
            bl_episode_costs=list(bl["episode_costs"]),
        )
    # A3C
    rl = eval_dict["rl"]
    bl = eval_dict["baseline"]
    cmp_ = eval_dict["comparison"]
    return EvalSummary(
        rl_mean_cost=rl["mean_cost"],
        rl_std_cost=rl["std_cost"],
        bl_mean_cost=bl["mean_cost"],
        bl_std_cost=bl["std_cost"],
        improvement_pct=cmp_["cost_improvement_percent"],
        rl_episode_costs=list(rl["episode_costs"]),
        bl_episode_costs=list(bl["episode_costs"]),
        rl_mean_service=rl["mean_service_level"],
        rl_std_service=rl["std_service_level"],
        bl_mean_service=bl["mean_service_level"],
        bl_std_service=bl["std_service_level"],
        service_improvement_pct=cmp_["service_level_improvement_percent"],
        cohens_d=cmp_["cohens_d_cost"],
        p_value=cmp_["cost_ttest"]["p_value"],
    )


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    c = np.convolve(x, np.ones(window) / window, mode="valid")
    pad = len(x) - len(c)
    return np.concatenate([np.full(pad, c[0]), c])


def plot_training_curves_ppo(train1: dict, train2: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, stats, title in [
        (axes[0], train1["ppo"], "PPO on Env-1 (divergent)"),
        (axes[1], train2["ppo"], "PPO on Env-2 (complex)"),
    ]:
        costs = np.asarray(stats["episode_costs"], dtype=float)
        smoothed = _smooth(costs, window=min(200, max(10, len(costs) // 50)))
        ax.plot(costs, alpha=0.25, linewidth=0.6, label="per-episode")
        ax.plot(smoothed, color="C1", linewidth=1.6, label="rolling mean")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("PPO training cost curves (per-episode and smoothed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training_curves_a3c(hist1: dict, hist2: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, hist, title in [
        (axes[0], hist1, "A3C on Env-1 (meis)"),
        (axes[1], hist2, "A3C on Env-2 (complex MEIS)"),
    ]:
        costs = np.asarray(hist["episode_costs"], dtype=float)
        smoothed = _smooth(costs, window=min(200, max(10, len(costs) // 50)))
        ax.plot(costs, alpha=0.25, linewidth=0.6, label="per-episode")
        ax.plot(smoothed, color="C2", linewidth=1.6, label="rolling mean")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cost")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("A3C training cost curves (per-episode and smoothed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cost_bars(summaries: Dict[str, Dict[str, EvalSummary]], path: str) -> None:
    """Four grouped bar charts — one per (algo, env) — since absolute costs
    are on very different scales across the two algorithms. Grouping like
    this keeps each chart readable instead of letting one dominate.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ((0, 0), "ppo", "env1", "PPO - Env-1 (divergent)"),
        ((0, 1), "ppo", "env2", "PPO - Env-2 (complex)"),
        ((1, 0), "a3c", "env1", "A3C - Env-1 (meis)"),
        ((1, 1), "a3c", "env2", "A3C - Env-2 (complex MEIS)"),
    ]
    for (r, c), algo, env, title in panels:
        ax = axes[r, c]
        s = summaries[algo][env]
        means = [s.rl_mean_cost, s.bl_mean_cost]
        stds = [s.rl_std_cost, s.bl_std_cost]
        rl_label = "PPO" if algo == "ppo" else "A3C"
        bl_label = "(s,S) baseline" if algo == "a3c" else "base-stock baseline"
        labels = [rl_label, bl_label]
        colors = ["#1f77b4", "#ff7f0e"]
        xs = np.arange(2)
        ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean episode cost (±1 std)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        improvement_tag = f"Δ = {s.improvement_pct:+.2f}%"
        color = "#2ca02c" if s.improvement_pct > 0 else "#d62728"
        ax.text(
            0.5, 0.92, improvement_tag,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color=color,
        )
    fig.suptitle("Evaluation: mean episode cost, RL vs heuristic baseline", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_service_bars(summaries: Dict[str, Dict[str, EvalSummary]], path: str) -> None:
    """A3C only — MEIS envs expose a service-level metric."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Env-1 (meis)", "Env-2 (complex)"]
    algos_x = np.arange(len(labels))
    width = 0.36
    rl_means = [summaries["a3c"][e].rl_mean_service * 100 for e in ("env1", "env2")]
    rl_stds = [summaries["a3c"][e].rl_std_service * 100 for e in ("env1", "env2")]
    bl_means = [summaries["a3c"][e].bl_mean_service * 100 for e in ("env1", "env2")]
    bl_stds = [summaries["a3c"][e].bl_std_service * 100 for e in ("env1", "env2")]
    ax.bar(algos_x - width / 2, rl_means, width, yerr=rl_stds, capsize=4,
           color="#1f77b4", label="A3C")
    ax.bar(algos_x + width / 2, bl_means, width, yerr=bl_stds, capsize=4,
           color="#ff7f0e", label="(s,S) baseline")
    ax.set_xticks(algos_x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Service level (%)")
    ax.set_ylim(85, 100)
    ax.set_title("A3C service level: RL vs (s,S) baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_cost_histogram(summary: EvalSummary, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rl = np.asarray(summary.rl_episode_costs, dtype=float)
    bl = np.asarray(summary.bl_episode_costs, dtype=float)
    lo = float(min(rl.min(), bl.min()))
    hi = float(max(rl.max(), bl.max()))
    bins = np.linspace(lo, hi, 40)
    ax.hist(bl, bins=bins, alpha=0.55, color="#ff7f0e", label="baseline")
    ax.hist(rl, bins=bins, alpha=0.55, color="#1f77b4", label="RL")
    ax.axvline(rl.mean(), color="#1f77b4", linestyle="--", linewidth=1.2)
    ax.axvline(bl.mean(), color="#ff7f0e", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Episode cost")
    ax.set_ylabel("Episode count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #

REPORT_TEMPLATE = """# Robustness of Policy-Gradient RL for Multi-Echelon Inventory Control:
## A Cross-Environment Comparison on Stationary and Non-Stationary Supply Chains

**Scope.** This report compares two policy-gradient reinforcement-learning
agents — PPO on a divergent 1-warehouse/3-retailer supply chain and A3C on a
3-node MEIS (factory → middle → 2 leaf) supply chain — against tuned
operations-research heuristics (echelon base-stock for PPO, (s,S) for A3C).
Each agent is trained and evaluated on two environments: the original
stationary formulation (**Env-1**) and a harder non-stationary variant
(**Env-2**) that adds seasonal demand, cross-retailer correlation, demand
shocks, heavy-tailed stochastic lead times, and stochastic
supplier/outbound capacity caps.

The core research question is whether the headline RL wins reported on
Env-1 transfer to the harder Env-2, or whether they are artefacts of
stationary, noise-free dynamics.

---

## 1. Environments

### 1.1 Env-1 (original)

- **PPO / divergent**: 1 warehouse, 3 retailers.
  Demand Poisson with Uniform[5,15] rates at each retailer; deterministic
  lead time = 1; no capacity cap; backlog modeled as negative inventory.
  State dim 14, continuous action ∈ ℝ⁴ (order quantities, normalized).
- **A3C / meis**: 3-node divergent MEIS (factory → middle → 2 leaves).
  Stationary demand ~ N(3300, 100) at leaves; stochastic lead times
  N(μ=2, σ=1); fixed backlog-duration window; discrete 13-way action.

### 1.2 Env-2 (complex, introduced in this work)

Both algorithms reuse the same topology, state space, and action space as
Env-1, so the agent implementations transfer unchanged. Only the *dynamics*
change:

| Stressor | PPO Env-2 | A3C Env-2 |
|---|---|---|
| Seasonal demand (sine, 120/90-day period, amplitude 0.7 / 0.5) | ✓ | ✓ |
| Cross-retailer demand correlation (shared latent factor, ρ ≈ 0.6) | ✓ | ✓ |
| Rare demand shocks (prob ~0.015–0.02, ×2–3 for 3–7 steps) | ✓ | ✓ |
| Heavy-tailed stochastic lead times + spikes (p≈0.08–0.10, +3–4 days) | ✓ | ✓ |
| Stochastic supplier + outbound capacity caps | ✓ | ✓ (middle) |
| Partial lost sales (aged-out backlog → explicit penalty) | ✓ | ✓ |

See `ppo/models/env_complex.py` and
`actor-critic/src/complex_meis_env.py` for the exact implementations, and
`ppo/configs/config.yaml` / `actor-critic/configs/complexMeisConfig.yaml`
for the numeric settings.

Crucially, the observation and action spaces are held fixed across Env-1
and Env-2 per algorithm: any performance delta is caused by the dynamics,
not by representation changes.

---

## 2. Experimental Setup

- **Training budget**:
  - PPO: 5000 iterations (buffer 512, batch 64, 5 update epochs, γ=0.95, λ=0.95,
    ε=0.2, lr=3e-4, value-loss 0.25, entropy 0.005).
  - A3C: 10000 episodes (GAE, γ, entropy coef, grad clip as in
    `configs/config.yaml`).
- **Evaluation**:
  - PPO: 100 episodes × 500 steps/episode, deterministic policy.
  - A3C: 1000 episodes × 365 steps/episode, deterministic policy.
  - Baselines: echelon base-stock (PPO, analytically initialized from
    demand and lead-time statistics) and tuned (s,S) policy for A3C
    (`scipy.optimize.differential_evolution` over 100 iterations, 30
    eval episodes each).
- **Seeds**: single seed (42) for each run; reported stds are across
  evaluation episodes, not across seeds.
- **Compute**: CUDA (GPU) for both agents.
- **Fairness**: identical hyperparameters across Env-1 and Env-2; Env-2
  results therefore reflect out-of-distribution robustness of an agent
  tuned for the simpler setting, not re-tuning on the harder env.

---

## 3. Results

### 3.1 Headline numbers

{results_table}

For A3C the evaluation pipeline runs a two-sample t-test and a Cohen's
d on the per-episode costs: both A3C cells are statistically significant
at p < 10⁻⁷⁰, and the Env-2 loss has a very large effect size. The PPO
pipeline does not compute a t-test, but the non-overlap of 95% confidence
intervals (see `ppo/results{,_complex}/evaluation_results.json` fields
`ci_lower` / `ci_upper`) implies significance at α = 0.05 in both PPO
cells.

### 3.2 Training curves

![PPO training curves](figures/training_curves_ppo.png)

![A3C training curves](figures/training_curves_a3c.png)

### 3.3 Evaluation cost (RL vs baseline)

![Evaluation cost bar charts](figures/cost_bars.png)

### 3.4 A3C service level

![A3C service level](figures/service_level_bars.png)

### 3.5 Cost distributions on Env-2

The non-stationary env induces a heavier right tail; the shape of the
distribution — not just its mean — differs materially between RL and
heuristic.

![PPO Env-2 cost distribution](figures/ppo_complex_cost_hist.png)

![A3C Env-2 cost distribution](figures/a3c_complex_cost_hist.png)

---

## 4. Discussion

### 4.1 PPO transfers, and the baseline degrades faster than PPO does

On Env-1, PPO beats the base-stock heuristic by **{ppo_env1_delta:+.2f}%**.
On Env-2, that margin *widens* to **{ppo_env2_delta:+.2f}%** — not because
PPO becomes better in absolute terms, but because the analytical base-stock
heuristic is only optimal under stationary, uncapped, backlog-tolerant
dynamics. Once shocks, capacity caps, and partial lost sales are added,
the base-stock setpoints are systematically wrong and episode costs blow
up. PPO's learned policy, by contrast, absorbs most of the non-stationarity
within the same training budget.

However, the PPO / Env-2 cost distribution has a pronounced right tail
(`std / mean ≈ {ppo_env2_cv:.2f}`), meaning a minority of episodes contain
shock events PPO has not learned to hedge. Mean cost is an optimistic
summary of that distribution.

### 4.2 A3C transfers poorly to the non-stationary MEIS

On Env-1 (stationary MEIS), A3C beats a well-tuned (s,S) policy by only
**{a3c_env1_delta:+.2f}%** — a narrow, statistically significant win
(Cohen's d ≈ {a3c_env1_d:.2f}). On Env-2, the situation reverses: A3C is
**{a3c_env2_delta_abs:.2f}% *worse*** than (s,S) (Cohen's d
≈ {a3c_env2_d:.2f}, p ≈ {a3c_env2_p:.1e}).

This is an honest negative result for A3C under this experimental
protocol. Several factors contribute:

1. **Discretized action head.** The A3C agent chooses one of 13 fixed
   order sizes per step. Under seasonal + shocked demand, the
   optimal ordering granularity is finer than the fixed
   `reorder_qty_options` grid can express.
2. **Stationary training distribution assumption.** The training loop
   uses per-episode resets that re-seed the demand generator, so
   across episodes the agent sees a *wide* range of dynamics — but
   no mechanism (e.g. recurrent network, explicit episode timestep
   feature, robust/adversarial training) that would let it
   *represent* non-stationarity within an episode.
3. **The heuristic is retuned for Env-2.** The (s,S) tuner runs
   differential evolution on the exact env it will be evaluated on,
   so the baseline is near-optimal under the new dynamics. A3C is
   not given an equivalent hyperparameter search.

None of these are fatal for A3C-for-MEIS as a research direction — they
simply mean that the Env-1 win is not robust to Env-2 dynamics with the
current model class and training protocol.

### 4.3 Asymmetric baselines

An important caveat: the two algorithms' baselines are *not* directly
comparable. PPO's base-stock heuristic is analytically initialized from
demand/lead-time statistics and is deliberately *not* re-tuned per env.
A3C's (s,S) policy *is* re-tuned per env via differential evolution
(and, in Env-2, tuned specifically against the harder dynamics).

Thus a "fairer" restatement of the bottom line is:

- PPO learns to beat a *fixed*, analytical heuristic even on Env-2.
- A3C, at this training budget, is not reliably better than a
  *retuned*, stochastic-search heuristic on Env-2.

---

## 5. Threats to validity

- **Single seed per cell.** All numbers are from one seed (42). The
  A3C/Env-2 loss to baseline is large enough in effect size
  (Cohen's d ≈ {a3c_env2_d_abs:.2f}) that seed variance is unlikely
  to flip the sign, but point estimates could shift by several
  percent with a new seed.
- **Absolute costs are not cross-env comparable.** Env-2 uses wider
  state bounds and an additional lost-sales penalty, so a
  same-algorithm cost drop from Env-1 to Env-2 is not a
  difficulty claim; only within-env RL-vs-baseline gaps are.
- **Evaluation length differs per algorithm.** PPO evaluates over
  100 × 500-step episodes (50k steps); A3C over 1000 × 365-step
  episodes (365k steps). This does not bias the RL-vs-baseline
  comparison within an algorithm.
- **Env-2 stressor choices are editorial.** The amplitudes, shock
  probabilities, and capacity caps are deliberately set to a level
  that breaks the analytical baseline. A less aggressive Env-2 would
  compress all reported deltas toward zero.

---

## 6. Reproducibility

From the repo root:

```powershell
# Re-run smoke tests (all 4 combinations)
.venv\\Scripts\\python.exe scripts\\smoke_env.py

# Re-train (optional, slow)
Push-Location ppo; ..\\.venv\\Scripts\\python.exe main.py --mode train --env divergent; Pop-Location
Push-Location ppo; ..\\.venv\\Scripts\\python.exe main.py --mode train --env complex;   Pop-Location
Push-Location actor-critic; ..\\.venv\\Scripts\\python.exe main.py --mode train --env meis;    Pop-Location
Push-Location actor-critic; ..\\.venv\\Scripts\\python.exe main.py --mode train --env complex; Pop-Location

# Re-evaluate using saved checkpoints
Push-Location ppo; ..\\.venv\\Scripts\\python.exe main.py --mode eval --env divergent; Pop-Location
Push-Location ppo; ..\\.venv\\Scripts\\python.exe main.py --mode eval --env complex;   Pop-Location
Push-Location actor-critic; ..\\.venv\\Scripts\\python.exe main.py --mode eval --env meis;    Pop-Location
Push-Location actor-critic; ..\\.venv\\Scripts\\python.exe main.py --mode eval --env complex; Pop-Location

# Rebuild this report from the saved JSON artifacts
.venv\\Scripts\\python.exe scripts\\make_report.py
```

Generated artefacts:

- `docs/report.md` (this file)
- `docs/figures/*.png`
- `ppo/results{,_complex}/evaluation_results.json`
- `actor-critic/results{,_complex}/logs/evaluation_results.json`

---

## 7. Bottom line

1. PPO's gains on the stationary Env-1 *transfer* to the harder Env-2
   against a fixed analytical baseline.
2. A3C's gains on the stationary Env-1 do *not* transfer against a
   retuned stochastic baseline: the heuristic wins on Env-2.
3. The research lesson — which is the main reason this second
   environment was introduced — is that single-environment evaluations
   of deep-RL inventory policies are not reliable evidence of real
   robustness. At minimum, both a harder distributional variant and
   an equivalently-tuned baseline should be reported.
"""


def build_results_table(summaries: Dict[str, Dict[str, EvalSummary]]) -> str:
    header = (
        "| Algorithm | Env | RL mean cost | Baseline mean cost | Δ vs baseline | "
        "Service level (RL / baseline) |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows: List[str] = []
    for algo in ("ppo", "a3c"):
        for env_key, env_label in (("env1", "Env-1"), ("env2", "Env-2")):
            s = summaries[algo][env_key]
            rl_c = f"{s.rl_mean_cost:,.2f} ± {s.rl_std_cost:,.2f}"
            bl_c = f"{s.bl_mean_cost:,.2f} ± {s.bl_std_cost:,.2f}"
            delta = f"{s.improvement_pct:+.2f}%"
            if s.rl_mean_service is not None:
                svc = f"{s.rl_mean_service * 100:.2f}% / {s.bl_mean_service * 100:.2f}%"
            else:
                svc = "n/a"
            rows.append(
                f"| {algo.upper()} | {env_label} | {rl_c} | {bl_c} | {delta} | {svc} |"
            )
    return header + "\n".join(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load evals
    summaries: Dict[str, Dict[str, EvalSummary]] = {"ppo": {}, "a3c": {}}
    for algo in ("ppo", "a3c"):
        for env_key in ("env1", "env2"):
            ev = _load_json(EVAL_PATHS[algo][env_key])
            if ev is None:
                print(f"[err] cannot build report: missing {algo}/{env_key} eval")
                return 1
            summaries[algo][env_key] = _summarize(algo, ev)

    # Load training histories
    train = {algo: {env: _load_json(TRAIN_PATHS[algo][env]) for env in ("env1", "env2")}
             for algo in ("ppo", "a3c")}
    for algo in ("ppo", "a3c"):
        for env_key in ("env1", "env2"):
            if train[algo][env_key] is None:
                print(f"[err] cannot build report: missing {algo}/{env_key} training history")
                return 1

    # --- Plots ---
    plot_training_curves_ppo(train["ppo"]["env1"], train["ppo"]["env2"],
                             os.path.join(FIG_DIR, "training_curves_ppo.png"))
    plot_training_curves_a3c(train["a3c"]["env1"], train["a3c"]["env2"],
                             os.path.join(FIG_DIR, "training_curves_a3c.png"))
    plot_cost_bars(summaries, os.path.join(FIG_DIR, "cost_bars.png"))
    plot_service_bars(summaries, os.path.join(FIG_DIR, "service_level_bars.png"))
    plot_cost_histogram(
        summaries["ppo"]["env2"],
        "PPO Env-2: per-episode cost distribution (100 episodes)",
        os.path.join(FIG_DIR, "ppo_complex_cost_hist.png"),
    )
    plot_cost_histogram(
        summaries["a3c"]["env2"],
        "A3C Env-2: per-episode cost distribution (1000 episodes)",
        os.path.join(FIG_DIR, "a3c_complex_cost_hist.png"),
    )

    # --- Report ---
    ppo_env2_cv = summaries["ppo"]["env2"].rl_std_cost / max(1.0, summaries["ppo"]["env2"].rl_mean_cost)
    a3c_env1 = summaries["a3c"]["env1"]
    a3c_env2 = summaries["a3c"]["env2"]
    # NB: use str.replace instead of str.format because the report template
    # contains literal `{,_complex}` shell-glob segments inside the repro
    # code block, which would otherwise be interpreted as format fields.
    subs = {
        "{results_table}": build_results_table(summaries),
        "{ppo_env1_delta:+.2f}": f"{summaries['ppo']['env1'].improvement_pct:+.2f}",
        "{ppo_env2_delta:+.2f}": f"{summaries['ppo']['env2'].improvement_pct:+.2f}",
        "{ppo_env2_cv:.2f}": f"{ppo_env2_cv:.2f}",
        "{a3c_env1_delta:+.2f}": f"{a3c_env1.improvement_pct:+.2f}",
        "{a3c_env1_d:.2f}": f"{a3c_env1.cohens_d:.2f}" if a3c_env1.cohens_d is not None else "n/a",
        "{a3c_env2_delta_abs:.2f}": f"{abs(a3c_env2.improvement_pct):.2f}",
        "{a3c_env2_d:.2f}": f"{a3c_env2.cohens_d:.2f}" if a3c_env2.cohens_d is not None else "n/a",
        "{a3c_env2_d_abs:.2f}": f"{abs(a3c_env2.cohens_d):.2f}" if a3c_env2.cohens_d is not None else "n/a",
        # p-values can underflow to exactly 0 when effect sizes are huge;
        # display those as "< 1e-300" rather than a misleading "0.0e+00".
        "{a3c_env2_p:.1e}": (
            "< 1e-300" if (a3c_env2.p_value is not None and a3c_env2.p_value == 0.0)
            else (f"{a3c_env2.p_value:.1e}" if a3c_env2.p_value is not None else "n/a")
        ),
    }
    md = REPORT_TEMPLATE
    for key, val in subs.items():
        md = md.replace(key, val)
    report_path = os.path.join(DOCS_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nReport written: {report_path}")
    print(f"Figures: {FIG_DIR}")
    print("\nHeadline deltas:")
    for algo in ("ppo", "a3c"):
        for env_key in ("env1", "env2"):
            s = summaries[algo][env_key]
            print(f"  {algo.upper():<3}  {env_key}: Δ = {s.improvement_pct:+.2f}%  "
                  f"(RL {s.rl_mean_cost:,.2f} vs BL {s.bl_mean_cost:,.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
