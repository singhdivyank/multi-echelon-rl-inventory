# PPO for divergent Multi-Echelon Inventory Optimization

Implementation of **Proximal Policy Optimization** for **Cost-Efficient** Supply Chain Control as defined in this [paper](https://link.springer.com/article/10.1007/s10100-023-00872-2).

## Problem Statement

Modern supply chains involve **multi-echelon inventory systems** where a central warehouse distributes goods to three retailers. The challenge lies in:

- Handling stochastic demand
- Managing inventory holding costs vs shortage penalties
- Making sequential ordering decisions under uncertainty

> We aim to learn an optimal policy 𝜋(𝑎∣𝑠) that minimizes total operational cost

$$
min\ E\left[\sum_{t=0}^{T}(C_{holding} + C_{backorder} + C_{ordering})\right]
$$

## Architecture

We use a shared backbone with two separate heads:
* Actor (Policy Network): outputs action distribution $\pi_{\theta}(a|s)$
* Critic (Value Network): estimates state value $V_{\phi}(s)$

Enables **low-variance** policy updates and stable convergence

Architecture configurations defined in: `configs/config.yaml`

Tunables:
- Learning rate
- Discount factor
- GAE
- Clip epsilon
- Batch size
- Buffer size
- Update epochs

## RL Formulation

**Objective Function** ([Clipped Surrogate Objective Function](https://campus.datacamp.com/courses/deep-reinforcement-learning-in-python/proximal-policy-optimization-and-drl-tips?ex=3)):

$$
L^{CLIP}(\theta) = E\left[\min\left(r_{t}(\theta)A_{t}, clip(r_{t}(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

**Value Loss**:

$$
L^{VF} = E\left[(V(s_{t}) - R_t)^2\right]
$$

**Entropy Bonus**:

$$
L^{ENT} = E\left[H(\pi(\cdot|s_t))\right]
$$

**Total Loss**: 

$$
L = L^{CLIP} + c_1 L^{VF} + c_2 L^{ENT}
$$

**Advantage Estimation** (reduce variance):

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t = \delta_t + \gamma\lambda\delta_{t+1} + \cdots
$$

## Results

> **PPO achieves a 33.5% cost reduction over the (s, S) baseline — saving ~1.38M per episode — while simultaneously improving service level from 95% to 99%.**

### Performance Comparison

| Metric | PPO | (s, S) Baseline | Improvement |
| ------ | --- | --------------- | ----------- |
| Avg Cost | **2,733,733** | 4,112,776 | ↓ **33.5%** (~1.38M/episode) |
| Cost Std Dev | 24,969 | 8,759 | — |
| 95% CI (cost) | [2.729M, 2.739M] | [4.111M, 4.115M] | Non-overlapping ✅ |
| Avg Reward | **−273.4** | −411.3 | ↑ 34% better |
| Service Level | **~99%** | ~95% | ↑ +4 pp |
| Avg Order Qty / Step | **2.67** | 14.0 | ↓ 5.2× fewer orders |
| Backorders | **0** | 0 | Maintained |

### Confidence Interval Analysis

The 95% confidence intervals for the two policies are **completely non-overlapping**, confirming the cost improvement is statistically significant without requiring a formal test:

```
PPO       ████████████████  [2,728,754 — 2,738,712]
Baseline                                              ██████████████████  [4,111,029 — 4,114,522]
```

The gap between the upper bound of PPO and the lower bound of baseline is **~1.37M** — more than 54× the width of either CI.

### Ordering Behaviour

The most revealing difference is in *how* each policy controls inventory:

| Behaviour | PPO | (s, S) Baseline |
| --------- | --- | --------------- |
| Avg orders/step | **2.67** | 14.0 |
| Strategy | Lean, demand-responsive | Fixed threshold reorder |
| Adaptability | Dynamic | Static |

> PPO learns a lean, just-in-time ordering strategy — ordering ~5× less per step yet achieving a **higher** service level. This shows the agent has internalized the cost structure and found a fundamentally better policy, not just a parameter tweak.

### Key Takeaways

- **Magnitude**: 33.5% cost reduction is the headline result — not marginal, but a structural improvement over the classical heuristic
- **Service level**: PPO improves service level from 95% → 99% while simultaneously cutting costs — both dimensions improve together
- **Ordering efficiency**: 5.2× fewer orders per step demonstrates the agent learns genuine demand anticipation rather than reactive restocking
- **Robustness**: Tight 95% CI ([2.729M, 2.739M]) shows consistent performance across all 100 evaluation episodes

![](./results/plots/comparison.png)

### Training Stability

* Converges after ~1K iterations with the clipped surrogate objective preventing destructive policy updates
* GAE smooths the advantage signal, suppressing gradient noise across the multi-echelon state space
* Stable, consistent policy post-convergence with tight episode-to-episode cost variance

## Execution Steps

``` bash
cd ppo
python3 main.py --mode train    # train agent
python3 main.py --mode eval     # evaluate performance against baseline
python3 main.py --mode plot     # visualise training curves and evaluation results
```