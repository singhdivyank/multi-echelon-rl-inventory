# A3C for Inventory Optimization (MEIS Environment)

Implementation of **Asynchronous Advantage Actor-Critic** for inventory optimization in a stochastic multi-echelon supply chain environment as defined in this [research paper](https://link.springer.com/article/10.1007/s10100-023-00872-2).

## Problem Statement

Modern supply chains face stochastic demand, lead times, and cost trade-offs, making inventory optimization a challenging sequential decision problem. The challenge lies in:

- Uncertain demand patterns
- Trade off between holding, ordering, and stockout costs
- Temporal dependencies (decisions affect future states)
- Balancing cost vs service level

We aim to:

> Minimise total inventory cost while maintaining high service levels

## Architecture

1. **Actor Network (Policy π)**
- Outputs probablility distribution over actions
- Learns what action to take

2. **Critic Network (Value Function V)**
- Estimates expected return from a state
- Helps reduce variance in training

3. **Environment (Custom MEIS environment)**
- Demand randomness
- Inventory levels
- Cost structure

Architecture configurations controlled via: `configs/config.yaml`

Environment settings defined in: `configs/meisConfig.yaml`

Key tunables:
* Learning Rate
* Discount Factor
* Entropy Regularization
* Episode Length

## RL Formulation

**Advantage Function**:

$$
A(s_t, a_t) = R_t - V(s_t)
$$

**Actor loss function (Policy Gradient)**

$$
Loss_{actor} = -\log\pi_{\theta}(a_t | s_t) \cdot A(s_t, a_t)
$$

**Critic loss function (Value Function)**

$$
Loss_{critic} = (R_t - V(s_t))^2
$$

**Total Loss**:

$$
L = Loss_{actor} + \lambda Loss_{critic}
$$


## Results

### Trade-off insights

| Metric | A3C (RL) | (s, S) baseline |
| ------ | -------- | --------------- |
| Avg Cost | 1933 | 1962 |
| Service Level | 95.6 | 95.4 |
| Adaptability | High | Static |
| Stability | Learned | Heuristic |

![](./results/plots/cost_per_period.png)

![](./results/plots/comparison.png)

## Training Stability

* Converges after 4K-5K episodes
* Reduced variance due to critic guidance
* Stable policy after convergence

## Execution Steps

``` bash
cd actor-critic
python3 main.py --mode train    # train agent
python3 main.py --mode eval     # evaluate performance against baseline
python3 main.py --mode plot     # visualise training curves and evaluation results
```