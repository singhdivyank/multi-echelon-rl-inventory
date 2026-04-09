# multi-echelon-rl-inventory

## Actor-Critic

![](./actor-critic/results/plots/cost_per_period.png)

(s,S) Baseline:
  Mean Cost: 1962.40 ± 52.39
  Mean Service Level: 95.44% ± 0.20%
  Cost Breakdown:
    Shortage: 18160470.44
    Holding: 163967.32
    Reordering: 1237604.00

--- Comparison Results ---

Improvement:
  Cost: 1.51%
  Service Level: 0.19%
  Statistical Significance (cost): True (p=0.0000)
  Effect Size (Cohen's d): 0.742

![](./actor-critic/results/plots/comparison.png)

> "The RL agent achieves a modest but statistically significant improvement (~1.5%) over the (s,S) policy, which is already near-optimal. This demonstrates the ability of reinforcement learning to fine-tune inventory decisions beyond classical heuristics, particularly in stochastic environments."

## PPO

> "PPO rapidly learns an optimal inventory policy within ~150 episodes and maintains it, with minor fluctuations due to stochastic exploration. PPO reduces cost by 33% compared to (s, S) baseline."

![](./ppo/results/plots/comparison.png)