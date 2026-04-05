"""
Evaluation Utilities

Functions for evaluating and comparing RL agent and baseline policies across:
- Multiple random seeds
- Different metrics (cost, service level, etc.)
- Statistical significance testing
"""

from typing import Dict

import numpy as np
from scipy import stats

def evaluate_agent(
    env, 
    n_episodes: int, 
    seed: int, 
    agent = None, 
    baseline_policy = None,
    verbose: bool=False
) -> Dict:
    """
    Evaluate trained RL agent
    
    Args:
        env: MEIS environment
        agent: Trained A3C agent
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary of evaluation metrics
    """

    env.seed(seed)
    if agent is not None:
        agent.eval_mode()

    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    service_levels = []
    cost_breakdowns = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward, episode_cost = 0, 0
        service_level_sum = 0
        steps = 0
        episode_breakdown = {'shortage': 0, 'holding': 0, 'reordering': 0}

        while not done:
            if baseline_policy:
                action = baseline_policy.select_action(state, env.reorder_quantities)
            else:
                action, _, _ = agent.select_action(state, deterministic=True)
            
            state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']

            for cost_type in ['shortage', 'holding']:
                for wh in info['cost_breakdown'][cost_type]:
                    episode_breakdown[cost_type] += info['cost_breakdown'][cost_type][wh]
            
            episode_breakdown['reordering'] += info['cost_breakdown']['reordering']
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(steps)
        service_levels.append(service_level_sum / steps if steps > 0 else 0)
        cost_breakdowns.append(episode_breakdown)

        if verbose and not (episode + 1) % 10:
            print(f"Episode {episode + 1}/{n_episodes}: Cost={episode_cost:.2f}, Service Level={service_levels[-1]:.2%}")
    
    avg_cost_breakdown = {
        'shortage': np.mean([cb['shortage'] for cb in cost_breakdowns]),
        'holding': np.mean([cb['holding'] for cb in cost_breakdowns]),
        'reordering': np.mean([cb['reordering'] for cb in cost_breakdowns])
    }

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_length': np.mean(episode_lengths),
        'mean_service_level': np.mean(service_levels),
        'std_service_level': np.std(service_levels),
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'service_levels': service_levels,
        'cost_breakdown': avg_cost_breakdown,
    }

def compare_policies(rl_metrics: Dict, baseline_metrics: Dict) -> Dict:
    """
    Statistical comparison between RL and baseline policies
    
    Args:
        rl_metrics: RL evaluation metrics
        baseline_metrics: Baseline evaluation metrics
    
    Returns:
        Dictionary of comparison statistics
    """

    comparison = {}

    cost_improvement = (
        (baseline_metrics['mean_cost'] - rl_metrics['mean_cost']) / 
        baseline_metrics['mean_cost'] * 100
    )
    comparison['cost_improvement_percent'] = cost_improvement
    
    t_stat, p_value = stats.ttest_ind(
        rl_metrics['episode_costs'],
        baseline_metrics['episode_costs']
    )
    comparison['cost_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': bool(p_value < 0.05)
    }
    
    sl_improvement = (
        (rl_metrics['mean_service_level'] - baseline_metrics['mean_service_level']) * 100
    )
    comparison['service_level_improvement_percent'] = sl_improvement

    t_stat, p_value = stats.ttest_ind(
        rl_metrics['service_levels'],
        baseline_metrics['service_levels']
    )
    comparison['service_level_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': bool(p_value < 0.05)
    }

    pooled_std_cost = np.sqrt(
        (rl_metrics['std_cost'] ** 2 + baseline_metrics['std_cost'] ** 2) / 2
    )
    comparison['cohens_d_cost'] = (
        (baseline_metrics['mean_cost'] - rl_metrics['mean_cost']) / pooled_std_cost
    )
    
    return comparison
