"""
Evaluation Utilities

Functions for evaluating and comparing RL agent and baseline policies across:
- Multiple random seeds
- Different metrics (cost, service level, etc.)
- Statistical significance testing

Author: AI/ML Engineering Team
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import json
import os


def evaluate_agent(
    env,
    agent,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False
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
    if seed is not None:
        env.seed(seed)
    
    agent.eval_mode()
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    service_levels = []
    cost_breakdowns = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        service_level_sum = 0
        steps = 0
        episode_breakdown = {'shortage': 0, 'holding': 0, 'reordering': 0}
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=deterministic)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']
            
            # Accumulate cost breakdown
            for cost_type in ['shortage', 'holding', 'reordering']:
                for wh in info['cost_breakdown'][cost_type]:
                    episode_breakdown[cost_type] += info['cost_breakdown'][cost_type][wh]
            
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(steps)
        service_levels.append(service_level_sum / steps if steps > 0 else 0)
        cost_breakdowns.append(episode_breakdown)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: Cost={episode_cost:.2f}, Service Level={service_levels[-1]:.2%}")
    
    # Aggregate cost breakdown
    avg_cost_breakdown = {
        'shortage': np.mean([cb['shortage'] for cb in cost_breakdowns]),
        'holding': np.mean([cb['holding'] for cb in cost_breakdowns]),
        'reordering': np.mean([cb['reordering'] for cb in cost_breakdowns]),
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


def evaluate_baseline(
    env,
    baseline_policy,
    n_episodes: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate baseline (s,S) policy
    
    Args:
        env: MEIS environment
        baseline_policy: Baseline policy instance
        n_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dictionary of evaluation metrics
    """
    if seed is not None:
        env.seed(seed)
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    service_levels = []
    cost_breakdowns = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        service_level_sum = 0
        steps = 0
        episode_breakdown = {'shortage': 0, 'holding': 0, 'reordering': 0}
        
        while not done:
            action = baseline_policy.select_action(state, env.reorder_quantities)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']
            
            # Accumulate cost breakdown
            for cost_type in ['shortage', 'holding', 'reordering']:
                for wh in info['cost_breakdown'][cost_type]:
                    episode_breakdown[cost_type] += info['cost_breakdown'][cost_type][wh]
            
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(steps)
        service_levels.append(service_level_sum / steps if steps > 0 else 0)
        cost_breakdowns.append(episode_breakdown)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}: Cost={episode_cost:.2f}, Service Level={service_levels[-1]:.2%}")
    
    # Aggregate cost breakdown
    avg_cost_breakdown = {
        'shortage': np.mean([cb['shortage'] for cb in cost_breakdowns]),
        'holding': np.mean([cb['holding'] for cb in cost_breakdowns]),
        'reordering': np.mean([cb['reordering'] for cb in cost_breakdowns]),
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


def compare_policies(
    rl_metrics: Dict,
    baseline_metrics: Dict,
    alpha: float = 0.05
) -> Dict:
    """
    Statistical comparison between RL and baseline policies
    
    Args:
        rl_metrics: RL evaluation metrics
        baseline_metrics: Baseline evaluation metrics
        alpha: Significance level
    
    Returns:
        Dictionary of comparison statistics
    """
    comparison = {}
    
    # Cost comparison
    cost_improvement = (
        (baseline_metrics['mean_cost'] - rl_metrics['mean_cost']) /
        baseline_metrics['mean_cost'] * 100
    )
    comparison['cost_improvement_percent'] = cost_improvement
    
    # T-test for cost
    t_stat, p_value = stats.ttest_ind(
        rl_metrics['episode_costs'],
        baseline_metrics['episode_costs']
    )
    comparison['cost_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': bool(p_value < alpha),
    }
    
    # Service level comparison
    sl_improvement = (
        (rl_metrics['mean_service_level'] - baseline_metrics['mean_service_level']) * 100
    )
    comparison['service_level_improvement_percent'] = sl_improvement
    
    # T-test for service level
    t_stat, p_value = stats.ttest_ind(
        rl_metrics['service_levels'],
        baseline_metrics['service_levels']
    )
    comparison['service_level_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': bool(p_value < alpha),
    }
    
    # Effect size (Cohen's d)
    pooled_std_cost = np.sqrt(
        (rl_metrics['std_cost']**2 + baseline_metrics['std_cost']**2) / 2
    )
    comparison['cohens_d_cost'] = (
        (baseline_metrics['mean_cost'] - rl_metrics['mean_cost']) / pooled_std_cost
    )
    
    return comparison


def evaluate_across_seeds(
    env_fn,
    agent_fn,
    baseline_policy_fn,
    seeds: List[int],
    n_eval_episodes: int = 100,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate policies across multiple random seeds
    
    Args:
        env_fn: Function to create environment
        agent_fn: Function to create and load agent
        baseline_policy_fn: Function to create baseline policy
        seeds: List of random seeds
        n_eval_episodes: Number of episodes per seed
        save_dir: Directory to save results
    
    Returns:
        Dictionary of results by seed
    """
    results = {}
    
    for seed in seeds:
        print(f"\nEvaluating with seed {seed}...")
        
        # Create environment
        env = env_fn()
        env.seed(seed)
        
        # Evaluate RL agent
        agent = agent_fn()
        rl_metrics = evaluate_agent(
            env,
            agent,
            n_episodes=n_eval_episodes,
            seed=seed,
            verbose=True
        )
        
        # Evaluate baseline
        baseline_policy = baseline_policy_fn()
        baseline_metrics = evaluate_baseline(
            env,
            baseline_policy,
            n_episodes=n_eval_episodes,
            seed=seed,
            verbose=True
        )
        
        # Compare
        comparison = compare_policies(rl_metrics, baseline_metrics)
        
        results[seed] = {
            'rl': rl_metrics,
            'baseline': baseline_metrics,
            'comparison': comparison,
        }
        
        print(f"Seed {seed} results:")
        print(f"  RL Cost: {rl_metrics['mean_cost']:.2f} ± {rl_metrics['std_cost']:.2f}")
        print(f"  Baseline Cost: {baseline_metrics['mean_cost']:.2f} ± {baseline_metrics['std_cost']:.2f}")
        print(f"  Cost Improvement: {comparison['cost_improvement_percent']:.2f}%")
        print(f"  RL Service Level: {rl_metrics['mean_service_level']:.2%}")
        print(f"  Baseline Service Level: {baseline_metrics['mean_service_level']:.2%}")
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        results_path = os.path.join(save_dir, 'seed_results.json')
        
        # Convert numpy arrays to lists for JSON
        results_json = {}
        for seed, result in results.items():
            results_json[str(seed)] = {
                'rl': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in result['rl'].items()},
                'baseline': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in result['baseline'].items()},
                'comparison': result['comparison'],
            }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    return results


def print_summary_statistics(results_by_seed: Dict):
    """
    Print summary statistics across seeds
    
    Args:
        results_by_seed: Dictionary of results by seed
    """
    seeds = list(results_by_seed.keys())
    
    # Aggregate metrics
    rl_costs = [results_by_seed[s]['rl']['mean_cost'] for s in seeds]
    baseline_costs = [results_by_seed[s]['baseline']['mean_cost'] for s in seeds]
    cost_improvements = [results_by_seed[s]['comparison']['cost_improvement_percent'] for s in seeds]
    
    rl_sls = [results_by_seed[s]['rl']['mean_service_level'] for s in seeds]
    baseline_sls = [results_by_seed[s]['baseline']['mean_service_level'] for s in seeds]
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS ACROSS SEEDS")
    print("="*60)
    
    print(f"\nNumber of seeds: {len(seeds)}")
    print(f"Seeds: {seeds}")
    
    print(f"\n--- Cost ---")
    print(f"RL Agent:")
    print(f"  Mean: {np.mean(rl_costs):.2f}")
    print(f"  Std: {np.std(rl_costs):.2f}")
    print(f"  Min: {np.min(rl_costs):.2f}")
    print(f"  Max: {np.max(rl_costs):.2f}")
    
    print(f"\nBaseline:")
    print(f"  Mean: {np.mean(baseline_costs):.2f}")
    print(f"  Std: {np.std(baseline_costs):.2f}")
    print(f"  Min: {np.min(baseline_costs):.2f}")
    print(f"  Max: {np.max(baseline_costs):.2f}")
    
    print(f"\nCost Improvement:")
    print(f"  Mean: {np.mean(cost_improvements):.2f}%")
    print(f"  Std: {np.std(cost_improvements):.2f}%")
    print(f"  Min: {np.min(cost_improvements):.2f}%")
    print(f"  Max: {np.max(cost_improvements):.2f}%")
    
    print(f"\n--- Service Level ---")
    print(f"RL Agent:")
    print(f"  Mean: {np.mean(rl_sls):.2%}")
    print(f"  Std: {np.std(rl_sls):.2%}")
    
    print(f"\nBaseline:")
    print(f"  Mean: {np.mean(baseline_sls):.2%}")
    print(f"  Std: {np.std(baseline_sls):.2%}")
    
    print("\n" + "="*60)