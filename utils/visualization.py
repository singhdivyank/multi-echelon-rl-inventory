"""
Visualization Utilities

Functions for creating plots and heatmaps to analyze:
1. Average cost per period and service level
2. Training curves (return vs steps)
3. Variance across seeds
4. Heatmap of learned policies vs baseline

Author: AI/ML Engineering Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def set_plot_style():
    """Set consistent plot style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves: rewards, costs, service levels
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display plot
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    rewards = history['episode_rewards']
    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, label='Episode reward')
    
    # Moving average
    window = min(100, len(rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, linewidth=2, label=f'{window}-ep moving avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Episode costs
    ax = axes[0, 1]
    costs = history['episode_costs']
    ax.plot(episodes, costs, alpha=0.3, label='Episode cost')
    
    if window > 1:
        moving_avg = np.convolve(costs, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, linewidth=2, label=f'{window}-ep moving avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Cost')
    ax.set_title('Training Cost Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Service levels
    ax = axes[1, 0]
    service_levels = history['service_levels']
    ax.plot(episodes, service_levels, alpha=0.3, label='Episode service level')
    
    if window > 1:
        moving_avg = np.convolve(service_levels, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, linewidth=2, label=f'{window}-ep moving avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Service Level')
    ax.set_title('Service Level Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Training losses
    ax = axes[1, 1]
    if len(history['losses']) > 0:
        ax.plot(history['losses'], label='Total loss', alpha=0.7)
        ax.plot(history['actor_losses'], label='Actor loss', alpha=0.7)
        ax.plot(history['critic_losses'], label='Critic loss', alpha=0.7)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    rl_metrics: Dict,
    baseline_metrics: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare RL agent vs baseline on cost and service level
    
    Args:
        rl_metrics: RL agent evaluation metrics
        baseline_metrics: Baseline policy metrics
        save_path: Path to save figure
        show: Whether to display plot
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cost comparison
    ax = axes[0]
    methods = ['A3C (RL)', '(s,S) Baseline']
    costs = [rl_metrics['mean_cost'], baseline_metrics['mean_cost']]
    errors = [rl_metrics['std_cost'], baseline_metrics['std_cost']]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, costs, yerr=errors, capsize=5, alpha=0.7, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Average Total Cost')
    ax.set_title('Cost Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Service level comparison
    ax = axes[1]
    service_levels = [rl_metrics['mean_service_level'], baseline_metrics['mean_service_level']]
    errors = [rl_metrics['std_service_level'], baseline_metrics['std_service_level']]
    
    bars = ax.bar(x, service_levels, yerr=errors, capsize=5, alpha=0.7, color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Service Level')
    ax.set_title('Service Level Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, sl) in enumerate(zip(bars, service_levels)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sl:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_variance_across_seeds(
    results_by_seed: Dict,
    metric: str = 'mean_cost',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot variance across different random seeds
    
    Args:
        results_by_seed: Dictionary {seed: metrics_dict}
        metric: Which metric to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seeds = sorted(results_by_seed.keys())
    
    # Extract RL and baseline values
    rl_values = [results_by_seed[seed]['rl'][metric] for seed in seeds]
    baseline_values = [results_by_seed[seed]['baseline'][metric] for seed in seeds]
    
    x = np.arange(len(seeds))
    width = 0.35
    
    ax.bar(x - width/2, rl_values, width, label='A3C (RL)', alpha=0.7, color='#2ecc71')
    ax.bar(x + width/2, baseline_values, width, label='(s,S) Baseline', alpha=0.7, color='#e74c3c')
    
    ax.set_xlabel('Random Seed')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Across Random Seeds')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean lines
    rl_mean = np.mean(rl_values)
    baseline_mean = np.mean(baseline_values)
    ax.axhline(rl_mean, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7, label=f'RL mean: {rl_mean:.2f}')
    ax.axhline(baseline_mean, color='#c0392b', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline mean: {baseline_mean:.2f}')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Variance plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_policy_heatmap(
    env,
    agent,
    baseline_policy,
    save_path: Optional[str] = None,
    show: bool = True,
    n_samples: int = 1000
):
    """
    Create heatmap showing action distributions for RL vs baseline
    
    Args:
        env: MEIS environment
        agent: Trained A3C agent
        baseline_policy: Baseline (s,S) policy
        save_path: Path to save figure
        show: Whether to display plot
        n_samples: Number of states to sample
    """
    set_plot_style()
    
    # Sample states from environment
    states = []
    for _ in range(n_samples):
        env.reset()
        for _ in range(np.random.randint(1, 100)):
            action = env.action_space.sample()
            state, _, done, _ = env.step(action)
            states.append(state)
            if done:
                break
    
    states = np.array(states[:n_samples])
    
    # Get RL actions
    agent.eval_mode()
    rl_actions = []
    for state in states:
        action, _, _ = agent.select_action(state, deterministic=True)
        rl_actions.append(action)
    rl_actions = np.array(rl_actions)
    
    # Get baseline actions
    baseline_actions = []
    for state in states:
        action = baseline_policy.select_action(state, env.reorder_quantities)
        baseline_actions.append(action)
    baseline_actions = np.array(baseline_actions)
    
    # Create action distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    action_labels = ['No order'] + \
                   [f'M-{i+1}' for i in range(4)] + \
                   [f'L1-{i+1}' for i in range(4)] + \
                   [f'L2-{i+1}' for i in range(4)]
    
    # RL action distribution
    ax = axes[0]
    rl_counts = np.bincount(rl_actions, minlength=13)
    rl_freq = rl_counts / n_samples
    bars = ax.bar(range(13), rl_freq, alpha=0.7, color='#2ecc71')
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('A3C Policy Action Distribution')
    ax.set_xticks(range(13))
    ax.set_xticklabels(action_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Baseline action distribution
    ax = axes[1]
    baseline_counts = np.bincount(baseline_actions, minlength=13)
    baseline_freq = baseline_counts / n_samples
    bars = ax.bar(range(13), baseline_freq, alpha=0.7, color='#e74c3c')
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.set_title('(s,S) Baseline Action Distribution')
    ax.set_xticks(range(13))
    ax.set_xticklabels(action_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy heatmap saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_cost_per_period(
    env,
    agent,
    baseline_policy,
    n_episodes: int = 10,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot average cost per time period for RL vs baseline
    
    Args:
        env: MEIS environment
        agent: Trained A3C agent
        baseline_policy: Baseline policy
        n_episodes: Number of episodes to average
        save_path: Path to save figure
        show: Whether to display plot
    """
    set_plot_style()
    
    # Collect costs per period for RL
    agent.eval_mode()
    rl_costs_per_period = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_costs = []
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_costs.append(-reward)
        
        rl_costs_per_period.append(episode_costs)
    
    # Collect costs per period for baseline
    baseline_costs_per_period = []
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_costs = []
        
        while not done:
            action = baseline_policy.select_action(state, env.reorder_quantities)
            state, reward, done, info = env.step(action)
            episode_costs.append(-reward)
        
        baseline_costs_per_period.append(episode_costs)
    
    # Compute mean and std
    max_len = max(max(len(ep) for ep in rl_costs_per_period),
                  max(len(ep) for ep in baseline_costs_per_period))
    
    rl_costs_padded = np.array([ep + [0]*(max_len - len(ep)) for ep in rl_costs_per_period])
    baseline_costs_padded = np.array([ep + [0]*(max_len - len(ep)) for ep in baseline_costs_per_period])
    
    rl_mean = rl_costs_padded.mean(axis=0)
    rl_std = rl_costs_padded.std(axis=0)
    baseline_mean = baseline_costs_padded.mean(axis=0)
    baseline_std = baseline_costs_padded.std(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    periods = np.arange(max_len)
    
    ax.plot(periods, rl_mean, label='A3C (RL)', linewidth=2, color='#2ecc71')
    ax.fill_between(periods, rl_mean - rl_std, rl_mean + rl_std, alpha=0.3, color='#2ecc71')
    
    ax.plot(periods, baseline_mean, label='(s,S) Baseline', linewidth=2, color='#e74c3c')
    ax.fill_between(periods, baseline_mean - baseline_std, baseline_mean + baseline_std, alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Time Period (Days)')
    ax.set_ylabel('Cost')
    ax.set_title('Average Cost per Period')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cost per period plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_training_curve(
    training_history: Dict, 
    save_path: str
):
    rewards = np.array(training_history['episode_rewards'])

    # Smooth rewards
    smoothed = moving_average(rewards, window=100)

    # Std deviation (rolling)
    stds = []
    window = 100
    for i in range(len(smoothed)):
        stds.append(np.std(rewards[i:i+window]))
    stds = np.array(stds)

    episodes = np.arange(len(smoothed))

    # Best so far
    best = np.maximum.accumulate(smoothed)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(episodes, smoothed, label='Training Scores', color='blue')
    plt.fill_between(
        episodes,
        smoothed - stds,
        smoothed + stds,
        color='blue',
        alpha=0.2
    )

    plt.plot(episodes, best, label='Best Result', color='orange')

    plt.xlabel('Episodes')
    plt.ylabel('Score [CHF]')
    plt.title('RL Training Performance')

    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_all_plots(
    env,
    agent,
    baseline_policy,
    training_history: Dict,
    rl_metrics: Dict,
    baseline_metrics: Dict,
    save_dir: str,
    show: bool = False
):
    """
    Create all visualization plots
    
    Args:
        env: MEIS environment
        agent: Trained A3C agent
        baseline_policy: Baseline policy
        training_history: Training history dictionary
        rl_metrics: RL evaluation metrics
        baseline_metrics: Baseline evaluation metrics
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Training curves
    plot_training_curves(
        training_history,
        save_path=os.path.join(save_dir, 'training_curves.png'),
        show=show
    )
    
    # Comparison
    plot_comparison(
        rl_metrics,
        baseline_metrics,
        save_path=os.path.join(save_dir, 'comparison.png'),
        show=show
    )
    
    # Policy heatmap
    plot_policy_heatmap(
        env,
        agent,
        baseline_policy,
        save_path=os.path.join(save_dir, 'policy_heatmap.png'),
        show=show
    )
    
    # Cost per period
    plot_cost_per_period(
        env,
        agent,
        baseline_policy,
        save_path=os.path.join(save_dir, 'cost_per_period.png'),
        show=show
    )

    plot_training_curve(
        training_history=training_history,
        save_path=os.path.join(save_dir, 'training_results.png')
    )
    
    print(f"All plots saved to: {save_dir}")