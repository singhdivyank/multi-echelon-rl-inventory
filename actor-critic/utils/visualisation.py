"""
Visualization Utilities

Functions for creating plots and heatmaps to analyze:
1. Average cost per period and service level
2. Training curves (return vs steps)
3. Agent performance comparison with baseline
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def set_plot_style():
    """Set consistent plot styles"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

def collect_data(env, policy, is_agent=True):
    all_costs = []
    n_episodes = 10

    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_costs = []

        while not done:
            if is_agent:
                action, _, _ = policy.select_action(state, deterministic=True)
            else:
                action = policy.select_action(state, env.reorder_quantities)
            
            state, reward, done, _ = env.step(action)
            episode_costs.append(-reward)
        all_costs.append(episode_costs)
        
    max_len = max(len(ep) for ep in all_costs)
    padded = np.full((n_episodes, max_len), np.nan)
    for i, ep in enumerate(all_costs):
        padded[i, :len(ep)] = ep
    
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

def plot_training_curves(history: Dict, save_path: str):
    """
    Plot training curves: rewards, costs, service levels
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """

    set_plot_style()

    _, ax = plt.subplots(figsize=(14, 10))
    
    loss_keys = ['losses', 'actor_losses', 'critic_losses']
    if all(k in history and len(history[k]) > 0 for k in loss_keys):
        for k in loss_keys:
            ax.plot(history[k], label=k.replace('_', ' ').capitalize(), alpha=0.7)
        ax.set(xlabel='Update Step', ylabel='Loss', title='Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")

def plot_comparison(rl_metrics: Dict, baseline_metrics: Dict, save_path: str):
    """
    Compare RL agent vs baseline on cost and service level
    
    Args:
        rl_metrics: RL agent evaluation metrics
        baseline_metrics: Baseline policy metrics
        save_path: Path to save figure
    """

    set_plot_style()
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = ['A3C (RL)', '(s,S) Baseline']
    colors = ['#2ecc71', '#e74c3c']
    x = np.arange(len(methods))

    plot_configs = [
        ('cost', 'Average Total Cost', 'Cost Comparison', axes[0], '.0f'),
        ('service_level', 'Service Level', 'Service Level Comparison', axes[1], '.2%')
    ]

    for key, ylabel, title, ax, fmt in plot_configs:
        means = [rl_metrics[f'mean_{key}'], baseline_metrics[f'mean_{key}']]
        errors = [rl_metrics[f'std_{key}'], baseline_metrics[f'std_{key}']]

        bars = ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7, color=colors)
        ax.set(ylabel=ylabel, title=title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.grid(True, alpha=0.3, axis='y')

        if key == 'service_level':
            ax.set_ylim([0, 1])
        
        for bar, val, err in zip(bars, means, errors):
            height = bar.get_height()
            label = f'{val:{fmt}}'
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + (err * 0.1),
                label,
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {save_path}")

def plot_cost_per_period(
    env, 
    agent, 
    baseline_policy, 
    save_path
):
    """
    Plot average cost per time period for RL vs baseline
    
    Args:
        env: MEIS environment
        agent: Trained A3C agent
        baseline_policy: Baseline policy
        save_path: Path to save figure
    """

    set_plot_style()
    agent.eval_mode()
    rl_mean, rl_std = collect_data(env, agent, is_agent=True)
    bs_mean, bs_std = collect_data(env, baseline_policy, is_agent=False)

    _, ax = plt.subplots(figsize=(14, 6))
    periods = np.arange(len(rl_mean))
    configs = [
        (rl_mean, rl_std, 'A3C (RL)', '#2ecc71'),
        (bs_mean, bs_std, '(s,S) Baseline', '#e74c3c')
    ]

    for mean, std, label, color in configs:
        ax.plot(periods, mean, label=label, lw=2, color=color)
        ax.fill_between(periods, mean - std, mean + std, alpha = 0.2, color = color)
    
    ax.set(xlabel='Time Period (Days)', ylabel='Cost', title='Average Cost per Period')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curve(training_history: Dict, save_path: str):
    rewards = np.array(training_history['episode_rewards'])
    smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
    stds = []
    
    for i in range(len(smoothed)):
        stds.append(np.std(rewards[i: i+100]))
    stds = np.array(stds)

    episodes = np.arange(len(smoothed))
    best = np.maximum.accumulate(smoothed)

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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
