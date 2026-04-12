"""
Visualization utilities for training and evaluation results
"""

import os
from typing import Dict

import matplotlib.pyplot as plt

def plot_training_curves(
    ppo_stats: Dict, 
    baseline_stats: Dict, 
    save_path: str
):
    """
    Plot training curves comparing PPO and baseline
    
    Args:
        ppo_stats: PPO training statistics
        baseline_stats: Baseline training statistics
        save_path: Path to save figure
    """

    from utils.visualise_helpers import (
        _plot_baseline, 
        _plot_with_confidence, 
        _plot_loss
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    plot_configs = [
        ('episode_costs', axes[0, 0], 'Cost vs Iteration', 'Cost', '#2E86AB'),
        ('episode_rewards', axes[0, 1], 'Reward vs Iteration', 'Reward', '#06A77D')
    ]
    
    iterations = ppo_stats.get('iterations', [])
    window = min(100, max(1, len(ppo_stats.get('episode_costs', [])) // 10))

    for key, ax, title, ylabel, color in plot_configs:
        data = ppo_stats.get(key, [])
        if data:
            _plot_with_confidence(
                ax=ax, 
                x=iterations, 
                y=data, 
                label='PPO', 
                color=color, 
                window=window
            )

            if baseline_stats and key in baseline_stats:
                stats = baseline_stats[key]
                _plot_baseline(stats, ax)
            
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Iteration')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    for ax, key, title, color in [
        (axes[1, 0], 'policy_loss', 'Policy Loss', '#F77F00'),
        (axes[1, 1], 'value_loss', 'Value Loss', '#9D4EDD')
    ]:
        loss_data = ppo_stats.get(key, [])
        _plot_loss(
            ax=ax, 
            data=loss_data, 
            key=key, 
            title=title, 
            color=color
        )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close() 

def plot_comparison(ppo_results: Dict, baseline_results: Dict, save_path: str):
    """
    Plot comparison between PPO and baseline with 95% confidence intervals
    Similar to the attached plot from the paper
    
    Args:
        ppo_results: PPO evaluation results
        baseline_results: Baseline evaluation results
        save_path: Path to save figure
    """

    from utils.visualise_helpers import compare_ppo_baseline

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    _, ax = plt.subplots(figsize=(10, 6))

    means, errors, x_pos, improvement = compare_ppo_baseline(
        ppo_results=ppo_results, baseline_results=baseline_results
    )

    ax.set_ylabel('Costs', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['PPO', 'Benchmark'], fontsize=12)
    ax.set_title(
        'Performance Comparison (95% Confidence Interval)', 
        fontsize=16, fontweight='bold'
    )
    ax.grid(True, axis='y', alpha=0.3)
    bars = ax.bar(
        x_pos, 
        means, 
        yerr=errors, 
        capsize=10, 
        color=['#2E86AB', '#E63946'], 
        alpha=0.7, 
        edgecolor='black', 
        linewidth=1.5
    )

    for _, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height, f'{mean:.0f}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
    
    ax.text(
        0.5, 0.95, f'Improvement: {improvement:.1f}%',
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', 
        alpha=0.5), transform=ax.transAxes
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()
