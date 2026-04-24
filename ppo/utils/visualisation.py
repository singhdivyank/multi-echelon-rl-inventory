"""
Utility functions for visualisations
"""

from typing import Dict, List, Tuple

import numpy as np

def compare_ppo_baseline(
    ppo_results: Dict, 
    baseline_results: Dict
) -> Tuple[List, List, np.array, float]:
    ppo_mean, baseline_mean = ppo_results['mean_cost'], baseline_results['mean_cost']
    means = [ppo_mean, baseline_mean]
    errors = [
        [ppo_mean - ppo_results['ci_lower'], baseline_mean - baseline_results['ci_lower']],
        [ppo_results['ci_upper'] - ppo_mean, baseline_results['ci_upper'] - baseline_mean]
    ]
    x_pos = np.arange(len(['PPO', 'Benchmark']))
    improvement = (baseline_mean - ppo_mean) / baseline_mean * 100
    return means, errors, x_pos, improvement

def _plot_with_confidence(
    ax, 
    x: List, 
    y: List, 
    label: str, 
    color: str, 
    window: int
):
    """Helper to plot smoothed line and confidence band."""

    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')
    
    def get_rolling_std(data, window):
        """Vectorized rolling standard deviation"""

        if len(data) < window:
            return np.zeros_like(data)
        
        data_sq = np.array(data) ** 2
        window_ones = np.ones(window) / window
        m_sq = np.convolve(data_sq, window_ones, mode='valid')
        m = np.convolve(data, window_ones, mode='valid')
        return np.sqrt(np.maximum(0, m_sq - m ** 2))
    
    if not y:
        return

    smoothed = moving_average(y, window)
    std = get_rolling_std(y, window)
    plot_x = x[:len(smoothed)] if len(x) >= len(smoothed) else np.arange(len(smoothed))
    ax.plot(plot_x, smoothed, label=label, color=color, linewidth=2)
    ax.fill_between(plot_x, smoothed - std, smoothed + std, alpha=0.2, color=color)

def _plot_baseline(stats, ax):
    b_data = np.array(stats)
    if b_data.size:
        b_mean, b_std = np.mean(b_data), np.std(b_data)
        ax.axhline(b_mean, color='#E63946', ls='--', lw=2, label='Benchmark', zorder=3)
        ax.axhspan(b_mean - b_std, b_mean + b_std, alpha=0.15, color='#E63946')

def _plot_loss(
    ax, 
    data: List, 
    key: str, 
    title: str, 
    color: str
):
    if not data:
        ax.text(0.5, 0.5, f'No {title} Data', ha='center', va='center')
    else:
        if key == 'value_loss':
            data = np.clip(data, *np.percentile(data, [1, 99]))
        ax.plot(data, color=color, lw=1.5)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
