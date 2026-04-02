"""
Visualization utilities for training and evaluation results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    ppo_stats: Dict,
    baseline_stats: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves comparing PPO and baseline
    
    Args:
        ppo_stats: PPO training statistics
        baseline_stats: Baseline training statistics
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Extract PPO data
    iterations = ppo_stats.get('iterations', [])
    costs = ppo_stats.get('episode_costs', [])
    rewards = ppo_stats.get('episode_rewards', [])
    
    # Smooth curves using moving average
    window = min(100, len(costs) // 10) if len(costs) > 0 else 1
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Plot 1: Cost over iterations
    ax = axes[0, 0]
    if len(costs) > 0:
        smoothed_costs = moving_average(costs, window)
        smoothed_iterations = iterations[:len(smoothed_costs)] if len(iterations) >= len(smoothed_costs) else range(len(smoothed_costs))
        
        ax.plot(smoothed_iterations, smoothed_costs, 
                label='PPO', linewidth=2, color='#2E86AB')
        
        # Add confidence band
        if len(costs) > window:
            costs_array = np.array(costs)
            # Calculate rolling std
            rolling_std = []
            for i in range(len(smoothed_costs)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(costs), i + window // 2)
                rolling_std.append(np.std(costs_array[start_idx:end_idx]))
            
            rolling_std = np.array(rolling_std)
            ax.fill_between(smoothed_iterations, 
                            smoothed_costs - rolling_std,
                            smoothed_costs + rolling_std,
                            alpha=0.3, color='#2E86AB')
    
    # Plot baseline as horizontal line with confidence band
    if baseline_stats and 'episode_costs' in baseline_stats:
        baseline_costs = np.array(baseline_stats['episode_costs'])
        if len(baseline_costs) > 0:
            baseline_mean = np.mean(baseline_costs)
            baseline_std = np.std(baseline_costs)
            
            # Plot mean as dashed line
            ax.axhline(y=baseline_mean, color='#E63946', linestyle='--', 
                      linewidth=2, label='Benchmark', zorder=5)
            
            # Add confidence band
            ax.axhspan(baseline_mean - baseline_std, 
                      baseline_mean + baseline_std,
                      alpha=0.2, color='#E63946', zorder=1)
    
    ax.set_xlabel('Iteration (×100)', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Cost vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    if len(costs) > 0:
        y_min = min(costs) * 0.8
        y_max = max(costs[:min(len(costs), 1000)]) * 1.2  # Use early training max
        ax.set_ylim([y_min, y_max])
    
    # Plot 2: Reward over iterations
    ax = axes[0, 1]
    if len(rewards) > 0:
        smoothed_rewards = moving_average(rewards, window)
        smoothed_iterations = iterations[:len(smoothed_rewards)] if len(iterations) >= len(smoothed_rewards) else range(len(smoothed_rewards))
        
        ax.plot(smoothed_iterations, smoothed_rewards,
                linewidth=2, color='#06A77D', label='PPO')
        
        # Add confidence band for rewards
        if len(rewards) > window:
            rewards_array = np.array(rewards)
            rolling_std = []
            for i in range(len(smoothed_rewards)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(rewards), i + window // 2)
                rolling_std.append(np.std(rewards_array[start_idx:end_idx]))
            
            rolling_std = np.array(rolling_std)
            ax.fill_between(smoothed_iterations, 
                            smoothed_rewards - rolling_std,
                            smoothed_rewards + rolling_std,
                            alpha=0.3, color='#06A77D')
    
    # Plot baseline reward as horizontal line
    if baseline_stats and 'episode_rewards' in baseline_stats:
        baseline_rewards = np.array(baseline_stats['episode_rewards'])
        if len(baseline_rewards) > 0:
            baseline_mean = np.mean(baseline_rewards)
            baseline_std = np.std(baseline_rewards)
            
            ax.axhline(y=baseline_mean, color='#E63946', linestyle='--', 
                      linewidth=2, label='Benchmark', zorder=5)
            ax.axhspan(baseline_mean - baseline_std, 
                      baseline_mean + baseline_std,
                      alpha=0.2, color='#E63946', zorder=1)
    
    ax.set_xlabel('Iteration (×100)', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward vs Iteration', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Policy loss (if available)
    ax = axes[1, 0]
    if 'policy_loss' in ppo_stats:
        policy_loss = ppo_stats['policy_loss']
        if len(policy_loss) > 0:
            ax.plot(policy_loss, linewidth=2, color='#F77F00')
            ax.set_xlabel('Update', fontsize=12)
            ax.set_ylabel('Policy Loss', fontsize=12)
            ax.set_title('Policy Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    else:
        # If no policy loss data, hide this subplot
        ax.text(0.5, 0.5, 'No Policy Loss Data', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot 4: Value loss (if available)
    ax = axes[1, 1]
    if 'value_loss' in ppo_stats:
        value_loss = ppo_stats['value_loss']
        if len(value_loss) > 0:
            # Clip extreme values for better visualization
            value_loss_clipped = np.clip(value_loss, 
                                        np.percentile(value_loss, 1),
                                        np.percentile(value_loss, 99))
            ax.plot(value_loss_clipped, linewidth=2, color='#9D4EDD')
            ax.set_xlabel('Update', fontsize=12)
            ax.set_ylabel('Value Loss', fontsize=12)
            ax.set_title('Value Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    else:
        # If no value loss data, hide this subplot
        ax.text(0.5, 0.5, 'No Value Loss Data', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    ppo_results: Dict,
    baseline_results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison between PPO and baseline with 95% confidence intervals
    Similar to the attached plot from the paper
    
    Args:
        ppo_results: PPO evaluation results
        baseline_results: Baseline evaluation results
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    agents = ['PPO', 'Benchmark']
    means = [ppo_results['mean_cost'], baseline_results['mean_cost']]
    
    # Confidence intervals (errors from mean)
    ci_lower_ppo = ppo_results['mean_cost'] - ppo_results['ci_lower']
    ci_upper_ppo = ppo_results['ci_upper'] - ppo_results['mean_cost']
    
    ci_lower_baseline = baseline_results['mean_cost'] - baseline_results['ci_lower']
    ci_upper_baseline = baseline_results['ci_upper'] - baseline_results['mean_cost']
    
    errors = [
        [ci_lower_ppo, ci_lower_baseline],
        [ci_upper_ppo, ci_upper_baseline]
    ]
    
    # Colors
    colors = ['#2E86AB', '#E63946']
    
    # Bar plot
    x_pos = np.arange(len(agents))
    bars = ax.bar(x_pos, means, yerr=errors, capsize=10, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_ylabel('Costs', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, fontsize=12)
    ax.set_title('Performance Comparison (95% Confidence Interval)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement percentage
    improvement = (baseline_results['mean_cost'] - ppo_results['mean_cost']) / baseline_results['mean_cost'] * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_heatmap(
    agent,
    env,
    save_path: Optional[str] = None,
    show: bool = True,
    num_points: int = 50
):
    """
    Plot policy heatmap showing action distribution
    Similar to Figure 4 from the paper
    
    Args:
        agent: Trained agent
        env: Environment
        save_path: Path to save figure
        show: Whether to show plot
        num_points: Grid resolution
    """
    from evaluation.evaluate import evaluate_policy_heatmap
    
    # Determine number of retailers
    num_actions = env.action_space.shape[0]
    num_retailers = num_actions - 1  # Subtract warehouse action
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1 + num_retailers, figsize=(5 * (num_retailers + 1), 4))
    
    if num_retailers == 0:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
    
    fig.suptitle('Actions of the Warehouse and Retailers', fontsize=16, fontweight='bold')
    
    # (a) Warehouse heatmap - vary warehouse inventory and first in-transit
    X, Y, actions = evaluate_policy_heatmap(
        agent, env, state_indices=(0, env.observation_space.shape[0]//2), num_points=num_points
    )
    
    warehouse_actions = actions[:, :, 0]  # First action is warehouse order
    
    # Denormalize actions for better interpretation
    max_order = env.max_orders
    warehouse_actions_denorm = (warehouse_actions + 1) * max_order / 2
    
    im = axes[0].contourf(X * env.max_inventory_warehouse, 
                          Y * env.max_in_transit, 
                          warehouse_actions_denorm, 
                          levels=20, cmap='YlOrRd')
    axes[0].set_xlabel('Inventory Warehouse', fontsize=12)
    axes[0].set_ylabel('In Transit', fontsize=12)
    axes[0].set_title('(a) Warehouse', fontsize=14)
    plt.colorbar(im, ax=axes[0], label='Order Quantity')
    
    # (b) Retailer heatmaps
    for i in range(num_retailers):
        # Vary retailer inventory and retailer in-transit
        retailer_inv_idx = 1 + i
        retailer_transit_idx = 1 + num_retailers + i
        
        X_r, Y_r, actions_r = evaluate_policy_heatmap(
            agent, env, 
            state_indices=(retailer_inv_idx, min(retailer_transit_idx, env.observation_space.shape[0]-1)), 
            num_points=num_points
        )
        
        retailer_actions = actions_r[:, :, i+1]
        retailer_actions_denorm = (retailer_actions + 1) * max_order / 2
        
        im = axes[i+1].contourf(X_r * env.max_inventory_buffer, 
                                Y_r * env.max_in_transit, 
                                retailer_actions_denorm, 
                                levels=20, cmap='YlOrRd')
        axes[i+1].set_xlabel(f'Inventory retailer {i+1}', fontsize=12)
        axes[i+1].set_ylabel('In Transit', fontsize=12)
        axes[i+1].set_title(f'(b) Retailer {i+1}', fontsize=14)
        plt.colorbar(im, ax=axes[i+1], label='Order Quantity')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_episode_metrics(
    episode_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot detailed metrics from a single episode
    
    Args:
        episode_data: Dictionary containing episode data
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Episode Metrics', fontsize=16, fontweight='bold')
    
    steps = range(len(episode_data.get('rewards', [])))
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(steps, episode_data['rewards'], linewidth=2, color='#06A77D')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward per Step', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Costs
    ax = axes[0, 1]
    ax.plot(steps, episode_data['costs'], linewidth=2, color='#E63946')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Cost per Step', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Inventory levels
    ax = axes[1, 0]
    if 'inventory_warehouse' in episode_data:
        ax.plot(steps, episode_data['inventory_warehouse'], 
                label='Warehouse', linewidth=2)
    if 'inventory_retailers' in episode_data:
        for i, inv in enumerate(episode_data['inventory_retailers'].T):
            ax.plot(steps, inv, label=f'Retailer {i+1}', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Inventory', fontsize=12)
    ax.set_title('Inventory Levels', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Actions
    ax = axes[1, 1]
    if 'actions' in episode_data:
        actions_array = np.array(episode_data['actions'])
        for i in range(actions_array.shape[1]):
            ax.plot(steps, actions_array[:, i], 
                   label=f'Action {i+1}', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Order Quantity', fontsize=12)
    ax.set_title('Actions (Orders)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_analysis(
    training_stats: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot convergence analysis showing learning progress
    
    Args:
        training_stats: Training statistics
        save_path: Path to save figure
        show: Whether to show plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    
    # Plot each metric
    metrics = [
        ('policy_loss', 'Policy Loss', '#F77F00'),
        ('value_loss', 'Value Loss', '#9D4EDD'),
        ('entropy', 'Entropy', '#06A77D'),
        ('kl_divergence', 'KL Divergence', '#E63946'),
        ('clip_fraction', 'Clip Fraction', '#2E86AB'),
    ]
    
    for idx, (metric, title, color) in enumerate(metrics):
        if metric in training_stats and len(training_stats[metric]) > 0:
            ax = axes[idx // 3, idx % 3]
            data = training_stats[metric]
            
            # Clip extreme values for value loss
            if metric == 'value_loss':
                data = np.clip(data, np.percentile(data, 1), np.percentile(data, 99))
            
            ax.plot(data, linewidth=2, color=color)
            ax.set_xlabel('Update', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Remove unused subplot
    if len(metrics) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()