"""
Main training script for Divergent environment
Trains both PPO and Baseline agents and compares performance
"""
import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
import json
from ruamel.yaml import YAML

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.divergent_env import DivergentInventoryEnv
from agents.ppo_agent import PPOAgent
from agents.baseline_agent import BaselineAgent
from training.train_ppo import train_ppo
from training.train_baseline import evaluate_baseline
from evaluation.evaluate import evaluate_agent
from evaluation.visualize import plot_training_curves, plot_comparison, plot_heatmap
from utils.logger import Logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PPO on Divergent Inventory Environment')
    
    parser.add_argument(
        '--mode', 
        type=str,
        default='train',
        choices=['train', 'eval', 'both'],
        help='Mode: train, eval, or both'
    )
    parser.add_argument(
        '--eval-episodes', 
        type=int, 
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint for evaluation'
    )
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--log-interval',
        type=int, 
        default=100,
        help='Logging interval'
    )
    parser.add_argument(
        '--save-interval', 
        type=int, 
        default=1000,
        help='Model save interval'
    )
    parser.add_argument(
        '--no-baseline', 
        action='store_true',
        help='Skip baseline training/evaluation'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use'
    )
    
    return parser.parse_args()


def set_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(save_dir: str):
    """Create necessary directories"""
    dirs = [
        save_dir,
        os.path.join(save_dir, 'checkpoints'),
        os.path.join(save_dir, 'logs'),
        os.path.join(save_dir, 'plots'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train_agents(args, config, env_config):
    """Train PPO and Baseline agents"""
    print("="*80)
    print("Training Divergent Inventory Optimization")
    print("="*80)
    
    # Create environment
    env = DivergentInventoryEnv(env_config)
    print(f"\nEnvironment created:")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.shape[0]}")
    print(f"  Max steps per episode: {env.max_steps}")
    
    # Create logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.save_dir, 'logs', f'divergent_{timestamp}')
    logger = Logger(log_dir)
    
    # Train PPO agent
    print("\n" + "="*80)
    print("Training PPO Agent")
    print("="*80)
    
    ppo_agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config
    )
    
    ppo_stats = train_ppo(
        agent=ppo_agent,
        env=env,
        num_iterations=config.get('iterations', 10000),
        logger=logger,
        save_dir=os.path.join(args.save_dir, 'checkpoints'),
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=config.get('eval_interval', 500),
        eval_episodes=config.get('eval_episodes', 10),
    )
    
    # Save final model
    ppo_save_path = os.path.join(args.save_dir, 'checkpoints', 'ppo_divergent_final.pt')
    ppo_agent.save(ppo_save_path)
    print(f"\nPPO agent saved to {ppo_save_path}")
    
    # Train/Evaluate Baseline
    if not args.no_baseline:
        print("\n" + "="*80)
        print("Evaluating Baseline Agent")
        print("="*80)
        
        baseline_agent = BaselineAgent(env)
        
        baseline_stats = evaluate_baseline(
            agent=baseline_agent,
            env=env,
            num_episodes=args.eval_episodes,
            logger=logger,
        )
        
        # Save baseline
        baseline_save_path = os.path.join(args.save_dir, 'checkpoints', 'baseline_divergent.npy')
        baseline_agent.save(baseline_save_path)
        print(f"\nBaseline agent saved to {baseline_save_path}")
    else:
        baseline_stats = None
        baseline_agent = None
    
    # Plot training curves
    print("\nGenerating training plots...")
    plot_path = os.path.join(args.save_dir, 'plots', 'training_curves.png')
    plot_training_curves(ppo_stats, baseline_stats, save_path=plot_path, show=False)
    print(f"Training curves saved to {plot_path}")
    
    # Save statistics
    stats = {
        'ppo': ppo_stats,
        'baseline': baseline_stats,
        'config': config,
        'env_config': {k: v for k, v in env_config.items() if not callable(v)},
        'args': vars(args),
    }
    
    stats_path = os.path.join(args.save_dir, 'training_stats.json')
    with open(stats_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        json.dump(stats, f, indent=2, default=convert)
    print(f"Training statistics saved to {stats_path}")
    
    return ppo_agent, baseline_agent, ppo_stats, baseline_stats


def evaluate_agents(args, config, env_config, ppo_agent=None, baseline_agent=None):
    """Evaluate trained agents"""
    print("\n" + "="*80)
    print("Evaluating Agents")
    print("="*80)
    
    # Create environment
    env = DivergentInventoryEnv(env_config)
    
    # Load PPO agent if not provided
    if ppo_agent is None:
        if args.checkpoint is None:
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', 'ppo_divergent_final.pt')
        else:
            checkpoint_path = args.checkpoint
            
        ppo_agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=config
        )
        ppo_agent.load(checkpoint_path)
        print(f"PPO agent loaded from {checkpoint_path}")
    
    ppo_agent.eval()
    
    # Evaluate PPO
    print("\nEvaluating PPO agent...")
    ppo_results = evaluate_agent(
        agent=ppo_agent,
        env=env,
        num_episodes=args.eval_episodes,
        deterministic=True
    )
    
    print(f"\nPPO Results:")
    print(f"  Mean Cost: {ppo_results['mean_cost']:.2f} ± {ppo_results['std_cost']:.2f}")
    print(f"  Mean Reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"  95% CI: [{ppo_results['ci_lower']:.2f}, {ppo_results['ci_upper']:.2f}]")
    
    # Evaluate Baseline
    if not args.no_baseline:
        if baseline_agent is None:
            baseline_agent = BaselineAgent(env)
            baseline_path = os.path.join(args.save_dir, 'checkpoints', 'baseline_divergent.npy')
            if os.path.exists(baseline_path):
                baseline_agent.load(baseline_path)
                print(f"Baseline agent loaded from {baseline_path}")
        
        print("\nEvaluating Baseline agent...")
        baseline_results = evaluate_agent(
            agent=baseline_agent,
            env=env,
            num_episodes=args.eval_episodes,
            deterministic=True
        )
        
        print(f"\nBaseline Results:")
        print(f"  Mean Cost: {baseline_results['mean_cost']:.2f} ± {baseline_results['std_cost']:.2f}")
        print(f"  Mean Reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
        print(f"  95% CI: [{baseline_results['ci_lower']:.2f}, {baseline_results['ci_upper']:.2f}]")
        
        # Compare agents
        print("\n" + "="*80)
        print("Comparison")
        print("="*80)
        
        improvement = (baseline_results['mean_cost'] - ppo_results['mean_cost']) / baseline_results['mean_cost'] * 100
        print(f"\nPPO Improvement over Baseline: {improvement:.2f}%")
        
        # Plot comparison
        print("\nGenerating comparison plots...")
        comparison_path = os.path.join(args.save_dir, 'plots', 'comparison.png')
        plot_comparison(ppo_results, baseline_results, save_path=comparison_path, show=False)
        print(f"Comparison plot saved to {comparison_path}")
    else:
        baseline_results = None
    
    # Generate heatmaps
    print("\nGenerating policy heatmaps...")
    heatmap_path = os.path.join(args.save_dir, 'plots', 'heatmap_ppo.png')
    plot_heatmap(ppo_agent, env, save_path=heatmap_path, show=False)
    print(f"Heatmap saved to {heatmap_path}")
    
    # Save evaluation results
    improvement = None  # default when baseline is skipped
    if baseline_results is not None:
        improvement = (baseline_results['mean_cost'] - ppo_results['mean_cost']) / baseline_results['mean_cost'] * 100

    eval_results = {
        'ppo': ppo_results,
        'baseline': baseline_results,
        'improvement': improvement,
    }

    eval_path = os.path.join(args.save_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        json.dump(eval_results, f, indent=2, default=convert)
    print(f"\nEvaluation results saved to {eval_path}")
    
    return ppo_results, baseline_results


def _get_config():
    with open('config.yaml', 'r') as f:
        config = YAML().load(f)
        return config

def main():
    """Main function"""
    args = parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create directories
    create_directories(args.save_dir)
    
    # Load config once
    full_config = _get_config()
    config = dict(full_config['ppo'])
    config['seed'] = args.seed
    env_config = dict(full_config['divergent'])
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nUsing device: {device}")
    
    # Run based on mode
    if args.mode == 'train':
        ppo_agent, baseline_agent, ppo_stats, baseline_stats = train_agents(
            args, config, env_config
        )
        
    elif args.mode == 'eval':
        ppo_results, baseline_results = evaluate_agents(
            args, config, env_config
        )
        
    elif args.mode == 'both':
        # Train
        ppo_agent, baseline_agent, ppo_stats, baseline_stats = train_agents(
            args, config, env_config
        )
        
        # Evaluate
        ppo_results, baseline_results = evaluate_agents(
            args, config, env_config, ppo_agent, baseline_agent
        )
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == '__main__':
    main()