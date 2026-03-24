"""
Main Training Script

Train A3C agent on MEIS environment and compare with baseline

Usage:
    python main.py --config configs/default.yaml --seed 42

Author: AI/ML Engineering Team
"""

import argparse
import yaml
import os
import sys
import json
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.meis_env import MEISEnv
from agents.a3c_agent import A3CAgent
from agents.trainer import train_agent
from baselines.s_s_policy import sSPolicy, sSPolicyTuner, evaluate_policy
from utils.evaluation import evaluate_agent, evaluate_baseline, compare_policies
from utils.visualization import create_all_plots


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or return default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        'environment': {
            'max_steps_per_episode': 365,
        },
        'agent': {
            'hidden_size': 64,
            'n_layers': 3,
            'lr': 1e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
            'max_grad_norm': 0.5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        'training': {
            'n_episodes': 500,
            'max_steps_per_episode': 365,
            'update_frequency': 1,
            'eval_frequency': 50,
            'n_eval_episodes': 10,
            'save_frequency': 100,
        },
        'baseline': {
            'tune': True,
            'tuning_episodes': 10,
            'tuning_maxiter': 50,
        },
        'evaluation': {
            'n_episodes': 100,
        },
    }


def setup_directories(base_dir: str = './results'):
    """Create necessary directories"""
    dirs = {
        'base': base_dir,
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'plots': os.path.join(base_dir, 'plots'),
        'logs': os.path.join(base_dir, 'logs'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def main(args):
    """Main training and evaluation pipeline"""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Setup directories
    dirs = setup_directories(args.save_dir)
    
    # Save config
    config_path = os.path.join(dirs['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("MULTI-ECHELON INVENTORY SYSTEM - A3C TRAINING")
    print("="*60)
    print(f"Random Seed: {args.seed}")
    print(f"Device: {config['agent']['device']}")
    print(f"Save Directory: {args.save_dir}")
    
    # Create environment
    print("\n--- Creating Environment ---")
    env = MEISEnv(config=config['environment'])
    env.seed(args.seed)
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.n}")
    
    # Create A3C agent
    print("\n--- Creating A3C Agent ---")
    agent = A3CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        **config['agent']
    )
    print(f"Network architecture: {config['agent']['n_layers']} layers, {config['agent']['hidden_size']} neurons")
    print(f"Learning rate: {config['agent']['lr']}")
    
    # Train agent
    if not args.eval_only:
        print("\n--- Training A3C Agent ---")
        training_history = train_agent(
            env,
            agent,
            config['training'],
            save_dir=dirs['checkpoints']
        )
    else:
        print("\n--- Loading Pre-trained Agent ---")
        checkpoint_path = os.path.join(dirs['checkpoints'], 'checkpoint_final.pt')
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        
        # Load training history if available
        history_path = os.path.join(dirs['checkpoints'], 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        else:
            training_history = None
    
    # Create and tune baseline policy
    print("\n--- Creating Baseline Policy ---")
    if config['baseline']['tune']:
        print("Tuning (s,S) policy parameters...")
        tuner = sSPolicyTuner(
            env,
            n_eval_episodes=config['baseline']['tuning_episodes']
        )
        best_params, _ = tuner.tune(
            maxiter=config['baseline']['tuning_maxiter'],
            seed=args.seed
        )
        baseline_policy = sSPolicy(best_params)
        
        # Save tuned parameters
        params_path = os.path.join(dirs['logs'], 'baseline_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Baseline parameters saved to: {params_path}")
    else:
        # Use default parameters
        baseline_policy = sSPolicy()
        print("Using default (s,S) parameters")
    
    # Evaluate both policies
    print("\n--- Evaluating Policies ---")
    
    print("Evaluating A3C agent...")
    rl_metrics = evaluate_agent(
        env,
        agent,
        n_episodes=config['evaluation']['n_episodes'],
        seed=args.seed,
        verbose=True
    )
    
    print("\nEvaluating baseline policy...")
    baseline_metrics = evaluate_baseline(
        env,
        baseline_policy,
        n_episodes=config['evaluation']['n_episodes'],
        seed=args.seed,
        verbose=True
    )
    
    # Compare policies
    print("\n--- Comparison Results ---")
    comparison = compare_policies(rl_metrics, baseline_metrics)
    
    print(f"\nA3C Agent:")
    print(f"  Mean Cost: {rl_metrics['mean_cost']:.2f} ± {rl_metrics['std_cost']:.2f}")
    print(f"  Mean Service Level: {rl_metrics['mean_service_level']:.2%} ± {rl_metrics['std_service_level']:.2%}")
    print(f"  Cost Breakdown:")
    print(f"    Shortage: {rl_metrics['cost_breakdown']['shortage']:.2f}")
    print(f"    Holding: {rl_metrics['cost_breakdown']['holding']:.2f}")
    print(f"    Reordering: {rl_metrics['cost_breakdown']['reordering']:.2f}")
    
    print(f"\n(s,S) Baseline:")
    print(f"  Mean Cost: {baseline_metrics['mean_cost']:.2f} ± {baseline_metrics['std_cost']:.2f}")
    print(f"  Mean Service Level: {baseline_metrics['mean_service_level']:.2%} ± {baseline_metrics['std_service_level']:.2%}")
    print(f"  Cost Breakdown:")
    print(f"    Shortage: {baseline_metrics['cost_breakdown']['shortage']:.2f}")
    print(f"    Holding: {baseline_metrics['cost_breakdown']['holding']:.2f}")
    print(f"    Reordering: {baseline_metrics['cost_breakdown']['reordering']:.2f}")
    
    print(f"\nImprovement:")
    print(f"  Cost: {comparison['cost_improvement_percent']:.2f}%")
    print(f"  Service Level: {comparison['service_level_improvement_percent']:.2f}%")
    print(f"  Statistical Significance (cost): {comparison['cost_ttest']['significant']} (p={comparison['cost_ttest']['p_value']:.4f})")
    print(f"  Effect Size (Cohen's d): {comparison['cohens_d_cost']:.3f}")
    
    # Save evaluation results
    eval_results = {
        'rl': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in rl_metrics.items()},
        'baseline': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in baseline_metrics.items()},
        'comparison': comparison,
    }
    
    eval_path = os.path.join(dirs['logs'], 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_path}")
    
    # Create visualizations
    if training_history is not None:
        print("\n--- Creating Visualizations ---")
        create_all_plots(
            env,
            agent,
            baseline_policy,
            training_history,
            rl_metrics,
            baseline_metrics,
            save_dir=dirs['plots'],
            show=args.show_plots
        )
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train A3C agent on MEIS environment')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate pre-trained model'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    main(args)