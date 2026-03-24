"""
Multi-Seed Evaluation Script

Evaluate trained agents across multiple random seeds to analyze variance

Usage:
    python run_multi_seed.py --seeds 42 123 456 789 1000 --n-eval 100

Author: AI/ML Engineering Team
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.meis_env import MEISEnv
from agents.a3c_agent import A3CAgent
from baselines.s_s_policy import sSPolicy, sSPolicyTuner
from utils.evaluation import evaluate_across_seeds, print_summary_statistics
from utils.visualization import plot_variance_across_seeds, set_plot_style
import matplotlib.pyplot as plt


def main(args):
    """Run multi-seed evaluation"""
    
    print("\n" + "="*60)
    print("MULTI-SEED EVALUATION")
    print("="*60)
    print(f"Seeds: {args.seeds}")
    print(f"Evaluation episodes per seed: {args.n_eval}")
    
    # Setup save directory
    save_dir = os.path.join(args.save_dir, 'multi_seed_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load baseline parameters if available
    baseline_params_path = os.path.join(args.checkpoint_dir, 'logs', 'baseline_params.json')
    if os.path.exists(baseline_params_path):
        import json
        with open(baseline_params_path, 'r') as f:
            baseline_params = json.load(f)
        print(f"\nUsing tuned baseline parameters from: {baseline_params_path}")
    else:
        baseline_params = None
        print("\nUsing default baseline parameters")
    
    # Factory functions
    def create_env():
        return MEISEnv()
    
    def create_agent():
        env = create_env()
        agent = A3CAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Load checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoints', 'checkpoint_final.pt')
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
            print(f"Loaded agent checkpoint from: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        return agent
    
    def create_baseline():
        if baseline_params:
            return sSPolicy(baseline_params)
        else:
            return sSPolicy()
    
    # Run evaluation across seeds
    results = evaluate_across_seeds(
        env_fn=create_env,
        agent_fn=create_agent,
        baseline_policy_fn=create_baseline,
        seeds=args.seeds,
        n_eval_episodes=args.n_eval,
        save_dir=save_dir
    )
    
    # Print summary statistics
    print_summary_statistics(results)
    
    # Create variance plots
    print("\n--- Creating Variance Plots ---")
    set_plot_style()
    
    # Cost variance
    plot_variance_across_seeds(
        results,
        metric='mean_cost',
        save_path=os.path.join(save_dir, 'cost_variance.png'),
        show=args.show_plots
    )
    
    # Service level variance
    plot_variance_across_seeds(
        results,
        metric='mean_service_level',
        save_path=os.path.join(save_dir, 'service_level_variance.png'),
        show=args.show_plots
    )
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-seed evaluation')
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456, 789, 1000],
        help='Random seeds to evaluate'
    )
    
    parser.add_argument(
        '--n-eval',
        type=int,
        default=100,
        help='Number of evaluation episodes per seed'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./results',
        help='Directory containing trained checkpoints'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    main(args)