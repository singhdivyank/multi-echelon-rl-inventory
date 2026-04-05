import argparse
import json
import os
from typing import Dict

import numpy as np
import yaml
import torch

from meis_env import MEISEnv
from a3c_agent import A3CAgent
from trainer import Trainer
from s_s_policy import sSPolicy, sSPolicyTuner
from utils.evaluation import compare_policies, evaluate_agent
from utils.visualisation import (
    plot_training_curves, 
    plot_comparison, 
    plot_cost_per_period, 
    plot_training_curve
)

def _load_config() -> Dict:
    """Load configurations from YAML file"""

    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def _save_config(config_path: str, content: Dict):
    """Save training config as JSON"""

    with open(config_path, 'w') as f:
        json.dump(content, f, indent=2)
    
    print(f"Configurations saved to: {config_path}")

def _setup_directory(base_path: str):
    """Create necessary directories"""
    dirs = {
        'base': base_path,
        'checkpoints': os.path.join(base_path, 'checkpoints'),
        'plots': os.path.join(base_path, 'plots'),
        'log': os.path.join(base_path, 'log')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def main(args):
    """Main training and evaluation pipeline"""

    config = _load_config()
    seed_val = config['general']['seed']
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    path_to_dir = config['general']['save_dir']
    dirs = _setup_directory(path_to_dir)
    _save_config(
        config_path=os.path.join(dirs['log'], 'config.json'),
        content=config
    )

    print("\n" + "="*60)
    print("MULTI-ECHELON INVENTORY SYSTEM - A3C TRAINING")
    print("="*60)
    print(f"Random Seed: {seed_val}")
    print(f"Device: {config['agent']['device']}")
    print(f"Save Directory: {path_to_dir}")

    # Create environment
    print("\n--- Creating Environment ---")
    env = MEISEnv()
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.n}")

    # Create A3C agent
    print("\n--- Creating A3C Agent ---")
    agent_config = config['agent']
    agent = A3CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        agent_config=agent_config
    )
    print(f"Network architecture: {agent_config['n_layers']} layers, {agent_config['hidden_size']} neurons")
    print(f"Learning rate: {agent_config['lr']}")

    # train model
    if not args.eval_only:
        print("\n--- Training A3C Agent ---")
        training_history = Trainer(
            env,
            agent,
            config['training'],
            save_dir=dirs['checkpoints']
        ).train()
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
        print("Tuning (s,S) policy parameters...")
        baseline_config = config['baseline']
        tuner = sSPolicyTuner(
            env,
            n_eval_episodes=baseline_config['tuning_episodes']
        )
        best_params, _ = tuner.tune(
            maxiter=baseline_config['tuning_maxiter'],
            seed=seed_val
        )
        baseline_policy = sSPolicy(best_params)
        
        # Save tuned parameters
        params_path = os.path.join(dirs['log'], 'baseline_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"Baseline parameters saved to: {params_path}")

        # Evaluate both policies
        print("\n--- Evaluating Policies ---")
        print("Evaluating A3C agent...")
        
        agent.load("./results/checkpoints/best_model.pt")
        rl_metrics = evaluate_agent(
            env,
            agent=agent,
            n_episodes=config['evaluation']['n_episodes'],
            seed=seed_val,
            verbose=True
        )
        
        print(f"\nA3C Agent:")
        print(f"  Mean Cost: {rl_metrics['mean_cost']:.2f} ± {rl_metrics['std_cost']:.2f}")
        print(f"  Mean Service Level: {rl_metrics['mean_service_level']:.2%} ± {rl_metrics['std_service_level']:.2%}")
        print(f"  Cost Breakdown:")
        print(f"    Shortage: {rl_metrics['cost_breakdown']['shortage']:.2f}")
        print(f"    Holding: {rl_metrics['cost_breakdown']['holding']:.2f}")
        print(f"    Reordering: {rl_metrics['cost_breakdown']['reordering']:.2f}")

        print("\nEvaluating baseline policy...")
        baseline_metrics = evaluate_agent(
            env,
            baseline_policy=baseline_policy,
            n_episodes=config['evaluation']['n_episodes'],
            seed=seed_val,
            verbose=True
        )

        print(f"\n(s,S) Baseline:")
        print(f"  Mean Cost: {baseline_metrics['mean_cost']:.2f} ± {baseline_metrics['std_cost']:.2f}")
        print(f"  Mean Service Level: {baseline_metrics['mean_service_level']:.2%} ± {baseline_metrics['std_service_level']:.2%}")
        print(f"  Cost Breakdown:")
        print(f"    Shortage: {baseline_metrics['cost_breakdown']['shortage']:.2f}")
        print(f"    Holding: {baseline_metrics['cost_breakdown']['holding']:.2f}")
        print(f"    Reordering: {baseline_metrics['cost_breakdown']['reordering']:.2f}")

        # Compare results
        print("\n--- Comparison Results ---")
        comparison = compare_policies(rl_metrics, baseline_metrics)
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
        eval_path = os.path.join(dirs['log'], 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation results saved to: {eval_path}")

        # Create visualizations
        if training_history is not None:
            print("\n--- Creating Visualizations ---")
            os.makedirs(dirs['plots'], exist_ok=True)
            print("\nGenerating visualisations...")

            plot_training_curves(
                training_history,
                save_path=os.path.join(dirs['plots'], 'training_curves.png')
            )
            plot_comparison(
                rl_metrics,
                baseline_metrics,
                save_path=os.path.join(dirs['plots'], 'comparison.png')
            )
            plot_cost_per_period(
                env,
                agent,
                baseline_policy,
                save_path=os.path.join(dirs['plots'], 'cost_per_period.png')
            )
            plot_training_curve(
                training_history,
                save_path=os.path.join(dirs['plots'], 'training_curve.png')
            )

        print("\n" + "="*60)
        print("TRAINING AND EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {dirs['plots']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A3C agent on MEIS environment")
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evauate pre-trained model'
    )
    args = parser.parse_args()
    main(args)
