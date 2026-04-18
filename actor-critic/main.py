"""
Main execution file
"""

import argparse
import sys
import os

import numpy as np
import torch

from src.meis_env import MEISEnv
from src.complex_meis_env import ComplexMEISEnv
from src.a3c_agent import A3CAgent
from src.trainer import Trainer
from src.s_s_policy import sSPolicy, sSPolicyTuner
from utils.evaluation import compare_policies, evaluate_agent
from utils.helpers import (
    print_eval_res,
    _get_complex_meis_env,
    _load_config,
    _read_metrics,
    _setup_directory,
    _save_config,
)
from utils.visualisation import (
    plot_training_curves, 
    plot_comparison, 
    plot_cost_per_period, 
    plot_training_curve
)


def main(args):
    """Main training and evaluation pipeline"""

    config = _load_config()
    seed_val = config['general']['seed']
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # Scope save_dir per env so Env-1 artifacts are never clobbered.
    base_save_dir = config['general']['save_dir']
    env_name = args.env
    if env_name == 'meis':
        path_to_dir = base_save_dir
    else:
        path_to_dir = f"{base_save_dir.rstrip('/\\')}_{env_name}"
    dirs = _setup_directory(path_to_dir)
    _save_config(
        config_path=os.path.join(dirs['logs'], 'config.json'),
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
    if env_name == 'complex':
        env = ComplexMEISEnv(config=_get_complex_meis_env())
    else:
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

    # Create and tune baseline policy
    print("\n--- Creating Baseline Policy ---")
    print("Tuning (s,S) policy parameters...")
    tuner = sSPolicyTuner(
        env,
        n_eval_episodes=config['baseline']['tuning_episodes']
    )
    best_params, _ = tuner.tune(
        maxiter=config['baseline']['tuning_maxiter'],
        seed=seed_val
    )
    baseline_policy = sSPolicy(best_params)

    # train model
    if args.mode == 'train':
        print("\n--- Training A3C Agent ---")
        training_history = Trainer(
            env,
            agent,
            config['training'],
            save_dir=dirs['checkpoints']
        ).train()
    # evaluate performance
    elif args.mode == 'eval':
        checkpoint_path = os.path.join(dirs['checkpoints'], 'checkpoint_final.pt')
        params_path = os.path.join(dirs['logs'], 'baseline_params.json')
        rl_metrics_path = os.path.join(dirs['eval'], 'rl_metrics.json')
        baseline_metrics_path = os.path.join(dirs['eval'], 'baseline_metrics.json')
        eval_res_path = os.path.join(dirs['logs'], 'evaluation_results.json')
        
        print("\n--- Loading Pre-trained Agent ---")
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        
        # Save tuned parameters
        _save_config(config_path=params_path, content=best_params)
        print(f"Baseline parameters saved to: {params_path}")

        # Evaluate both policies
        print("\n--- Evaluating Policies ---")
        print("Evaluating A3C agent...")
        best_model_path = os.path.join(dirs['checkpoints'], 'best_model.pt')
        agent.load(best_model_path)
        rl_metrics = evaluate_agent(
            env,
            agent=agent,
            n_episodes=config['evaluation']['n_episodes'],
            seed=seed_val,
            verbose=True
        )
        _save_config(config_path=rl_metrics_path, content=rl_metrics)
        print_eval_res(eval_res=rl_metrics, agent='a3c')

        print("\nEvaluating baseline policy...")
        baseline_metrics = evaluate_agent(
            env,
            baseline_policy=baseline_policy,
            n_episodes=config['evaluation']['n_episodes'],
            seed=seed_val,
            verbose=True
        )
        _save_config(config_path=baseline_metrics_path, content=baseline_metrics)
        print_eval_res(eval_res=baseline_metrics, agent='baseline')

        # Compare results
        print("\n--- Comparison Results ---")
        comparison = compare_policies(rl_metrics, baseline_metrics)
        print_eval_res(eval_res=comparison)

        # Save evaluation results
        eval_results = {
            'rl': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in rl_metrics.items()
            },
            'baseline': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in baseline_metrics.items()
            },
            'comparison': comparison,
        }
        _save_config(config_path=eval_res_path, content=eval_results)
        print(f"\nEvaluation results saved to: {eval_res_path}")
    # visualise results
    else:
        history_path = os.path.join(dirs['checkpoints'], 'training_history.json')
        training_history = _read_metrics(metrics_path=history_path)

        if training_history is None:
            print("No training history found ... Please train the model.")
            sys.exit(1)
        
        print("\nGenerating visualisations...")
        plot_training_curves(
            training_history,
            save_path=os.path.join(dirs['plots'], 'training_curves.png')
        )
        plot_training_curve(
            training_history,
            save_path=os.path.join(dirs['plots'], 'training_curve.png')
        )

        rl_metrics =_read_metrics(metrics_path=rl_metrics_path)
        if rl_metrics is None:
            print("No rl training metrics found.. Exiting")
            sys.exit(1)
        
        baseline_metrics = _read_metrics(metrics_path=baseline_metrics_path)
        if baseline_metrics is None:
            print("No baseline metrics found.. Exiting")
            sys.exit(1)
        
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

        print("\n" + "="*60)
        print("TRAINING AND EVALUATION COMPLETE")
        print("="*60)
        print(f"Results saved to: {dirs['plots']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train A3C agent on MEIS environment")
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval', 'plot'],
        help='Mode: train, eval, or plot'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='meis',
        choices=['meis', 'complex'],
        help='Environment variant: original MEIS (Env-1) or complex MEIS (Env-2)'
    )
    args = parser.parse_args()
    main(args)
