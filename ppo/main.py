"""
Main training script for Divergent environment.
Trains both PPO and Baseline agents and compares performance.
"""

import argparse
import os
import sys

from src.evaluate import eval_agents
from src.visualise import plot_comparison, plot_training_curves
from src.train import train_agents
from utils.helpers import (
    create_directories, 
    get_paths,
    load_config,
    read_results,
    save_eval_results,
    save_stats,
    set_seeds, 
    set_device,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(args):
    config = load_config()
    ppo_config = dict(config['ppo'])

    seed_val = ppo_config['seed']
    set_seeds(seed_val)
    general_config = dict(config['general'])
    save_dir = general_config['save_dir']
    device_name = set_device(general_config['device'])
    
    dirs = create_directories(save_dir)
    env_name = args.env
    _paths = get_paths(dirs=dirs, env_name=env_name)

    if args.mode == 'train':
        training_stats = train_agents(
            config=ppo_config,
            log_dir = _paths.log_dir,
            device=device_name,
            best_model_path=_paths.best_model_path,
            ppo_save_path=_paths.ppo_save_path,
            baseline_save_path=_paths.baseline_save_path,
            metrics_save_path=_paths.metrics_path,
            env_name=env_name,
        )
        save_stats(stats=training_stats, save_path=_paths.stats_path)
    else:
        eval_results = eval_agents(
            config=ppo_config,
            ppo_load_path=_paths.ppo_save_path,
            baseline_path=_paths.baseline_save_path,
            device_name=device_name,
            env_name=env_name,
        )
        save_eval_results(results=eval_results, save_path=_paths.eval_path)

        print("\nGenerating training plots...")
        train_results = read_results(_paths.train_stats_path)
        plot_training_curves(
            ppo_stats=train_results['ppo'], 
            baseline_stats=train_results['baseline'], 
            save_path=_paths.plot_path
        )
        print(f"Training curves saved to {_paths.plot_path}")

        print("\nGenerating comparison plots...")
        eval_results = read_results(_paths.eval_path)
        plot_comparison(
            ppo_results=eval_results['ppo'], 
            baseline_results=eval_results['baseline'], 
            save_path=_paths.comparison_path
        )
        print(f"Comparison plots saved to {eval_results}")
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on Divergent Inventory Environment')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval'],
        help='Mode: train or eval'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='divergent',
        choices=['divergent', 'complex'],
        help='Environment variant: original divergent (Env-1) or complex (Env-2)'
    )
    args = parser.parse_args()
    main(args)
