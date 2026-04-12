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
    dirs = create_directories(save_dir)

    device_name = set_device(general_config['device'])
    env_config = dict(config['divergent'])

    if args.mode == 'train':
        best_model_path = os.path.join(dirs[1], 'ppo_divergent.pt')
        stats_path = os.path.join(save_dir, 'training_stats.json')
        training_stats = train_agents(
            env_config=env_config, 
            config=ppo_config, 
            save_dirs=dirs,
            device=device_name,
            best_model_path=best_model_path
        )
        save_stats(stats=training_stats, save_path=stats_path)
    elif args.mode == 'eval':
        eval_path = os.path.join(save_dir, 'evaluation_results.json')
        eval_results = eval_agents(
            config=ppo_config, 
            env_config=env_config, 
            checkpoint_path=dirs[1],
            device_name=device_name
        )
        save_eval_results(results=eval_results, save_path=eval_path)
    else:
        eval_path = os.path.join(save_dir, 'evaluation_results.json')
        train_stats_path = os.path.join(save_dir, 'training_stats.json')
        plot_path = os.path.join(dirs[-1], 'training_curves.png')
        comparison_path = os.path.join(dirs[-1], 'comparison.png')

        print("\nGenerating training plots...")
        train_results = read_results(train_stats_path)
        plot_training_curves(
            ppo_stats=train_results['ppo'], 
            baseline_stats=train_results['baseline'], 
            save_path=plot_path
        )
        print(f"Training curves saved to {plot_path}")

        print("\nGenerating comparison plots...")
        eval_results = read_results(eval_path)
        plot_comparison(
            ppo_results=eval_results['ppo'], 
            baseline_results=eval_results['baseline'], 
            save_path=comparison_path
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
        choices=['train', 'eval', 'plot'],
        help='Mode: train, eval, or plot'
    )
    args = parser.parse_args()
    main(args)
