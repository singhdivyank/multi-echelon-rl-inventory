"""
Main training script for Divergent environment.
Trains both PPO and Baseline agents and compares performance.
"""

import argparse
import os
import sys
from typing import Dict

from ruamel.yaml import YAML

from evaluation.evaluate import eval_agents
from utils.helpers import (
    create_directories, 
    save_eval_results,
    save_stats,
    set_seeds, 
    set_device,
)
from utils.train import train_agents

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _load_config() -> Dict:
    """Load configurations from YAML file"""

    with open('config.yaml', 'r') as f:
        config = YAML().load(f)
        return config


def main(args):
    config = _load_config()
    ppo_config = dict(config['ppo'])

    seed_val = ppo_config['seed']
    set_seeds(seed_val)

    general_config = dict(config['general'])
    save_dir = general_config['save_dir']
    dirs = create_directories(save_dir)

    device_name = set_device(general_config['device'])
    env_config = dict(config['divergent'])

    if args.mode == 'train':
        best_model_path = os.path.join(save_dir, 'ppo_divergent.pt')
        training_stats = train_agents(
            env_config=env_config, 
            config=ppo_config, 
            save_dirs=dirs,
            device=device_name,
            best_model_path=best_model_path
        )
        stats_path = os.path.join(save_dir, 'training_stats.json')
        save_stats(stats=training_stats, save_path=stats_path)
    else:
        eval_results = eval_agents(
            config=config, 
            env_config=env_config, 
            checkpoint_path=dirs[0],
            plots_path=dirs[-1]
        )
        eval_path = os.path.join(save_dir, 'evaluation_results.json')
        save_eval_results(results=eval_results, save_path=eval_path)
    
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
    args = parser.parse_args()
    main(args)
