"""Utility functions for file read and write operations"""

import json
import os
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Optional

import yaml

def _load_config() -> Dict:
    """Load configurations from YAML file"""

    with open('./configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def _get_meis_env() -> Dict:
    with open('./configs/meisConfig.yaml', 'r') as f:
        return yaml.safe_load(f)

def _get_complex_meis_env() -> Dict:
    with open('./configs/complexMeisConfig.yaml', 'r') as f:
        return yaml.safe_load(f)

def _save_config(base_path, content: Dict):
    """Save training config as JSON"""
    if isinstance(base_path, str):
        config_path = os.path.join(base_path, 'config.json')
    else:
        config_path= base_path
    with open(config_path, 'w') as f:
        json.dump(content, f, indent=2)
    
    print(f"Configurations saved to: {config_path}")

def _setup_directory(base_path: str):
    """Create necessary directories"""
    dirs = {
        'base': base_path,
        'checkpoints': os.path.join(base_path, 'checkpoints'),
        'plots': os.path.join(base_path, 'plots'),
        'logs': os.path.join(base_path, 'logs'),
        'eval': os.path.join(base_path, 'eval')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def print_eval_res(eval_res: Dict, agent: Optional[str] = None):
    """Print statements for agent and baseline evaluation results"""

    # comparison results
    if not agent:
        print(f"\nImprovement:")
        print(f"  Cost: {eval_res['cost_improvement_percent']:.2f}%")
        print(f"  Service Level: {eval_res['service_level_improvement_percent']:.2f}%")
        print(f"  Statistical Significance (cost): {eval_res['cost_ttest']['significant']} (p={eval_res['cost_ttest']['p_value']:.4f})")
        print(f"  Effect Size (Cohen's d): {eval_res['cohens_d_cost']:.3f}")
    else:
        # agent evaluation results
        if agent == 'a3c':
            print(f"\nA3C Agent:")
        else:
            print(f"\n(s,S) Baseline:")

        print(f"  Mean Cost: {eval_res['mean_cost']:.2f} ± {eval_res['std_cost']:.2f}")
        print(f"  Mean Service Level: {eval_res['mean_service_level']:.2%} ± {eval_res['std_service_level']:.2%}")
        print(f"  Cost Breakdown:")
        print(f"    Shortage: {eval_res['cost_breakdown']['shortage']:.2f}")
        print(f"    Holding: {eval_res['cost_breakdown']['holding']:.2f}")
        print(f"    Reordering: {eval_res['cost_breakdown']['reordering']:.2f}")

def _read_metrics(metrics_path: str) -> Optional[Dict]:
    """Load training history if available"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            training_history = json.load(f)
            return training_history
    
    return None


@dataclass
class PathConfig:
    checkpoint_path: str
    best_model_path: str
    history_path: str
    rl_metrics_path: str
    baseline_metrics_path: str
    params_path: str
    eval_res_path: str

    def validate(self):
        """Checks if all paths exist, otherwise exit the program"""
        for field in fields(self):
            path = getattr(self, field.name)
            if not path.exists():
                sys.exit(f"Critical Error: Required path not found at: {path}")
        print("All paths validated successfully")

def get_paths(dirs: Dict) -> PathConfig:
    return PathConfig(
        checkpoint_path = Path(dirs['checkpoints']) / 'checkpoint_final.pt',
        best_model_path = Path(dirs['checkpoints']) / 'best_model.pt',
        history_path = Path(dirs['checkpoints']) / 'training_history.json',
        rl_metrics_path = Path(dirs['eval']) / 'rl_metrics.json',
        baseline_metrics_path = Path(dirs['eval']) / 'baseline_metrics.json',
        params_path = Path(dirs['logs']) / 'baseline_params.json',
        eval_res_path = Path(dirs['logs']) / 'evaluation_results.json',
    )
