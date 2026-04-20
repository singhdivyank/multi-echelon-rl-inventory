"""Utility functions for file read and write operations"""

import json
import os
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