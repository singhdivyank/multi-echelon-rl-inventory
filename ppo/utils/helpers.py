"""
Utility functions for model training and evaluation
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ruamel.yaml import YAML

def load_config() -> Dict:
    """Load configurations from YAML file"""

    with open('./configs/config.yaml', 'r') as f:
        config = YAML().load(f)
        return config

def load_complex_config() -> Dict:
    """Load configurations of complex environemnt from YAML file"""

    with open('./configs/complexConfig.yaml', 'r') as f:
        config = YAML().load(f)
        return config

def load_divergent_config() -> Dict:
    """Load divergent env configurations from YAML file"""

    with open('./configs/divergentConfig.yaml', 'r') as f:
        config = YAML().load(f)
        return config

def set_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories(save_dir: str) -> List[str]:
    """Create necessary directories"""
    dirs = [
        save_dir,
        os.path.join(save_dir, 'checkpoints'),
        os.path.join(save_dir, 'logs'),
        os.path.join(save_dir, 'plots'),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs

def set_device(device: str) -> str:
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"\nUsing device: {device}")
    return device

def save_stats(stats: Dict, save_path: str):
    """Save agent training stats"""

    with open(save_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        json.dump(stats, f, indent=2, default=convert)
    
    print(f"Training statistics saved to {save_path}")

def save_eval_results(results: Dict, save_path: str):
    """Save agent evaluation stats"""

    with open(save_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"Evaluation results saved to {save_path}")

def read_results(save_path: str) -> Dict:
    """Return agent evaluation stats"""

    with open(save_path, 'r') as f:
        data = json.load(f)
    
    return data

@dataclass
class PathConfig:
    best_model_path: str
    stats_path: str
    eval_path: str
    train_stats_path: str
    plot_path: str
    comparison_path: str
    ppo_save_path: str
    baseline_save_path: str
    metrics_path: str
    log_dir: str

def get_paths(dirs: Dict, env_name: str) -> PathConfig:
    return PathConfig(
        best_model_path = Path(dirs[1]) / 'ppo_divergent.pt',
        stats_path = Path(dirs[0]) / 'training_stats.json',
        eval_path = Path(dirs[0]) / 'evaluation_results.json',
        train_stats_path = Path(dirs[0]) / 'training_stats.json',
        plot_path = Path(dirs[-1]) / 'training_curves.png',
        comparison_path = Path(dirs[-1]) / 'comparison.png',
        ppo_save_path= Path(dirs[1]) / 'ppo_divergent_final.pt',
        baseline_save_path = Path(dirs[1]) / 'baseline_divergent.npy',
        metrics_path = Path(dirs[2], 'metrics.json'),
        log_dir = Path(dirs[2]) / f"{env_name}"
    )
