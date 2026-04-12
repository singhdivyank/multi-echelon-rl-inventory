"""
Utility functions for model training and evaluation
"""

import json
import os
from typing import Dict, List

import numpy as np
import torch
from ruamel.yaml import YAML

def load_config() -> Dict:
    """Load configurations from YAML file"""

    with open('./configs/config.yaml', 'r') as f:
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
