"""
Logger with TensorBoard integration for tracking training progress
"""

import json
import numpy as np
import os

from datetime import datetime
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for tracking training metrics and writing to TensorBoard
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_to_file = True
        self.metrics = {
            'train': {},
            'eval': {}
        }
        self.current_iter = 0
        self._create_log()
    
    def _create_log(self):
        self.log_file = os.path.join(self.log_dir, 'training.log')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*80 + "\n")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value
        
        Args:
            tag: Tag for the metric (e.g., 'train/loss', 'eval/reward')
            step: Step number
        """

        self.writer.add_scalar(tag, value, step)
        if tag not in self.metrics['train'] and tag not in self.metrics['eval']:
            category = tag.split('/')[0] if '/' in tag else 'train'
            if category not in self.metrics:
                self.metrics[category] = {}
            
            metric_name = tag.split('/')[-1]
            if metric_name not in self.metrics[category]:
                self.metrics[category][metric_name] = []
            
            self.metrics[category][metric_name].append((step, value))
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars at once
        
        Args:
            main_tag: Main tag (e.g., 'loss')
            tag_scalar_dict: Dictionary of {tag: value}
            step: Step number
        """

        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.log_scalar(full_tag, value, step)
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        prefix: str, 
        step: Optional[int] = None
    ):
        """
        Log a dictionary of metrics
        
        Args:
            metrics: Dictionary of metric_name: value
            prefix: Prefix for tags (e.g., 'train', 'eval')
            step: Step number
        """

        if step is None:
            step = self.current_iter
        
        for name, value in metrics.items():
            tag = f"{prefix}/{name}"
            self.log_scalar(tag, value, step)

    def log_episode(
        self, 
        episode: int, 
        episode_reward: float,
        episode_cost: float, 
        episode_length: int, 
        additional_metrics: Optional[Dict] = None
    ):
        """
        Log episode statistics
        
        Args:
            episode: Episode number
            episode_reward: Total episode reward
            episode_cost: Total episode cost
            episode_length: Episode length
            additional_metrics: Additional metrics to log
        """

        self.log_scalar('episode/reward', episode_reward, episode)
        self.log_scalar('episode/rcost', episode_cost, episode)
        self.log_scalar('episode/length', episode_length, episode)

        if additional_metrics:
            for key, value in additional_metrics.items():
                self.log_scalar(f'episode/{key}', value, episode)
        
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Length={episode_length}")
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Length={episode_length}\n")
    
    def save_metrics(self):
        """Save all metrics to JSON"""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        
        # Convert metrics to serializable format
        serializable_metrics = {}
        for category, metrics_dict in self.metrics.items():
            serializable_metrics[category] = {}
            for metric_name, values in metrics_dict.items():
                serializable_metrics[category][metric_name] = [
                    {'step': int(step), 'value': float(value)} 
                    for step, value in values
                ]
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

    def close(self):
        """Close logger and save all data"""
        self.save_metrics()
        self.writer.close()
        
        with open(self.log_file, 'a') as f:
            f.write("="*80 + "\n")
            f.write(f"Training ended at {datetime.now()}\n")
