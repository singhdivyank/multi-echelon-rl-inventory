"""
Logger with TensorBoard integration for tracking training progress
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for tracking training metrics and writing to TensorBoard
    """
    
    def __init__(self, log_dir: str, log_to_file: bool = True):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for logs
            log_to_file: Whether to write logs to file
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Log to file
        self.log_to_file = log_to_file
        if log_to_file:
            self.log_file = os.path.join(log_dir, 'training.log')
            with open(self.log_file, 'w') as f:
                f.write(f"Training started at {datetime.now()}\n")
                f.write("="*80 + "\n")
        
        # Metrics storage
        self.metrics = {
            'train': {},
            'eval': {},
        }
        
        self.current_iteration = 0
        
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value
        
        Args:
            tag: Tag for the metric (e.g., 'train/loss', 'eval/reward')
            step: Step number (uses current_iteration if None)
        """
        if step is None:
            step = self.current_iteration
            
        # Write to TensorBoard
        self.writer.add_scalar(tag, value, step)
        
        # Store in metrics
        if tag not in self.metrics['train'] and tag not in self.metrics['eval']:
            category = tag.split('/')[0] if '/' in tag else 'train'
            metric_name = tag.split('/')[-1]
            
            if category not in self.metrics:
                self.metrics[category] = {}
            if metric_name not in self.metrics[category]:
                self.metrics[category][metric_name] = []
                
            self.metrics[category][metric_name].append((step, value))
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple scalars at once
        
        Args:
            main_tag: Main tag (e.g., 'loss')
            tag_scalar_dict: Dictionary of {tag: value}
            step: Step number
        """
        if step is None:
            step = self.current_iteration
            
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        for tag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{tag}"
            self.log_scalar(full_tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log a histogram
        
        Args:
            tag: Tag for the histogram
            values: Array of values
            step: Step number
        """
        if step is None:
            step = self.current_iteration
            
        self.writer.add_histogram(tag, values, step)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = 'train', step: Optional[int] = None):
        """
        Log a dictionary of metrics
        
        Args:
            metrics: Dictionary of metric_name: value
            prefix: Prefix for tags (e.g., 'train', 'eval')
            step: Step number
        """
        if step is None:
            step = self.current_iteration
            
        for name, value in metrics.items():
            tag = f"{prefix}/{name}"
            self.log_scalar(tag, value, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text
        
        Args:
            tag: Tag for the text
            text: Text to log
            step: Step number
        """
        if step is None:
            step = self.current_iteration
            
        self.writer.add_text(tag, text, step)
        
        # Also write to log file
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"[Step {step}] {tag}: {text}\n")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Optional[Dict[str, float]] = None):
        """
        Log hyperparameters
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Optional dictionary of metrics
        """
        # Convert to serializable format
        hparams_serializable = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparams_serializable[key] = value
            else:
                hparams_serializable[key] = str(value)
        
        if metrics is None:
            metrics = {}
            
        self.writer.add_hparams(hparams_serializable, metrics)
        
        # Save to JSON
        hparams_path = os.path.join(self.log_dir, 'hyperparameters.json')
        with open(hparams_path, 'w') as f:
            json.dump(hparams_serializable, f, indent=2)
    
    def log_episode(self, episode: int, episode_reward: float, episode_cost: float, 
                    episode_length: int, additional_metrics: Optional[Dict] = None):
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
        self.log_scalar('episode/cost', episode_cost, episode)
        self.log_scalar('episode/length', episode_length, episode)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.log_scalar(f'episode/{key}', value, episode)
        
        # Print to console
        print(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Length={episode_length}")
        
        # Write to log file
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Episode {episode}: Reward={episode_reward:.2f}, Cost={episode_cost:.2f}, Length={episode_length}\n")
    
    def log_training_step(self, iteration: int, metrics: Dict[str, float]):
        """
        Log training step metrics
        
        Args:
            iteration: Training iteration
            metrics: Dictionary of metrics
        """
        self.current_iteration = iteration
        
        for name, value in metrics.items():
            self.log_scalar(f'train/{name}', value, iteration)
        
        # Print summary every 100 iterations
        if iteration % 100 == 0:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"Iteration {iteration}: {metrics_str}")
    
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
        
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write("="*80 + "\n")
                f.write(f"Training ended at {datetime.now()}\n")
    
    def __del__(self):
        """Destructor to ensure writer is closed"""
        try:
            self.close()
        except:
            pass