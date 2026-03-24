"""
Training Loop for A3C Agent

Implements the training procedure with:
- Episode rollouts
- Advantage computation
- Policy updates
- Logging and checkpointing

Author: AI/ML Engineering Team
"""

import numpy as np
import torch
import re
from typing import Dict, Optional
from tqdm import tqdm
import json
import os
from collections import deque


class Trainer:
    """
    Trainer class for A3C agent on MEIS environment
    """
    
    def __init__(
        self,
        env,
        agent,
        config: Dict,
        save_dir: str = './results'
    ):
        """
        Args:
            env: MEIS environment
            agent: A3C agent
            config: Training configuration
            save_dir: Directory to save results
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training parameters
        self.n_episodes = config.get('n_episodes', 500)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 365)
        self.update_frequency = config.get('update_frequency', 1)  # Episodes between updates
        self.eval_frequency = config.get('eval_frequency', 50)
        self.n_eval_episodes = config.get('n_eval_episodes', 10)
        self.save_frequency = config.get('save_frequency', 100)
        
        # Logging
        self.training_history = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_lengths': [],
            'service_levels': [],
            'losses': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'eval_rewards': [],
            'eval_costs': [],
            'eval_service_levels': [],
        }
        
        # For tracking
        self.total_steps = 0
        self.episode_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }
    
    def train(self) -> Dict:
        """
        Main training loop with automatic checkpoint resume support.
        
        If a previous checkpoint exists in save_dir, training will
        automatically resume from the last saved episode.
        
        Returns:
            Training history dictionary
        """
        # --- Checkpoint resume ---
        start_episode = 0
        latest_ckpt = self._find_latest_checkpoint()
        if latest_ckpt is not None:
            start_episode = self._load_checkpoint(latest_ckpt)
            print(f"Resumed from checkpoint: {latest_ckpt} (episode {start_episode})")
        
        print(f"Starting training for {self.n_episodes} episodes "
              f"(episodes {start_episode + 1}–{self.n_episodes})...")
        print(f"Environment: {self.env.__class__.__name__}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Save directory: {self.save_dir}")
        
        if start_episode >= self.n_episodes:
            print("Training already complete based on checkpoint. Skipping.")
            return self.training_history
        
        # Recent episodes for moving average
        recent_rewards = deque(maxlen=100)
        # Pre-fill from restored history
        for r in self.training_history['episode_rewards'][-100:]:
            recent_rewards.append(r)
        
        # Progress bar starting from the resumed episode
        remaining = self.n_episodes - start_episode
        pbar = tqdm(range(remaining), desc="Training", initial=0, total=remaining)
        
        for i in pbar:
            episode = start_episode + i
            
            # Run episode
            episode_metrics = self._run_episode(episode)
            
            # Log metrics
            self.training_history['episode_rewards'].append(episode_metrics['reward'])
            self.training_history['episode_costs'].append(episode_metrics['cost'])
            self.training_history['episode_lengths'].append(episode_metrics['length'])
            self.training_history['service_levels'].append(episode_metrics['service_level'])
            
            recent_rewards.append(episode_metrics['reward'])
            
            # Update progress bar
            pbar.set_postfix({
                'ep': episode + 1,
                'reward': f"{episode_metrics['reward']:.2f}",
                'avg_reward_100': f"{np.mean(recent_rewards):.2f}",
                'service_level': f"{episode_metrics['service_level']:.2%}",
            })
            
            # Update agent
            if (episode + 1) % self.update_frequency == 0:
                update_metrics = self._update_agent()
                if update_metrics:
                    self.training_history['losses'].append(update_metrics['loss'])
                    self.training_history['actor_losses'].append(update_metrics['actor_loss'])
                    self.training_history['critic_losses'].append(update_metrics['critic_loss'])
                    self.training_history['entropies'].append(update_metrics['entropy'])
            
            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self.training_history['eval_rewards'].append(eval_metrics['mean_reward'])
                self.training_history['eval_costs'].append(eval_metrics['mean_cost'])
                self.training_history['eval_service_levels'].append(eval_metrics['mean_service_level'])
                
                print(f"\nEvaluation at episode {episode + 1}:")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Mean cost: {eval_metrics['mean_cost']:.2f} ± {eval_metrics['std_cost']:.2f}")
                print(f"  Service level: {eval_metrics['mean_service_level']:.2%}")
            
            # Save checkpoint
            if (episode + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode + 1)
        
        # Final save
        self._save_checkpoint(self.n_episodes, final=True)
        self._save_training_history()
        
        print("\nTraining completed!")
        return self.training_history
    
    def _run_episode(self, episode: int) -> Dict:
        """
        Run a single training episode
        
        Returns:
            Episode metrics
        """
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        service_level_sum = 0
        steps = 0
        
        # Clear episode buffer
        self.episode_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }
        
        while not done and steps < self.max_steps_per_episode:
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.episode_buffer['states'].append(state)
            self.episode_buffer['actions'].append(action)
            self.episode_buffer['rewards'].append(reward)
            self.episode_buffer['values'].append(value)
            self.episode_buffer['dones'].append(done)
            
            # Update metrics
            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']
            steps += 1
            self.total_steps += 1
            
            state = next_state
        
        return {
            'reward': episode_reward,
            'cost': episode_cost,
            'length': steps,
            'service_level': service_level_sum / steps if steps > 0 else 0,
        }
    
    def _update_agent(self) -> Optional[Dict]:
        """
        Update agent using collected experience
        
        Returns:
            Update metrics
        """
        if len(self.episode_buffer['states']) == 0:
            return None
        
        # Get next value (bootstrap)
        last_state = self.episode_buffer['states'][-1]
        state_tensor = torch.FloatTensor(last_state).to(self.agent.device)
        with torch.no_grad():
            _, next_value = self.agent.network.forward(state_tensor.unsqueeze(0))
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.agent.compute_gae(
            self.episode_buffer['rewards'],
            self.episode_buffer['values'],
            self.episode_buffer['dones'],
            next_value
        )
        
        # Update policy
        update_metrics = self.agent.update(
            self.episode_buffer['states'],
            self.episode_buffer['actions'],
            advantages,
            returns
        )
        
        return update_metrics
    
    def _evaluate(self) -> Dict:
        """
        Evaluate current policy
        
        Returns:
            Evaluation metrics
        """
        self.agent.eval_mode()
        
        episode_rewards = []
        episode_costs = []
        service_levels = []
        
        for _ in range(self.n_eval_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            service_level_sum = 0
            steps = 0
            
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_cost -= reward
                service_level_sum += info['service_level']
                steps += 1
            
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            service_levels.append(service_level_sum / steps if steps > 0 else 0)
        
        self.agent.train_mode()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'mean_service_level': np.mean(service_levels),
            'std_service_level': np.std(service_levels),
        }
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint with full training state for resuming."""
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{suffix}.pt')
        
        # Build checkpoint dict with training metadata
        checkpoint = {
            'network_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode': episode,
            'total_steps': self.total_steps,
            'training_history': self.training_history,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path} (episode {episode})")
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        Scan save_dir for the most recent checkpoint_ep*.pt file.
        
        Returns:
            Path to latest checkpoint, or None if none found.
        """
        if not os.path.isdir(self.save_dir):
            return None
        
        pattern = re.compile(r'checkpoint_ep(\d+)\.pt$')
        best_ep = -1
        best_path = None
        
        for fname in os.listdir(self.save_dir):
            m = pattern.match(fname)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    best_path = os.path.join(self.save_dir, fname)
        
        return best_path
    
    def _load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and restore training state.
        
        Args:
            path: Path to checkpoint file.
        
        Returns:
            The episode number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.agent.device)
        
        # Restore model + optimizer
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training metadata
        episode = checkpoint.get('episode', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        saved_history = checkpoint.get('training_history', None)
        if saved_history is not None:
            self.training_history = saved_history
        
        return episode
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.training_history.items():
            if isinstance(value, np.ndarray):
                history_json[key] = value.tolist()
            elif isinstance(value, list):
                history_json[key] = value
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved: {history_path}")


def train_agent(env, agent, config: Dict, save_dir: str) -> Dict:
    """
    Convenience function to train agent
    
    Args:
        env: MEIS environment
        agent: A3C agent
        config: Training configuration
        save_dir: Save directory
    
    Returns:
        Training history
    """
    trainer = Trainer(env, agent, config, save_dir)
    return trainer.train()