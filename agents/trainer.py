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
        Main training loop
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {self.n_episodes} episodes...")
        print(f"Environment: {self.env.__class__.__name__}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Save directory: {self.save_dir}")
        
        # Recent episodes for moving average
        recent_rewards = deque(maxlen=100)
        
        # Progress bar
        pbar = tqdm(range(self.n_episodes), desc="Training")
        
        for episode in pbar:
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
        if len(self.episode_buffer['states']) > 0:
            last_state = self.episode_buffer['states'][-1]
            state_tensor = torch.FloatTensor(last_state).to(self.agent.device)
            with torch.no_grad():
                _, next_value = self.agent.network.forward(state_tensor.unsqueeze(0))
                next_value = next_value.item()
        else:
            next_value = 0
        
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
        """Save model checkpoint"""
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{suffix}.pt')
        self.agent.save(checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
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