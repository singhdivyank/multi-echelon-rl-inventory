"""
Training Loop for A3C Agent

Implements the training procedure with:
- Episode rollouts
- Advantage computation
- Policy updates
- Logging and checkpointing
"""

import json
import os
import re
from collections import deque
from typing import Dict
from tqdm import tqdm

import numpy as np
import torch


class Trainer:
    """
    Trainer class for A3C agent on MEIS environment
    """

    def __init__(self, env, agent, save_dir: str, config: Dict, history_path: str, best_model_path: str):
        self.env = env
        self.agent = agent
        self.config = config
        self.save_dir = save_dir
        self.history_path = history_path
        self.best_path = best_model_path
        os.makedirs(save_dir, exist_ok=True)
        self._get_parameters()
        self._for_logging_tracking()
        self.total_steps = 0
    
    def _get_parameters(self):
        """Get training parameters"""
        self.n_episodes = self.config['n_episodes']
        self.max_steps_per_episode = self.config['max_steps_per_episode']
        self.update_frequency = self.config['update_frequency']
        self.eval_frequency = self.config['eval_frequency']
        self.n_eval_episodes = self.config['n_eval_episodes']
        self.save_frequency = self.config['save_frequency']
    
    def _for_logging_tracking(self):
        """Initialise dictionaries for logging and tracking"""
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
            'eval_service_levels': []
        }
        self.episode_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def _find_latest_checkpoint(self):
        """
        Scan save_dir for the most recent checkpoint_ep*.pt file.
        """
        
        pattern = re.compile(r'checkpoint_ep(\d+)\.pt$')
        best_ep, self.latest_path = -1, None

        for fname in os.listdir(self.save_dir):
            m = pattern.match(fname)
            if m:
                ep = int(m.group(1))
                if ep > best_ep:
                    best_ep = ep
                    self.latest_path = os.path.join(self.save_dir, fname)
    
    def _load_chkpt(self, path: str) -> int:
        """
        Load checkpoint and restore training state.
        
        Args:
            path: Path to checkpoint file.
        
        Returns:
            The episode number to resume from.
        """

        checkpoint = torch.load(path, map_location=self.agent.device, weights_only=False)
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        
        saved_history = checkpoint.get('training_history', None)
        if saved_history is not None:
            self.training_history = saved_history
        
        episode = checkpoint.get('episode', 0)
        return episode
    
    def _run_episode(self) -> Dict:
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

        self.episode_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

        while not done and steps < self.max_steps_per_episode:
            action, _, value = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            self.episode_buffer['states'].append(state)
            self.episode_buffer['actions'].append(action)
            self.episode_buffer['rewards'].append(reward)
            self.episode_buffer['values'].append(value)
            self.episode_buffer['dones'].append(done)

            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']
            steps += 1
            self.total_steps += 1

            state = next_state
        
        service_level = service_level_sum / steps if steps > 0 else 0
        return {
            'reward': episode_reward,
            'cost': episode_cost,
            'length': steps,
            'service_level': service_level
        }
    
    def _update_agent(self):
        """Update agent using collected experience"""
        if not len(self.episode_buffer['states']):
            return None
        
        last_state = self.episode_buffer['states'][-1]
        state_tensor = torch.FloatTensor(last_state).to(self.agent.device)
        with torch.no_grad():
            _, next_val = self.agent.network.forward(state_tensor.unsqueeze(0))
            next_val = next_val.item()
        
        advantages, returns = self.agent.compute_gae(
            self.episode_buffer['rewards'],
            self.episode_buffer['values'],
            self.episode_buffer['dones'],
            next_val
        )

        update_metrics = self.agent.update(
            self.episode_buffer['states'],
            self.episode_buffer['actions'],
            advantages,
            returns
        )

        if update_metrics:
            self.training_history['losses'].append(update_metrics['loss'])
            self.training_history['actor_losses'].append(update_metrics['actor_loss'])
            self.training_history['critic_losses'].append(update_metrics['critic_loss'])
            self.training_history['entropies'].append(update_metrics['entropy'])
    
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
            episode_reward, episode_cost = 0, 0
            service_level_sum, steps = 0, 0

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

        self.eval_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'std_cost': np.std(episode_costs),
            'mean_service_level': np.mean(service_levels),
            'std_service_level': np.std(service_levels)
        }
        
        self.training_history['eval_rewards'].append(self.eval_metrics['mean_reward'])
        self.training_history['eval_costs'].append(self.eval_metrics['mean_cost'])
        self.training_history['eval_service_levels'].append(self.eval_metrics['mean_service_level'])
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint with full training state for resuming."""
        
        checkpoint = {
            'network_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode': episode,
            'total_steps': self.total_steps
        }
        suffix = 'final' if final else f'ep{episode}'
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_{suffix}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path} (episode {episode})")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_json = {}

        for key, value in self.training_history.items():
            if isinstance(value, np.ndarray):
                history_json[key] = value.tolist()
            elif isinstance(value, list):
                history_json[key] = value
            else:
                history_json[key] = value
            
        with open(self.history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved: {self.history_path}")

    def train(self) -> Dict:
        """
        Main training loop with automatic checkpoint resume support.
        
        If a previous checkpoint exists in save_dir, training will
        automatically resume from the last saved episode.
        
        Returns:
            Training history dictionary
        """

        start_episode = 0
        self._find_latest_checkpoint()
        if self.best_path is not None:
            start_episode = self._load_chkpt(self.best_path)
            print(f"Resumed from checkpoint: {self.best_path} (episode {start_episode})")
        
        print(f"Starting training for {self.n_episodes} episodes "
              f"(episodes {start_episode + 1}-{self.n_episodes})...")
        print(f"Environment: {self.env.__class__.__name__}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Save directory: {self.save_dir}")

        if start_episode >= self.n_episodes:
            print("Training already complete based on checkpoint. Skipping.")
            return self.training_history

        recent_rewards = deque(maxlen=100)
        best_cost = float('inf')
        best_episode = 0

        for r in self.training_history['episode_rewards'][-100:]:
            recent_rewards.append(r)
        
        remaining = self.n_episodes - start_episode
        pbar = tqdm(
            range(remaining), 
            desc="Training", 
            initial=0, 
            total=remaining
        )
        for i in pbar:
            episode = start_episode + i
            episode_metrics = self._run_episode()

            self.training_history['episode_rewards'].append(episode_metrics['reward'])
            self.training_history['episode_costs'].append(episode_metrics['cost'])
            self.training_history['episode_lengths'].append(episode_metrics['length'])
            self.training_history['service_levels'].append(episode_metrics['service_level'])

            recent_rewards.append(episode_metrics['reward'])

            pbar.set_postfix({
                'ep': episode + 1,
                'reward': f"{episode_metrics['reward']:.2f}",
                'avg_reward_100': f"{np.mean(recent_rewards):.2f}",
                "service_level": f"{episode_metrics['service_level']:.2f}"
            })

            # update agent
            if not (episode + 1) % self.update_frequency:
                self._update_agent()
            
            # Evaluation
            if not (episode + 1) % self.eval_frequency:
                self._evaluate()
                current_cost = self.eval_metrics['mean_cost']

                print(f"\nEvaluation at episode {episode + 1}:")
                print(f"  Mean reward: {self.eval_metrics['mean_reward']:.2f} ± {self.eval_metrics['std_reward']:.2f}")
                print(f"  Mean cost: {self.eval_metrics['mean_cost']:.2f} ± {self.eval_metrics['std_cost']:.2f}")
                print(f"  Service level: {self.eval_metrics['mean_service_level']:.2%}")

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_episode = episode + 1

                    if self.best_path is None:
                        raise ValueError("Error: path is None. Check if the path config was initialized correctly.")
                    
                    torch.save({
                        'network_state_dict': self.agent.network.state_dict(),
                        'optimizer_state_dict': self.agent.optimizer.state_dict(),
                        'episode': best_episode,
                        'best_cost': best_cost
                    }, self.latest_path)

                    print(f"BEST model saved at episode {best_episode} with cost {best_cost:.2f}")
            
        self._save_checkpoint(self.n_episodes, final=True)
        self._save_training_history()
            
        print("\n Training completed!")
        return self.training_history
