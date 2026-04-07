"""
Proximal Policy Optimization with clipped surrogate objective
"""

import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_critic import ActorCritic
from utils.replay_buffer import PPOBuffer
from utils.metrics import compute_gae


class PPOAgent:
    """
    PPO Agent with Actor-Critic architecture
    
    Implements:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple epochs of minibatch updates
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        config: Dict,
        device: torch.device
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
        }
        self.training_modes = True
        self._get_parameters()
        self._init_models()
    
    def _get_parameters(self):
        self.gamma = self.config['gamma']
        self.gae_lambda = self.config['gae_lambda']
        self.epsilon = self.config['epsilon']
        self.learning_rate = self.config['learning_rate']
        self.buffer_size = self.config['buffer_size']
        self.batch_size = self.config['batch_size']
        self.update_epochs = self.config['update_epochs']
        self.value_loss_coef = self.config['value_loss_coef']
        self.entropy_coef = self.config['entropy_coef']
        self.max_grad_norm = self.config['max_grad_norm']
        self.normalize_advantages = self.config['normalize_advantages']
        self.hidden_dims = self.config['hidden_layers']
    
    def _init_models(self):
        self.actor_critic = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate
        )

        self.buffer = PPOBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )
    
    def get_action(
        self, 
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy
        
        Args:
            state: Current state
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate of state
        """

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor_critic.get_action(state_tensor, deterministic)
            value = self.actor_critic.critic(state_tensor)
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item() if log_prob is not None else 0.0
            value = value.cpu().numpy()[0][0]
        
        return action, log_prob, value
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool, 
        log_prob: float, 
        value: float
    ):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of action
            value: Value estimate
        """

        self.buffer.store(state, action, reward, next_state, done, log_prob, value)
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm
        
        Returns:
            Dictionary of training statistics
        """

        if len(self.buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones, old_log_probs, values = self.buffer.get()

        # Compute returns and advantages using GAE
        with torch.no_grad():
            next_values = self.actor_critic.critic(next_states).squeeze()
            returns, advantages = compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
        
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss, total_value_loss = 0, 0
        total_entropy, total_kl = 0, 0
        total_clip_fraction, num_updates = 0, 0
        
        early_stop = False
        
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start: end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, values_new, entropy = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_pred = values_new.squeeze()

                value_loss_unclipped = (value_pred - batch_returns) ** 2
                value_loss_clipped = (value_pred - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm(
                        self.actor_critic.parameters(),
                        self.max_grad_norm
                    )
                
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()

                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    if kl > 0.05:
                        early_stop = True
                        break
                    
                    total_kl += kl.item()
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()
                    total_clip_fraction += clip_fraction.item()
                
                num_updates += 1
            
            if early_stop:
                break
        
        self.buffer.clear()
        if not num_updates > 0:
            return {}
        
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'total_loss': (total_policy_loss + total_value_loss) / num_updates,
            'kl_divergence': total_kl / num_updates,
            'clip_fraction': total_clip_fraction / num_updates
        }
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a state
        
        Args:
            state: State to evaluate
            
        Returns:
            Value estimate
        """

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.actor_critic.critic(state_tensor)
            return value.cpu().numpy()[0][0]
    
    def train(self):
        """Set agent to training mode"""

        self._training_mode = True
        self.actor_critic.train()
    
    def eval(self):
        """Set agent to evaluation mode"""

        self._training_mode = False
        self.actor_critic.eval()
    
    def save(self, path: str):
        """
        Save agent
        
        Args:
            path: Save path
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """
        Load agent
        
        Args:
            path: Load path
        """

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        print(f"Agent loaded from {path}")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics"""

        return self.training_stats
