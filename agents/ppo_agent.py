"""
PPO Agent implementation
Proximal Policy Optimization with clipped surrogate objective
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
import os

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
        config: Optional[Dict] = None
    ):
        """
        Initialize PPO agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.epsilon = self.config.get('epsilon', 0.2)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.buffer_size = self.config.get('buffer_size', 256)
        self.batch_size = self.config.get('batch_size', 64)
        self.update_epochs = self.config.get('update_epochs', 10)
        self.value_loss_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.normalize_advantages = self.config.get('normalize_advantages', True)
        
        # Networks
        hidden_dims = self.config.get('hidden_layers', [64, 64])
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate
        )
        
        # Replay buffer
        self.buffer = PPOBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=self.buffer_size,
            device=self.device
        )
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': [],
        }
        
        self._training_mode = True
        
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
            
            # Get action from actor
            action, log_prob = self.actor_critic.get_action(
                state_tensor,
                deterministic=deterministic
            )
            
            # Get value from critic
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
        self.buffer.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO algorithm
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) < self.batch_size:
            return {}
            
        # Get data from buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = \
            self.buffer.get()
            
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
            
        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Multiple epochs of updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_fraction = 0
        num_updates = 0
        
        for epoch in range(self.update_epochs):
            # Generate random indices for minibatches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                    
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = values[batch_indices]  # critic predictions at collection time
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions with current policy
                log_probs, values_new, entropy = self.actor_critic.evaluate_actions(
                    batch_states,
                    batch_actions
                )
                
                # Compute ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.epsilon,
                    1.0 + self.epsilon
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping (standard PPO value clipping)
                value_pred = values_new.squeeze()
                value_old = batch_old_values.squeeze().detach()

                value_clipped = value_old + torch.clamp(
                    value_pred - value_old,
                    -self.epsilon,
                    self.epsilon
                )

                value_loss_unclipped = (value_pred - batch_returns) ** 2
                value_loss_clipped = (value_clipped - batch_returns) ** 2

                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        self.max_grad_norm
                    )
                    
                self.optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                
                # KL divergence (approximate)
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    total_kl += kl.item()
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()
                    total_clip_fraction += clip_fraction.item()
                    
                num_updates += 1
                
        # Clear buffer
        self.buffer.clear()
        
        # Average statistics
        if num_updates > 0:
            stats = {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'total_loss': (total_policy_loss + total_value_loss) / num_updates,
                'kl_divergence': total_kl / num_updates,
                'clip_fraction': total_clip_fraction / num_updates,
            }
            
            # Store in history
            for key, value in stats.items():
                self.training_stats[key].append(value)
                
            return stats
        else:
            return {}
            
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
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"Agent loaded from {path}")
        
    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return self.training_stats