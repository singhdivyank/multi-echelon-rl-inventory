"""
Asynchronous Advantage Actor-Critic (A3C) Agent

Implementation following the paper specifications:
- Fully Connected MLP for both Actor and Critic
- 3 layers with 64 neurons each
- Policy entropy regularization
- Decentralized agents with asynchronous updates

Author: AI/ML Engineering Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, List


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network with separate heads for policy and value
    
    Architecture:
    - Shared: None (separate networks as per paper)
    - Actor: 3-layer FCMLP with 64 neurons per layer
    - Critic: 3-layer FCMLP with 64 neurons per layer
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        n_layers: int = 3
    ):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        actor_layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            actor_layers.append(nn.Linear(input_dim, hidden_size))
            actor_layers.append(nn.ReLU())
            input_dim = hidden_size
        actor_layers.append(nn.Linear(hidden_size, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (value function)
        critic_layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            critic_layers.append(nn.Linear(input_dim, hidden_size))
            critic_layers.append(nn.ReLU())
            input_dim = hidden_size
        critic_layers.append(nn.Linear(hidden_size, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
        """
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor [state_dim]
            deterministic: If True, select argmax action
        
        Returns:
            action: Selected action (int)
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self.forward(state.unsqueeze(0))
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size]
        
        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: State value estimates [batch_size]
            entropy: Policy entropy [batch_size]
        """
        action_logits, values = self.forward(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy


class A3CAgent:
    """
    A3C Agent for training
    
    Features:
    - Advantage Actor-Critic with GAE
    - Policy entropy regularization
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        n_layers: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.device = self._get_device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim,
            action_dim,
            hidden_size,
            n_layers
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training mode
        self.network.train()
    
    def _get_device(self, device: str = 'auto'):
        """
        Get the compute device.
        If device is 'auto', auto-detect best available device.
        Otherwise use the specified device.
        """
        if device not in ('auto', None):
            resolved = torch.device(device)
            print(f"Using {str(resolved).upper()} device")
            return resolved

        # Auto-detect best device
        if torch.cuda.is_available():
            try:
                # Validate CUDA actually works
                _ = torch.zeros(1, device='cuda')
                print("Using CUDA device")
                return torch.device("cuda")
            except RuntimeError:
                print("CUDA reported available but failed — falling back to CPU")
        
        print("Using CPU device")
        return torch.device("cpu")

    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given state
        
        Args:
            state: Current state
            deterministic: If True, select best action
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                state_tensor,
                deterministic
            )
        
        return action, log_prob.item(), value.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
        
        Returns:
            advantages: Computed advantages
            returns: Computed returns (targets for value function)
        """
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] +
                self.gamma * values[step + 1] * (1 - dones[step]) -
                values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values[:-1])
        
        return advantages, returns
    
    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Update policy and value function
        
        Args:
            states: List of states
            actions: List of actions
            advantages: Computed advantages
            returns: Computed returns
        
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (
            (advantages_tensor - advantages_tensor.mean()) /
            (advantages_tensor.std() + 1e-8)
        )
        
        # Evaluate actions
        log_probs, values, entropy = self.network.evaluate_actions(
            states_tensor,
            actions_tensor
        )
        
        # Actor loss (policy gradient with entropy regularization)
        actor_loss = -(log_probs * advantages_tensor).mean()
        
        # Critic loss (value function MSE)
        critic_loss = F.mse_loss(values, returns_tensor)
        
        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = (
            actor_loss +
            self.value_loss_coef * critic_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.max_grad_norm
        )
        self.optimizer.step()
        
        # Return metrics
        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': -entropy_loss.item(),
            'value_mean': values.mean().item(),
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def eval_mode(self):
        """Set network to evaluation mode"""
        self.network.eval()
    
    def train_mode(self):
        """Set network to training mode"""
        self.network.train()


def create_agent(config: Dict, state_dim: int, action_dim: int) -> A3CAgent:
    """
    Factory function to create A3C agent from config
    
    Args:
        config: Configuration dictionary
        state_dim: State space dimension
        action_dim: Action space dimension
    
    Returns:
        Initialized A3C agent
    """
    agent_config = config.get('agent', {})
    
    return A3CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=agent_config.get('hidden_size', 64),
        n_layers=agent_config.get('n_layers', 3),
        lr=agent_config.get('lr', 1e-4),
        gamma=agent_config.get('gamma', 0.99),
        gae_lambda=agent_config.get('gae_lambda', 0.95),
        entropy_coef=agent_config.get('entropy_coef', 0.01),
        value_loss_coef=agent_config.get('value_loss_coef', 0.5),
        max_grad_norm=agent_config.get('max_grad_norm', 0.5),
        device=agent_config.get('device', 'cpu')
    )