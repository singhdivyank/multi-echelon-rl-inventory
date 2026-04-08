"""
Actor-Critic neural network architecture
Architecture from Table 4: 2 hidden layers with 64 nodes each, tanh activation
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Tuple, List


class Actor(nn.Module):
    """
    Policy network (Actor)
    Outputs mean of action distribution (std is learned separately)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int, int]):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self._build_network()
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.apply(self._init_weights)
    
    def _build_network(self):
        layers = []
        
        input_dim = self.state_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, self.action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Glorot-Uniform initialization as per Table 4"""

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            mean: Action mean [batch_size, action_dim]
            std: Action std [batch_size, action_dim]
        """

        mean = self.network(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_action(self, state: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """

        mean, std = self.forward(state)

        if deterministic:
            action, log_prob = torch.tanh(mean), None
        else:
            dist = Normal(mean, std)
            z = dist.rsample()
            action = torch.tanh(z)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        action = torch.clamp(action, -1, 1)
        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """

        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """"""

    def __init__(self, state_dim: int, hidden_dims: List[int, int]):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self._build_network()
        self.apply(self._init_weights)
    
    def _build_network(self):
        layers = []
        
        input_dim = self.state_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Glorot-Uniform initialization"""

        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            value: State value estimate [batch_size, 1]
        """

        return self.network(state)
    

class ActorCritic(nn.Module):
    """Combined Actor-Critic model"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int, int]):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dims)
        self.critic = Critic(state_dim, hidden_dims)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks
        
        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: State value estimate
        """

        action_mean, action_std = self.actor(state)
        value = self.critic(state)
        return action_mean, action_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get action from actor"""

        return self.actor.get_action(state=state, deterministic=deterministic)
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate actions"""

        log_prob, entropy = self.actor.evaluate_actions(state=state, action=action)
        value = self.critic(state)
        return log_prob, value, entropy
