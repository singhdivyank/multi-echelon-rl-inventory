"""
Replay buffer for PPO algorithm
"""
import torch
import numpy as np
from typing import Tuple


class PPOBuffer:
    """
    Buffer for storing trajectories during PPO training
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: torch.device
    ):
        """
        Initialize buffer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            buffer_size: Maximum buffer size
            device: Device to store tensors
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # Initialize storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.ptr = 0
        self.size = 0
        
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """Store a transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        self.size = min(self.size + 1, self.buffer_size)
        
    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all stored transitions as tensors
        
        Returns:
            Tuple of tensors (states, actions, rewards, next_states, dones, log_probs, values)
        """
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        return states, actions, rewards, next_states, dones, log_probs, values
        
    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        self.ptr = 0
        self.size = 0
        
    def __len__(self):
        """Return current buffer size"""
        return len(self.states)