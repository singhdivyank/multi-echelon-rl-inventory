"""
Utility functions for computing metrics and advantages
"""

from typing import Tuple

import torch

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Tensor of rewards [batch_size]
        values: Tensor of value estimates [batch_size]
        next_values: Tensor of next value estimates [batch_size]
        dones: Tensor of done flags [batch_size]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        returns: Discounted returns
        advantages: GAE advantages
    """

    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0

    for t in reversed(range(batch_size)):
        next_values = next_values[t] if t == batch_size - 1 else values[t + 1]
    
        delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return returns, advantages
