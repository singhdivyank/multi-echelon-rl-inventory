"""
Utility functions for computing metrics and advantages
"""

from typing import Tuple

import numpy as np
import torch

def compute_confidence_interval(
    data: np.ndarray, 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval
    
    Args:
        data: Array of data points
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        mean: Mean of data
        lower: Lower bound of CI
        upper: Upper bound of CI
    """

    from scipy import stats

    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    lower_bound, upper_bound = mean - ci, mean + ci
    return mean, lower_bound, upper_bound

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
