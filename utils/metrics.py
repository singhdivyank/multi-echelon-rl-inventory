"""
Utility functions for computing metrics and advantages
"""
import torch
import numpy as np
from typing import Tuple


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    GAE(γ, λ) combines multiple n-step returns with exponential weights
    
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
        if t == batch_size - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
            
        # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # GAE: A_t = δ_t + γλδ_{t+1} + (γλ)^2 δ_{t+2} + ...
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
        
    # Returns = advantages + values
    returns = advantages + values
    
    return returns, advantages


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute discounted returns
    
    Args:
        rewards: Tensor of rewards
        dones: Tensor of done flags
        gamma: Discount factor
        
    Returns:
        returns: Discounted returns
    """
    returns = torch.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return
        
    return returns


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
    
    # t-distribution for small samples
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean, mean - ci, mean + ci