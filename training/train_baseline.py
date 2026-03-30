"""
Training/Evaluation for baseline agent
"""
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm

from utils.logger import Logger


def evaluate_baseline(
    agent,
    env,
    num_episodes: int,
    logger: Optional[Logger] = None,
) -> Dict:
    """
    Evaluate baseline agent (no training needed)
    
    Args:
        agent: Baseline agent
        env: Environment
        num_episodes: Number of episodes
        logger: Logger instance
        
    Returns:
        Evaluation statistics
    """
    print(f"\nEvaluating baseline agent for {num_episodes} episodes...")
    
    stats = {
        'iterations': [],
        'episode_costs': [],
        'episode_rewards': [],
        'episode_lengths': [],
    }
    
    for episode in tqdm(range(num_episodes), desc="Baseline Evaluation"):
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from baseline
            action = agent.get_action(state, deterministic=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_cost += info.get('cost', 0)
            episode_length += 1
            
            state = next_state
        
        stats['episode_costs'].append(episode_cost)
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['iterations'].append(episode)
        
        if logger and episode % 100 == 0:
            logger.log_episode(
                episode=episode,
                episode_reward=episode_reward,
                episode_cost=episode_cost,
                episode_length=episode_length
            )
    
    print(f"\nBaseline Results:")
    print(f"  Avg Cost: {np.mean(stats['episode_costs']):.2f} ± {np.std(stats['episode_costs']):.2f}")
    print(f"  Avg Reward: {np.mean(stats['episode_rewards']):.2f} ± {np.std(stats['episode_rewards']):.2f}")
    
    return stats