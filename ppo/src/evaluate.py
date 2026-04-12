"""
Evaluation Utilities

Functions for evaluating and comparing RL agent and baseline policies across:
- Multiple random seeds
- Different metrics (cost, service level, etc.)
- Statistical significance testing
"""

import os
from typing import Dict

import numpy as np
from tqdm import tqdm

from models.baseline import BaselineAgent
from models.env import DivergentInventoryEnv
from utils.logger import Logger
from utils.metrics import compute_confidence_interval

def evaluate_agent(
    agent, 
    env, 
    num_episodes: int, 
    deterministic: bool, 
) -> Dict:
    """
    Evaluate an agent on an environment
    
    Args:
        agent: Agent to evaluate (PPO or Baseline)
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        
    Returns:
        Dictionary of evaluation metrics
    """

    agent.eval()

    episode_rewards = []
    episode_costs = []
    episode_lengths = []

    inventory_levels = []
    backorder_levels = []
    order_quantities = []

    iterator = tqdm(range(num_episodes), desc = 'Evaluating')

    for episode in iterator:
        state, _ = env.reset(seed=episode)
        episode_reward, episode_cost = 0, 0
        episode_length, done = 0, False

        episode_inventories = []
        episode_backorders = []
        episode_orders = []

        while not done:
            from models.ppo import PPOAgent
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.get_action(state, deterministic=deterministic)
            else:
                action = agent.get_action(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_cost += info.get('cost', 0)
            episode_length += 1

            if 'inventory_warehouse' in info:
                episode_inventories.append(info['inventory_warehouse'])
            if 'inventory_retailers' in info:
                episode_backorders.append(np.sum(np.maximum(0, -info['inventory_retailers'])))
            episode_orders.append(action)

            state = next_state
        
        inventory_levels.append(np.mean(episode_inventories) if episode_inventories else 0)
        backorder_levels.append(np.mean(episode_backorders) if episode_backorders else 0)
        order_quantities.append(np.mean([np.abs(a).sum() for a in episode_orders]))
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
    
    episode_rewards = np.array(episode_rewards)
    episode_costs = np.array(episode_costs)
    episode_lengths = np.array(episode_lengths)
    mean_reward, ci_reward_lower, ci_reward_upper = compute_confidence_interval(episode_rewards)
    mean_cost, ci_cost_lower, ci_cost_upper = compute_confidence_interval(episode_costs)
    return {
        'mean_reward': mean_reward,
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'ci_reward_lower': ci_reward_lower,
        'ci_reward_upper': ci_reward_upper,
        'mean_cost': mean_cost,
        'std_cost': np.std(episode_costs),
        'min_cost': np.min(episode_costs),
        'max_cost': np.max(episode_costs),
        'median_cost': np.median(episode_costs),
        'ci_lower': ci_cost_lower,
        'ci_upper': ci_cost_upper,
        'mean_length': np.mean(inventory_levels),
        'mean_backorders': np.mean(backorder_levels),
        'mean_orders': np.mean(order_quantities),
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_lengths': episode_lengths
    }


def evaluate_baseline(
    agent, 
    env, 
    num_episodes: int, 
    logger: Logger
) -> Dict[str, list]:
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
        'episode_lengths': []
    }

    for episode in tqdm(range(num_episodes), desc="Baseline Evaluation"):
        state, _ = env.reset()
        episode_reward, episode_cost = 0, 0
        episode_length = 0
        done = False

        while not done:
            action = agent.get_action(state)
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

        if not episode % 100:
            logger.log_episode(
                episode=episode,
                episode_cost=episode_cost,
                episode_reward=episode_reward,
                episode_length=episode_length
            )
    
    print(f"\nBaseline Results:")
    print(f"  Avg Cost: {np.mean(stats['episode_costs']):.2f} ± {np.std(stats['episode_costs']):.2f}")
    print(f"  Avg Reward: {np.mean(stats['episode_rewards']):.2f} ± {np.std(stats['episode_rewards']):.2f}")
    return stats

def evaluate_policy(agent, env, num_episodes: int):
    """
    Evaluate policy during training
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of episodes
        
    Returns:
        Tuple of (avg_cost, avg_reward)
    """

    costs, rewards = [], []
    agent.eval()

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward, episode_cost = 0, 0
        done = False

        while not done:
            action, _, _ = agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_cost += info.get('cost', 0)
            state = next_state
        
        costs.append(episode_cost)
        rewards.append(episode_reward)
    
    agent.train()
    return np.mean(costs), np.mean(rewards)

def eval_agents(
    config: Dict, 
    env_config: Dict, 
    checkpoint_path: str,
    device_name: str
) -> Dict:
    """Evaluate trained agents"""

    from models.ppo import PPOAgent

    print("\n" + "="*80)
    print("Evaluating Agents")
    print("="*80)

    env = DivergentInventoryEnv(env_config)
    
    # Load PPO agent
    ppo_agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config,
        device=device_name
    )
    ppo_load_path=os.path.join(checkpoint_path, 'ppo_divergent_final.pt')
    ppo_agent.load(ppo_load_path)
    print(f"PPO agent loaded from {ppo_load_path}")
    
    # Evaluate PPO agent
    ppo_agent.eval()
    print("\nEvaluating PPO agent...")
    ppo_results = evaluate_agent(
        agent=ppo_agent,
        env=env,
        num_episodes=config['eval_episodes'],
        deterministic=True
    )
    print(f"\nPPO Results:")
    print(f"  Mean Cost: {ppo_results['mean_cost']:.2f} ± {ppo_results['std_cost']:.2f}")
    print(f"  Mean Reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
    print(f"  95% CI: [{ppo_results['ci_lower']:.2f}, {ppo_results['ci_upper']:.2f}]")

    # Load Baseline agent
    baseline_agent = BaselineAgent(env)
    baseline_path = os.path.join(checkpoint_path, 'baseline_divergent.npy')
    if os.path.exists(baseline_path):
        baseline_agent.load(baseline_path)
        print(f"Baseline agent loaded from {baseline_path}")
    
    print("\nEvaluating Baseline agent...")
    baseline_results = evaluate_agent(
        agent=baseline_agent,
        env=env,
        num_episodes=config['eval_episodes'],
        deterministic=True
    )
    print(f"\nBaseline Results:")
    print(f"  Mean Cost: {baseline_results['mean_cost']:.2f} ± {baseline_results['std_cost']:.2f}")
    print(f"  Mean Reward: {baseline_results['mean_reward']:.2f} ± {baseline_results['std_reward']:.2f}")
    print(f"  95% CI: [{baseline_results['ci_lower']:.2f}, {baseline_results['ci_upper']:.2f}]")
    
    # Compare PPO against baseline
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    improvement = (baseline_results['mean_cost'] - ppo_results['mean_cost']) / baseline_results['mean_cost'] * 100
    print(f"\nPPO Improvement over Baseline: {improvement:.2f}%")

    return {
        'ppo': ppo_results,
        'baseline': baseline_results,
        'improvement': improvement,
    }
