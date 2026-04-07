import os
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from models.baseline import BaselineAgent
from models.env import DivergentInventoryEnv
from evaluation.visualise import plot_comparison, plot_heatmap
from utils.logger import Logger

def evaluate_agent(
    agent, 
    env, 
    num_episodes: int, 
    deterministic: bool, 
    verbose: Optional[bool] = True
) -> Dict:
    """
    Evaluate an agent on an environment
    
    Args:
        agent: Agent to evaluate (PPO or Baseline)
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        verbose: Whether to print progress
        
    Returns:
        Dictionary of evaluation metrics
    """

    agent.eval()


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
    plots_path: str
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
        config=config
    )
    ppo_agent.load(checkpoint_path)
    print(f"PPO agent loaded from {checkpoint_path}")
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
    
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    improvement = (baseline_results['mean_cost'] - ppo_results['mean_cost']) / baseline_results['mean_cost'] * 100
    print(f"\nPPO Improvement over Baseline: {improvement:.2f}%")

    print("\nGenerating comparison plots...")
    comparison_path = os.path.join(plots_path, 'comparison.png')
    plot_comparison(ppo_results, baseline_results, save_path=comparison_path)
    print("\nGenerating policy heatmaps...")
    heatmap_path = os.path.join(plots_path, 'heatmap.png')
    plot_heatmap(ppo_agent, env, save_path=heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")

    eval_results = {
        'ppo': ppo_results,
        'baseline': baseline_results,
        'improvement': improvement,
    }
    return eval_results
