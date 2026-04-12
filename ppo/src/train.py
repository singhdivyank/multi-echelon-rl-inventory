"""
Training Loop for PPO Agent

Implements the training procedure with:
- Episode rollouts
- Advantage computation
- Policy updates
- Logging and checkpointing
"""

import os

from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

import numpy as np

from models.baseline import BaselineAgent
from models.env import DivergentInventoryEnv
from models.ppo import PPOAgent
from utils.logger import Logger

def train_ppo(
    agent, 
    env, 
    num_iterations: int, 
    logger: Logger,
    log_interval: int,
    save_interval: int,
    eval_interval: int,
    eval_episodes: int,
    best_model_path: str
) -> Dict:
    """
    Train PPO agent
    
    Args:
        agent: PPO agent
        env: Environment
        num_iterations: Number of training iterations
        logger: Logger instance
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints
        eval_interval: Interval for evaluation
        eval_episodes: Number of evaluation episodes
        
    Returns:
        Training statistics
    """

    from src.evaluate import evaluate_policy

    print(f"\nStarting PPO training for {num_iterations} iterations...")
    stats = {
        'iterations': [],
        'episode_costs': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'total_loss': [],
        'kl_divergence': [],
        'clip_fraction': [],
        'eval_costs': [],
        'eval_rewards': [],
    }

    agent.train()
    iteration, episode = 0, 0
    best_eval_cost = float('inf')
    progress_bar = tqdm(total=num_iterations, desc="Training")

    while iteration < num_iterations:
        state, _ = env.reset()
        episode_reward, episode_cost = 0, 0
        episode_length, done = 0, False

        while not done:
            action, log_prob, value = agent.get_action(state, deterministic=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(
                state=state, 
                action=action, 
                reward=reward, 
                next_state=next_state, 
                done=done, 
                log_prob=log_prob, 
                value=value
            )

            episode_reward += reward
            episode_cost += info.get('cost', 0)
            episode_length += 1
            state = next_state

            if len(agent.buffer) >= agent.buffer_size:
                update_stats = agent.update()
                if update_stats:
                    for key, value in update_stats.items():
                        if key in stats:
                            stats[key].append(value)
                    
                    logger.log_metrics(update_stats, prefix='train', step=iteration)
                
                iteration += 1
                progress_bar.update(1)

                if iteration >= num_iterations:
                    done = True
                    break
        
        episode += 1
        stats['episode_costs'].append(episode_cost)
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['iterations'].append(iteration)
        logger.log_episode(
            episode=episode,
            episode_cost=episode_cost,
            episode_length=episode_length,
            episode_reward=episode_reward
        )

        if not episode % log_interval:
            recent_costs = stats['episode_costs'][-log_interval:]
            recent_rewards = stats['episode_rewards'][-log_interval:]

            print(f"\nEpisode {episode} (Iteration {iteration}):")
            print(f"  Avg Cost (last {log_interval}): {np.mean(recent_costs):.2f}")
            print(f"  Avg Reward (last {log_interval}): {np.mean(recent_rewards):.2f}")

            if stats['policy_loss']:
                print(f"  Policy Loss: {stats['policy_loss'][-1]:.4f}")
                print(f"  Value Loss: {stats['value_loss'][-1]:.4f}")
        
        if iteration > 0 and not iteration % eval_interval:
            eval_cost, eval_reward = evaluate_policy(
                agent=agent,
                env=env,
                num_episodes=eval_episodes
            )

            stats['eval_costs'].append(eval_cost)
            stats['eval_rewards'].append(eval_reward)

            print(f"\nEvaluation at iteration {iteration}:")
            print(f"  Avg Cost: {eval_cost:.2f}")
            print(f"  Avg Reward: {eval_reward:.2f}")

            logger.log_scalars('eval/cost', {'cost': float(eval_cost)}, iteration)
            logger.log_scalars('eval/reward', {'reward': float(eval_reward)}, iteration)

            if eval_cost < best_eval_cost:
                best_eval_cost = eval_cost
                agent.save(best_model_path)
                print(f"  New best model saved! Cost: {eval_cost:.2f}")
        
        if iteration > 0 and not iteration % save_interval:
            model_path_base = os.path.basename(best_model_path)
            checkpoint_path = os.path.join(model_path_base, f'ppo_divergent_iter_{iteration}.pt')
            agent.save(checkpoint_path)
    
    progress_bar.close()
    
    print("\nTraining completed!")
    print(f"Total iterations: {iteration}")
    print(f"Total episodes: {episode}")
    print(f"Final avg cost (last 100): {np.mean(stats['episode_costs'][-100:]):.2f}")

    logger.close()
    return stats

def train_agents(
    env_config: Dict, 
    config: Dict, 
    save_dirs: List[str],
    device: str,
    best_model_path: str
) -> Dict:
    """Train PPO and Baseline agents"""

    from src.evaluate import evaluate_baseline

    print("="*80)
    print("Training Divergent Inventory Optimization")
    print("="*80)

    # Create environment
    env = DivergentInventoryEnv(config=env_config)
    print(f"\nEnvironment created:")
    print(f"  State dimension: {env.observation_space.shape[0]}")
    print(f"  Action dimension: {env.action_space.shape[0]}")
    print(f"  Max steps per episode: {env.max_steps}")

    # Create Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(save_dirs[2], f'divergent_{timestamp}')
    logger = Logger(log_dir)

    # Train PPO agent
    print("\n" + "="*80)
    print("Training PPO Agent")
    print("="*80)

    ppo_agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config,
        device=device,
    )
    ppo_stats = train_ppo(
        agent=ppo_agent,
        env=env,
        num_iterations=config['iterations'],
        logger=logger,
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        eval_interval=config['eval_interval'],
        eval_episodes=config['eval_episodes'],
        best_model_path=best_model_path
    )
    ppo_save_path = os.path.join(save_dirs[0], 'ppo_divergent_final.pt')
    ppo_agent.save(ppo_save_path)
    print(f"\nPPO agent saved to {ppo_save_path}")

    print("\n" + "="*80)
    print("Evaluating Baseline Agent")
    print("="*80)

    baseline_agent = BaselineAgent(env)
    baseline_stats = evaluate_baseline(
        agent=baseline_agent,
        env=env,
        num_episodes=config['eval_episodes'],
        logger=logger
    )
    baseline_save_path = os.path.join(save_dirs[0], 'baseline_divergent.npy')
    baseline_agent.save(baseline_save_path)

    # Generate stats
    stats = {
        'ppo': ppo_stats,
        'baseline': baseline_stats,
        'config': config,
        'env_config': {k: v for k, v in env_config.items() if not callable(v)},
    }
    return stats
