"""
Training loop for PPO agent
"""
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm
import os

from utils.logger import Logger


def train_ppo(
    agent,
    env,
    num_iterations: int,
    logger: Optional[Logger] = None,
    save_dir: Optional[str] = None,
    log_interval: int = 100,
    save_interval: int = 1000,
    eval_interval: int = 500,
    eval_episodes: int = 10,
) -> Dict:
    """
    Train PPO agent
    
    Args:
        agent: PPO agent
        env: Environment
        num_iterations: Number of training iterations
        logger: Logger instance
        save_dir: Directory to save checkpoints
        log_interval: Interval for logging
        save_interval: Interval for saving checkpoints
        eval_interval: Interval for evaluation
        eval_episodes: Number of evaluation episodes
        
    Returns:
        Training statistics
    """
    print(f"\nStarting PPO training for {num_iterations} iterations...")
    
    # Training statistics
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
    
    # Set agent to training mode
    agent.train()
    
    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    iteration = 0
    episode = 0
    best_eval_cost = float('inf')
    
    progress_bar = tqdm(total=num_iterations, desc="Training")
    
    while iteration < num_iterations:
        # Run one episode
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action
            action, log_prob, value = agent.get_action(state, deterministic=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
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
            
            # Update if buffer is full
            if len(agent.buffer) >= agent.buffer_size:
                update_stats = agent.update()
                
                if update_stats:
                    for key, value in update_stats.items():
                        if key in stats:
                            stats[key].append(value)
                    
                    if logger:
                        logger.log_metrics(update_stats, prefix='train', step=iteration)
                
                iteration += 1
                progress_bar.update(1)
                
                # Check if we've reached the iteration limit
                if iteration >= num_iterations:
                    done = True
                    break
        
        # Episode finished
        episode += 1
        stats['episode_costs'].append(episode_cost)
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['iterations'].append(iteration)
        
        if logger:
            logger.log_episode(
                episode=episode,
                episode_reward=episode_reward,
                episode_cost=episode_cost,
                episode_length=episode_length
            )
        
        # Logging
        if episode % log_interval == 0:
            recent_costs = stats['episode_costs'][-log_interval:]
            recent_rewards = stats['episode_rewards'][-log_interval:]
            
            print(f"\nEpisode {episode} (Iteration {iteration}):")
            print(f"  Avg Cost (last {log_interval}): {np.mean(recent_costs):.2f}")
            print(f"  Avg Reward (last {log_interval}): {np.mean(recent_rewards):.2f}")
            
            if stats['policy_loss']:
                print(f"  Policy Loss: {stats['policy_loss'][-1]:.4f}")
                print(f"  Value Loss: {stats['value_loss'][-1]:.4f}")
        
        # Evaluation
        if iteration % eval_interval == 0 and iteration > 0:
            eval_cost, eval_reward = evaluate_policy(
                agent, env, num_episodes=eval_episodes
            )
            
            stats['eval_costs'].append(eval_cost)
            stats['eval_rewards'].append(eval_reward)
            
            print(f"\nEvaluation at iteration {iteration}:")
            print(f"  Avg Cost: {eval_cost:.2f}")
            print(f"  Avg Reward: {eval_reward:.2f}")
            
            if logger:
                logger.log_scalar('eval/cost', eval_cost, iteration)
                logger.log_scalar('eval/reward', eval_reward, iteration)
            
            # Save best model
            if save_dir and eval_cost < best_eval_cost:
                best_eval_cost = eval_cost
                best_path = os.path.join(save_dir, 'ppo_divergent_best.pt')
                agent.save(best_path)
                print(f"  New best model saved! Cost: {eval_cost:.2f}")
        
        # Save checkpoint
        if save_dir and iteration % save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(save_dir, f'ppo_divergent_iter_{iteration}.pt')
            agent.save(checkpoint_path)
    
    progress_bar.close()
    
    print("\nTraining completed!")
    print(f"Total iterations: {iteration}")
    print(f"Total episodes: {episode}")
    print(f"Final avg cost (last 100): {np.mean(stats['episode_costs'][-100:]):.2f}")
    
    if logger:
        logger.close()
    
    return stats


def evaluate_policy(agent, env, num_episodes: int = 10) -> tuple:
    """
    Evaluate policy during training
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of episodes
        
    Returns:
        Tuple of (avg_cost, avg_reward)
    """
    agent.eval()
    
    costs = []
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
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