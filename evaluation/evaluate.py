"""
Evaluation utilities for trained agents
"""
import numpy as np
from typing import Dict, Tuple
from tqdm import tqdm
import gymnasium as gym

from utils.metrics import compute_confidence_interval


def evaluate_agent(
    agent,
    env: gym.Env,
    num_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate an agent on an environment
    
    Args:
        agent: Agent to evaluate (PPO or Baseline)
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        render: Whether to render environment
        verbose: Whether to print progress
        
    Returns:
        Dictionary of evaluation metrics
    """
    agent.eval()
    
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    
    # Track detailed metrics
    inventory_levels = []
    backorder_levels = []
    order_quantities = []
    
    iterator = tqdm(range(num_episodes), desc="Evaluating") if verbose else range(num_episodes)
    
    for episode in iterator:
        state, _ = env.reset(seed=episode)
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        done = False
        
        episode_inventories = []
        episode_backorders = []
        episode_orders = []
        
        while not done:
            # Get action from agent
            # PPOAgent.get_action returns (action, log_prob, value);
            # BaselineAgent.get_action returns just action
            from agents.ppo_agent import PPOAgent
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.get_action(state, deterministic=deterministic)
            else:
                action = agent.get_action(state, deterministic=deterministic)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_cost += info.get('cost', 0)
            episode_length += 1
            
            # Track metrics
            if 'inventory_warehouse' in info:
                episode_inventories.append(info['inventory_warehouse'])
            if 'inventory_retailers' in info:
                episode_backorders.append(np.sum(np.maximum(0, -info['inventory_retailers'])))
            episode_orders.append(action)
            
            state = next_state
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        
        inventory_levels.append(np.mean(episode_inventories) if episode_inventories else 0)
        backorder_levels.append(np.mean(episode_backorders) if episode_backorders else 0)
        order_quantities.append(np.mean([np.abs(a).sum() for a in episode_orders]))
    
    # Compute statistics
    episode_rewards = np.array(episode_rewards)
    episode_costs = np.array(episode_costs)
    episode_lengths = np.array(episode_lengths)
    
    # Compute confidence intervals
    mean_reward, ci_reward_lower, ci_reward_upper = compute_confidence_interval(episode_rewards)
    mean_cost, ci_cost_lower, ci_cost_upper = compute_confidence_interval(episode_costs)
    
    results = {
        # Rewards
        'mean_reward': mean_reward,
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'ci_reward_lower': ci_reward_lower,
        'ci_reward_upper': ci_reward_upper,
        
        # Costs
        'mean_cost': mean_cost,
        'std_cost': np.std(episode_costs),
        'min_cost': np.min(episode_costs),
        'max_cost': np.max(episode_costs),
        'median_cost': np.median(episode_costs),
        'ci_lower': ci_cost_lower,
        'ci_upper': ci_cost_upper,
        
        # Episode lengths
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        
        # Detailed metrics
        'mean_inventory': np.mean(inventory_levels),
        'mean_backorders': np.mean(backorder_levels),
        'mean_orders': np.mean(order_quantities),
        
        # Raw data
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_lengths': episode_lengths,
    }
    
    if verbose:
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(f"Episodes: {num_episodes}")
        print(f"\nReward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  95% CI: [{ci_reward_lower:.2f}, {ci_reward_upper:.2f}]")
        print(f"\nCost: {mean_cost:.2f} ± {np.std(episode_costs):.2f}")
        print(f"  95% CI: [{ci_cost_lower:.2f}, {ci_cost_upper:.2f}]")
        print(f"\nEpisode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"\nInventory Level: {np.mean(inventory_levels):.2f}")
        print(f"Backorder Level: {np.mean(backorder_levels):.2f}")
        print(f"Order Quantity: {np.mean(order_quantities):.2f}")
        print("="*80)
    
    return results


def compare_agents(
    agents: Dict[str, any],
    env: gym.Env,
    num_episodes: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Compare multiple agents
    
    Args:
        agents: Dictionary of {agent_name: agent}
        env: Environment
        num_episodes: Number of episodes per agent
        verbose: Whether to print results
        
    Returns:
        Dictionary of results for each agent
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            deterministic=True,
            verbose=verbose
        )
    
    # Print comparison
    if verbose:
        print("\n" + "="*80)
        print("Agent Comparison")
        print("="*80)
        print(f"{'Agent':<20} {'Mean Cost':<15} {'95% CI':<25} {'Improvement':<15}")
        print("-"*80)
        
        baseline_cost = None
        for name in agents.keys():
            if 'baseline' in name.lower():
                baseline_cost = results[name]['mean_cost']
                break
        
        for name, result in results.items():
            mean_cost = result['mean_cost']
            ci_lower = result['ci_lower']
            ci_upper = result['ci_upper']
            
            if baseline_cost and name.lower() != 'baseline':
                improvement = (baseline_cost - mean_cost) / baseline_cost * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement_str = "-"
            
            print(f"{name:<20} {mean_cost:<15.2f} [{ci_lower:.2f}, {ci_upper:.2f}]    {improvement_str:<15}")
        
        print("="*80)
    
    return results


def evaluate_policy_heatmap(
    agent,
    env: gym.Env,
    state_indices: Tuple[int, int] = (0, 1),
    num_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for policy heatmap visualization
    
    Args:
        agent: Agent to evaluate
        env: Environment
        state_indices: Which state dimensions to vary (e.g., warehouse, retailer inventory)
        num_points: Number of grid points per dimension
        
    Returns:
        X: Grid of first state dimension
        Y: Grid of second state dimension
        actions: Mean action at each grid point
    """
    agent.eval()
    
    # Get a reference state
    state, _ = env.reset()
    
    # Create grid
    state_min = -1.0
    state_max = 1.0
    
    x = np.linspace(state_min, state_max, num_points)
    y = np.linspace(state_min, state_max, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate policy at each grid point
    actions = np.zeros((num_points, num_points, env.action_space.shape[0]))
    
    for i in range(num_points):
        for j in range(num_points):
            # Create state with grid values
            test_state = state.copy()
            test_state[state_indices[0]] = X[i, j]
            test_state[state_indices[1]] = Y[i, j]
            
            # Get action
            from agents.ppo_agent import PPOAgent
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.get_action(test_state, deterministic=True)
            else:
                action = agent.get_action(test_state, deterministic=True)
            
            actions[i, j] = action
    
    return X, Y, actions


def monte_carlo_evaluation(
    agent,
    env: gym.Env,
    num_episodes: int = 1000,
    confidence_level: float = 0.95
) -> Dict:
    """
    Perform Monte Carlo evaluation with statistical analysis
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of Monte Carlo samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Detailed statistical results
    """
    print(f"\nRunning Monte Carlo evaluation with {num_episodes} episodes...")
    
    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        deterministic=True,
        verbose=False
    )
    
    # Additional statistical analysis
    costs = results['episode_costs']
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    cost_percentiles = {
        f'p{p}': np.percentile(costs, p) for p in percentiles
    }
    
    # Standard error
    se = np.std(costs) / np.sqrt(len(costs))
    
    results.update({
        'percentiles': cost_percentiles,
        'standard_error': se,
        'confidence_level': confidence_level,
    })
    
    print("\nMonte Carlo Results:")
    print(f"  Mean Cost: {results['mean_cost']:.2f}")
    print(f"  Standard Error: {se:.2f}")
    print(f"  95% CI: [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]")
    print(f"\nPercentiles:")
    for p, val in cost_percentiles.items():
        print(f"  {p}: {val:.2f}")
    
    return results