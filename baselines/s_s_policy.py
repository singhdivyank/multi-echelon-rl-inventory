"""
Classical (s, S) Inventory Policy

This module implements the classical (s, S) reorder point policy as a baseline.
The policy works as follows:
- s: reorder point (trigger level)
- S: order-up-to level (target level)
- When inventory falls below s, order enough to reach S

The policy parameters are tuned using grid search or optimization.

Author: AI/ML Engineering Team
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import differential_evolution


class sSPolicy:
    """
    (s, S) Inventory Policy
    
    For each warehouse:
    - s: reorder point
    - S: order-up-to level
    - When IOH + pipeline inventory < s, order (S - IOH - pipeline)
    """
    
    def __init__(self, policy_params: Optional[Dict] = None):
        """
        Initialize (s, S) policy
        
        Args:
            policy_params: Dictionary with keys 'middle', 'leaf_1', 'leaf_2'
                          Each value is a dict with 's' and 'S' parameters
        """
        if policy_params is None:
            # Default parameters (will need tuning)
            self.policy_params = {
                'middle': {'s': 8000, 'S': 15000},
                'leaf_1': {'s': 4000, 'S': 8000},
                'leaf_2': {'s': 4000, 'S': 8000},
            }
        else:
            self.policy_params = policy_params
    
    def select_action(
        self,
        state: np.ndarray,
        reorder_quantities: Dict
    ) -> int:
        """
        Select action based on (s, S) policy
        
        Args:
            state: Current state [12-dim]
            reorder_quantities: Available reorder quantities per warehouse
        
        Returns:
            action: Integer in [0, 12]
        """
        # Parse state
        state_dict = self._parse_state(state)
        
        # Check each warehouse and decide which one should order
        for warehouse in ['middle', 'leaf_1', 'leaf_2']:
            ioh = state_dict[warehouse]['ioh']
            pipeline = state_dict[warehouse]['total_reorder']
            inventory_position = ioh + pipeline
            
            s = self.policy_params[warehouse]['s']
            S = self.policy_params[warehouse]['S']
            
            # Check if we should order
            if inventory_position < s:
                # Calculate order quantity
                order_qty = S - inventory_position
                
                # Find closest available reorder quantity
                action = self._map_to_action(
                    warehouse,
                    order_qty,
                    reorder_quantities
                )
                return action
        
        # No warehouse needs to order
        return 0
    
    def _parse_state(self, state: np.ndarray) -> Dict:
        """
        Parse state vector into structured dictionary
        
        State format: For each warehouse [IOH, oldest_qty, days_waiting, total_reorder]
        """
        state_dict = {}
        warehouses = ['middle', 'leaf_1', 'leaf_2']
        
        for i, wh in enumerate(warehouses):
            idx = i * 4
            state_dict[wh] = {
                'ioh': state[idx],
                'oldest_order_qty': state[idx + 1],
                'days_since_oldest': state[idx + 2],
                'total_reorder': state[idx + 3],
            }
        
        return state_dict
    
    def _map_to_action(
        self,
        warehouse: str,
        desired_qty: float,
        reorder_quantities: Dict
    ) -> int:
        """
        Map desired order quantity to closest available action
        
        Args:
            warehouse: Which warehouse is ordering
            desired_qty: Desired order quantity
            reorder_quantities: Available quantities
        
        Returns:
            action: Mapped action
        """
        available_qtys = reorder_quantities[warehouse]
        
        # Find closest quantity
        closest_idx = np.argmin([abs(q - desired_qty) for q in available_qtys])
        
        # Map to action
        if warehouse == 'middle':
            return 1 + closest_idx
        elif warehouse == 'leaf_1':
            return 5 + closest_idx
        elif warehouse == 'leaf_2':
            return 9 + closest_idx
        
        return 0


class sSPolicyTuner:
    """
    Tune (s, S) policy parameters using optimization
    """
    
    def __init__(self, env, n_eval_episodes: int = 10):
        """
        Args:
            env: MEIS environment
            n_eval_episodes: Number of episodes for evaluation
        """
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.reorder_quantities = env.reorder_quantities
    
    def evaluate_params(self, params: np.ndarray) -> float:
        """
        Evaluate policy parameters
        
        Args:
            params: Flattened array of [s_middle, S_middle, s_leaf1, S_leaf1, s_leaf2, S_leaf2]
        
        Returns:
            Average total cost (to minimize)
        """
        # Unpack parameters
        policy_params = {
            'middle': {'s': params[0], 'S': params[1]},
            'leaf_1': {'s': params[2], 'S': params[3]},
            'leaf_2': {'s': params[4], 'S': params[5]},
        }
        
        # Create policy
        policy = sSPolicy(policy_params)
        
        # Evaluate over multiple episodes
        total_costs = []
        
        for episode in range(self.n_eval_episodes):
            state = self.env.reset()
            episode_cost = 0
            done = False
            
            while not done:
                action = policy.select_action(state, self.reorder_quantities)
                state, reward, done, info = self.env.step(action)
                episode_cost -= reward  # Convert reward back to cost
            
            total_costs.append(episode_cost)
        
        return np.mean(total_costs)
    
    def tune(
        self,
        bounds: Optional[list] = None,
        maxiter: int = 100,
        seed: Optional[int] = None
    ) -> Tuple[Dict, float]:
        """
        Tune policy parameters using differential evolution
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            maxiter: Maximum iterations for optimization
            seed: Random seed
        
        Returns:
            best_params: Best found parameters
            best_cost: Best average cost
        """
        if bounds is None:
            # Default bounds based on environment config
            bounds = [
                (5000, 20000),   # s_middle
                (10000, 30000),  # S_middle
                (2000, 10000),   # s_leaf1
                (4000, 15000),   # S_leaf1
                (2000, 10000),   # s_leaf2
                (4000, 15000),   # S_leaf2
            ]
        
        print("Tuning (s, S) policy parameters...")
        print(f"Bounds: {bounds}")
        
        result = differential_evolution(
            self.evaluate_params,
            bounds,
            maxiter=maxiter,
            seed=seed,
            disp=True,
            workers=1,  # Use single worker for determinism
            updating='deferred',
            polish=True
        )
        
        # Extract best parameters
        best_params = {
            'middle': {'s': result.x[0], 'S': result.x[1]},
            'leaf_1': {'s': result.x[2], 'S': result.x[3]},
            'leaf_2': {'s': result.x[4], 'S': result.x[5]},
        }
        
        print(f"\nBest parameters found:")
        for wh, params in best_params.items():
            print(f"  {wh}: s={params['s']:.0f}, S={params['S']:.0f}")
        print(f"Best average cost: {result.fun:.2f}")
        
        return best_params, result.fun


def evaluate_policy(
    env,
    policy: sSPolicy,
    n_episodes: int = 100,
    seed: Optional[int] = None
) -> Dict:
    """
    Evaluate (s, S) policy on environment
    
    Args:
        env: MEIS environment
        policy: sSPolicy instance
        n_episodes: Number of evaluation episodes
        seed: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    if seed is not None:
        env.seed(seed)
    
    episode_costs = []
    episode_rewards = []
    service_levels = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_cost = 0
        episode_reward = 0
        done = False
        service_level_sum = 0
        steps = 0
        
        while not done:
            action = policy.select_action(state, env.reorder_quantities)
            state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_cost -= reward
            service_level_sum += info['service_level']
            steps += 1
        
        episode_costs.append(episode_cost)
        episode_rewards.append(episode_reward)
        service_levels.append(service_level_sum / steps)
    
    return {
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_service_level': np.mean(service_levels),
        'std_service_level': np.std(service_levels),
        'episode_costs': episode_costs,
        'service_levels': service_levels,
    }
