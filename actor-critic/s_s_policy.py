"""
Classical (s, S) Inventory Policy

This module implements the classical (s, S) reorder point policy as a baseline.
The policy works as follows:
- s: reorder point (trigger level)
- S: order-up-to level (target level)
- When inventory falls below s, order enough to reach S

The policy parameters are tuned using grid search or optimization
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

WAREHOUSES = ['middle', 'leaf1', 'leaf2']


class sSPolicy:
    """
    (s, S) Inventory Policy
    
    For each warehouse:
    - s: reorder point
    - S: order-up-to level
    - When IOH + pipeline inventory < s, order (S - IOH - pipeline)
    """

    def __init__(self, policy_params: Optional[Dict]):
        self.policy_params = policy_params
    
    def _parse_state(self, state: np.ndarray):
        """Parse state vector into structured dictionary"""

        self.state_dict = {}

        for i, wh in enumerate(WAREHOUSES):
            idx = i * 4
            self.state_dict[wh] = {
                'ioh': state[idx],
                'oldest_order_qty': state[idx + 1],
                'days_since_oldest': state[idx + 2],
                'total_reorder': state[idx + 3]
            }
    
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
        closest_idx = np.argmin((np.array(available_qtys) - desired_qty) ** 2)

        if warehouse == 'middle':
            return 1 + closest_idx
        elif warehouse == 'leaf1':
            return 5 + closest_idx
        else:
            return 9 + closest_idx
    
    def select_action(
        self, 
        state: np.ndarray, 
        reorder_quantities: Dict
    ) -> int:
        """
        Select actions based on (s, S) policy

        Args:
            state: Current state [12-dim]
            reorder_quantities: Available reorder quantities per warehouse
        
        Returns:
            action: Integer in [0, 12]
        """

        candidates = []
        self._parse_state(state)

        for warehouse in WAREHOUSES:
            inventory_pos = self.state_dict[warehouse]['ioh'] + self.state_dict[warehouse]['total_reorder']
            s = self.policy_params[warehouse]['s']
            S = self.policy_params[warehouse]['S']

            if inventory_pos < s:
                gap = s - inventory_pos
                candidates.append((warehouse, gap, inventory_pos, S))
        
        if candidates:
            warehouse, _, inventory_pos, S = max(candidates, key=lambda x: x[1])
            order_qty = S - inventory_pos
            action = self._map_to_action(
                warehouse, 
                order_qty,
                reorder_quantities
            )
            return action
        
        return 0


class sSPolicyTuner:
    """Tune (s, S) policy parameters using optimization"""

    def __init__(self, env, n_eval_episodes):
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.reorder_quantities = env.reorder_quantities
        self.bounds = [
            (5000, 20000),
            (10000, 30000),
            (2000, 10000),
            (4000, 15000),
            (2000, 10000),
            (4000, 15000),
        ]
    
    def evaluate_params(self, params: np.ndarray) -> float:
        """
        Evaluate policy parameters
        
        Args:
            params: Flattened array of [s_middle, S_middle, s_leaf1, S_leaf1, s_leaf2, S_leaf2]
        
        Returns:
            Average total cost (to minimize)
        """

        policy_params = {
            'middle': {'s': params[0], 'S': params[1]},
            'leaf1': {'s': params[2], 'S': params[3]},
            'leaf2': {'s': params[4], 'S': params[5]},
        }
        policy = sSPolicy(policy_params=policy_params)
        total_costs = []

        for _ in range(self.n_eval_episodes):
            state = self.env.reset()
            episode_cost = 0
            done = False

            while not done:
                action = policy.select_action(state=state, reorder_quantities=self.reorder_quantities)
                state, reward, done, _ = self.env.step(action)
                episode_cost -= reward
            
            total_costs.append(episode_cost)
        
        return np.mean(total_costs)

    def tune(
        self, 
        maxiter: int, 
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

        print("Tuning (s, S) policy parameters...")
        print(f"Bounds: {self.bounds}")
        result = differential_evolution(
            self.evaluate_params,
            self.bounds,
            maxiter=maxiter,
            seed=seed,
            disp=True,
            workers=1,
            updating='deferred',
            polish=True
        )

        best_params = {
            'middle': {'s': result.x[0], 'S': result.x[1]},
            'leaf1': {'s': result.x[2], 'S': result.x[3]},
            'leaf2': {'s': result.x[4], 'S': result.x[5]}
        }

        print(f"\nBest parameters found:")
        for wh, params in best_params.items():
            print(f" {wh}: s={params['s']:.0f}, S={params['S']:.0f}")
        print(f"Best average cost: {result.fun:.2f}")

        return best_params, result.fun
