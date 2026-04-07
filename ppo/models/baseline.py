"""
Baseline agent implementing the Divergent policy
Simple rule-based policy for comparison with PPO
"""

import numpy as np


class BaselineAgent:
    """
    Baseline agent using simple inventory control policy
    
    Policy:
    - Order up to target inventory level
    - Target = base stock level (s, S policy)
    - Simple heuristic without learning
    """

    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.observation_space.shape[0]
        self._init_base_stock_levels()
        self.episode_costs = []
        self.episode_rewards = []
    
    def _init_base_stock_levels(self):
        """
        Initialize base stock levels using heuristic
        
        Simple heuristic:
            - Target inventory = average demand * lead time + safety stock
            - Safety stock = z * sqrt(lead_time) * std(demand)
        """

        avg_demand = self.env.lambda_w_r
        avg_lead_time = self.env.lead_time
        z_score = 1.65
        safety_stock = z_score * np.sqrt(avg_demand) * np.sqrt(avg_lead_time)
        self.base_stock_warehouse = avg_demand * avg_lead_time + safety_stock

        num_retailers = self.action_dim - 1
        self.base_stock_retailers = np.ones(num_retailers) * (
            avg_demand * avg_lead_time * 0.5 + safety_stock * 0.8
        )

        self.base_stock_warehouse = min(
            self.base_stock_warehouse,
            self.env.max_inventory_warehouse * 0.6
        )
        self.base_stock_retailers = np.minimum(
            self.base_stock_retailers,
            self.env.max_inventory_buffer * 0.5
        )
    
    def _denormalized_state(self, state: np.ndarray) -> np.ndarray:
        """
        Denormalize state from [-1, 1] to actual values
        
        Args:
            state: Normalized state
            
        Returns:
            Denormalized state
        """

        max_values = np.array([
            self.env.max_inventory_warehouse,
            *[self.env.max_inventory_buffer] * (self.action_dim - 1),
            *[self.env.max_backorder] * (self.action_dim - 1),
            *[self.env.max_in_transit] * (self.action_dim - 1),
            self.env.max_outstanding,
            *[self.env.max_outstanding] * (self.action_dim - 1),
        ])[:len(state)]
        return (state + 1) * max_values / 2

    def _normalise_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to [-1, 1] range
        
        Args:
            action: Denormalized order quantities
            
        Returns:
            Normalized action
        """

        max_order = self.env.max_orders
        return np.clip((2.0 * action / max_order) - 1.0, -1.0, 1.0)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action using base stock policy
        
        Args:
            state: Current state (normalized)
            
        Returns:
            action: Order quantities (normalized to [-1, 1])
        """

        denormalized_state = self._denormalized_state(state)
        num_retailers = self.action_dim - 1
        inventory_warehouse = denormalized_state[0]
        inventory_retailers = denormalized_state[1: 1 + num_retailers]

        inv_position_warehouse = inventory_warehouse
        inv_position_retailers = inventory_retailers

        order_warehouse = max(0, self.base_stock_warehouse - inv_position_warehouse)
        order_retailers = np.maximum(0, self.base_stock_retailers - inv_position_retailers)
        orders = np.concatenate([[order_warehouse], order_retailers])
        normalised_orders = self._normalise_action(orders)
        return normalised_orders

    def update(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def save(self, path: str):
        """
        Save baseline parameters
        
        Args:
            path: Save path
        """

        params = {
            'base_stock_warehouse': self.base_stock_warehouse,
            'base_stock_retailers': self.base_stock_retailers,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        np.save(path, params)
    
    def load(self, path: str):
        """
        Load baseline parameters
        
        Args:
            path: Load path
        """

        params = np.load(path, allow_pickle=True).item()
        self.base_stock_warehouse = params['base_stock_warehouse']
        self.base_stock_retailers = params['base_stock_retailers']
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get state value estimate (not applicable for baseline)
        Returns 0 for API compatibility
        """

        return 0.0
