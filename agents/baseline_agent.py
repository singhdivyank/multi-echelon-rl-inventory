"""
Baseline agent implementing the Divergent policy
Simple rule-based policy for comparison with PPO
"""
import numpy as np
from typing import Dict, Optional


class BaselineAgent:
    """
    Baseline agent using simple inventory control policy
    
    Policy:
    - Order up to target inventory level
    - Target = base stock level (s, S policy)
    - Simple heuristic without learning
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        """
        Initialize baseline agent
        
        Args:
            env: Environment instance
            config: Configuration dictionary
        """
        self.env = env
        self.config = config or {}
        
        # Get action and state dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Base stock levels (to be tuned or set heuristically)
        self._initialize_base_stock_levels()
        
        # Statistics tracking
        self.episode_costs = []
        self.episode_rewards = []
        
    def _initialize_base_stock_levels(self):
        """
        Initialize base stock levels using heuristic
        
        Simple heuristic:
        - Target inventory = average demand * lead time + safety stock
        - Safety stock = z * sqrt(lead_time) * std(demand)
        """
        # Get environment parameters
        if hasattr(self.env, 'lambda_w_r'):
            avg_demand = self.env.lambda_w_r
        else:
            avg_demand = 1.0  # Default
            
        # Estimate average lead time (simplified)
        avg_lead_time = 10  # Approximate from config
        
        # Safety factor (z-score for 95% service level)
        z_score = 1.65
        
        # Calculate base stock levels
        safety_stock = z_score * np.sqrt(avg_demand) * np.sqrt(avg_lead_time)
        
        # Warehouse base stock
        self.base_stock_warehouse = avg_demand * avg_lead_time + safety_stock
        
        # Retailer base stocks (one per retailer)
        num_retailers = self.action_dim - 1
        self.base_stock_retailers = np.ones(num_retailers) * (
            avg_demand * avg_lead_time * 0.5 + safety_stock * 0.8
        )
        
        # Adjust based on environment max values
        self.base_stock_warehouse = min(
            self.base_stock_warehouse, 
            self.env.max_inventory_warehouse * 0.6
        )
        self.base_stock_retailers = np.minimum(
            self.base_stock_retailers,
            self.env.max_inventory_buffer * 0.5
        )
        
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get action using base stock policy
        
        Args:
            state: Current state (normalized)
            deterministic: Not used for baseline (always deterministic)
            
        Returns:
            action: Order quantities (normalized to [-1, 1])
        """
        # Denormalize state to get actual inventory levels
        denormalized_state = self._denormalize_state(state)
        
        # Extract inventory levels from state
        # State structure: [I_w, I_r1, I_r2, I_r3, BO_r1, BO_r2, BO_r3, IT_..., O_...]
        num_retailers = self.action_dim - 1
        
        inventory_warehouse = denormalized_state[0]
        inventory_retailers = denormalized_state[1:1+num_retailers]
        
        # Calculate inventory position (on-hand + on-order)
        # Simplified: just use on-hand inventory
        inv_position_warehouse = inventory_warehouse
        inv_position_retailers = inventory_retailers
        
        # Calculate order quantities using base stock policy
        # Order = max(0, base_stock - inventory_position)
        
        order_warehouse = max(0, self.base_stock_warehouse - inv_position_warehouse)
        orders_retailers = np.maximum(
            0, 
            self.base_stock_retailers - inv_position_retailers
        )
        
        # Combine orders
        orders = np.concatenate([[order_warehouse], orders_retailers])
        
        # Normalize to [-1, 1] for environment
        normalized_orders = self._normalize_action(orders)
        
        return normalized_orders
        
    def _denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Denormalize state from [-1, 1] to actual values
        
        Args:
            state: Normalized state
            
        Returns:
            Denormalized state
        """
        # State bounds from environment
        max_values = np.array([
            self.env.max_inventory_warehouse,
            *[self.env.max_inventory_buffer] * (self.action_dim - 1),
            *[self.env.max_backorder] * (self.action_dim - 1),
            *[self.env.max_in_transit] * (self.action_dim - 1),
            self.env.max_outstanding,
            *[self.env.max_outstanding] * (self.action_dim - 1),
        ])[:len(state)]
        
        return (state + 1) * max_values / 2
        
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to [-1, 1] range
        
        Args:
            action: Denormalized order quantities
            
        Returns:
            Normalized action
        """
        # Scale from [0, max_orders] to [-1, 1]
        max_order = self.env.max_orders
        return np.clip((2.0 * action / max_order) - 1.0, -1.0, 1.0)
        
    def update(self, *args, **kwargs):
        """
        No learning for baseline agent
        This method exists for API compatibility with PPO agent
        """
        pass
        
    def train(self):
        """Set to training mode (no-op for baseline)"""
        pass
        
    def eval(self):
        """Set to evaluation mode (no-op for baseline)"""
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


class ImprovedBaselineAgent(BaselineAgent):
    """
    Improved baseline with adaptive base stock levels
    Uses recent demand history to adjust targets
    """
    
    def __init__(self, env, config: Optional[Dict] = None):
        super().__init__(env, config)
        
        # Demand history for adaptation
        self.demand_history = []
        self.history_window = 100
        
    def update_demand_history(self, demands: np.ndarray):
        """Update demand history for adaptive policy"""
        self.demand_history.append(demands)
        if len(self.demand_history) > self.history_window:
            self.demand_history.pop(0)
            
    def _adapt_base_stock_levels(self):
        """Adapt base stock levels based on recent demand"""
        if len(self.demand_history) < 10:
            return
            
        recent_demands = np.array(self.demand_history[-50:])
        avg_demand = np.mean(recent_demands)
        std_demand = np.std(recent_demands)
        
        # Update base stock levels
        avg_lead_time = 10
        z_score = 1.65
        
        safety_stock = z_score * np.sqrt(avg_lead_time * std_demand)
        self.base_stock_warehouse = avg_demand * avg_lead_time + safety_stock
        
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action with adaptive base stock"""
        self._adapt_base_stock_levels()
        return super().get_action(state, deterministic)