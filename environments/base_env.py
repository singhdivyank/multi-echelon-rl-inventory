"""
Base Gym-style environment for multi-echelon inventory management
Follows the MDP formulation from the paper
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional


class BaseInventoryEnv(gym.Env):
    """
    Base environment for multi-echelon inventory optimization.
    
    State space includes:
    - Inventory levels (warehouse and retailers)
    - Backorders
    - In-transit quantities
    - Outstanding orders
    
    Action space:
    - Order quantities for each edge in the network
    
    Reward:
    - Negative cost (holding + backorder costs)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.structure = config['structure']
        
        # Time step
        self.current_step = 0
        self.max_steps = config.get('max_steps', 1000)
        self.warmup_steps = config.get('warmup_steps', 100)
        
        # Cost parameters
        self.h_w = config['h_w']  # Warehouse holding cost
        self.h_r = config['h_r']  # Retailer holding cost
        self.b_w = config['b_w']  # Warehouse backorder cost
        self.b_r = config['b_r']  # Retailer backorder cost
        
        # Demand parameters
        self.lambda_w = config['lambda_w']
        self.lambda_w_r = config['lambda_w_r']
        self.lead_time_dist = lambda: np.random.poisson(np.random.uniform(5, 15))
        
        # Network structure
        self.K = config['K']  # Number of stock points
        
        # State bounds
        self.max_inventory_warehouse = config['max_inventory_warehouse']
        self.max_inventory_buffer = config['max_inventory_buffer']
        self.max_in_transit = config['max_in_transit_w_r']
        self.max_backorder = config['max_backorder_r']
        self.max_outstanding = config['max_outstanding_w']
        self.max_orders = config['max_orders_w']
        
        # Initialize state components
        self.inventory_warehouse = 0
        self.inventory_retailers = None
        self.backorders_warehouse = None
        self.backorders_retailers = None
        self.in_transit = None
        self.outstanding_orders = None
        
        # Define observation and action spaces (to be overridden by subclasses)
        self.observation_space = None
        self.action_space = None
        
        # Episode tracking
        self.episode_costs = []
        self.episode_demands = []
        
    def _get_state_dimension(self) -> int:
        """Calculate state dimension based on network structure"""
        raise NotImplementedError
        
    def _get_action_dimension(self) -> int:
        """Calculate action dimension based on network structure"""
        raise NotImplementedError
        
    def _initialize_state(self):
        """Initialize state variables"""
        raise NotImplementedError
        
    def _compute_cost(self) -> float:
        """
        Compute instantaneous cost: holding costs + backorder costs
        As per equation (3) in the paper
        """
        # Warehouse costs
        warehouse_holding = self.h_w * max(0, self.inventory_warehouse)
        warehouse_backorder = self.b_w * max(0, -self.inventory_warehouse)
        
        # Retailer costs
        retailer_holding = np.sum(self.h_r * np.maximum(0, self.inventory_retailers))
        retailer_backorder = np.sum(self.b_r * np.maximum(0, -self.inventory_retailers))
        
        total_cost = (warehouse_holding + warehouse_backorder + 
                     retailer_holding + retailer_backorder)
        
        return total_cost
        
    def _generate_demand(self) -> Tuple[float, np.ndarray]:
        """Generate stochastic demand for warehouse and retailers"""
        # Warehouse demand
        if isinstance(self.lambda_w, (int, float)):
            warehouse_demand = np.random.poisson(self.lambda_w)
        else:
            warehouse_demand = self.lambda_w  # For uniform case
            
        # Retailer demands
        if isinstance(self.lambda_w_r, (int, float)):
            retailer_demands = np.random.poisson(self.lambda_w_r, 
                                                 size=len(self.inventory_retailers))
        else:
            retailer_demands = np.full(len(self.inventory_retailers), self.lambda_w_r)
            
        return warehouse_demand, retailer_demands
        
    def _generate_lead_time(self) -> int:
        """Generate stochastic lead time"""
        return self.lead_time_dist()
        
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range"""
        # Simple normalization using max bounds
        normalized = state / np.array([
            self.max_inventory_warehouse,
            self.max_inventory_buffer,
            self.max_in_transit,
            self.max_backorder,
            self.max_outstanding,
        ] * (len(state) // 5 + 1))[:len(state)]
        
        return np.clip(normalized, -1, 1)
        
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to actual order quantities"""
        # Scale from [-1, 1] to [0, max_orders]
        denormalized = (action + 1) * self.max_orders / 2
        return np.clip(denormalized, 0, self.max_orders)
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Returns:
            state: Initial state observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.episode_costs = []
        self.episode_demands = []
        
        # Initialize state
        self._initialize_state()
        
        state = self._get_observation()
        info = {}
        
        return state, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: Order quantities (normalized to [-1, 1])
            
        Returns:
            next_state: Next state observation
            reward: Reward (negative cost)
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional information
        """
        # Denormalize action
        orders = self._denormalize_action(action)
        
        # Execute the 5 events as described in the paper
        self._event_1_receive_shipments()
        self._event_2_supplier_shipment(orders)
        self._event_3_fulfill_retailer_orders()
        self._event_4_transport_orders(orders)
        self._event_5_customer_demand()
        
        # Compute cost and reward
        cost = self._compute_cost()
        reward = -cost  # Minimize cost = maximize negative cost
        
        self.episode_costs.append(cost)
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next state
        next_state = self._get_observation()
        
        # Info
        info = {
            'cost': cost,
            'step': self.current_step,
            'inventory_warehouse': self.inventory_warehouse,
            'inventory_retailers': self.inventory_retailers.copy(),
        }
        
        return next_state, reward, terminated, truncated, info
        
    def _event_1_receive_shipments(self):
        """Event 1: Shipment of upstream stock points received"""
        raise NotImplementedError
        
    def _event_2_supplier_shipment(self, orders: np.ndarray):
        """Event 2: Supplier sends outgoing shipment to warehouse"""
        raise NotImplementedError
        
    def _event_3_fulfill_retailer_orders(self):
        """Event 3: Retailer fulfills orders from retailers with on-hand inventory"""
        raise NotImplementedError
        
    def _event_4_transport_orders(self, orders: np.ndarray):
        """Event 4: Transport orders with stochastic lead time"""
        raise NotImplementedError
        
    def _event_5_customer_demand(self):
        """Event 5: Customer demand fulfilled at retailer or backlogged"""
        raise NotImplementedError
        
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        raise NotImplementedError
        
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Warehouse Inventory: {self.inventory_warehouse:.2f}")
            print(f"Retailer Inventories: {self.inventory_retailers}")
            print(f"Current Cost: {self._compute_cost():.2f}")
            
    def close(self):
        """Clean up environment"""
        pass
        
    def seed(self, seed: Optional[int] = None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)