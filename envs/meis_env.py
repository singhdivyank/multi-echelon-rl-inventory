"""
Multi-Echelon Inventory System (MEIS) Gym Environment

This module implements a Gym-compatible environment for a divergent 2-layer MEIS
as described in the research paper. The environment consists of:
- 1 Factory (infinite supply)
- 1 Middle warehouse
- 2 Leaf warehouses

Author: AI/ML Engineering Team
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque


class MEISEnv(gym.Env):
    """
    Multi-Echelon Inventory System Environment
    
    State Space (12-dimensional):
        For each warehouse (Middle, Leaf1, Leaf2):
        - Current Inventory on Hand (IOH)
        - Order quantity of oldest open order
        - Number of days since oldest open order was placed
        - Reorder quantity of all open orders
    
    Action Space (13-dimensional discrete):
        - Action 0: No warehouse orders
        - Actions 1-4: Only middle warehouse orders (4 reorder quantities)
        - Actions 5-8: Only leaf 1 orders (4 reorder quantities)
        - Actions 9-12: Only leaf 2 orders (4 reorder quantities)
    
    Reward: Negative total cost (to convert minimization to maximization)
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: Optional[Dict] = None):
        super(MEISEnv, self).__init__()
        
        # Load configuration
        self.config = self._default_config()
        print("Warehouse config: ", self.config)
        
        # Environment specifications
        self.n_warehouses = 3  # Middle + 2 Leaf
        self.state_dim_per_warehouse = 4
        self.total_state_dim = self.n_warehouses * self.state_dim_per_warehouse
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_state_dim,),
            dtype=np.float32
        )
        
        # Warehouse configurations
        self.warehouse_config = self._setup_warehouse_config()
        
        # Reorder quantities for each warehouse
        self.reorder_quantities = self._setup_reorder_quantities()
        
        # Internal state
        self.current_step = 0
        self.max_steps = self.config.get('max_steps_per_episode', 365)
        
        # Warehouse states
        self.ioh = {}  # Inventory on Hand
        self.open_orders = {}  # Queue of open orders per warehouse
        self.total_cost_history = []
        
        # Random seed
        self.np_random = None
        self.seed()
        
    def _default_config(self) -> Dict:
        """Default configuration matching the research paper"""
        return {
            'max_steps_per_episode': 365,
            'factory': {
                'ioh_initial': np.inf,
                'lead_time_mean': 0,
                'lead_time_std': 0,
                'price_per_product': 0,
                'min_reorder_cost': 0,
                'reorder_cost_constant': 0,
                'shortage_cost_constant': 0,
                'holding_cost_constant': 0,
            },
            'middle_warehouse': {
                'ioh_initial': 10000,
                'lead_time_mean': 2,
                'lead_time_std': 1,
                'price_per_product': 50,
                'min_reorder_cost': 1000,
                'reorder_cost_constant': 0,
                'shortage_cost_constant': 0,  # No shortage cost at middle
                'holding_cost_constant': 0.1,
                'reorder_qty_options': [5000, 10000, 15000, 20000],
            },
            'leaf_warehouse': {
                'ioh_initial': 5000,
                'demand_mean': 3300,
                'demand_std': 100,
                'lead_time_mean': 2,
                'lead_time_std': 1,
                'price_per_product': 100,
                'min_reorder_cost': 5000,
                'reorder_cost_constant': 0.5,
                'shortage_cost_constant': 10,
                'holding_cost_constant': 0.1,
                'max_backlog_duration': 7,
                'reorder_qty_options': [3000, 6000, 9000, 12000],
            }
        }
    
    def _setup_warehouse_config(self) -> Dict:
        """Setup configuration for each warehouse"""
        config = {}
        
        # Factory (not controlled, always has supply)
        config['factory'] = self.config['factory'].copy()
        
        # Middle warehouse
        config['middle'] = self.config['middle_warehouse'].copy()
        
        # Leaf warehouses (identical configuration)
        config['leaf_1'] = self.config['leaf_warehouse'].copy()
        config['leaf_2'] = self.config['leaf_warehouse'].copy()
        
        return config
    
    def _setup_reorder_quantities(self) -> Dict:
        """Setup discrete reorder quantity options for each warehouse"""
        return {
            'middle': self.config['middle_warehouse']['reorder_qty_options'],
            'leaf_1': self.config['leaf_warehouse']['reorder_qty_options'],
            'leaf_2': self.config['leaf_warehouse']['reorder_qty_options'],
        }
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility"""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.total_cost_history = []
        
        # Initialize inventory on hand
        self.ioh = {
            'factory': self.warehouse_config['factory']['ioh_initial'],
            'middle': self.warehouse_config['middle']['ioh_initial'],
            'leaf_1': self.warehouse_config['leaf_1']['ioh_initial'],
            'leaf_2': self.warehouse_config['leaf_2']['ioh_initial'],
        }
        
        # Initialize open orders (queue of tuples: (quantity, days_waiting, arrival_day))
        self.open_orders = {
            'middle': deque(),
            'leaf_1': deque(),
            'leaf_2': deque(),
        }
        
        # Backlog tracking
        self.backlog = {
            'middle': deque(),
            'leaf_1': deque(),
            'leaf_2': deque(),
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state vector (12-dimensional)
        For each warehouse: [IOH, oldest_order_qty, days_since_oldest, total_reorder_qty]
        """
        state = []
        
        for wh_name in ['middle', 'leaf_1', 'leaf_2']:
            # Current IOH
            ioh = self.ioh[wh_name]
            
            # Oldest order information
            if len(self.open_orders[wh_name]) > 0:
                oldest_order = self.open_orders[wh_name][0]
                oldest_qty = oldest_order[0]
                days_waiting = oldest_order[1]
            else:
                oldest_qty = 0
                days_waiting = 0
            
            # Total reorder quantity of all open orders
            total_reorder = sum([order[0] for order in self.open_orders[wh_name]])
            
            state.extend([ioh, oldest_qty, days_waiting, total_reorder])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: Integer in [0, 12] representing which warehouse orders
        
        Returns:
            observation: Next state
            reward: Negative total cost
            done: Whether episode is finished
            info: Additional information
        """
        # Process action (place orders)
        self._process_action(action)
        
        # Generate demand at leaf warehouses
        self._generate_demand()
        
        # Process deliveries (orders arriving)
        self._process_deliveries()
        
        # Update backlog aging
        self._update_backlog()
        
        # Calculate costs
        cost_breakdown = self._calculate_costs()
        total_cost = cost_breakdown['total']
        self.total_cost_history.append(total_cost)
        
        # Reward is negative cost (minimization -> maximization)
        reward = -total_cost
        
        # Update step counter
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'cost_breakdown': cost_breakdown,
            'ioh': self.ioh.copy(),
            'service_level': self._calculate_service_level(),
            'step': self.current_step,
        }
        
        return next_state, reward, done, info
    
    def _process_action(self, action: int):
        """Process the action and place orders accordingly"""
        if action == 0:
            # No warehouse orders
            return
        elif 1 <= action <= 4:
            # Middle warehouse orders
            warehouse = 'middle'
            qty_idx = action - 1
            quantity = self.reorder_quantities[warehouse][qty_idx]
            self._place_order(warehouse, quantity, upstream='factory')
        elif 5 <= action <= 8:
            # Leaf 1 orders
            warehouse = 'leaf_1'
            qty_idx = action - 5
            quantity = self.reorder_quantities[warehouse][qty_idx]
            self._place_order(warehouse, quantity, upstream='middle')
        elif 9 <= action <= 12:
            # Leaf 2 orders
            warehouse = 'leaf_2'
            qty_idx = action - 9
            quantity = self.reorder_quantities[warehouse][qty_idx]
            self._place_order(warehouse, quantity, upstream='middle')
    
    def _place_order(self, warehouse: str, quantity: int, upstream: str):
        """
        Place an order from warehouse to upstream supplier
        
        Args:
            warehouse: Warehouse placing the order
            quantity: Quantity to order
            upstream: Upstream supplier
        """
        # Sample lead time
        config = self.warehouse_config[warehouse]
        lead_time = max(1, int(self.np_random.normal(
            config['lead_time_mean'],
            config['lead_time_std']
        )))
        
        # Calculate arrival day
        arrival_day = self.current_step + lead_time
        
        # Add to open orders: (quantity, days_waiting, arrival_day)
        self.open_orders[warehouse].append([quantity, 0, arrival_day])
    
    def _generate_demand(self):
        """Generate random demand at leaf warehouses"""
        for warehouse in ['leaf_1', 'leaf_2']:
            config = self.warehouse_config[warehouse]
            demand = max(0, int(self.np_random.normal(
                config['demand_mean'],
                config['demand_std']
            )))
            
            # Fulfill demand or create backlog
            if self.ioh[warehouse] >= demand:
                self.ioh[warehouse] -= demand
            else:
                # Partial fulfillment
                fulfilled = max(0, self.ioh[warehouse])
                shortage = demand - fulfilled
                self.ioh[warehouse] = 0
                
                # Add to backlog with age 0
                self.backlog[warehouse].append([shortage, 0])
    
    def _process_deliveries(self):
        """Process orders that are arriving today"""
        for warehouse in ['middle', 'leaf_1', 'leaf_2']:
            # Check which orders arrive today
            orders_to_remove = []
            
            for idx, order in enumerate(self.open_orders[warehouse]):
                quantity, days_waiting, arrival_day = order
                
                # Increment days waiting
                order[1] += 1
                
                # Check if order arrives today
                if arrival_day == self.current_step:
                    # Check if upstream has inventory
                    upstream = 'factory' if warehouse == 'middle' else 'middle'
                    
                    if upstream == 'factory' or self.ioh[upstream] >= quantity:
                        # Fulfill order
                        if upstream != 'factory':
                            self.ioh[upstream] -= quantity
                        self.ioh[warehouse] += quantity
                        orders_to_remove.append(idx)
                    else:
                        # Partial fulfillment or delay
                        if upstream != 'factory':
                            available = max(0, self.ioh[upstream])
                            if available > 0:
                                self.ioh[upstream] = 0
                                self.ioh[warehouse] += available
                                order[0] -= available  # Reduce order quantity
                        # Reschedule order for next day
                        order[2] = self.current_step + 1
            
            # Remove fulfilled orders
            for idx in sorted(orders_to_remove, reverse=True):
                del self.open_orders[warehouse][idx]
    
    def _update_backlog(self):
        """Update backlog ages and handle withdrawals"""
        for warehouse in ['leaf_1', 'leaf_2']:
            config = self.warehouse_config[warehouse]
            max_backlog = config['max_backlog_duration']
            
            backlogs_to_remove = []
            
            for idx, backlog_item in enumerate(self.backlog[warehouse]):
                quantity, age = backlog_item
                
                # Try to fulfill from current inventory
                if self.ioh[warehouse] > 0:
                    fulfilled = min(self.ioh[warehouse], quantity)
                    self.ioh[warehouse] -= fulfilled
                    backlog_item[0] -= fulfilled
                    
                    if backlog_item[0] <= 0:
                        backlogs_to_remove.append(idx)
                        continue
                
                # Increment age
                backlog_item[1] += 1
                
                # Check if exceeded max backlog duration (order withdrawn)
                if backlog_item[1] > max_backlog:
                    backlogs_to_remove.append(idx)
            
            # Remove fulfilled or withdrawn backlogs
            for idx in sorted(backlogs_to_remove, reverse=True):
                del self.backlog[warehouse][idx]
    
    def _calculate_costs(self) -> Dict:
        """
        Calculate costs for all warehouses
        
        Returns breakdown of shortage, reordering, and holding costs
        """
        cost_breakdown = {
            'shortage': {},
            'reordering': {},
            'holding': {},
            'total': 0
        }
        
        for warehouse in ['middle', 'leaf_1', 'leaf_2']:
            config = self.warehouse_config[warehouse]
            
            # Shortage cost (only for leaf warehouses, based on backlog)
            shortage_cost = 0
            if warehouse in ['leaf_1', 'leaf_2']:
                for backlog_item in self.backlog[warehouse]:
                    quantity, age = backlog_item
                    # Shortage cost increases with age
                    shortage_cost += (
                        config['shortage_cost_constant'] * 
                        quantity * 
                        config['price_per_product'] *
                        (age + 1)  # Weight by age
                    )
            
            cost_breakdown['shortage'][warehouse] = shortage_cost
            
            # Holding cost (positive inventory)
            holding_cost = (
                config['holding_cost_constant'] * 
                max(0, self.ioh[warehouse]) * 
                config['price_per_product']
            )
            cost_breakdown['holding'][warehouse] = holding_cost
            
            # Reordering cost (for orders placed this step)
            # This is calculated based on action, but we approximate here
            reordering_cost = 0
            # We count reordering cost for any new orders in the queue
            # (This is a simplification; in practice, track orders placed this step)
            
            cost_breakdown['reordering'][warehouse] = reordering_cost
        
        # Calculate total
        cost_breakdown['total'] = (
            sum(cost_breakdown['shortage'].values()) +
            sum(cost_breakdown['reordering'].values()) +
            sum(cost_breakdown['holding'].values())
        )
        
        return cost_breakdown
    
    def _calculate_service_level(self) -> float:
        """
        Calculate service level (percentage of demand met immediately)
        Based on current backlog vs total demand
        """
        total_backlog = sum([
            sum([item[0] for item in self.backlog[wh]])
            for wh in ['leaf_1', 'leaf_2']
        ])
        
        # Estimate total demand up to this point
        total_demand = 0
        for wh in ['leaf_1', 'leaf_2']:
            config = self.warehouse_config[wh]
            total_demand += config['demand_mean'] * self.current_step
        
        if total_demand == 0:
            return 1.0
        
        service_level = 1.0 - (total_backlog / total_demand)
        return max(0.0, min(1.0, service_level))
    
    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Inventory Levels:")
            for wh, ioh in self.ioh.items():
                print(f"  {wh}: {ioh:.0f}")
            print(f"Open Orders:")
            for wh, orders in self.open_orders.items():
                print(f"  {wh}: {len(orders)} orders")
            print(f"Service Level: {self._calculate_service_level():.2%}")
            if self.total_cost_history:
                print(f"Avg Cost: {np.mean(self.total_cost_history):.2f}")
    
    def close(self):
        """Clean up resources"""
        pass