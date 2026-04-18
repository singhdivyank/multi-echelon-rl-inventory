"""
Multi-Echelon Inventory System (MEIS) Gym Environment.

This module implements a Gym-compatible environment for a divergent 2-layer MEIS
as described in the research paper. The environment consists of:
- 1 Factory (infinite supply)
- 1 Middle warehouse
- 2 Leaf warehouses
"""

from collections import deque
from typing import Dict, Tuple

import numpy as np
import gym

from utils.helpers import _get_meis_env

WAREHOUSES = ['middle', 'leaf1', 'leaf2']
LEAF_WAREHOUSES = ['leaf1', 'leaf2']

class MEISEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: 'dict | None' = None):
        super(MEISEnv, self).__init__()
        self.ioh = {}
        self.open_orders = {}
        self.total_cost_history = []
        # Allow caller to pass an already-parsed config (e.g. the complex env
        # variant); otherwise fall back to the default meisConfig.yaml.
        self.config = config if config is not None else _get_meis_env()
        print('Warehouse Config: ', self.config)
        self.current_step = 0
        # self.np_random = np.random.RandomState(42)
        self.seed(42)
        self.max_steps = self.config['max_steps_per_episode']
        self.env_specs()
        self.env_spaces()
        self._setup_warehouse_configs()
        self._setup_reorder_quantity()
    
    def seed(self, seed=None):
        """Set random seed for reproducability"""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def env_specs(self):
        """Define environment specifications"""
        specs = self.config['env']
        self.n_warehouses = specs['n_warehouses']
        self.state_dim_per_warehouse = specs['state_dim_per_warehouse']
        self.total_state_dim = specs['total_state_dim']
    
    def env_spaces(self):
        """Define action space and observation space"""
        self.action_space = gym.spaces.Discrete(13)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_state_dim,),
            dtype=np.float32
        )
    
    def _setup_warehouse_configs(self):
        """Setup configuration for each warehouse"""
        self.warehouse_config = {
            'factory': self.config['factory'].copy(),
            'middle': self.config['middle'].copy(),
            'leaf1': self.config['leaf'].copy(),
            'leaf2': self.config['leaf'].copy()
        }
    
    def _setup_reorder_quantity(self):
        """Setup discrete reorder quantity options for each warehouse"""
        self.reorder_quantities = {
            'middle': self.config['middle']['reorder_qty_options'],
            'leaf1': self.config['leaf']['reorder_qty_options'],
            'leaf2': self.config['leaf']['reorder_qty_options'],
        }
    
    def _get_state(self):
        """Construct 12 dimension state-vector"""

        state = []

        for wh_name in WAREHOUSES:
            ioh = float(self.ioh[wh_name])

            # oldest order information
            if len(self.open_orders[wh_name]) > 0:
                oldest_order = self.open_orders[wh_name][0]
                oldest_qty = float(oldest_order['qty']) 
                days_waiting = float(oldest_order['age'])
            else:
                oldest_qty, days_waiting = 0, 0
            
            total_reorder = sum([float(order['qty']) for order in self.open_orders[wh_name]])
            state.extend([ioh, oldest_qty, days_waiting, total_reorder])
        
        return np.array(state, dtype=np.float32)
    
    def _process_action(self, action: int):
        """Process the action and place orders accordingly"""

        if not action:
            return
        elif 1 <= action <= 4:
            warehouse = 'middle'
            qty_idx = action - 1
            quantity = self.reorder_quantities[warehouse][qty_idx]
            self._place_order(warehouse, quantity, upstream='factory')
        elif 5 <= action <= 8:
            warehouse = 'leaf1'
            qty_idx = action - 5
            quantity = self.reorder_quantities[warehouse][qty_idx]
            self._place_order(warehouse, quantity, upstream='middle')
        elif 9 <= action <= 12:
            warehouse = 'leaf2'
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

        config = self.warehouse_config[warehouse]
        reorder_cost = max(
            config['min_reorder_cost'],
            config['reorder_cost_constant'] * quantity
        )
        self.last_reorder_cost += reorder_cost
        lead_time = max(1, int(round(
            self.np_random.normal(
                config['lead_time_mean'], 
                config['lead_time_std']
            )
        )))
        arrival_day = self.current_step + lead_time
        self.open_orders[warehouse].append({
            'qty': quantity, 
            'age': 0, 
            'arrival': arrival_day,
            'upstream': upstream
        })
    
    def _generate_demand(self):
        """Generate random demand at leaf warehouses"""

        for warehouse in LEAF_WAREHOUSES:
            config = self.warehouse_config[warehouse]
            demand = max(0, int(round(
                self.np_random.normal(
                    config['demand_mean'],
                    config['demand_std']
                )
            )))

            if self.ioh[warehouse] >= demand:
                self.ioh[warehouse] -= demand
            else:
                fulfilled = max(0, self.ioh[warehouse])
                shortage = demand - fulfilled
                self.ioh[warehouse] = 0
                self.backlog[warehouse].append([shortage, 0])
    
    def _process_delivery(self):
        """Process order that are arriving today"""

        for warehouse in WAREHOUSES:
            upstream = 'factory' if warehouse == 'middle' else 'middle'
            is_from_factory = (upstream == 'factory')

            for order in self.open_orders[warehouse]:
                order['age'] += 1

            orders = self.open_orders[warehouse]
            arriving_today = sorted(
                [o for o in orders if o['arrival'] == self.current_step],
                key=lambda x: -x['age']
            )
            still_waiting = [o for o in orders if o['arrival'] != self.current_step]
            orders_to_keep = still_waiting

            for order in arriving_today:
                quantity = order['qty']
                if is_from_factory:
                    self.ioh[warehouse] += quantity
                else:
                    available = self.ioh[upstream]
                    if available >= quantity:
                        self.ioh[upstream] -= quantity
                        self.ioh[warehouse] += quantity
                    else:
                        if available > 0:
                            self.ioh[warehouse] += available
                            order['qty'] -= available
                            self.ioh[upstream] = 0
                        
                        order['arrival'] = self.current_step + 1
                        orders_to_keep.append(order)
            
            self.open_orders[warehouse] = deque(orders_to_keep)
    
    def _update_backlog(self):
        """Update backlog ages and handle withdrawl"""

        for warehouse in LEAF_WAREHOUSES:
            ioh = self.ioh[warehouse]
            max_backlog = self.warehouse_config[warehouse]['max_backlog_duration']
            updated_backlog = []

            for quantity, age in self.backlog[warehouse]:
                if ioh > 0:
                    fulfilled = min(ioh, quantity)
                    ioh -= fulfilled
                    quantity -= fulfilled
                
                if quantity > 0:
                    new_age = age + 1
                    if new_age <= max_backlog:
                        updated_backlog.append([quantity, new_age])
            
            self.ioh[warehouse] = ioh
            self.backlog[warehouse] = updated_backlog
    
    def _get_total_cost(self) -> Dict:
        """Calculate costs for all warehouses."""

        cost_breakdown = {
            'shortage': {},
            'holding': {},
            'reordering': 0,
            'pipeline': 0,
            'total': 0
        }

        total_shortage_cost = 0
        total_holding_cost = 0
        pipeline_penalty = 0

        for warehouse in WAREHOUSES:
            config = self.warehouse_config[warehouse]
            shortage_cost = 0

            total_pipeline = sum(o['qty'] for o in self.open_orders[warehouse])
            pipeline_penalty += 0.01 * total_pipeline
            
            if warehouse in LEAF_WAREHOUSES:
                for quantity, _ in self.backlog[warehouse]:
                    shortage_cost += (
                        config['shortage_cost_constant'] * quantity
                    )
            
            cost_breakdown['shortage'][warehouse] = shortage_cost
            total_shortage_cost += shortage_cost
            
            holding_cost = (
                config['holding_cost_constant'] *
                max(0, self.ioh[warehouse])
            )
            cost_breakdown['holding'][warehouse] = holding_cost
            total_holding_cost += holding_cost

        total_reordering_cost = self.last_reorder_cost
        cost_breakdown['reordering'] = total_reordering_cost
        cost_breakdown['pipeline'] = pipeline_penalty
        total_cost = total_shortage_cost + total_holding_cost + total_reordering_cost + pipeline_penalty
        cost_breakdown['total'] = total_cost
        self.total_cost_history.append(cost_breakdown)
        return cost_breakdown

    def _calculate_service_level(self) -> float:
        """Calculate service level based on current backlog vs total demand"""

        total_demand = 0
        warehouses = LEAF_WAREHOUSES
        total_backlog = sum([
            sum([item[0] for item in self.backlog[wh]])
            for wh in warehouses
        ])

        for wh in warehouses:
            config = self.warehouse_config[wh]
            total_demand += config['demand_mean'] * self.current_step
        
        if not total_demand:
            return 1.0

        service_level = 1.0 - (total_backlog / total_demand)
        service_level = min(1.0, service_level)
        return max(0.0, service_level)

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = 0
        self.last_reorder_cost = 0
        self.total_cost_history = []
        self.ioh = {
            'factory': self.warehouse_config['factory']['ioh_initial'],
            'middle': self.warehouse_config['middle']['ioh_initial'],
            'leaf1': self.warehouse_config['leaf1']['ioh_initial'],
            'leaf2': self.warehouse_config['leaf2']['ioh_initial']
        }
        self.open_orders = {
            'middle': deque(),
            'leaf1': deque(),
            'leaf2': deque()
        }
        self.backlog = {
            'middle': deque(),
            'leaf1': deque(),
            'leaf2': deque()
        }
        return self._get_state()
    
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

        self.last_reorder_cost = 0
        self._process_action(action=action)
        self._generate_demand()
        self._process_delivery()
        self._update_backlog()
        cost_breakdown = self._get_total_cost()
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        reward = -cost_breakdown['total'] / 10000
        info = {
            'cost_breakdown': cost_breakdown,
            'ioh': self.ioh.copy(),
            'service_level': self._calculate_service_level(),
            'step': self.current_step
        }

        return next_state, reward, done, info
