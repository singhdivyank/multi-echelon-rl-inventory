"""
Divergent supply chain environment Structure: 1 warehouse -> 3 retailers.
Base Gym-style environment for multi-echelon inventory management
Follows the MDP formulation from the paper
"""

import re
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gym import spaces

from utils.helpers import load_divergent_config


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
        self.current_step = 0
        self.observation_space = None
        self.action_space = None
        self.episode_costs = []
        self.episode_demands = []
        self._get_parameters()
        self._init_state_components()
    
    def _get_parameters(self):
        self.structure = self.config['structure']
        self.max_steps = self.config['max_steps']
        self.warmup_steps = self.config['warmup_steps']
        # Warehouse holding cost
        self.h_w = self.config['h_w']
        # Retailer holding cost
        self.h_r = self.config['h_r']
        # Warehouse backorder cost
        self.b_w = self.config['b_w']
        # Retailer backorder cost
        self.b_r = self.config['b_r']
        # Demand Parameters
        self.lambda_w = self.config['lambda_w']
        self.lambda_w_r = self.config['lambda_w_r']
        self._gen_samples()
        # Network Structure
        self.K = self.config['K']
        # State Space Upper Bounds
        self.max_inventory_warehouse = self.config['max_inventory_warehouse']
        self.max_inventory_buffer = self.config['max_inventory_buffer']
        self.max_in_transit = self.config['max_in_transit_w_r']
        self.max_backorder = self.config['max_backorder_r']
        self.max_outstanding = self.config['max_outstanding_w']
        self.max_orders = self.config['max_orders_w']
    
    def _init_state_components(self):
        self.inventory_warehouse = 0
        self.inventory_retailers = None
        self.backorders_warehouse = None
        self.backorders_retailers = None
        self.in_transit = None
        self.outstanding_orders = None
    
    def _gen_samples(self):
        config_str = self.config['d_i']
        match = re.search(r'\[(\d+),\s*(\d+)\]', config_str)
        if match:
            low, high = map(int, match.groups())
            rates = np.random.uniform(low, high, size=1)
            demand = np.random.poisson(rates)
            return demand[0]
        
        return 0
    
    def _initialize_state(self) -> int:
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
        retailer_backorder = np.sum(self.b_r * self.backorders_retailers)

        cost = warehouse_holding + warehouse_backorder + retailer_holding + retailer_backorder
        return cost

    def _generate_demand(self):
        """Demand: Poisson(Uniform[5,15])"""
        rates = np.random.uniform(5, 15, size=self.num_retailers)
        retailer_demands = np.random.poisson(rates)
        warehouse_demands = np.random.poisson(10)
        return warehouse_demands, retailer_demands
    
    def _normalise_state(self, state: np.ndarray) -> np.ndarray:
        """Normalise state to [-1, 1] range"""
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
        denormalized = (action + 1) * self.max_orders / 2
        return np.clip(denormalized, 0, self.max_orders)
    
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
    
    def _get_observation(self):
        """Get current state observation"""
        raise NotImplementedError

    def close(self):
        """Clean up the environment"""
        pass

    def seed(self, seed):
        """Set random seed"""
        np.random.seed(seed)

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
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

        orders = self._denormalize_action(action)
        orders = np.clip(orders, 0, 15)
        
        self._event_1_receive_shipments()
        self._event_2_supplier_shipment(orders)
        self._event_3_fulfill_retailer_orders()
        self._event_4_transport_orders(orders)
        self._event_5_customer_demand()
        
        cost = self._compute_cost()
        self.episode_costs.append(cost)
        self.current_step += 1

        next_state = self._get_observation()
        reward = -cost/10000
        # reward = -np.log1p(cost / 100)
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {
            'cost': cost,
            'step': self.current_step,
            'inventory_warehouse': self.inventory_warehouse,
            'inventory_retailers': self.inventory_retailers.copy(),
        }

        return next_state, reward, terminated, truncated, info
        

class DivergentInventoryEnv(BaseInventoryEnv):
    def __init__(self):
        self.config = load_divergent_config()['divergent']
        self.num_retailers = self.config['num_retailers']
        super().__init__(config=self.config)
        self._define_spaces()
        self.lead_time = 1
        self.in_transit_warehouse = []
        self.in_transit_retailers = [[] for _ in range(self.num_retailers)]
    
    def _define_spaces(self):
        """
        Create action space and observation space
        """

        state_dim = 1 + self.num_retailers * 4 + 1
        action_dim = 1 + self.num_retailers

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def _initialize_state(self):
        """Initialize all state variables"""
        self.inventory_warehouse = self.max_inventory_warehouse * 0.5
        self.inventory_retailers = np.ones(self.num_retailers) * self.max_inventory_buffer * 0.3
        self.backorders_retailers = np.zeros(self.num_retailers)
        self.in_transit_warehouse = []
        self.in_transit_retailers = [[] for _ in range(self.num_retailers)]
        self.outstanding_order_warehouse = 0
        self.outstanding_orders_retailers = np.zeros(self.num_retailers)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation
        
        Returns normalized state vector
        """

        it_retailers = np.array([
            sum(qty for qty, _ in self.in_transit_retailers[i])
            for i in range(self.num_retailers)
        ])

        state = np.concatenate([
            [self.inventory_warehouse],
            self.inventory_retailers,
            self.backorders_retailers,
            it_retailers,
            [self.outstanding_order_warehouse],
            self.outstanding_orders_retailers,
        ])

        max_values = np.array([
            self.max_inventory_warehouse,
            *[self.max_inventory_buffer] * self.num_retailers,
            *[self.max_backorder] * self.num_retailers,
            *[self.max_in_transit] * self.num_retailers,
            self.max_outstanding,
            *[self.max_outstanding] * self.num_retailers,
        ])

        normalised_state = state / max_values
        normalised_state = np.clip(normalised_state, -1, 1).astype(np.float32)
        return normalised_state
    
    def _event_1_receive_shipments(self):
        """Event 1: Receive shipments that have arrived"""
        
        # Check warehouse shipment
        arrived_warehouse = [
            qty for qty, arrival in self.in_transit_warehouse
            if arrival <= self.current_step
        ]
        if arrived_warehouse:
            self.inventory_warehouse = min(
                self.inventory_warehouse + sum(arrived_warehouse), 
                self.max_inventory_warehouse
            )
            self.in_transit_warehouse = [
                (qty, arrival) for qty, arrival in self.in_transit_warehouse
                if arrival > self.current_step
            ]
        
        # Check retail shipments
        for i in range(self.num_retailers):
            arrived_retailer = [
                qty for qty, arrival in self.in_transit_retailers[i]
                if arrival <= self.current_step
            ]
            if arrived_retailer:
                self.inventory_retailers[i] = min(
                    self.inventory_retailers[i] + sum(arrived_retailer), 
                    self.max_inventory_buffer
                )
                self.in_transit_retailers[i] = [
                    (qty, arrival) for qty, arrival in self.in_transit_retailers[i]
                    if arrival > self.current_step
                ]
        
    def _event_2_supplier_shipment(self, orders: np.ndarray):
        """Event 2: Supplier sends shipment to warehouse"""
        qty = max(0, orders[0])
        if qty > 0:
            arrival_time = self.current_step + self.lead_time
            self.in_transit_warehouse.append((qty, arrival_time))
            self.outstanding_order_warehouse = qty
        
    def _event_3_fulfill_retailer_orders(self):
        """Event 3: Retailer fulfills orders from retailers with on-hand inventory"""
        for i in range(self.num_retailers):
            if self.outstanding_orders_retailers[i] > 0 and self.inventory_warehouse > 0:
                fulfillment = min(self.outstanding_orders_retailers[i], self.inventory_warehouse)
                self.inventory_warehouse -= fulfillment
                arrival_time = self.current_step + self.lead_time
                self.in_transit_retailers[i].append((fulfillment, arrival_time))
                self.outstanding_orders_retailers[i] -= fulfillment
        
    def _event_4_transport_orders(self, orders: np.ndarray):
        """Event 4: Place orders from retailers to warehouse"""
        retailer_orders = orders[1:]

        for i in range(self.num_retailers):
            if retailer_orders[i] > 0:
                self.outstanding_orders_retailers[i] = retailer_orders[i]
            
            self.outstanding_orders_retailers[i] = min(
                self.outstanding_orders_retailers[i],
                self.max_outstanding
            )
        
    def _event_5_customer_demand(self):
        """Event 5: Customer demand at retailers"""
        warehouse_demand, retailer_demands = self._generate_demand()
        self.inventory_warehouse = max(0, self.inventory_warehouse - warehouse_demand)

        for i in range(self.num_retailers):
            demand = retailer_demands[i]
            self.episode_demands.append(demand)

            if self.backorders_retailers[i] > 0 and self.inventory_retailers[i] > 0:
                cleared = min(self.inventory_retailers[i], self.backorders_retailers[i])
                self.inventory_retailers[i] -= cleared
                self.backorders_retailers[i] -= cleared
            
            fulfilled = min(demand, self.inventory_retailers[i])
            self.inventory_retailers[i] -= fulfilled

            if demand > fulfilled:
                self.backorders_retailers[i] += (demand - fulfilled)
            
            self.backorders_retailers[i] = min(self.backorders_retailers[i], self.max_backorder)
