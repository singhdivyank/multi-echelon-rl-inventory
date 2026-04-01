"""
Divergent supply chain environment Structure: 1 warehouse -> 3 retailers
As described in Table 5 and 6 of the paper
"""

import numpy as np
from gymnasium import spaces
from typing import Dict
from .base_env import BaseInventoryEnv


class DivergentInventoryEnv(BaseInventoryEnv):
    """
    Divergent inventory environment with:
    - 1 warehouse
    - 3 retailers
    - Stochastic demand at retailers
    - Stochastic lead times
    - No backlogging at warehouse
    """
    
    def __init__(self, config: Dict):
        # Ensure we have the right number of retailers
        config['num_retailers'] = config.get('num_retailers', 3)
        self.num_retailers = config['num_retailers']
        
        super().__init__(config)
        
        # Define observation space
        # State: [I_w, I_r1, I_r2, I_r3, BO_r1, BO_r2, BO_r3, IT_w_r1, IT_w_r2, IT_w_r3, O_w, O_r1, O_r2, O_r3]
        state_dim = self._get_state_dimension()
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        # Define action space
        # Actions: [order_warehouse, order_r1, order_r2, order_r3]
        action_dim = self._get_action_dimension()
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(action_dim,), 
            dtype=np.float32
        )
        
        # Initialize in-transit tracking (list of tuples: (quantity, arrival_time))
        self.in_transit_warehouse = []
        self.in_transit_retailers = [[] for _ in range(self.num_retailers)]
        
    def _get_state_dimension(self) -> int:
        """
        State dimension for divergent structure:
        - Warehouse inventory: 1
        - Retailer inventories: num_retailers
        - Retailer backorders: num_retailers
        - In-transit to retailers: num_retailers
        - Outstanding orders warehouse: 1
        - Outstanding orders retailers: num_retailers
        """
        return 1 + self.num_retailers * 4 + 1
        
    def _get_action_dimension(self) -> int:
        """
        Action dimension:
        - Order for warehouse: 1
        - Orders for each retailer: num_retailers
        """
        return 1 + self.num_retailers
        
    def _initialize_state(self):
        """Initialize all state variables"""
        # Start with some initial inventory
        self.inventory_warehouse = self.max_inventory_warehouse * 0.5
        self.inventory_retailers = np.ones(self.num_retailers) * self.max_inventory_buffer * 0.3
        
        # No initial backorders
        self.backorders_retailers = np.zeros(self.num_retailers)
        
        # No initial in-transit
        self.in_transit_warehouse = []
        self.in_transit_retailers = [[] for _ in range(self.num_retailers)]
        
        # No initial outstanding orders
        self.outstanding_order_warehouse = 0
        self.outstanding_orders_retailers = np.zeros(self.num_retailers)
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation
        
        Returns normalized state vector
        """
        # Calculate total in-transit quantities
        it_retailers = np.array([
            sum([qty for qty, _ in self.in_transit_retailers[i]]) 
            for i in range(self.num_retailers)
        ])
        
        # Construct state vector
        state = np.concatenate([
            [self.inventory_warehouse],
            self.inventory_retailers,
            self.backorders_retailers,
            it_retailers,
            [self.outstanding_order_warehouse],
            self.outstanding_orders_retailers,
        ])
        
        # Normalize
        max_values = np.array([
            self.max_inventory_warehouse,
            *[self.max_inventory_buffer] * self.num_retailers,
            *[self.max_backorder] * self.num_retailers,
            *[self.max_in_transit] * self.num_retailers,
            self.max_outstanding,
            *[self.max_outstanding] * self.num_retailers,
        ])
        
        normalized_state = state / max_values
        normalized_state = np.clip(normalized_state, -1, 1).astype(np.float32)
        
        return normalized_state
        
    def _event_1_receive_shipments(self):
        """Event 1: Receive shipments that have arrived"""
        # Check warehouse shipments
        arrived_warehouse = [
            qty for qty, arrival in self.in_transit_warehouse 
            if arrival <= self.current_step
        ]
        if arrived_warehouse:
            self.inventory_warehouse += sum(arrived_warehouse)
            self.in_transit_warehouse = [
                (qty, arrival) for qty, arrival in self.in_transit_warehouse
                if arrival > self.current_step
            ]
            
        # Check retailer shipments
        for i in range(self.num_retailers):
            arrived_retailer = [
                qty for qty, arrival in self.in_transit_retailers[i]
                if arrival <= self.current_step
            ]
            if arrived_retailer:
                self.inventory_retailers[i] += sum(arrived_retailer)
                self.in_transit_retailers[i] = [
                    (qty, arrival) for qty, arrival in self.in_transit_retailers[i]
                    if arrival > self.current_step
                ]
                
    def _event_2_supplier_shipment(self, orders: np.ndarray):
        """Event 2: Supplier sends shipment to warehouse"""
        warehouse_order = orders[0]
        
        if warehouse_order > 0:
            # Generate stochastic lead time
            lead_time = self._generate_lead_time()
            arrival_time = self.current_step + lead_time
            
            # Add to in-transit
            self.in_transit_warehouse.append((warehouse_order, arrival_time))
            self.outstanding_order_warehouse = warehouse_order
            
    def _event_3_fulfill_retailer_orders(self):
        """Event 3: Warehouse fulfills retailer orders with on-hand inventory"""
        for i in range(self.num_retailers):
            if self.outstanding_orders_retailers[i] > 0:
                # Can only fulfill from available warehouse inventory
                available = max(0, self.inventory_warehouse)
                fulfillment = min(self.outstanding_orders_retailers[i], available)
                
                if fulfillment > 0:
                    self.inventory_warehouse -= fulfillment
                    # Add to in-transit to retailer
                    lead_time = self._generate_lead_time()
                    arrival_time = self.current_step + lead_time
                    self.in_transit_retailers[i].append((fulfillment, arrival_time))
                    
                    self.outstanding_orders_retailers[i] -= fulfillment
                    
    def _event_4_transport_orders(self, orders: np.ndarray):
        """Event 4: Place orders from retailers to warehouse"""
        retailer_orders = orders[1:]
        
        for i in range(self.num_retailers):
            if retailer_orders[i] > 0:
                self.outstanding_orders_retailers[i] += retailer_orders[i]
                
    def _event_5_customer_demand(self):
        """Event 5: Customer demand at retailers"""
        _, retailer_demands = self._generate_demand()
        
        for i in range(self.num_retailers):
            demand = retailer_demands[i]
            self.episode_demands.append(demand)
            
            # Try to fulfill from inventory
            available = max(0, self.inventory_retailers[i])
            fulfilled = min(demand, available)
            
            # Backorder unfulfilled demand
            self.inventory_retailers[i] -= fulfilled
            if demand > fulfilled:
                self.backorders_retailers[i] += (demand - fulfilled)
                
            # Clear backorders if we have inventory
            if self.inventory_retailers[i] > 0 and self.backorders_retailers[i] > 0:
                backorder_fulfillment = min(
                    self.inventory_retailers[i], 
                    self.backorders_retailers[i]
                )
                self.inventory_retailers[i] -= backorder_fulfillment
                self.backorders_retailers[i] -= backorder_fulfillment