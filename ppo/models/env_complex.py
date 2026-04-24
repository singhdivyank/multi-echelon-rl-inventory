"""
Complex Divergent Supply Chain Environment.

Second environment used to test whether the PPO policy trained on the original
`DivergentInventoryEnv` generalizes to a harder setting, or whether the
original result was a fluke of benign dynamics.

Same topology (1 warehouse -> N retailers) and identical observation/action
space shapes as `DivergentInventoryEnv` so no network changes are required.
Only the *dynamics* are harder:

    1. Non-stationary seasonal demand (sinusoidal base + noise).
    2. Stochastic lead times with occasional delay spikes.
    3. Stochastic supplier capacity cap (upstream can't always ship the full
       order).
    4. Demand shocks (low-prob burst multiplier for a few consecutive steps).
    5. Correlated retailer demand via a shared latent factor.
    6. Partial lost sales (a fraction of unmet demand is lost, not backlogged).
"""

import numpy as np

from .env import DivergentInventoryEnv
from utils.helpers import load_complex_config


class ComplexDivergentInventoryEnv(DivergentInventoryEnv):
    """More realistic / stressful version of `DivergentInventoryEnv`."""

    def __init__(self):
        super().__init__()
        self.config = load_complex_config()['complex']
        self._get_complex_parameters()
        # Shock book keeping
        self._shock_steps_remaining = 0
        self._shock_multiplier = 1.0

    def _get_complex_parameters(self):
        """Pull complex-env-only hyperparameters with sane defaults."""
        # Seasonality
        self._seasonality_params()
        # Correlation: shared latent factor strength in [0, 1]
        self.demand_corr = float(self.config.get('demand_corr', 0.5))
        # Shocks
        self.shock_prob = float(self.config.get('shock_prob', 0.01))
        self.shock_multiplier_range = tuple(self.config.get('shock_multiplier_range', [2.0, 3.0]))
        self.shock_duration_range = tuple(self.config.get('shock_duration_range', [3, 7]))
        # Stochastic lead time
        self._lead_times()
        # Supplier capacity cap
        self._supplier_cap()
        # Lost sales fraction of unmet demand
        self.lost_sales_fraction = float(self.config.get('lost_sales_fraction', 0.3))
        self.lost_sales_penalty = float(self.config.get('lost_sales_penalty', 5.0))
        # Running lost-sales counter
        self._lost_sales_step = 0.0
    
    def _seasonality_params(self):
        self.demand_base = float(self.config.get('demand_base', 10.0))
        self.demand_amplitude = float(self.config.get('demand_amplitude', 0.6))
        self.demand_period = float(self.config.get('demand_period', 120.0))
        self.demand_trend = float(self.config.get('demand_trend', 0.0))
        self.demand_noise_std = float(self.config.get('demand_noise_std', 2.0))
    
    def _lead_times(self):
        self.lead_time_mean = float(self.config.get('lead_time_mean', 2.0))
        self.lead_time_std = float(self.config.get('lead_time_std', 1.0))
        self.lead_time_spike_prob = float(self.config.get('lead_time_spike_prob', 0.05))
        self.lead_time_spike_extra = int(self.config.get('lead_time_spike_extra', 3))
        self.lead_time_max = int(self.config.get('lead_time_max', 8))

    def _supplier_cap(self):
        self.supplier_capacity_mean = float(self.config.get('supplier_capacity_mean', 60.0))
        self.supplier_capacity_std = float(self.config.get('supplier_capacity_std', 10.0))
        self.warehouse_capacity_mean = float(self.config.get('warehouse_capacity_mean', 40.0))
        self.warehouse_capacity_std = float(self.config.get('warehouse_capacity_std', 8.0))

    def _sample_lead_time(self) -> int:
        lt = self.np_random_normal(self.lead_time_mean, self.lead_time_std)
        if np.random.random() < self.lead_time_spike_prob:
            lt += self.lead_time_spike_extra
        return int(np.clip(round(lt), 1, self.lead_time_max))

    def np_random_normal(self, mean: float, std: float) -> float:
        return float(np.random.normal(mean, std))

    def _generate_demand(self):
        """Seasonal, correlated, shock-prone demand."""
        t = self.current_step
        season = 1.0 + self.demand_amplitude * np.sin(
            2.0 * np.pi * t / max(1.0, self.demand_period)
        )
        mean_t = max(0.1, self.demand_base * season + self.demand_trend * t)

        if self._shock_steps_remaining <= 0 and np.random.random() < self.shock_prob:
            self._shock_steps_remaining = int(
                np.random.randint(self.shock_duration_range[0],
                                  self.shock_duration_range[1] + 1)
            )
            self._shock_multiplier = float(np.random.uniform(
                self.shock_multiplier_range[0],
                self.shock_multiplier_range[1]
            ))

        active_mult = self._shock_multiplier if self._shock_steps_remaining > 0 else 1.0
        if self._shock_steps_remaining > 0:
            self._shock_steps_remaining -= 1
        
        shared = np.random.normal(0.0, self.demand_noise_std)
        idio = np.random.normal(0.0, self.demand_noise_std, size=self.num_retailers)
        noise = self.demand_corr * shared + np.sqrt(
            max(0.0, 1.0 - self.demand_corr ** 2)
        ) * idio

        rates = np.maximum(0.1, mean_t * active_mult + noise)
        retailer_demands = np.random.poisson(rates).astype(float)
        warehouse_rate = max(0.1, 0.3 * mean_t * active_mult)
        warehouse_demands = float(np.random.poisson(warehouse_rate))
        return warehouse_demands, retailer_demands

    def _event_2_supplier_shipment(self, orders: np.ndarray):
        """Supplier ships to warehouse, capped by stochastic capacity and
        with a stochastic lead time."""
        requested = max(0.0, float(orders[0]))
        if not requested:
            return

        cap = max(0.0, float(np.random.normal(
            self.supplier_capacity_mean, self.supplier_capacity_std
        )))
        shipped = min(requested, cap)
        lead_time = self._sample_lead_time()
        arrival_time = self.current_step + lead_time
        if shipped > 0:
            self.in_transit_warehouse.append((shipped, arrival_time))
        self.outstanding_order_warehouse = requested

    def _event_3_fulfill_retailer_orders(self):
        """Warehouse ships to retailers, capped by stochastic outbound capacity,
        with stochastic lead times on each shipment."""
        if self.inventory_warehouse <= 0:
            return
        outbound_cap = max(0.0, float(np.random.normal(
            self.warehouse_capacity_mean, self.warehouse_capacity_std
        )))
        remaining_cap = outbound_cap

        # Round-robin fulfillment so no single retailer starves the others.
        order = np.argsort(-self.outstanding_orders_retailers)
        for i in order:
            if self.outstanding_orders_retailers[i] <= 0:
                continue
            if self.inventory_warehouse <= 0 or remaining_cap <= 0:
                break
            fulfillment = min(
                self.outstanding_orders_retailers[i],
                self.inventory_warehouse,
                remaining_cap,
            )
            if fulfillment <= 0:
                continue
            self.inventory_warehouse -= fulfillment
            remaining_cap -= fulfillment
            arrival_time = self.current_step + self._sample_lead_time()
            self.in_transit_retailers[i].append((fulfillment, arrival_time))
            self.outstanding_orders_retailers[i] -= fulfillment

    def _event_5_customer_demand(self):
        """Customer demand at retailers with partial lost sales."""
        self._lost_sales_step = 0.0
        warehouse_demand, retailer_demands = self._generate_demand()
        self.inventory_warehouse = max(0.0, self.inventory_warehouse - warehouse_demand)

        for i in range(self.num_retailers):
            demand = float(retailer_demands[i])
            self.episode_demands.append(demand)

            # First clear any existing backlog if inventory allows.
            if self.backorders_retailers[i] > 0 and self.inventory_retailers[i] > 0:
                cleared = min(self.inventory_retailers[i], self.backorders_retailers[i])
                self.inventory_retailers[i] -= cleared
                self.backorders_retailers[i] -= cleared

            fulfilled = min(demand, self.inventory_retailers[i])
            self.inventory_retailers[i] -= fulfilled
            unmet = demand - fulfilled

            if unmet > 0:
                lost = unmet * self.lost_sales_fraction
                backlog = unmet - lost
                self.backorders_retailers[i] += backlog
                self._lost_sales_step += lost

            self.backorders_retailers[i] = min(
                self.backorders_retailers[i], self.max_backorder
            )

    def _compute_cost(self) -> float:
        """Base cost + lost-sales penalty."""
        base = super()._compute_cost()
        return base + self.lost_sales_penalty * self._lost_sales_step

    def reset(self, seed=None):
        self._shock_steps_remaining = 0
        self._shock_multiplier = 1.0
        self._lost_sales_step = 0.0
        return super().reset(seed=seed)

    def step(self, action: np.ndarray):
        """Same event ordering as the base env, but without the base env's
        hard-coded 15-unit order cap (which silently limits throughput when
        seasonal / shocked demand spikes far above that)."""
        orders = self._denormalize_action(action)
        orders = np.clip(orders, 0, self.max_orders)

        self._event_1_receive_shipments()
        self._event_2_supplier_shipment(orders)
        self._event_3_fulfill_retailer_orders()
        self._event_4_transport_orders(orders)
        self._event_5_customer_demand()

        cost = self._compute_cost()
        self.episode_costs.append(cost)
        self.current_step += 1

        next_state = self._get_observation()
        reward = -cost / 10000.0
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {
            'cost': cost,
            'step': self.current_step,
            'inventory_warehouse': self.inventory_warehouse,
            'inventory_retailers': self.inventory_retailers.copy(),
            'lost_sales': self._lost_sales_step,
            'shock_active': self._shock_steps_remaining > 0,
        }
        return next_state, reward, terminated, truncated, info
