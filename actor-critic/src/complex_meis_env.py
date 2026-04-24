"""
Complex Multi-Echelon Inventory System (MEIS) environment.

Second environment for A3C. Subclasses `MEISEnv` so state/action spaces and
the discrete action mapping remain identical (the actor head does not need to
change). Only the dynamics are harder:

    1. Non-stationary seasonal demand at leaf warehouses.
    2. Correlated demand across leaves via a shared latent factor.
    3. Low-prob demand shocks that last a few steps.
    4. Heavier-tailed lead times (bigger std + occasional spikes).
    5. Stochastic supplier capacity cap at the middle warehouse when shipping
       to leaves.
    6. Lost sales cost on backlog items aged out of the system.

The complex-env-only parameters live in `configs/complexMeisConfig.yaml`.
"""

from collections import deque
from typing import Dict

import numpy as np

from src.meis_env import MEISEnv, WAREHOUSES, LEAF_WAREHOUSES
from utils.helpers import _get_complex_meis_env


class ComplexMEISEnv(MEISEnv):
    """Harder variant of `MEISEnv`."""

    def __init__(self):
        super().__init__()
        self.config = _get_complex_meis_env()
        self._load_complex_params()
        self._shock_steps_remaining = 0
        self._shock_multiplier = 1.0
        self._lost_sales_step = 0.0

    def _load_complex_params(self):
        complex_cfg = self.config.get('complex', {})
        # Seasonal demand around the leaf `demand_mean`.
        self.demand_amplitude = float(complex_cfg.get('demand_amplitude', 0.5))
        self.demand_period = float(complex_cfg.get('demand_period', 90.0))
        self.demand_trend = float(complex_cfg.get('demand_trend', 0.0))
        # Correlation across leaves via shared latent factor.
        self.demand_corr = float(complex_cfg.get('demand_corr', 0.5))
        # Shocks
        self.shock_prob = float(complex_cfg.get('shock_prob', 0.01))
        self.shock_multiplier_range = tuple(complex_cfg.get('shock_multiplier_range', [1.8, 2.5]))
        self.shock_duration_range = tuple(complex_cfg.get('shock_duration_range', [3, 7]))
        # Lead time heavy tails
        self.lead_time_spike_prob = float(complex_cfg.get('lead_time_spike_prob', 0.08))
        self.lead_time_spike_extra = int(complex_cfg.get('lead_time_spike_extra', 3))
        self.lead_time_std_mult = float(complex_cfg.get('lead_time_std_mult', 2.0))
        self.lead_time_max = int(complex_cfg.get('lead_time_max', 10))
        # Middle-warehouse outbound capacity cap
        self.middle_capacity_mean = float(complex_cfg.get('middle_capacity_mean', 12000.0))
        self.middle_capacity_std = float(complex_cfg.get('middle_capacity_std', 3000.0))
        # Lost sales
        self.lost_sales_penalty = float(complex_cfg.get('lost_sales_penalty', 3.0))

    def _seasonal_mean(self, base_mean: float) -> float:
        t = self.current_step
        season = 1.0 + self.demand_amplitude * np.sin(
            2.0 * np.pi * t / max(1.0, self.demand_period)
        )
        return max(1.0, base_mean * season + self.demand_trend * t)

    def _maybe_trigger_shock(self):
        if self._shock_steps_remaining <= 0 and self.np_random.rand() < self.shock_prob:
            self._shock_steps_remaining = int(
                self.np_random.randint(self.shock_duration_range[0],
                                       self.shock_duration_range[1] + 1)
            )
            self._shock_multiplier = float(self.np_random.uniform(
                self.shock_multiplier_range[0],
                self.shock_multiplier_range[1]
            ))

    def _generate_demand(self):
        """Seasonal + correlated + shock-prone demand at leaves."""
        self._maybe_trigger_shock()
        active_mult = self._shock_multiplier if self._shock_steps_remaining > 0 else 1.0
        if self._shock_steps_remaining > 0:
            self._shock_steps_remaining -= 1

        # Shared latent factor (per-step, standard normal).
        shared = self.np_random.normal(0.0, 1.0)

        for warehouse in LEAF_WAREHOUSES:
            cfg = self.warehouse_config[warehouse]
            base_mean = float(cfg['demand_mean'])
            base_std = float(cfg['demand_std'])

            mean_t = self._seasonal_mean(base_mean) * active_mult
            idio = self.np_random.normal(0.0, 1.0)
            noise_z = (
                self.demand_corr * shared
                + np.sqrt(max(0.0, 1.0 - self.demand_corr ** 2)) * idio
            )
            demand = max(0, int(round(mean_t + base_std * noise_z)))

            if self.ioh[warehouse] >= demand:
                self.ioh[warehouse] -= demand
            else:
                fulfilled = max(0, self.ioh[warehouse])
                shortage = demand - fulfilled
                self.ioh[warehouse] = 0
                self.backlog[warehouse].append([shortage, 0])

    def _place_order(self, warehouse: str, quantity: int, upstream: str):
        """Stochastic lead time with heavier tails + occasional spikes."""
        config = self.warehouse_config[warehouse]
        reorder_cost = max(
            config['min_reorder_cost'],
            config['reorder_cost_constant'] * quantity,
        )
        self.last_reorder_cost += reorder_cost

        base_std = float(config['lead_time_std']) * self.lead_time_std_mult
        lt = self.np_random.normal(config['lead_time_mean'], base_std)
        if self.np_random.rand() < self.lead_time_spike_prob:
            lt += self.lead_time_spike_extra
        lead_time = int(np.clip(round(lt), 1, self.lead_time_max))

        arrival_day = self.current_step + lead_time
        self.open_orders[warehouse].append({
            'qty': quantity,
            'age': 0,
            'arrival': arrival_day,
            'upstream': upstream,
        })

    def _process_delivery(self):
        """Same as base, but middle warehouse has a stochastic outbound
        capacity cap when shipping down to leaves on any given day."""
        middle_cap_remaining = max(
            0.0,
            float(self.np_random.normal(self.middle_capacity_mean, self.middle_capacity_std)),
        )

        for warehouse in WAREHOUSES:
            upstream = 'factory' if warehouse == 'middle' else 'middle'

            for order in self.open_orders[warehouse]:
                order['age'] += 1

            orders = self.open_orders[warehouse]
            arriving_today = sorted(
                [o for o in orders if o['arrival'] == self.current_step],
                key=lambda x: -x['age'],
            )
            still_waiting = [o for o in orders if o['arrival'] != self.current_step]
            orders_to_keep = still_waiting

            for order in arriving_today:
                quantity = order['qty']
                if upstream == 'factory':
                    self.ioh[warehouse] += quantity
                else:
                    available_stock = self.ioh[upstream]
                    # Cap by both inventory AND today's outbound capacity.
                    available = min(available_stock, middle_cap_remaining)
                    if available >= quantity:
                        self.ioh[upstream] -= quantity
                        self.ioh[warehouse] += quantity
                        middle_cap_remaining -= quantity
                    else:
                        if available > 0:
                            self.ioh[warehouse] += available
                            order['qty'] -= available
                            self.ioh[upstream] -= available
                            middle_cap_remaining -= available
                        # Push delivery of the remainder to tomorrow.
                        order['arrival'] = self.current_step + 1
                        orders_to_keep.append(order)

            self.open_orders[warehouse] = deque(orders_to_keep)

    def _update_backlog(self):
        """Same as base, but count items aged-out as lost sales for cost."""
        self._lost_sales_step = 0.0
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
                    else:
                        # Aged out -> lost sale.
                        self._lost_sales_step += quantity

            self.ioh[warehouse] = ioh
            self.backlog[warehouse] = updated_backlog

    def _get_total_cost(self) -> Dict:
        """Base cost breakdown + an explicit lost-sales line."""
        cost_breakdown = super()._get_total_cost()
        lost_sales_cost = self.lost_sales_penalty * float(self._lost_sales_step)
        cost_breakdown['lost_sales'] = lost_sales_cost
        cost_breakdown['total'] += lost_sales_cost
        # Replace the last history entry written by super() so total matches.
        if self.total_cost_history:
            self.total_cost_history[-1] = cost_breakdown
        return cost_breakdown

    def reset(self) -> np.ndarray:
        self._shock_steps_remaining = 0
        self._shock_multiplier = 1.0
        self._lost_sales_step = 0.0
        return super().reset()
