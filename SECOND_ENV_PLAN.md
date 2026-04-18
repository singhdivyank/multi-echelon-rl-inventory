# Plan: Second Environment for Train / Eval / Comparison

**Branch:** `env-second-environment`

**Prof's feedback:** Training + evaluation was shown on a single environment per algorithm. If that environment's results are a fluke, we cannot tell. Re-run on at least one additional, different environment and explicitly explain *why* each algorithm succeeds or fails on each environment, then compare.

**My scope (Ishan):** Add one new, more complex environment for each algorithm (PPO and A3C), re-run training + evaluation there, and produce a side-by-side comparison of Env-1 vs Env-2 results with success/failure reasoning. Divyank handles the other poster changes.

---

## 1. Current State (Env-1, the "original" envs)

| Algo | File | Env Class | Action | State dim | Horizon | Key dynamics |
|------|------|-----------|--------|-----------|---------|--------------|
| PPO | `ppo/models/env.py` | `DivergentInventoryEnv` | Continuous `Box(4,)` in `[-1,1]` | 11 | 500 steps | 1 warehouse → 3 retailers, fixed lead time = 1, stationary Poisson(Unif[5,15]) demand per retailer, backorders only |
| A3C | `actor-critic/src/meis_env.py` | `MEISEnv` | Discrete(13) | 12 | 365 steps | Factory → middle → 2 leaves, Normal lead time (mean 2, std 1), stationary Normal demand (mean 3300, std 100), backlog with max age = 7 |

Both envs are **stationary, mostly benign, small**. The learned policy could be exploiting the stationarity.

---

## 2. Env-2 Design — "Complex" Variants

Goal: introduce realistic supply-chain stressors so the *same* algorithms must generalize. Keep the interface identical (same obs/action space shapes where feasible) so we do not need to re-architect the networks.

### Common stressors added (both algos)

1. **Non-stationary, seasonal demand** — sinusoidal mean + noise:
   `mean_t = base * (1 + A * sin(2π t / period)) + trend * t`
2. **Stochastic lead times with higher variance** — lead time `L ~ max(1, round(N(μ, σ)))` with larger σ and an occasional delay spike (prob `p_delay` adds `+k` days).
3. **Supplier capacity cap** — upstream cannot always ship the full order: `shipped = min(ordered, capacity_t)` where `capacity_t` itself is stochastic.
4. **Demand shocks** — with small probability per step, demand scales by a shock multiplier (`×2`–`×3`) for a few consecutive steps.
5. **Correlated retailer demand** — shared latent factor `z_t` drives all retailers, so shortages tend to happen simultaneously (worst case for pooling).
6. **Partial lost sales** — a fraction `α` of unmet demand is lost (revenue loss penalty) instead of fully backlogged.
7. **Slightly larger network** — PPO: 3 → 5 retailers; A3C: 2 → 3 leaf warehouses (only if it fits existing discrete action head without redesign — otherwise keep 2 leaves and only add stressors 1–6).

### PPO Env-2: `ComplexDivergentInventoryEnv`
- File: `ppo/models/env_complex.py`
- Config block: `complex:` in `ppo/configs/config.yaml` (or new `config_complex.yaml`)
- Subclasses `BaseInventoryEnv` (or `DivergentInventoryEnv`) and overrides `_generate_demand`, `_event_2_supplier_shipment`, `_event_4_transport_orders`, `_event_5_customer_demand`, plus stochastic lead-time sampling.
- Keeps action space continuous `Box` (dim = 1 + num_retailers) and observation space normalized Box.

### A3C Env-2: `ComplexMEISEnv`
- File: `actor-critic/src/complex_meis_env.py`
- Config: `actor-critic/configs/complexMeisConfig.yaml`
- Subclasses `MEISEnv` and overrides `_generate_demand`, `_place_order` (stochastic/capped lead times), `_process_delivery` (capacity cap at middle), cost function (adds lost-sales penalty).
- Keeps `Discrete(13)` action space so the existing actor head works unchanged.

### Episode length
- Keep horizons identical to Env-1 (500 / 365 steps) so only dynamics differ.

---

## 3. Wiring

### PPO `ppo/main.py`
- Add CLI flag `--env {divergent,complex}` (default `divergent`).
- Select env class + config block based on flag.
- Change `save_dir` to `./results_<env>` to avoid clobbering Env-1 artifacts.

### A3C `actor-critic/main.py`
- Add CLI flag `--env {meis,complex}` (default `meis`).
- `MEISEnv` currently loads config from `_get_meis_env()`; change that helper (or add `_get_complex_meis_env()`) to route based on flag.
- Change `save_dir` to `./results_<env>`.

### Baselines
- (s, S) tuner must be re-run on Env-2 (different dynamics → different optimal s, S).
- Same code path, just run against the new env instance.

---

## 4. Execution Pipeline

### Phase 0 — Plan review *(current commit)*
- This file only. No code changes.

### Phase 1 — Pre-training *(me, this branch)*
1. Implement `ComplexDivergentInventoryEnv` + config.
2. Implement `ComplexMEISEnv` + config.
3. Wire `--env` CLI flag in both `main.py` files; isolate result dirs.
4. Re-tune (s, S) on Env-2 for both algos (part of `main.py` flow already).
5. Smoke test each env (reset + 20 random steps, assert shapes/bounds, print a cost trace). No training yet.
6. Commit. **Stop and hand off.**

### Phase 2 — Training *(user, manual)*
```powershell
# PPO on Env-2
cd ppo
python main.py --mode train --env complex

# A3C on Env-2
cd ..\actor-critic
python main.py --mode train --env complex
```
User confirms training finished and checkpoints exist in `results_complex/checkpoints/`.

### Phase 3 — Post-training *(me, same branch)*
1. Run evaluation:
   - `python ppo/main.py --mode eval --env complex`
   - `python actor-critic/main.py --mode eval --env complex`
2. Generate plots (training curves, cost distributions, cost-per-period) for Env-2.
3. Build **cross-env comparison artifacts**:
   - `docs/env_comparison.md` with:
     - Table A: Env-1 vs Env-2 spec diff (demand model, lead time, capacity, shocks, correlation, lost sales).
     - Table B: Metrics side-by-side — for each algo × env: Avg Cost (model vs baseline vs %imp), Service Level, Cost Std Dev, training wall-clock / episodes.
     - Success/failure narrative per (algo, env) cell explaining *why*:
       - Env-1 PPO succeeds (stationary → clipped updates converge on optimal replenishment curve).
       - Env-1 A3C marginal gain (small discrete action grid + small network + stable demand leaves little room over (s,S)).
       - Env-2 PPO: hypothesis — retains improvement but smaller (seasonality + shocks raise variance; GAE helps). If it fails, reason = non-stationarity breaks value estimate / reward scaling.
       - Env-2 A3C: hypothesis — likely underperforms (discrete action grid too coarse for shocks + capacity caps; workers see different demand regimes → noisier gradients).
   - `docs/env_comparison.png` — grouped bar chart (algo × env) of % improvement over baseline.
4. Update top-level `README.md` Results table with Env-2 row.
5. Draft LaTeX snippet for poster/report update (table + 2–3 bullet insights) — saved as `docs/poster_update_snippet.tex` for Divyank to merge if wanted.
6. Commit. Open PR back into `main`.

---

## 5. Deliverables Checklist

- [ ] `ppo/models/env_complex.py`
- [ ] `ppo/configs/config.yaml` — `complex:` block (or new file)
- [ ] `ppo/main.py` — `--env` flag + env-scoped `save_dir`
- [ ] `actor-critic/src/complex_meis_env.py`
- [ ] `actor-critic/configs/complexMeisConfig.yaml`
- [ ] `actor-critic/main.py` — `--env` flag + env-scoped `save_dir`
- [ ] `actor-critic/utils/helpers.py` — loader for complex config
- [ ] Smoke test evidence (printed in Phase 1 commit message or a short `scripts/smoke_env.py`)
- [ ] *(After manual training)* `results_complex/` populated for both algos
- [ ] `docs/env_comparison.md`
- [ ] `docs/env_comparison.png`
- [ ] `README.md` updated Results section
- [ ] `docs/poster_update_snippet.tex`

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Env-2 action/state normalization caps (`max_orders`, `max_inventory_*`) are too small for the new demand scale | Widen caps in the `complex` config; verify via smoke test that observations don't saturate at ±1 most of the time. |
| Discrete A3C action head (13 actions) is too coarse for Env-2 capacity constraints | Keep the same head; this IS a legitimate "why A3C fails" story if it fails. |
| Reward magnitude changes break value-function scaling | Keep the same `-cost/10000` scaling; if training diverges, note it as a finding rather than re-tuning. |
| Seed-sensitivity makes Env-2 result look like a fluke again | Run eval over ≥100 episodes (already the default); in post-training also report std dev and quantiles. |
| Baseline (s,S) tuner takes too long on Env-2 | Reuse `tuning_episodes` / `tuning_maxiter` from existing config; bump only if needed after smoke test. |

---

## 7. Open Questions for You Before Phase 1

1. **Both algos or just one?** Plan covers both PPO and A3C. If you only want to demo Env-2 for one of them (e.g., just PPO since it's the headline 33% result), say so and I'll trim.
2. **How "complex" is too complex?** Default plan enables all 6 stressors at once. If you want a milder Env-2 (to give methods a fair shot), I can stage them behind config flags and we start with only seasonality + stochastic lead time.
3. **Network size** for complex env — stay as-is (64, 64) or bump to (128, 128)? Default: stay as-is so the comparison isolates *env difficulty*, not capacity.
4. **Bigger network / more retailers** — should I actually scale the topology (3 → 5 retailers for PPO), or keep the topology identical and only change dynamics? Default: keep topology, only change dynamics (cleaner comparison).

---

## 8. Timeline (rough)

- Phase 1 (pre-training, me): ~1–2 hrs of coding + smoke tests
- Phase 2 (training, you): whatever PPO 5k iters + A3C 10k eps take on your box
- Phase 3 (post-training, me): ~1–2 hrs (eval, plots, write-up)
