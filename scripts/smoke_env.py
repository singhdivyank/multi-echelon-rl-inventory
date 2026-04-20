"""Smoke tests for the second-environment pipeline.

Verifies that *before* kicking off the long manual training run, the new
complex envs and their integration with the existing agents actually work:

    1. Both complex envs can be instantiated from their configs.
    2. `reset()` returns a correctly-shaped observation.
    3. ~50 random steps execute without raising, without NaN, and with finite
       costs.
    4. Each agent can compute one full update on a rollout from the new env
       (PPO = buffer-sized rollout; A3C = single-episode update).
    5. Env-1 still works (regression check).

Run as:

    python scripts/smoke_env.py

Exits 0 on success; prints a concise pass/fail table.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

# Make both algorithm packages importable and allow them to use their usual
# relative-path config loading (they do open('./configs/...') from cwd).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def _check_finite(arr, label: str) -> None:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{label} contains non-finite values")


# --------------------------------------------------------------------------- #
# PPO smoke
# --------------------------------------------------------------------------- #

def smoke_ppo(env_name: str) -> None:
    """Drive PPO on `divergent` or `complex` for one buffer update."""
    cwd_prev = os.getcwd()
    os.chdir(os.path.join(REPO_ROOT, "ppo"))
    sys.path.insert(0, os.getcwd())
    try:
        from models.env import DivergentInventoryEnv
        from models.env_complex import ComplexDivergentInventoryEnv
        from models.ppo import PPOAgent
        from models.baseline import BaselineAgent
        from utils.helpers import load_config, set_device

        cfg = load_config()
        ppo_cfg = dict(cfg['ppo'])
        # Shrink the buffer so a single rollout is enough to trigger an update.
        ppo_cfg['buffer_size'] = 64
        ppo_cfg['batch_size'] = 16
        ppo_cfg['update_epochs'] = 2

        env_cfg = dict(cfg[env_name])
        env_cls = ComplexDivergentInventoryEnv if env_name == 'complex' else DivergentInventoryEnv
        env = env_cls(config=env_cfg)

        state, _ = env.reset(seed=0)
        assert state.shape == env.observation_space.shape, (
            f"obs shape mismatch: {state.shape} vs {env.observation_space.shape}"
        )
        _check_finite(state, "initial state")

        # Random rollout sanity
        total_cost = 0.0
        for _ in range(30):
            a = env.action_space.sample()
            s, r, term, trunc, info = env.step(a)
            _check_finite(s, "state")
            assert np.isfinite(r), "reward not finite"
            assert np.isfinite(info['cost']), "cost not finite"
            total_cost += info['cost']
            if term or trunc:
                state, _ = env.reset(seed=1)
        assert total_cost >= 0, "cost went negative (impossible)"

        # Baseline sanity
        baseline = BaselineAgent(env)
        state, _ = env.reset(seed=2)
        a = baseline.get_action(state)
        assert a.shape == env.action_space.shape, "baseline action shape mismatch"

        # PPO single-update sanity
        device = set_device('auto')
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=ppo_cfg,
            device=device,
        )
        agent.train()
        state, _ = env.reset(seed=3)
        did_update = False
        for _ in range(ppo_cfg['buffer_size'] + 5):
            a, lp, v = agent.get_action(state, deterministic=False)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.store_transition(
                state=state, action=a, reward=r, next_state=ns,
                done=done, log_prob=lp, value=v,
            )
            state = ns if not done else env.reset(seed=4)[0]
            if len(agent.buffer) >= agent.buffer_size:
                stats = agent.update()
                if stats:
                    for k in ('policy_loss', 'value_loss', 'entropy'):
                        assert np.isfinite(stats[k]), f"PPO {k} not finite"
                    did_update = True
                    break
        assert did_update, "PPO buffer never filled; smoke did not exercise update()"
        print(f"  [ppo:{env_name}] OK  | random-rollout cost={total_cost:.1f}  update stats={list(stats.keys())}")
    finally:
        os.chdir(cwd_prev)
        # Leave sys.path alone; leaking a ppo path entry for the rest of the
        # script would break A3C (conflicting `models` package). Pop it.
        if sys.path and sys.path[0].endswith('ppo'):
            sys.path.pop(0)
        # Drop imported ppo modules so A3C section gets a clean slate.
        for mod in list(sys.modules):
            if mod.startswith('models') or mod.startswith('utils') or mod.startswith('src'):
                del sys.modules[mod]


# --------------------------------------------------------------------------- #
# A3C smoke
# --------------------------------------------------------------------------- #

def smoke_a3c(env_name: str) -> None:
    """Drive A3C on `meis` or `complex` for one episode + one update."""
    cwd_prev = os.getcwd()
    os.chdir(os.path.join(REPO_ROOT, "actor-critic"))
    sys.path.insert(0, os.getcwd())
    try:
        from src.meis_env import MEISEnv
        from src.complex_meis_env import ComplexMEISEnv
        from src.a3c_agent import A3CAgent
        from utils.helpers import _load_config, _get_complex_meis_env

        cfg = _load_config()
        agent_cfg = dict(cfg['agent'])

        if env_name == 'complex':
            env = ComplexMEISEnv(config=_get_complex_meis_env())
        else:
            env = MEISEnv()

        state = env.reset()
        assert state.shape == env.observation_space.shape, "A3C obs shape mismatch"
        _check_finite(state, "initial state")

        # Shortened episode for speed.
        env.max_steps = 30

        agent = A3CAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            agent_config=agent_cfg,
        )

        states, actions, rewards, values, dones = [], [], [], [], []
        state = env.reset()
        done = False
        step = 0
        while not done and step < env.max_steps:
            a, lp, v = agent.select_action(state, deterministic=False)
            ns, r, done, info = env.step(a)
            _check_finite(ns, "next_state")
            assert np.isfinite(r), "A3C reward not finite"
            assert 'cost_breakdown' in info and 'service_level' in info
            states.append(state); actions.append(a); rewards.append(r)
            values.append(v); dones.append(done)
            state = ns
            step += 1

        assert len(states) > 0, "A3C rollout was empty"
        advs, rets = agent.compute_gae(rewards, values, dones, next_value=0.0)
        metrics = agent.update(states, actions, advs, rets)
        for k in ('actor_loss', 'critic_loss', 'entropy'):
            assert np.isfinite(metrics[k]), f"A3C {k} not finite"

        # Deterministic path check (relevant since we fixed a bug that made
        # eval stochastic). Calling twice from the same state should produce
        # the same argmax action.
        state = env.reset()
        a1, _, _ = agent.select_action(state, deterministic=True)
        a2, _, _ = agent.select_action(state, deterministic=True)
        assert a1 == a2, f"deterministic action not reproducible: {a1} vs {a2}"
        print(f"  [a3c:{env_name}] OK  | steps={len(states)}  update={list(metrics.keys())}")
    finally:
        os.chdir(cwd_prev)
        if sys.path and sys.path[0].endswith('actor-critic'):
            sys.path.pop(0)
        for mod in list(sys.modules):
            if mod.startswith('src') or mod.startswith('utils') or mod.startswith('models'):
                del sys.modules[mod]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    results = []
    for label, fn in [
        ("PPO / divergent (Env-1)", lambda: smoke_ppo('divergent')),
        ("PPO / complex   (Env-2)", lambda: smoke_ppo('complex')),
        ("A3C / meis      (Env-1)", lambda: smoke_a3c('meis')),
        ("A3C / complex   (Env-2)", lambda: smoke_a3c('complex')),
    ]:
        _banner(f"SMOKE: {label}")
        try:
            fn()
            results.append((label, True, ""))
        except Exception as e:
            traceback.print_exc()
            results.append((label, False, f"{type(e).__name__}: {e}"))

    _banner("SMOKE SUMMARY")
    ok = True
    for label, passed, err in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {label}" + (f"   ({err})" if err else ""))
        ok = ok and passed
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
