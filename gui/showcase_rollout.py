"""
Greedy showcase rollouts with per-step callbacks for Streamlit / demos.
Branched by phase: A (Gymnasium), B/C (SumoEnvironment multi-agent).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import sumo_rl  # noqa: F401  — registers gym id

from agent import DQNAgent

ShouldStop = Callable[[], bool]
OnStepA = Callable[[int, dict[str, Any]], None]
OnStepMA = Callable[[int, dict[str, Any]], None]


def repo_root() -> Path:
    return _REPO


def load_config_phase(phase: str, config_path: str | None) -> dict:
    phase = phase.upper()
    if phase == "A":
        from train import load_config

        return load_config(config_path)
    if phase == "B":
        from train_phase_b import load_config

        return load_config(config_path)
    if phase == "C":
        from train_phase_c import load_config

        return load_config(config_path)
    raise ValueError(f"Unknown phase {phase}")


def run_showcase_phase_a(
    cfg: dict,
    *,
    gui: bool,
    checkpoint_path: str | None,
    step_delay_sec: float,
    on_step: OnStepA,
    should_stop: ShouldStop,
) -> dict[str, Any]:
    from train import make_env

    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("SUMO_HOME is not set.")

    root = repo_root()
    out = cfg["outputs"]
    ck_dir = root / out["checkpoint_dir"]
    ck = Path(checkpoint_path) if checkpoint_path else ck_dir / out["best_name"]
    if not ck.is_file():
        alt = ck_dir / "last_phase_a.pth"
        ck = alt if alt.is_file() else ck

    env = make_env(cfg, gui=gui)
    obs_dim = env.observation_space.shape[0]
    n_act = int(env.action_space.n)
    hs = int(cfg["training"]["hidden_size"])
    agent = DQNAgent(
        state_size=obs_dim,
        action_size=n_act,
        hidden_size=hs,
    )
    if not agent.load(ck):
        env.close()
        raise FileNotFoundError(f"No checkpoint at {ck}")

    state, info = env.reset()
    step = 0
    summary: dict[str, Any] = {"steps": 0, "total_reward": 0.0, "error": None}
    try:
        while True:
            if should_stop():
                break
            action = int(agent.act(state, epsilon=0.0))
            q_flat = agent.q_values(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            payload = {
                "phase_key": "A",
                "obs": np.asarray(state),
                "action": action,
                "q_values": np.asarray(q_flat, dtype=np.float64).ravel().tolist(),
                "reward": float(reward),
                "info": dict(info) if isinstance(info, dict) else {},
                "done": done,
                "n_green_phases": n_act,
            }
            on_step(step, payload)
            summary["total_reward"] += float(reward)
            summary["steps"] = step + 1
            state = next_state
            step += 1
            if step_delay_sec > 0:
                time.sleep(step_delay_sec)
            if done:
                break
    except Exception as e:
        summary["error"] = str(e)
        raise
    finally:
        env.close()
    return summary


def _build_agents_b(
    cfg: dict, env
) -> tuple[dict[str, DQNAgent], list[str]]:
    from sumo_rl.environment.env import SumoEnvironment

    assert isinstance(env, SumoEnvironment)
    tls_ids = sorted(env.ts_ids)
    ck_dir = repo_root() / cfg["outputs"]["checkpoint_dir"]
    hs = int(cfg["training"]["hidden_size"])
    obs_dim = env.observation_space.shape[0]
    n_act = int(env.action_space.n)
    agents: dict[str, DQNAgent] = {}
    for tid in tls_ids:
        agents[tid] = DQNAgent(
            state_size=obs_dim,
            action_size=n_act,
            hidden_size=hs,
        )
        ck = ck_dir / f"best_{tid}.pth"
        if not ck.is_file():
            ck = ck_dir / f"last_{tid}.pth"
        if not agents[tid].load(ck):
            raise FileNotFoundError(f"No checkpoint for {tid} at {ck}")
    return agents, tls_ids


def _build_agents_c(cfg: dict, env) -> tuple[dict[str, DQNAgent], list[str]]:
    from sumo_rl.environment.env import SumoEnvironment

    assert isinstance(env, SumoEnvironment)
    tls_ids = sorted(env.ts_ids)
    ck_dir = repo_root() / cfg["outputs"]["checkpoint_dir"]
    hs = int(cfg["training"]["hidden_size"])
    agents: dict[str, DQNAgent] = {}
    for tid in tls_ids:
        odim = int(env.observation_spaces(tid).shape[0])
        n_act = int(env.action_spaces(tid).n)
        agents[tid] = DQNAgent(
            state_size=odim,
            action_size=n_act,
            hidden_size=hs,
        )
        ck = ck_dir / f"best_{tid}.pth"
        if not ck.is_file():
            ck = ck_dir / f"last_{tid}.pth"
        if not agents[tid].load(ck):
            raise FileNotFoundError(f"No checkpoint for {tid} at {ck}")
    return agents, tls_ids


def run_showcase_phase_b(
    cfg: dict,
    *,
    gui: bool,
    step_delay_sec: float,
    on_step: OnStepMA,
    should_stop: ShouldStop,
) -> dict[str, Any]:
    from train_phase_b import make_env

    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("SUMO_HOME is not set.")

    env = make_env(cfg, gui=gui)
    agents, tls_ids = _build_agents_b(cfg, env)
    obs = env.reset()
    step = 0
    total = {tid: 0.0 for tid in tls_ids}
    summary: dict[str, Any] = {"steps": 0, "per_tls_return": dict(total), "error": None}
    try:
        while True:
            if should_stop():
                break
            actions = {tid: agents[tid].act(obs[tid], epsilon=0.0) for tid in tls_ids}
            q_val_map = {
                tid: np.asarray(agents[tid].q_values(obs[tid]), dtype=np.float64).ravel().tolist()
                for tid in tls_ids
            }
            next_obs, rewards, dones, info = env.step(actions)
            payload = {
                "phase_key": "B",
                "obs": {tid: np.asarray(obs[tid]) for tid in tls_ids},
                "actions": {tid: int(actions[tid]) for tid in tls_ids},
                "q_values": q_val_map,
                "rewards": {tid: float(rewards[tid]) for tid in tls_ids},
                "info": dict(info) if isinstance(info, dict) else {},
                "dones": dict(dones),
                "tls_ids": tls_ids,
                "n_green_phases": int(env.action_space.n),
            }
            on_step(step, payload)
            for tid in tls_ids:
                total[tid] += float(rewards[tid])
            summary["steps"] = step + 1
            summary["per_tls_return"] = {k: float(v) for k, v in total.items()}
            obs = next_obs
            step += 1
            if step_delay_sec > 0:
                time.sleep(step_delay_sec)
            if dones.get("__all__"):
                break
    except Exception as e:
        summary["error"] = str(e)
        raise
    finally:
        env.close()
    return summary


def run_showcase_phase_c(
    cfg: dict,
    *,
    gui: bool,
    step_delay_sec: float,
    on_step: OnStepMA,
    should_stop: ShouldStop,
) -> dict[str, Any]:
    from train_phase_c import make_env

    if "SUMO_HOME" not in os.environ:
        raise RuntimeError("SUMO_HOME is not set.")

    env = make_env(cfg, gui=gui)
    agents, tls_ids = _build_agents_c(cfg, env)
    obs = env.reset()
    step = 0
    total = {tid: 0.0 for tid in tls_ids}
    summary: dict[str, Any] = {"steps": 0, "per_tls_return": dict(total), "error": None}
    try:
        while True:
            if should_stop():
                break
            actions = {tid: agents[tid].act(obs[tid], epsilon=0.0) for tid in tls_ids}
            q_val_map = {
                tid: np.asarray(agents[tid].q_values(obs[tid]), dtype=np.float64).ravel().tolist()
                for tid in tls_ids
            }
            next_obs, rewards, dones, info = env.step(actions)
            n_green = {tid: int(env.action_spaces(tid).n) for tid in tls_ids}
            payload = {
                "phase_key": "C",
                "obs": {tid: np.asarray(obs[tid]) for tid in tls_ids},
                "actions": {tid: int(actions[tid]) for tid in tls_ids},
                "q_values": q_val_map,
                "rewards": {tid: float(rewards[tid]) for tid in tls_ids},
                "info": dict(info) if isinstance(info, dict) else {},
                "dones": dict(dones),
                "tls_ids": tls_ids,
                "n_green_phases": n_green,
            }
            on_step(step, payload)
            for tid in tls_ids:
                total[tid] += float(rewards[tid])
            summary["steps"] = step + 1
            summary["per_tls_return"] = {k: float(v) for k, v in total.items()}
            obs = next_obs
            step += 1
            if step_delay_sec > 0:
                time.sleep(step_delay_sec)
            if dones.get("__all__"):
                break
    except Exception as e:
        summary["error"] = str(e)
        raise
    finally:
        env.close()
    return summary
