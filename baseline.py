"""
Fixed-time traffic signal baseline (SUMO program in net.xml): same horizon as training.
Use to compare Phase A validation KPIs (lower mean waiting time is better).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import sumo_rl  # noqa: F401

from train import _root, load_config, make_env


def run_baseline() -> dict:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    env = make_env(cfg, fixed_ts=True)
    state, info = env.reset()
    total_r = 0.0
    waits: list[float] = []
    queues: list[float] = []
    n = 0
    while True:
        # Action ignored when fixed_ts=True
        state, reward, term, trunc, info = env.step(0)
        total_r += float(reward)
        if isinstance(info, dict):
            if "system_mean_waiting_time" in info:
                waits.append(float(info["system_mean_waiting_time"]))
            if "agents_total_accumulated_waiting_time" in info:
                queues.append(float(info["agents_total_accumulated_waiting_time"]))
        n += 1
        if term or trunc:
            break
    env.close()

    out = {
        "fixed_ts_total_reward": total_r,
        "mean_system_mean_waiting_time": float(np.mean(waits)) if waits else None,
        "final_agents_total_accumulated_waiting_time": queues[-1] if queues else None,
        "rl_steps": n,
        "horizon_seconds": cfg["env"]["num_seconds"],
    }
    p = _root() / "runs" / "phase_a_baseline.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {p}", flush=True)
    return out


if __name__ == "__main__":
    run_baseline()
