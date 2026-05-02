"""
Fixed-time baseline for Phase B (two TLS): same KPI family as training (system mean wait).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from train_phase_b import _root, load_config, make_env


def run_baseline(cfg: dict | None = None) -> dict:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)
    cfg = cfg if cfg is not None else load_config()
    env = make_env(cfg, fixed_ts=True)
    env.reset()
    waits: list[float] = []
    n = 0
    while True:
        _, _, dones, info = env.step({})
        if isinstance(info, dict) and "system_mean_waiting_time" in info:
            waits.append(float(info["system_mean_waiting_time"]))
        n += 1
        if dones["__all__"]:
            break
    env.close()
    out = {
        "mean_system_mean_waiting_time": float(np.mean(waits)) if waits else None,
        "rl_steps": n,
        "horizon_seconds": cfg["env"]["num_seconds"],
        "sumo_seed": cfg["env"].get("sumo_seed"),
    }
    p = _root() / "runs" / "phase_b_baseline.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {p}", flush=True)
    return out


if __name__ == "__main__":
    run_baseline()
