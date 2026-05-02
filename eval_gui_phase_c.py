"""Greedy GUI eval for Phase C: load best_<tls_id>.pth (heterogeneous obs/act per TLS)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from agent import DQNAgent
from train_phase_c import load_config, make_env


def main() -> None:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent
    cfg = load_config()
    ck_dir = root / cfg["outputs"]["checkpoint_dir"]
    env = make_env(cfg, gui=True)
    tls_ids = sorted(env.ts_ids)
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
            print(
                f"No checkpoint for {tid} at {ck}; run train_phase_c.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

    obs = env.reset()
    total = {tid: 0.0 for tid in tls_ids}
    steps = 0
    while True:
        actions = {tid: agents[tid].act(obs[tid], epsilon=0.0) for tid in tls_ids}
        obs, rewards, dones, _info = env.step(actions)
        for tid in tls_ids:
            total[tid] += float(rewards[tid])
        steps += 1
        time.sleep(0.03)
        if dones["__all__"]:
            break
    env.close()
    print(
        f"Phase C GUI episode | steps={steps} | per-TLS return { {k: round(v, 2) for k, v in total.items()} }",
        flush=True,
    )


if __name__ == "__main__":
    main()
