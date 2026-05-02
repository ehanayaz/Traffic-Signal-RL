"""Load best Phase A weights and run SUMO-GUI for one episode (greedy policy)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import sumo_rl  # noqa: F401

from agent import DQNAgent
from train import load_config, make_env


def main() -> None:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent
    cfg = load_config()
    ck = root / cfg["outputs"]["checkpoint_dir"] / cfg["outputs"]["best_name"]
    if not ck.is_file():
        ck = root / cfg["outputs"]["checkpoint_dir"] / "last_phase_a.pth"

    env = make_env(cfg, gui=True)
    obs_dim = env.observation_space.shape[0]
    n_act = int(env.action_space.n)
    agent = DQNAgent(
        state_size=obs_dim,
        action_size=n_act,
        hidden_size=int(cfg["training"]["hidden_size"]),
    )
    if not agent.load(ck):
        print(f"No checkpoint at {ck}; train first.", file=sys.stderr)
        sys.exit(1)

    state, _ = env.reset()
    total_r = 0.0
    while True:
        a = agent.act(state, epsilon=0.0)
        state, r, term, trunc, _ = env.step(a)
        total_r += float(r)
        time.sleep(0.03)
        if term or trunc:
            break
    env.close()
    print(f"Eval GUI episode total reward: {total_r:.2f}", flush=True)


if __name__ == "__main__":
    main()
