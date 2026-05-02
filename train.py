"""
Phase A training: Gymnasium + SUMO-RL single-agent, Double Dueling DQN, uniform replay.
Validation: periodic greedy rollouts; best checkpoint by lowest mean system waiting time.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

# Registers gymnasium env id sumo-rl-v0
import sumo_rl  # noqa: F401

from agent import DQNAgent


def _root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("TRAFFIC_RL_CONFIG", _root() / "config.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: dict, *, gui: bool = False, fixed_ts: bool = False) -> gym.Env:
    e = cfg["env"]
    paths = cfg["paths"]
    root = _root()
    net = str(root / paths["net_file"])
    route = str(root / paths["route_file"])
    seed = e.get("sumo_seed", "random")
    if seed != "random":
        seed = int(seed)

    return gym.make(
        "sumo-rl-v0",
        net_file=net,
        route_file=route,
        use_gui=gui,
        num_seconds=int(e["num_seconds"]),
        delta_time=int(e["delta_time"]),
        yellow_time=int(e["yellow_time"]),
        min_green=int(e["min_green"]),
        waiting_time_memory=int(e["waiting_time_memory"]),
        max_depart_delay=int(e["max_depart_delay"]),
        sumo_warnings=bool(e.get("sumo_warnings", False)),
        reward_fn=e.get("reward_fn", "diff-waiting-time"),
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
        additional_sumo_cmd="--no-step-log",
        out_csv_name=None,
    )


def run_rollout(
    env: gym.Env,
    agent: DQNAgent,
    *,
    epsilon: float,
    learn: bool,
    batch_size: int,
    tau: float,
    sync_every: int,
    step_offset: int = 0,
) -> tuple[float, float, int, float, int]:
    """
    Returns (total_reward, mean_system_mean_waiting_time, n_rl_steps, mean_loss_or_0, new_global_step).
    """
    state, info = env.reset()
    total_r = 0.0
    wait_accum = 0.0
    losses: list[float] = []
    n = 0
    gstep = step_offset

    while True:
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if learn:
            agent.push(state, action, float(reward), next_state, done)
            if len(agent.buffer) >= batch_size:
                loss = agent.learn(batch_size)
                if loss is not None:
                    losses.append(loss)
            gstep += 1
            if gstep % sync_every == 0:
                agent.soft_update_target(tau)

        total_r += float(reward)
        if isinstance(info, dict) and "system_mean_waiting_time" in info:
            wait_accum += float(info["system_mean_waiting_time"])

        state = next_state
        n += 1
        if done:
            break

    mean_wait = wait_accum / max(n, 1)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return total_r, mean_wait, n, mean_loss, gstep


def train(cfg: dict | None = None) -> dict:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME to your SUMO installation.", file=sys.stderr)
        sys.exit(1)

    cfg = cfg if cfg is not None else load_config()
    root = _root()
    tcfg = cfg["training"]
    vcfg = cfg["validation"]
    out = cfg["outputs"]
    ck_dir = root / out["checkpoint_dir"]
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / out["log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    n_act = int(env.action_space.n)

    agent = DQNAgent(
        state_size=obs_dim,
        action_size=n_act,
        lr=float(tcfg["lr"]),
        gamma=float(tcfg["gamma"]),
        max_memory=int(tcfg["max_memory"]),
        max_grad=float(tcfg["max_grad"]),
        hidden_size=int(tcfg["hidden_size"]),
    )

    episodes = int(tcfg["episodes"])
    batch_size = int(tcfg["batch_size"])
    tau = float(tcfg["target_update_tau"])
    sync_every = int(tcfg["target_sync_steps"])

    eps_start = float(tcfg["epsilon_start"])
    eps_end = float(tcfg["epsilon_end"])
    eps_decay = float(tcfg["epsilon_decay"])
    epsilon = eps_start

    best_path = ck_dir / out["best_name"]
    best_val_wait = float("inf")

    fields = [
        "episode",
        "train_return",
        "train_mean_wait",
        "train_loss",
        "val_mean_wait",
        "val_return",
        "epsilon",
        "best_val_wait",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

    global_step = 0
    print(
        f"Phase A train | obs_dim={obs_dim} actions={n_act} "
        f"episodes={episodes} device={agent.device}",
        flush=True,
    )

    for ep in range(1, episodes + 1):
        tr, train_wait, steps, train_loss, global_step = run_rollout(
            env,
            agent,
            epsilon=epsilon,
            learn=True,
            batch_size=batch_size,
            tau=tau,
            sync_every=sync_every,
            step_offset=global_step,
        )

        val_wait = float("nan")
        val_ret = float("nan")
        if ep % int(vcfg["every_episodes"]) == 0:
            vw_list: list[float] = []
            vr_list: list[float] = []
            for _ in range(int(vcfg["episodes"])):
                vr, vw, _, _, _ = run_rollout(
                    env,
                    agent,
                    epsilon=0.0,
                    learn=False,
                    batch_size=batch_size,
                    tau=tau,
                    sync_every=sync_every,
                    step_offset=0,
                )
                vw_list.append(vw)
                vr_list.append(vr)
            val_wait = float(np.mean(vw_list))
            val_ret = float(np.mean(vr_list))
            if val_wait < best_val_wait:
                best_val_wait = val_wait
                agent.save(best_path)
                print(f"  [best] ep={ep} val_mean_wait={val_wait:.4f} -> saved {best_path}", flush=True)

        row = {
            "episode": ep,
            "train_return": round(tr, 3),
            "train_mean_wait": round(train_wait, 6),
            "train_loss": round(train_loss, 6),
            "val_mean_wait": round(val_wait, 6) if val_wait == val_wait else "",
            "val_return": round(val_ret, 3) if val_ret == val_ret else "",
            "epsilon": round(epsilon, 5),
            "best_val_wait": round(best_val_wait, 6) if best_val_wait < float("inf") else "",
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        print(
            f"ep {ep:4d}/{episodes} | train_R={tr:8.2f} train_wait={train_wait:8.4f} "
            f"loss={train_loss:8.5f} eps={epsilon:.4f}",
            flush=True,
        )
        if val_wait == val_wait:
            print(
                f"         val_mean_wait={val_wait:.4f} val_R={val_ret:.2f} "
                f"(lower wait is better)",
                flush=True,
            )

        epsilon = max(eps_end, epsilon * eps_decay)

    env.close()
    final_path = ck_dir / "last_phase_a.pth"
    agent.save(final_path)
    print(f"Saved last checkpoint -> {final_path}", flush=True)

    summary = {
        "best_val_system_mean_waiting_time": best_val_wait
        if best_val_wait < float("inf")
        else None,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "log_csv": str(log_path),
        "note": "Primary Phase A metric: validation mean system_mean_waiting_time (lower better).",
    }
    with open(root / out["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {root / out['summary_json']}", flush=True)
    summary["sumo_seed"] = cfg["env"].get("sumo_seed")
    return summary


if __name__ == "__main__":
    train()
