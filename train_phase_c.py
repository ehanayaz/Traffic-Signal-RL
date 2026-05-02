"""
Phase C: six traffic signals (2×3 grid), asymmetric spacing — IDQN with uniform replay.
Per-TLS obs/action sizes come from sumo_rl (often uniform once fringe approaches exist).

Primary validation KPI: mean system_mean_waiting_time (greedy rollouts), same as Phase B.
Progression-oriented metrics: trip duration mean / p95 from SUMO tripinfo (when enabled).
Per-TLS checkpoints: checkpoints/phase_c/best_<tls_id>.pth
"""

from __future__ import annotations

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from sumo_rl.environment.env import SumoEnvironment

from agent import DQNAgent


def _root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("TRAFFIC_RL_CONFIG_C", _root() / "config_phase_c.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tripinfo_durations(tripinfo_path: Path) -> list[float]:
    """Extract trip durations (seconds) from SUMO tripinfo output."""
    if not tripinfo_path.is_file():
        return []
    try:
        tree = ET.parse(tripinfo_path)
        root_el = tree.getroot()
        out: list[float] = []
        for el in root_el.iter():
            if el.tag == "tripinfo" or el.tag.endswith("}tripinfo"):
                d = el.get("duration")
                if d is not None:
                    out.append(float(d))
        return out
    except (ET.ParseError, OSError, ValueError):
        return []


def make_env(cfg: dict, *, fixed_ts: bool = False, gui: bool = False) -> SumoEnvironment:
    e = cfg["env"]
    paths = cfg["paths"]
    root = _root()
    net = str(root / paths["net_file"])
    route = str(root / paths["route_file"])
    seed = e.get("sumo_seed", "random")
    if seed != "random":
        seed = int(seed)

    extra = "--no-step-log"
    prog = cfg.get("progression") or {}
    if prog.get("tripinfo") and paths.get("tripinfo_output"):
        trip_p = (root / paths["tripinfo_output"]).resolve()
        trip_p.parent.mkdir(parents=True, exist_ok=True)
        extra = f"--no-step-log --tripinfo-output {trip_p}"

    return SumoEnvironment(
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
        single_agent=False,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
        additional_sumo_cmd=extra,
        out_csv_name=None,
    )


def run_episode(
    env: SumoEnvironment,
    agents: dict[str, DQNAgent],
    tls_ids: list[str],
    *,
    epsilon: float,
    learn: bool,
    batch_size: int,
    tau: float,
    sync_every: int,
    step_offset: int,
    close_after: bool = False,
) -> tuple[dict[str, float], float, int, float, int]:
    """
    Returns (returns_per_tls, mean_system_wait, n_steps, mean_loss, new_global_step).
    """
    obs = env.reset()
    losses: list[float] = []
    ret = {tid: 0.0 for tid in tls_ids}
    wait_accum = 0.0
    n_steps = 0
    gstep = step_offset

    while True:
        actions: dict[str, int] = {}
        for tid in tls_ids:
            actions[tid] = agents[tid].act(obs[tid], epsilon)

        next_obs, rewards, dones, info = env.step(actions)
        done_all = dones["__all__"]

        if learn:
            for tid in tls_ids:
                agents[tid].push(
                    obs[tid],
                    actions[tid],
                    float(rewards[tid]),
                    next_obs[tid],
                    done_all,
                )
                if len(agents[tid].buffer) >= batch_size:
                    loss = agents[tid].learn(batch_size)
                    if loss is not None:
                        losses.append(loss)
            gstep += 1
            if gstep % sync_every == 0:
                for tid in tls_ids:
                    agents[tid].soft_update_target(tau)

        for tid in tls_ids:
            ret[tid] += float(rewards[tid])
        if isinstance(info, dict) and "system_mean_waiting_time" in info:
            wait_accum += float(info["system_mean_waiting_time"])

        obs = next_obs
        n_steps += 1
        if done_all:
            break

    # SUMO writes tripinfo when the process exits; sumo_rl only tears down on reset()/close().
    if close_after:
        env.close()

    mean_wait = wait_accum / max(n_steps, 1)
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return ret, mean_wait, n_steps, mean_loss, gstep


def train(cfg: dict | None = None) -> dict:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)

    cfg = cfg if cfg is not None else load_config()
    root = _root()
    tcfg = cfg["training"]
    vcfg = cfg["validation"]
    out = cfg["outputs"]
    paths = cfg["paths"]
    ck_dir = root / out["checkpoint_dir"]
    ck_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, fixed_ts=False)
    tls_ids = sorted(env.ts_ids)
    prog_cfg = cfg.get("progression") or {}
    use_tripinfo = bool(prog_cfg.get("tripinfo")) and bool(paths.get("tripinfo_output"))
    tripinfo_path = (root / paths["tripinfo_output"]).resolve() if use_tripinfo else None

    agents: dict[str, DQNAgent] = {}
    dim_summary: list[str] = []
    per_tls_dims: dict[str, dict[str, int]] = {}
    for tid in tls_ids:
        obs_sp = env.observation_spaces(tid)
        act_sp = env.action_spaces(tid)
        odim = int(obs_sp.shape[0])
        n_act = int(act_sp.n)
        per_tls_dims[tid] = {"obs_dim": odim, "n_actions": n_act}
        dim_summary.append(f"{tid}:obs{odim}/act{n_act}")
        agents[tid] = DQNAgent(
            state_size=odim,
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

    log_path = root / out["log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ret_fields = [f"train_ret_{tid}" for tid in tls_ids]
    fields = [
        "episode",
        "epsilon",
        *ret_fields,
        "train_mean_wait",
        "train_loss",
        "val_system_mean_wait",
        "best_val_system_wait",
        "val_mean_trip_duration",
        "val_p95_trip_duration",
    ]

    best_system_wait = float("inf")
    global_step = 0
    last_val_trip_mean = float("nan")
    last_val_trip_p95 = float("nan")

    print(
        f"Phase C (IDQN, per-TLS networks) | tls={tls_ids} | "
        f"{' | '.join(dim_summary)} | episodes={episodes}",
        flush=True,
    )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for ep in range(1, episodes + 1):
        tr, train_wait, steps, train_loss, global_step = run_episode(
            env,
            agents,
            tls_ids,
            epsilon=epsilon,
            learn=True,
            batch_size=batch_size,
            tau=tau,
            sync_every=sync_every,
            step_offset=global_step,
        )

        val_sys_wait = float("nan")
        val_trip_mean = float("nan")
        val_trip_p95 = float("nan")
        if ep % int(vcfg["every_episodes"]) == 0:
            sys_waits: list[float] = []
            trip_ep_means: list[float] = []
            trip_ep_p95: list[float] = []
            for _ in range(int(vcfg["episodes"])):
                _, vw, _, _, _ = run_episode(
                    env,
                    agents,
                    tls_ids,
                    epsilon=0.0,
                    learn=False,
                    batch_size=batch_size,
                    tau=tau,
                    sync_every=sync_every,
                    step_offset=0,
                    close_after=bool(use_tripinfo),
                )
                sys_waits.append(vw)
                if use_tripinfo and tripinfo_path is not None:
                    durs = parse_tripinfo_durations(tripinfo_path)
                    if durs:
                        trip_ep_means.append(float(np.mean(durs)))
                        trip_ep_p95.append(float(np.percentile(durs, 95)))
            val_sys_wait = float(np.mean(sys_waits))
            if trip_ep_means:
                val_trip_mean = float(np.mean(trip_ep_means))
                val_trip_p95 = float(np.mean(trip_ep_p95))
                last_val_trip_mean = val_trip_mean
                last_val_trip_p95 = val_trip_p95
            if val_sys_wait < best_system_wait:
                best_system_wait = val_sys_wait
                for tid in tls_ids:
                    agents[tid].save(ck_dir / f"best_{tid}.pth")
                print(
                    f"  [best] ep={ep} val_system_mean_wait={val_sys_wait:.4f} "
                    f"-> saved {ck_dir}/best_*.pth",
                    flush=True,
                )

        row = {
            "episode": ep,
            "epsilon": round(epsilon, 5),
            **{f"train_ret_{tid}": round(tr.get(tid, 0.0), 3) for tid in tls_ids},
            "train_mean_wait": round(train_wait, 6),
            "train_loss": round(train_loss, 6),
            "val_system_mean_wait": round(val_sys_wait, 6)
            if val_sys_wait == val_sys_wait
            else "",
            "best_val_system_wait": round(best_system_wait, 6)
            if best_system_wait < float("inf")
            else "",
            "val_mean_trip_duration": round(val_trip_mean, 6)
            if val_trip_mean == val_trip_mean
            else "",
            "val_p95_trip_duration": round(val_trip_p95, 6)
            if val_trip_p95 == val_trip_p95
            else "",
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        ret_s = " ".join(f"{tid}={tr.get(tid, 0):5.2f}" for tid in tls_ids)
        print(
            f"ep {ep:4d}/{episodes} | {ret_s} "
            f"sys_wait={train_wait:7.3f} loss={train_loss:7.5f} eps={epsilon:.4f}",
            flush=True,
        )
        if val_sys_wait == val_sys_wait:
            extra = (
                f"         val_system_mean_wait={val_sys_wait:.4f} (lower is better)"
            )
            if val_trip_mean == val_trip_mean:
                extra += (
                    f" | trip_mean={val_trip_mean:.2f}s trip_p95={val_trip_p95:.2f}s "
                    f"(progression; lower is better)"
                )
            print(extra, flush=True)

        epsilon = max(eps_end, epsilon * eps_decay)

    env.close()
    for tid in tls_ids:
        agents[tid].save(ck_dir / f"last_{tid}.pth")

    summary = {
        "tls_ids": tls_ids,
        "per_tls_dims": per_tls_dims,
        "best_val_system_mean_waiting_time": best_system_wait
        if best_system_wait < float("inf")
        else None,
        "last_val_mean_trip_duration_s": last_val_trip_mean
        if last_val_trip_mean == last_val_trip_mean
        else None,
        "last_val_p95_trip_duration_s": last_val_trip_p95
        if last_val_trip_p95 == last_val_trip_p95
        else None,
        "checkpoint_dir": str(ck_dir),
        "log_csv": str(log_path),
        "note": (
            "Independent DQN per TLS (obs/act from sumo_rl per signal). "
            "Primary KPI: validation mean system_mean_waiting_time. "
            "Trip duration stats from SUMO tripinfo complement progression analysis."
        ),
    }
    with open(root / out["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {root / out['summary_json']}", flush=True)
    return summary


if __name__ == "__main__":
    train()
