#!/usr/bin/env python3
"""
Phase A stress test: same net/routes, multiple SUMO seeds.
Runs baseline + shortened training per seed; writes aggregate summary.

Usage:
  python stress_phase_a.py
  python stress_phase_a.py --seeds 42,123,456 --episodes 80

Requires SUMO_HOME. Uses runs/stress/seed_<id>/ and checkpoints/stress/seed_<id>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from baseline import run_baseline
from train import _root, load_config, train


def main() -> None:
    if "SUMO_HOME" not in os.environ:
        print("ERROR: Set SUMO_HOME.", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Multi-seed Phase A stress test.")
    ap.add_argument(
        "--seeds",
        type=str,
        default="7,42,1337",
        help="Comma-separated integer SUMO seeds (default: 7,42,1337).",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Training episodes per seed (default: 50; use 200 for full runs).",
    )
    args = ap.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    base = load_config()
    root = _root()
    out_dir = root / "runs" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg["env"]["sumo_seed"] = seed
        cfg["training"]["episodes"] = int(args.episodes)
        sub = f"seed_{seed}"
        cfg["outputs"] = {
            "log_csv": f"runs/stress/{sub}/train.csv",
            "summary_json": f"runs/stress/{sub}/summary.json",
            "checkpoint_dir": f"checkpoints/stress/{sub}",
            "best_name": "best.pth",
        }
        (root / "runs" / "stress" / sub).mkdir(parents=True, exist_ok=True)
        (root / "checkpoints" / "stress" / sub).mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60, flush=True)
        print(f" SEED {seed} | baseline", flush=True)
        print("=" * 60, flush=True)
        bl = run_baseline(cfg, write_default_path=False)
        base_wait = bl["mean_system_mean_waiting_time"]
        assert base_wait is not None
        print(
            f"  baseline mean_system_mean_waiting_time = {base_wait:.4f}",
            flush=True,
        )

        print("\n" + "=" * 60, flush=True)
        print(f" SEED {seed} | train ({args.episodes} ep)", flush=True)
        print("=" * 60, flush=True)
        summ = train(cfg)
        best = summ.get("best_val_system_mean_waiting_time")
        rows.append(
            {
                "sumo_seed": seed,
                "baseline_mean_wait": round(base_wait, 6),
                "best_val_mean_wait": round(best, 6) if best is not None else None,
                "beats_baseline": bool(best is not None and best < base_wait),
                "stress_episodes": args.episodes,
                "log_csv": summ.get("log_csv"),
                "best_checkpoint": summ.get("best_checkpoint"),
            }
        )

    wins = sum(1 for r in rows if r["beats_baseline"])
    summary = {
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "beats_baseline_count": wins,
        "total_runs": len(rows),
        "rows": rows,
        "note": "beats_baseline: best_val_mean_wait < baseline_mean_wait for that seed.",
    }
    out_json = out_dir / "phase_a_stress_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Copy for git (runs/ is often gitignored)
    report_root = root / "phase_a_stress_report.json"
    with open(report_root, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(" STRESS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nWrote {out_json}", flush=True)
    print(f"Wrote {report_root} (commit this file to record the sweep)", flush=True)


if __name__ == "__main__":
    main()
