# Traffic RL — Phase A (SUMO-RL)

Single-intersection adaptive traffic signal control using **Gymnasium + [sumo-rl](https://github.com/LucasAlegre/sumo-rl)** (Lucas Alegre), **Double Dueling DQN**, and **uniform experience replay** — aligned with simple ATSC stacks such as [Traffic-Control-RL](https://github.com/CodeKnight314/Traffic-Control-RL) and common paper practice (queue / waiting-time–based rewards; default sumo-rl reward `diff-waiting-time`).

## Prerequisites

- **SUMO** installed and **`SUMO_HOME`** set (sumo-rl requires this).
- Python 3.10+ recommended.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Phase A workflow

1. **Baseline (fixed-time program in `single.net.xml`)** — KPI to beat:

   ```bash
   python baseline.py
   ```

   Writes `runs/phase_a_baseline.json` (mean system waiting time, etc.).

2. **Train** (logs CSV + saves **best** checkpoint by **lowest validation mean waiting time**):

   ```bash
   python train.py
   # or
   python main.py
   ```

   - Training log: `runs/phase_a_train.csv`
   - Summary: `runs/phase_a_summary.json`
   - Checkpoints: `checkpoints/best_phase_a.pth`, `checkpoints/last_phase_a.pth`

3. **Evaluate with GUI** (greedy policy):

   ```bash
   python eval_gui.py
   # or
   python test.py
   ```

Hyperparameters and paths live in **`config.yaml`** (episode length, `delta_time`, `min_green`, epsilon schedule, validation cadence).

### Lock + multi-seed stress (optional)

After a good training run, tag the repo (`git tag phase-a-complete`) and check **generalization across SUMO seeds** (same net/routes, different insertions randomness):

```bash
python stress_phase_a.py --seeds 7,42,1337 --episodes 80
```

Writes `runs/stress/phase_a_stress_summary.json` and a repo-root **`phase_a_stress_report.json`** (for committing). Each seed gets baseline metrics plus a shortened train run; `beats_baseline` compares best validation wait to that seed’s baseline.

## Success criterion (Phase A)

Compare **validation `val_mean_wait`** in `runs/phase_a_train.csv` (lower is better) to **`mean_system_mean_waiting_time`** from `runs/phase_a_baseline.json`. RL should beat fixed-time on that metric after sufficient episodes.

## Phase B — two intersections (corridor)

Two TLS (**`B`** and **`E`**) on the SUMO-RL **double-intersection** scenario (`two.net.xml`, `two.rou.xml`, copied from the sumo-rl package). **Independent Double DQN (IDQN):** one `DQNAgent` per signal, separate replay buffers; actions every RL step are joint `{B: a0, E: a1}`.

Training uses **`SumoEnvironment`** with `single_agent=False` (PettingZoo `parallel_env` is avoided here due to version quirks).

1. **Baseline**

   ```bash
   python baseline_phase_b.py
   ```

   Writes `runs/phase_b_baseline.json` (`mean_system_mean_waiting_time`).

2. **Train**

   ```bash
   python train_phase_b.py
   ```

   Config: **`config_phase_b.yaml`**. Logs **`runs/phase_b_train.csv`**. Saves **`checkpoints/phase_b/best_<tls_id>.pth`** when **validation mean `system_mean_waiting_time`** (greedy, averaged over val episodes) improves.

3. **Success criterion**

   Compare validation **`val_system_mean_wait`** (from the CSV) to Phase B baseline **`mean_system_mean_waiting_time`**. Lower is better.

4. **SUMO-GUI (greedy, both TLS)**

   ```bash
   python eval_gui_phase_b.py
   # or
   python test_phase_b.py
   ```

   Loads **`checkpoints/phase_b/best_B.pth`** and **`best_E.pth`** (falls back to **`last_*.pth`**).

## Phase C — 2×3 grid (six TLS), open perimeter + asymmetric spacing

Six signalized junctions **`A0`, `A1`, `B0`, `B1`, `C0`, `C1`** on a three-column × two-row mesh (`phase_c.net.xml`). Edges use **three lanes** per direction (similar spirit to the multi-lane Phase B corridor — richer than a single-lane grid). **`netgenerate`** added **`--grid.x-attach-length`** / **`--grid.y-attach-length`** so each side of the rectangle has **approach edges** to **`dead_end`** fringe nodes (`top*`, `bottom*`, `left*`, `right*`). Demand uses **fringe O/Ds** (see **`phase_c.rou.xml`**, regenerated with **`randomTrips`** using **`--min-distance`** and **`--fringe-factor`**) — not a sealed internal-only cordon.

The middle column **`B`** is shifted horizontally in **`phase_c_plain.nod.xml`** (unequal A–B vs B–C); **`B1`** is slightly offset vertically versus **`A1`/`C1`** before **`netconvert`**.

**IDQN:** one **`DQNAgent` per TLS**, each sized from **`env.observation_spaces(tid)`** / **`env.action_spaces(tid)`** (with the open network, these are often **uniform** across the six signals; the code still supports different sizes if the net changes).

**Primary KPI** remains validation **`val_system_mean_wait`**. **Progression-oriented metrics** (optional trip-level): **`val_mean_trip_duration`** / **`val_p95_trip_duration`** from SUMO **`tripinfo`** (see `progression.tripinfo` in **`config_phase_c.yaml`**).

Rebuild **`phase_c.net.xml`**: generate a base grid with attach + **`--default.lanenumber 3`**, export plain with **`netconvert --sumo-net-file … --plain-output-prefix phase_c_plain`**, edit **`phase_c_plain.nod.xml`** for **B** asymmetry, then:

```bash
netconvert --node-files phase_c_plain.nod.xml --edge-files phase_c_plain.edg.xml \
  --connection-files phase_c_plain.con.xml --tllogic-files phase_c_plain.tll.xml \
  -o phase_c.net.xml
```

Changing lane counts **changes observation size**; retrain Phase C or delete old **`checkpoints/phase_c/*.pth`**.

Regenerate trips (example):

```bash
python $SUMO_HOME/tools/randomTrips.py -n phase_c.net.xml -o phase_c.rou.xml \
  -b 0 -e 3000 -p 4.0 --seed 42 --fringe-factor 6 --min-distance 180 --validate
```

1. **Baseline**

   ```bash
   python baseline_phase_c.py
   ```

   Writes `runs/phase_c_baseline.json` (system wait + mean / p95 trip duration when tripinfo is enabled).

2. **Train**

   ```bash
   python train_phase_c.py
   ```

   Config: **`config_phase_c.yaml`**. Logs **`runs/phase_c_train.csv`**. Checkpoints **`checkpoints/phase_c/best_<tls_id>.pth`**.

3. **SUMO-GUI**

   ```bash
   python eval_gui_phase_c.py
   # or
   python test_phase_c.py
   ```

## Repository layout

| File | Role |
|------|------|
| `config.yaml` | Phase A paths & hyperparameters |
| `config_phase_b.yaml` | Phase B paths & hyperparameters |
| `config_phase_c.yaml` | Phase C paths & hyperparameters |
| `train.py` | Phase A training + validation |
| `train_phase_b.py` | Phase B multi-agent (IDQN) training |
| `train_phase_c.py` | Phase C multi-agent (IDQN per TLS) |
| `baseline.py` / `baseline_phase_b.py` / `baseline_phase_c.py` | Fixed-time baselines |
| `agent.py` | Double Dueling DQN, uniform replay |
| `model.py` | Dueling network (compact 128 trunk) |
| `replay.py` | Uniform replay buffer |
| `eval_gui.py` / `eval_gui_phase_b.py` / `eval_gui_phase_c.py` | GUI eval |
| `test_phase_b.py` / `test_phase_c.py` | Shortcuts for Phase B / C GUI |
| `stress_phase_a.py` | Multi-seed Phase A stress sweep |
| `single.*` | Phase A SUMO scenario |
| `two.*` | Phase B SUMO scenario (double corridor) |
| `phase_c.*`, `phase_c_plain.*` | Phase C SUMO scenario & editable plain inputs |

## References

See `atsc_refs.txt` (IntelliLight, PressLight, surveys, **sumo-rl**).
