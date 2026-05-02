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

## Phase C (next)

- **2×2 grid:** four TLS, same IDQN pattern or shared weights + PettingZoo parallel API once versions align; add neighbor-aware observations if needed.

## Repository layout

| File | Role |
|------|------|
| `config.yaml` | Phase A paths & hyperparameters |
| `config_phase_b.yaml` | Phase B paths & hyperparameters |
| `train.py` | Phase A training + validation |
| `train_phase_b.py` | Phase B multi-agent (IDQN) training |
| `baseline.py` / `baseline_phase_b.py` | Fixed-time baselines |
| `agent.py` | Double Dueling DQN, uniform replay |
| `model.py` | Dueling network (compact 128 trunk) |
| `replay.py` | Uniform replay buffer |
| `eval_gui.py` / `eval_gui_phase_b.py` | Phase A / Phase B GUI eval |
| `test_phase_b.py` | Shortcut for Phase B GUI |
| `stress_phase_a.py` | Multi-seed Phase A stress sweep |
| `single.*` | Phase A SUMO scenario |
| `two.*` | Phase B SUMO scenario (double corridor) |

## References

See `atsc_refs.txt` (IntelliLight, PressLight, surveys, **sumo-rl**).
