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

## Scaling (later phases)

- **2 lights / 2×2:** use sumo-rl **PettingZoo `parallel_env`**, one agent per TLS, neighbor features — same repo layout; extend `config.yaml` and add a multi-agent train script.

## Repository layout

| File | Role |
|------|------|
| `config.yaml` | Paths, env kwargs, training & validation |
| `train.py` | Phase A training + validation |
| `baseline.py` | Fixed-time baseline KPIs |
| `agent.py` | Double Dueling DQN, uniform replay |
| `model.py` | Dueling network (compact 128 trunk) |
| `replay.py` | Uniform replay buffer |
| `eval_gui.py` | Load best weights, SUMO-GUI |
| `single.net.xml`, `single.rou.xml`, `single.sumocfg` | SUMO scenario |

## References

See `atsc_refs.txt` (IntelliLight, PressLight, surveys, **sumo-rl**).
