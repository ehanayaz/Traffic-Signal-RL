# Traffic RL — phases A, B, and C (SUMO-RL)

Adaptive traffic signal control with **Gymnasium + [sumo-rl](https://github.com/LucasAlegre/sumo-rl)** (Lucas Alegre), **Double Dueling DQN**, and **uniform experience replay** — aligned with common ATSC practice (default reward `diff-waiting-time`). The repo builds up in three stages:

| Phase | Scenario | Agents | What it exercises |
|-------|----------|--------|-------------------|
| **A** | One intersection (`single.net.xml`) | Single DQN | Basics: Gym `sumo-rl-v0`, validation, one checkpoint |
| **B** | Two TLS on a corridor (`two.net.xml`) | IDQN (2 agents) | Multi-agent joint actions, per-TLS checkpoints |
| **C** | Six TLS on a 2×3 grid (`phase_c.net.xml`) | IDQN (6 agents) | Open perimeter, asymmetric links, optional trip metrics |

## Prerequisites

- **SUMO** installed and **`SUMO_HOME`** set (required by sumo-rl).
- Python 3.10+ recommended.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Every phase assumes:

```bash
export SUMO_HOME=/usr/share/sumo   # Linux typical path; adjust for your install
source venv/bin/activate
```

## GUI dashboard (optional)

Browser UI (**Streamlit**) to **start/stop training** for phases A/B/C (subprocess + live CSV plots) and run **Showcase** mode: greedy rollouts with **SUMO-GUI** plus charts of the **same observation** the DQN sees (lane densities/queues, phase one-hot).

```bash
pip install -r gui/requirements-gui.txt
# from repo root:
streamlit run gui/app.py
```

Leave **`SUMO_HOME`** set in the environment where you launch Streamlit. Training writes the same `runs/phase_*_train.csv` files as the CLI; Showcase loads checkpoints from `checkpoints/` like `eval_gui*.py`.

---

## Quick reference — how to run each phase

Use this order: **baseline → train → GUI eval** (GUI needs trained weights).

| Phase | Baseline | Train | Config | Logs / checkpoints | GUI eval |
|-------|----------|-------|--------|--------------------|----------|
| **A** | `python baseline.py` | `python train.py` or `python main.py` | `config.yaml` | `runs/phase_a_*.csv/json`, `checkpoints/best_phase_a.pth` | `python eval_gui.py` or `python test.py` |
| **B** | `python baseline_phase_b.py` | `python train_phase_b.py` | `config_phase_b.yaml` | `runs/phase_b_train.csv`, `checkpoints/phase_b/best_<tls>.pth` | `python eval_gui_phase_b.py` or `python test_phase_b.py` |
| **C** | `python baseline_phase_c.py` | `python train_phase_c.py` or `python main_phase_c.py` | `config_phase_c.yaml` | `runs/phase_c_train.csv`, `checkpoints/phase_c/best_<tls>.pth` | `python eval_gui_phase_c.py` or `python test_phase_c.py` |

**Optional env overrides:** `TRAFFIC_RL_CONFIG`, `TRAFFIC_RL_CONFIG_B`, `TRAFFIC_RL_CONFIG_C` point to alternate YAML paths if you fork configs.

**Success (all phases):** compare validation **mean system waiting time** in the training CSV to the same metric from the baseline JSON — **lower is better**. Phase C also logs optional **trip duration** columns when `progression.tripinfo` is enabled in `config_phase_c.yaml`.

---

## Phase A — single intersection

- **SUMO files:** `single.net.xml`, `single.rou.xml` (and related).
- **Algorithm:** one **Double Dueling DQN** via `gym.make("sumo-rl-v0", ...)`.
- **Baseline:** fixed-time program in the net.

1. **Baseline**

   ```bash
   python baseline.py
   ```

   Writes `runs/phase_a_baseline.json`.

2. **Train**

   ```bash
   python train.py
   # or
   python main.py
   ```

   Hyperparameters: **`config.yaml`**. Outputs: `runs/phase_a_train.csv`, `runs/phase_a_summary.json`, `checkpoints/best_phase_a.pth`, `checkpoints/last_phase_a.pth`.

3. **Evaluate (SUMO-GUI, greedy)**

   ```bash
   python eval_gui.py
   # or
   python test.py
   ```

### Phase A — optional stress sweep

Same net/routes, multiple SUMO seeds:

```bash
python stress_phase_a.py --seeds 7,42,1337 --episodes 80
```

Writes `runs/stress/phase_a_stress_summary.json` and `phase_a_stress_report.json`.

---

## Phase B — two intersections (corridor)

- **SUMO files:** `two.net.xml`, `two.rou.xml`, `two.sumocfg` (SUMO-RL “double” scenario; TLS **B** and **E**).
- **Algorithm:** **IDQN** — one `DQNAgent` per TLS; `SumoEnvironment(..., single_agent=False)` (PettingZoo `parallel_env` is not used here for stability).

1. **Baseline**

   ```bash
   python baseline_phase_b.py
   ```

   Writes `runs/phase_b_baseline.json`.

2. **Train**

   ```bash
   python train_phase_b.py
   ```

   Config: **`config_phase_b.yaml`**. Checkpoints: **`checkpoints/phase_b/best_B.pth`**, **`best_E.pth`** (and `last_*`).

3. **Evaluate (SUMO-GUI)**

   ```bash
   python eval_gui_phase_b.py
   # or
   python test_phase_b.py
   ```

---

## Phase C — 2×3 grid (six TLS), open perimeter

- **SUMO files:** `phase_c.net.xml`, `phase_c.rou.xml`, `phase_c.sumocfg`; editable rebuild inputs: `phase_c_plain.{nod,edg,con,tll}.xml`.
- **Scenario:** six TLS **`A0`…`C1`**; **three lanes** per edge; **attach** roads on the rectangle perimeter (fringe **O/D**); middle column **B** shifted in **`phase_c_plain.nod.xml`** for unequal link lengths.
- **Algorithm:** **IDQN** — one agent per TLS; observation/action sizes come from sumo-rl per signal (often uniform with the current net).

1. **Baseline**

   ```bash
   python baseline_phase_c.py
   ```

   Writes `runs/phase_c_baseline.json` (system wait; trip mean / p95 when tripinfo is enabled).

2. **Train**

   ```bash
   python train_phase_c.py
   # or
   python main_phase_c.py
   ```

   Config: **`config_phase_c.yaml`**. Checkpoints: **`checkpoints/phase_c/best_<tls_id>.pth`**. Training CSV may include **`val_mean_trip_duration`** / **`val_p95_trip_duration`** if `progression.tripinfo` is on.

3. **Evaluate (SUMO-GUI)**

   ```bash
   python eval_gui_phase_c.py
   # or
   python test_phase_c.py
   ```

### Rebuilding Phase C from plain XML

After editing geometry or lane counts, rebuild the net and regenerate demand:

```bash
netconvert --node-files phase_c_plain.nod.xml --edge-files phase_c_plain.edg.xml \
  --connection-files phase_c_plain.con.xml --tllogic-files phase_c_plain.tll.xml \
  -o phase_c.net.xml

python $SUMO_HOME/tools/randomTrips.py -n phase_c.net.xml -o phase_c.rou.xml \
  -b 0 -e 3000 -p 4.0 --seed 42 --fringe-factor 6 --min-distance 180 --validate
```

Changing lane counts changes **observation size** — remove old **`checkpoints/phase_c/*.pth`** and train again.

---

## Repository layout

| File | Role |
|------|------|
| `config.yaml` | Phase A paths & hyperparameters |
| `config_phase_b.yaml` | Phase B paths & hyperparameters |
| `config_phase_c.yaml` | Phase C paths & hyperparameters |
| `train.py` / `main.py` | Phase A training + validation |
| `train_phase_b.py` | Phase B multi-agent (IDQN) |
| `train_phase_c.py` / `main_phase_c.py` | Phase C multi-agent (IDQN) |
| `baseline.py` / `baseline_phase_b.py` / `baseline_phase_c.py` | Fixed-time baselines |
| `agent.py`, `model.py`, `replay.py` | Double Dueling DQN + replay |
| `eval_gui.py` / `eval_gui_phase_b.py` / `eval_gui_phase_c.py` | Greedy GUI eval |
| `test.py`, `test_phase_b.py`, `test_phase_c.py` | Shortcuts to GUI eval |
| `stress_phase_a.py` | Multi-seed Phase A stress sweep |
| `single.*` | Phase A SUMO scenario |
| `two.*` | Phase B SUMO scenario |
| `phase_c.*`, `phase_c_plain.*` | Phase C SUMO scenario & plain inputs |
| `gui/` | Streamlit dashboard (`app.py`, `obs_decode.py`, `showcase_rollout.py`, `train_runner.py`) |

## References

See `atsc_refs.txt` (IntelliLight, PressLight, surveys, **sumo-rl**).
