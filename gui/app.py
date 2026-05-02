"""
Streamlit dashboard: Train (subprocess + CSV plots) | Showcase (live obs/reward + SUMO-GUI).

Run from repo root:
  export SUMO_HOME=...
  streamlit run gui/app.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from typing import cast
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Repo root = parent of gui/
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gui.obs_decode import DecodedObservation, decode_default_observation, lane_labels
from gui.showcase_rollout import (
    load_config_phase,
    run_showcase_phase_a,
    run_showcase_phase_b,
    run_showcase_phase_c,
)
from gui.train_runner import Phase, TrainJob, csv_path_for_phase, train_script_path

st.set_page_config(page_title="Traffic RL dashboard", layout="wide")


def _sync_showcase_if_phase_changed(phase: Phase) -> None:
    """Avoid stale B/C payloads and decode errors after switching scenario phase."""
    key = "_dashboard_showcase_sync_phase"
    if st.session_state.get(key) == phase:
        return
    st.session_state[key] = phase
    st.session_state.show_snap = None
    st.session_state.show_q = queue.Queue(maxsize=4)
    ev = st.session_state.get("show_stop_ev")
    if ev is not None:
        ev.set()


def _train_columns(phase: Phase) -> dict[str, list[str]]:
    """CSV column groups for Plotly."""
    if phase == "A":
        return {
            "wait": ["val_mean_wait"],
            "loss": ["train_loss"],
            "eps": ["epsilon"],
        }
    return {
        "wait": ["val_system_mean_wait"],
        "loss": ["train_loss"],
        "eps": ["epsilon"],
    }


def render_train_tab(phase: Phase) -> None:
    st.subheader("Training")
    if not os.environ.get("SUMO_HOME"):
        st.warning("Set **SUMO_HOME** before starting training.")

    if "train_job" not in st.session_state:
        st.session_state.train_job = TrainJob(phase=phase)

    job: TrainJob = st.session_state.train_job
    if job.phase != phase:
        job.stop()
        st.session_state.train_job = TrainJob(phase=phase)
        job = st.session_state.train_job

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Start training", disabled=job.is_running()):
            job.phase = phase
            job.start()
            st.rerun()
    with c2:
        if st.button("Stop training", disabled=not job.is_running()):
            job.stop()
            st.rerun()

    with c3:
        st.caption(f"Script: `{train_script_path(phase).name}`")
    st.caption(f"Process: **{'running' if job.is_running() else 'idle'}** · CSV: `{csv_path_for_phase(phase)}`")

    @st.fragment(run_every=timedelta(seconds=1))
    def train_live_panel() -> None:
        """Poll CSV and subprocess log so curves update while training (no extra clicks)."""
        j: TrainJob = st.session_state.train_job
        if j.phase != phase:
            return

        if j.log_lines:
            with st.expander("Training log (tail)", expanded=False):
                st.code("\n".join(j.log_lines)[-8000:], language="text")

        csv_path = csv_path_for_phase(phase)
        if not csv_path.is_file():
            st.info(f"No `{csv_path.name}` yet — start training or run CLI once.")
            return
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                st.info("CSV exists but is empty.")
                return
            cols = _train_columns(phase)
            x = df["episode"] if "episode" in df.columns else df.index
            fig = go.Figure()
            for name in cols["wait"]:
                if name in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pd.to_numeric(df[name], errors="coerce"),
                            name=name,
                        )
                    )
            for name in cols["loss"]:
                if name in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=pd.to_numeric(df[name], errors="coerce"),
                            name=name,
                            yaxis="y2",
                        )
                    )
            fig.update_layout(
                height=420,
                margin=dict(l=40, r=60, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=dict(title="validation wait"),
                yaxis2=dict(title="train loss", overlaying="y", side="right", showgrid=False),
            )
            st.plotly_chart(fig, width="stretch")

            last = df.iloc[-1].to_dict()
            st.markdown("**Latest row**")
            st.json({k: last[k] for k in last if pd.notna(last[k])})
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    train_live_panel()


def _current_phase_from_one_hot(phase_one_hot: list[float]) -> int:
    if not phase_one_hot:
        return 0
    return int(np.argmax(np.asarray(phase_one_hot, dtype=np.float64)))


def _render_phase_activity_strip(n_phases: int, active_idx: int, chosen_idx: int) -> None:
    """Show which green phase the observation encodes vs which phase index the policy picks."""
    st.caption(
        "Green phases **g0 … gN−1** — ● **active** = argmax(phase one-hot) in the incoming obs · "
        "★ **pick** = greedy action argmax Q(s,·) this step"
    )
    cols = st.columns(max(n_phases, 1))
    for i in range(n_phases):
        with cols[i]:
            tags: list[str] = []
            if i == active_idx:
                tags.append("● active")
            if i == chosen_idx:
                tags.append("★ pick")
            line = " · ".join(tags) if tags else "idle"
            if i == active_idx and i == chosen_idx:
                st.success(f"**g{i}**\n{line}")
            elif i == chosen_idx:
                st.warning(f"**g{i}**\n{line}")
            elif i == active_idx:
                st.info(f"**g{i}**\n{line}")
            else:
                st.caption(f"g{i}\n{line}")


def _render_q_values_chart(q_values: list[float], chosen_action: int, *, title: str) -> None:
    if not q_values:
        return
    n = len(q_values)
    labels = [f"g{i}" for i in range(n)]
    colors = ["#AB63FA"] * n
    if 0 <= chosen_action < n:
        colors[chosen_action] = "#00CC96"
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=q_values,
            marker_color=colors,
            text=[f"{float(v):.2f}" for v in q_values],
            textposition="outside",
            hovertemplate="%{x}<br>Q=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Q(s,·)",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def _render_observation_inputs(dec: DecodedObservation, title_prefix: str = "") -> None:
    labels = lane_labels(dec.n_lanes)
    df = pd.DataFrame(
        {
            "lane": labels,
            "density": [float(x) for x in dec.densities],
            "queue": [float(x) for x in dec.queues],
        }
    )
    st.markdown(f"##### {title_prefix}Lane densities & queues (middle / tail of obs vector)")
    st.dataframe(df, hide_index=True, width="stretch")

    ph = np.asarray(dec.phase_one_hot, dtype=np.float64)
    ph_df = pd.DataFrame(
        {"phase": [f"g{i}" for i in range(len(ph))], "value": ph}
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"##### {title_prefix}Phase one-hot (first block)")
        st.dataframe(ph_df, hide_index=True, width="stretch")
    with c2:
        st.markdown(f"##### {title_prefix}Min green flag")
        st.metric("min_green_ok", f"{dec.min_green_ok:.3f}")


def _render_obs_bars(dec: DecodedObservation, title_prefix: str = "") -> None:
    labels = lane_labels(dec.n_lanes)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=dec.densities, name="density"))
    fig.add_trace(go.Bar(x=labels, y=dec.queues, name="queue"))
    fig.update_layout(
        title=f"{title_prefix}Lane density / queue (bars)",
        barmode="group",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    ph = np.asarray(dec.phase_one_hot)
    fig2 = go.Figure(go.Bar(x=[f"g{i}" for i in range(len(ph))], y=ph, name="phase one-hot"))
    fig2.update_layout(title=f"{title_prefix}Phase one-hot · min_green_ok={dec.min_green_ok:.2f}", height=260)
    st.plotly_chart(fig2, width="stretch")


def _render_showcase_step(
    dec: DecodedObservation,
    action: int,
    q_values: list[float] | None,
    *,
    title_prefix: str = "",
    reward: float | None = None,
    sys_w=None,
    include_system_wait: bool = True,
) -> None:
    nga = dec.n_green_phases
    active = _current_phase_from_one_hot(dec.phase_one_hot)
    st.markdown(f"### {title_prefix}Policy output vs inputs")
    n_metrics = 4 if include_system_wait else 3
    mcols = st.columns(n_metrics)
    with mcols[0]:
        st.metric("step reward", f"{reward:.4f}" if reward is not None else "—")
    with mcols[1]:
        st.metric("greedy action", f"g{action}", help="Discrete index; usually targets that green phase.")
    with mcols[2]:
        st.metric("active phase in obs", f"g{active}")
    if include_system_wait and n_metrics == 4:
        with mcols[3]:
            st.metric("system_mean_waiting_time", sys_w if sys_w is not None else "—")

    _render_phase_activity_strip(nga, active, action)

    if q_values and len(q_values) > 0:
        _render_q_values_chart(
            q_values,
            action,
            title=f"{title_prefix}Q(s,·) per action — green bar = argmax (decision)",
        )

    with st.expander("Full input breakdown (tables + charts)", expanded=False):
        _render_observation_inputs(dec, title_prefix)
        _render_obs_bars(dec, title_prefix)


def render_showcase_tab(phase: Phase) -> None:
    st.subheader("Showcase (greedy policy + SUMO-GUI)")
    if not os.environ.get("SUMO_HOME"):
        st.error("Set **SUMO_HOME**.")
        return

    ck_override = st.text_input("Checkpoint override (Phase A only, optional)", value="", placeholder="Leave empty for best/last from config")

    use_gui = st.checkbox("Open SUMO-GUI", value=True)
    delay = st.slider("Step delay (seconds)", 0.0, 0.5, 0.03, 0.01)

    if phase == "B":
        tls_pick = st.radio(
            "Inspect TLS",
            ["(all)", "B", "E"],
            horizontal=True,
            width="stretch",
            key="inspect_tls_B",
        )
    elif phase == "C":
        tls_pick = st.radio(
            "Inspect TLS",
            ["(all)", "A0", "A1", "B0", "B1", "C0", "C1"],
            horizontal=True,
            width="stretch",
            key="inspect_tls_C",
        )
    else:
        tls_pick = "(all)"

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        run_btn = st.button("Run one episode", key="show_run")
    with col_b:
        stop_btn = st.button("Stop", key="show_stop")

    if "show_q" not in st.session_state:
        st.session_state.show_q = queue.Queue(maxsize=4)
    if "show_snap" not in st.session_state:
        st.session_state.show_snap = None

    if stop_btn:
        ev = st.session_state.get("show_stop_ev")
        if ev is not None:
            ev.set()

    if run_btn:
        # Worker threads must NOT touch st.session_state (wrong thread / session).
        stop_ev = threading.Event()
        q: queue.Queue = queue.Queue(maxsize=4)
        st.session_state.show_stop_ev = stop_ev
        st.session_state.show_q = q

        def worker() -> None:
            stop_ev.clear()

            def should_stop() -> bool:
                return stop_ev.is_set()

            def push(step: int, payload: dict) -> None:
                try:
                    q.put_nowait((step, payload))
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait((step, payload))
                    except queue.Full:
                        pass

            cfg = load_config_phase(phase, None)
            try:
                if phase == "A":
                    run_showcase_phase_a(
                        cfg,
                        gui=use_gui,
                        checkpoint_path=ck_override.strip() or None,
                        step_delay_sec=delay,
                        on_step=lambda s, p: push(s, p),
                        should_stop=should_stop,
                    )
                elif phase == "B":
                    run_showcase_phase_b(
                        cfg,
                        gui=use_gui,
                        step_delay_sec=delay,
                        on_step=lambda s, p: push(s, p),
                        should_stop=should_stop,
                    )
                else:
                    run_showcase_phase_c(
                        cfg,
                        gui=use_gui,
                        step_delay_sec=delay,
                        on_step=lambda s, p: push(s, p),
                        should_stop=should_stop,
                    )
            except Exception as e:
                push(-1, {"error": str(e)})

        threading.Thread(target=worker, daemon=True).start()
        st.success("Showcase started — watch SUMO-GUI and live metrics below.")

    @st.fragment(run_every=timedelta(milliseconds=350))
    def poll_showcase() -> None:
        q = st.session_state.show_q
        try:
            while True:
                step, payload = q.get_nowait()
                st.session_state.show_snap = (step, payload)
        except queue.Empty:
            pass

        snap = st.session_state.show_snap
        if not snap:
            return
        step, payload = snap
        if isinstance(payload, dict) and payload.get("error"):
            st.error(payload["error"])
            return

        pk = payload.get("phase_key")
        if pk != phase:
            return

        st.caption(f"Step **{step}**")

        if pk == "A":
            info = payload.get("info") or {}
            sys_w = info.get("system_mean_waiting_time", "—")
            q_raw = payload.get("q_values")
            q_list: list[float] | None = (
                [float(x) for x in q_raw] if isinstance(q_raw, list) else None
            )
            obs = payload.get("obs")
            nga = int(payload.get("n_green_phases", 0))
            if obs is not None and nga > 0:
                dec = decode_default_observation(obs, n_green_phases=nga)
                _render_showcase_step(
                    dec,
                    int(payload.get("action", 0)),
                    q_list,
                    reward=float(payload.get("reward", 0.0)),
                    sys_w=sys_w,
                    include_system_wait=True,
                )
            return

        # Multi-agent
        info = payload.get("info") or {}
        sys_w = info.get("system_mean_waiting_time", "—")
        rewards = payload.get("rewards") or {}
        q_map = payload.get("q_values") or {}

        g1, g2 = st.columns(2)
        with g1:
            st.metric("system_mean_waiting_time", sys_w)
        with g2:
            st.caption("Per-TLS step rewards")
            st.json(rewards)

        tls_ids = payload.get("tls_ids") or []
        obs_map = payload.get("obs") or {}
        act_map = payload.get("actions") or {}
        ng_map = payload.get("n_green_phases")

        pick = tls_pick
        ids_show = tls_ids if pick == "(all)" or pick.startswith("(") else [pick] if pick in obs_map else tls_ids
        to_show = [tid for tid in ids_show if tid in obs_map]
        if not to_show:
            return
        multi = len(to_show) > 1

        def render_one_tls(tid: str) -> None:
            obs = obs_map[tid]
            if phase == "B":
                ng = int(payload.get("n_green_phases", 2))
            else:
                ng = int((ng_map or {}).get(tid, 2))
            try:
                dec = decode_default_observation(obs, n_green_phases=ng)
                q_raw = q_map.get(tid) if isinstance(q_map, dict) else None
                q_list = [float(x) for x in q_raw] if isinstance(q_raw, list) else None
                rew = float(rewards[tid]) if isinstance(rewards, dict) and tid in rewards else None
                _render_showcase_step(
                    dec,
                    int(act_map.get(tid, 0)),
                    q_list,
                    title_prefix=f"`{tid}` · ",
                    reward=rew,
                    sys_w=sys_w,
                    include_system_wait=not multi,
                )
            except Exception as ex:
                st.warning(f"{tid}: decode error {ex}")

        if len(to_show) == 1:
            render_one_tls(to_show[0])
        else:
            # Full-width control (st.tabs labels stay tiny); selection persists via key.
            view_tid = st.segmented_control(
                "View traffic light (TLS)",
                options=to_show,
                default=to_show[0],
                key=f"showcase_tls_view_{phase}",
                label_visibility="visible",
                width="stretch",
            )
            chosen = view_tid if view_tid in to_show else to_show[0]
            render_one_tls(str(chosen))

    poll_showcase()


def main() -> None:
    left, right = st.columns([2, 3], vertical_alignment="bottom")
    with left:
        st.title("Traffic RL dashboard")
    with right:
        sel = st.segmented_control(
            "Scenario phase",
            options=["A", "B", "C"],
            default="A",
            key="dashboard_phase",
            label_visibility="visible",
            width="stretch",
        )
    phase: Phase = cast(Phase, sel if sel in ("A", "B", "C") else "A")
    _sync_showcase_if_phase_changed(phase)

    if not os.environ.get("SUMO_HOME"):
        st.warning(
            "Set **SUMO_HOME** before Train or Showcase (e.g. `export SUMO_HOME=/usr/share/sumo`)."
        )

    tab_train, tab_show, tab_about = st.tabs(["Train", "Showcase", "About"])

    with tab_train:
        render_train_tab(phase)

    with tab_show:
        render_showcase_tab(phase)

    with tab_about:
        st.markdown(
            """
### Observation layout (sumo-rl default)

`[ phase_one_hot | min_green | lane_densities… | lane_queues… ]`

Reward (training): **`diff-waiting-time`** — change in accumulated waiting on incoming lanes.

### Running

```bash
export SUMO_HOME=/usr/share/sumo   # your path
cd /path/to/traffic_rl_project
source venv/bin/activate
pip install -r gui/requirements-gui.txt
streamlit run gui/app.py
```

Train tab runs `train.py` / `train_phase_b.py` / `train_phase_c.py` as subprocesses.  
Showcase loads checkpoints from `checkpoints/` paths in each phase config (same as CLI `eval_gui*.py`).
"""
        )


if __name__ == "__main__":
    main()
