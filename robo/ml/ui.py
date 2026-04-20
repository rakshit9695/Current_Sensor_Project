"""
robo/ml/ui.py
Streamlit ML page for EPSON Robot.

4 tabs:
  1. Data & Labels  – visualise auto-generated state / cycle / job labels
  2. Train Model    – configure hyperparameters and run training
  3. Evaluation     – confusion matrices, per-class report
  4. Inference View – overlay ML predictions on raw current signal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .label_generator import (
    generate_labels,
    DEFAULT_STANDBY_MAX,
    DEFAULT_ACTIVE_MIN,
    DEFAULT_MIN_RUN,
    DEFAULT_N_JOBS,
)
from .dataset import build_datasets
from .trainer import (
    train as ml_train,
    CHECKPOINT_PATH,
    checkpoint_exists,
    load_checkpoint,
)
from .predictor import RoboPredictor

_STATE_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
_STATE_NAMES  = {0: "STANDBY", 1: "IDLE", 2: "ACTIVE"}
_JOB_COLORS   = ["#4e9af1", "#f1914e", "#9b59b6", "#2ecc71"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ss(key, default=None):
    full = f"robo_ml_{key}"
    if full not in st.session_state:
        st.session_state[full] = default
    return st.session_state[full]


def _ss_set(key, val):
    st.session_state[f"robo_ml_{key}"] = val


# ── Main entry point ──────────────────────────────────────────────────────────

def render_ml_page(df: pd.DataFrame):
    """
    Render the full ML diagnostics page for EPSON Robot.

    Parameters
    ----------
    df : DataFrame from data_loader.load_data() — single CSV, full session.
    """
    st.title("🤖 ML Diagnostics — EPSON Robot")
    st.caption(
        "Single-phase robot data (I1 + I_Avg). Labels auto-generated from "
        "I_Avg thresholds. States: **STANDBY** / **IDLE** / **ACTIVE**."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data & Labels", "🏋️ Train Model", "📈 Evaluation", "🔍 Inference View"]
    )

    with tab1:
        _tab_data_labels(df)
    with tab2:
        _tab_train(df)
    with tab3:
        _tab_evaluate(df)
    with tab4:
        _tab_inference(df)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Data & Labels
# ══════════════════════════════════════════════════════════════════════════════

def _no_data_warning():
    st.warning(
        "No data loaded. Go to **Data Validation**, upload the robot CSV file, "
        "then return here."
    )


def _tab_data_labels(df: pd.DataFrame | None):
    if df is None:
        _no_data_warning()
        return
    st.subheader("Label Generation Settings")

    with st.expander("⚙️ Threshold parameters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        standby_max = c1.number_input(
            "STANDBY max I_Avg (A)", min_value=0.01, max_value=0.20,
            value=DEFAULT_STANDBY_MAX, step=0.005, key="robo_standby_max",
        )
        active_min  = c2.number_input(
            "ACTIVE min I_Avg (A)", min_value=0.05, max_value=0.30,
            value=DEFAULT_ACTIVE_MIN, step=0.005, key="robo_active_min",
        )
        min_run = c3.number_input(
            "Min run samples", min_value=1, max_value=30,
            value=DEFAULT_MIN_RUN, step=1, key="robo_min_run",
        )
        n_jobs = c4.number_input(
            "Job clusters (K)", min_value=2, max_value=5,
            value=DEFAULT_N_JOBS, step=1, key="robo_n_jobs",
        )

    regen = st.button("🔄 (Re)generate Labels", key="robo_regen_labels")

    df_lab = _ss("df_labeled")
    meta   = _ss("label_meta")

    if df_lab is None or regen:
        with st.spinner("Generating labels…"):
            df_lab, meta = generate_labels(
                df,
                standby_max = standby_max,
                active_min  = active_min,
                min_run     = int(min_run),
                n_job_types = int(n_jobs),
            )
        _ss_set("df_labeled", df_lab)
        _ss_set("label_meta",  meta)

    if df_lab is None:
        return

    # ── Summary metrics ────────────────────────────────────────────────
    st.markdown("---")
    n_total   = len(df_lab)
    n_standby = (df_lab["state_label"] == 0).sum()
    n_idle    = (df_lab["state_label"] == 1).sum()
    n_active  = (df_lab["state_label"] == 2).sum()
    n_cycles  = meta["n_cycles"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Samples",     f"{n_total:,}")
    c2.metric("STANDBY (RED)",     f"{n_standby:,}", f"{100*n_standby/n_total:.1f}%")
    c3.metric("IDLE (AMBER)",      f"{n_idle:,}",    f"{100*n_idle/n_total:.1f}%")
    c4.metric("ACTIVE (GREEN)",    f"{n_active:,}",  f"{100*n_active/n_total:.1f}%")
    c5.metric("Cycles Detected",   f"{n_cycles:,}")

    # ── State distribution pie ─────────────────────────────────────────
    col_pie, col_hist = st.columns(2)

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["STANDBY", "IDLE", "ACTIVE"],
            values=[n_standby, n_idle, n_active],
            marker_colors=["#e74c3c", "#f39c12", "#27ae60"],
            hole=0.4,
        ))
        fig_pie.update_layout(
            title="State Distribution", template="plotly_dark", height=320,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_hist:
        durs = meta.get("cycle_durations", [])
        if durs:
            fig_hist = go.Figure(go.Histogram(
                x=durs, nbinsx=40,
                marker_color="#27ae60", opacity=0.8,
            ))
            fig_hist.update_layout(
                title="ACTIVE Segment Durations (s)",
                xaxis_title="Duration (s)", yaxis_title="Count",
                template="plotly_dark", height=320,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_hist, width='stretch')
        else:
            st.info("No cycles detected with current thresholds.")

    # ── Job cluster scatter ────────────────────────────────────────────
    cf_dict = meta.get("cycle_features", {})
    if cf_dict and "cycle_id" in cf_dict:
        cf = pd.DataFrame(cf_dict)
        if "job_type" in cf.columns and len(cf) > 0:
            st.subheader("Operational Mode Clusters (per ACTIVE segment)")
            fig_sc = go.Figure()
            for jt in sorted(cf["job_type"].unique()):
                sub = cf[cf["job_type"] == jt]
                fig_sc.add_trace(go.Scatter(
                    x=sub["duration_s"],
                    y=sub["mean_iavg"],
                    mode="markers",
                    name=f"Mode {jt}",
                    marker=dict(color=_JOB_COLORS[jt % len(_JOB_COLORS)], size=6, opacity=0.7),
                    text=[f"Duration: {d:.0f}s<br>I_avg: {i:.3f}A"
                          for d, i in zip(sub["duration_s"], sub["mean_iavg"])],
                    hoverinfo="text+name",
                ))
            fig_sc.update_layout(
                title="Cluster: Duration vs Mean Current per ACTIVE Segment",
                xaxis_title="Segment Duration (s)",
                yaxis_title="Mean I_Avg (A)",
                template="plotly_dark", height=360,
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_sc, width='stretch')

            st.markdown("**Cluster Summary:**")
            for jt in sorted(cf["job_type"].unique()):
                sub = cf[cf["job_type"] == jt]
                label = {0: "Light Load (short bursts)", 1: "Heavy Load (sustained runs)"}.get(jt, f"Mode {jt}")
                st.markdown(
                    f"- **Mode {jt} — {label}**: {len(sub)} segments, "
                    f"median duration = **{sub['duration_s'].median():.0f}s**, "
                    f"mean I_Avg = **{sub['mean_iavg'].mean():.3f}A**"
                )

    # ── Raw timeline with labels ───────────────────────────────────────
    st.subheader("Labelled Time-Series (first 1,000 samples)")
    view = df_lab.head(1000)
    t    = view["timestamp"] if "timestamp" in view.columns else pd.RangeIndex(len(view))

    fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("I_Avg with State Labels", "Cycle Position"),
                           vertical_spacing=0.12)

    fig_ts.add_trace(go.Scatter(x=t, y=view["i_avg"], name="I_Avg",
                                line=dict(color="#4e9af1", width=1)), row=1, col=1)

    for state, color in _STATE_COLORS.items():
        mask = view["state_label"] == state
        fig_ts.add_trace(go.Scatter(
            x=t[mask], y=view.loc[mask, "i_avg"],
            mode="markers", name=_STATE_NAMES[state],
            marker=dict(color=color, size=4, opacity=0.8),
        ), row=1, col=1)

    cp_mask = view["cycle_pos"].notna()
    fig_ts.add_trace(go.Scatter(
        x=t[cp_mask], y=view.loc[cp_mask, "cycle_pos"],
        name="Cycle Pos", line=dict(color="#2ecc71", width=1.5),
    ), row=2, col=1)

    fig_ts.update_layout(height=500, template="plotly_dark",
                         margin=dict(l=40, r=20, t=50, b=30))
    fig_ts.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig_ts.update_yaxes(title_text="Cycle Pos", row=2, col=1)
    st.plotly_chart(fig_ts, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Train Model
# ══════════════════════════════════════════════════════════════════════════════

def _tab_train(df: pd.DataFrame | None):
    if df is None:
        _no_data_warning()
        return
    st.subheader("Model Hyperparameters")

    df_lab = _ss("df_labeled")
    meta   = _ss("label_meta")

    if df_lab is None:
        st.warning("Generate labels first in the **Data & Labels** tab.")
        return

    c1, c2, c3, c4 = st.columns(4)
    look_back  = c1.slider("Look-back window (s)", 10, 120, 30, 5,  key="robo_look_back")
    epochs     = c2.slider("Epochs",               5,  50,  20, 5,  key="robo_epochs")
    batch_size = c3.selectbox("Batch size", [32, 64, 128, 256], index=1, key="robo_batch")
    lr         = c4.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, key="robo_lr")

    c5, c6, c7 = st.columns(3)
    d_model    = c5.selectbox("d_model",  [32, 64, 128], index=1, key="robo_dmodel")
    n_layers   = c6.selectbox("n_layers", [2, 3, 4],     index=1, key="robo_nlayers")
    n_heads    = c7.selectbox("n_heads",  [2, 4, 8],     index=1, key="robo_nheads")

    c8, c9, c10 = st.columns(3)
    alpha = c8.number_input("α (state loss weight)",    min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="robo_alpha")
    beta  = c9.number_input("β (cycle pos weight)",     min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="robo_beta")
    gamma = c10.number_input("γ (job type weight)",     min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="robo_gamma")

    if checkpoint_exists(CHECKPOINT_PATH):
        ckpt = load_checkpoint(CHECKPOINT_PATH)
        h    = ckpt.get("train_history", {})
        epochs_done = len(h.get("train_loss", []))
        val_acc = h.get("val_state_acc", [None])[-1]
        st.info(
            f"Existing checkpoint: {epochs_done} epochs trained, "
            f"val state acc = {val_acc*100:.1f}%" if val_acc else
            f"Existing checkpoint: {epochs_done} epochs trained."
        )

    if not st.button("🚀 Start Training", type="primary", key="robo_train_btn"):
        return

    # ── Build training dataset ────────────────────────────────────────
    from robo.ml.label_generator import generate_labels as _gen

    df_full_lab, _ = _gen(
        df,
        standby_max = meta.get("standby_max", DEFAULT_STANDBY_MAX),
        active_min  = meta.get("active_min",  DEFAULT_ACTIVE_MIN),
        min_run     = meta.get("min_run",     DEFAULT_MIN_RUN),
        n_job_types = meta.get("n_job_types", DEFAULT_N_JOBS),
    )

    try:
        train_ds, val_ds, test_ds, norm_stats = build_datasets(df_full_lab, look_back=look_back)
    except ValueError as e:
        st.error(str(e))
        return

    st.markdown("---")
    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    loss_chart    = st.empty()

    train_losses: list[float] = []
    val_losses:   list[float] = []

    def progress_cb(epoch: int, total: int, metrics: dict):
        frac = epoch / total
        progress_bar.progress(frac)
        status_text.markdown(
            f"**Epoch {epoch}/{total}** — "
            f"train loss: `{metrics['train_loss']:.4f}` | "
            f"val loss: `{metrics['val_loss']:.4f}` | "
            f"state acc: `{metrics['state_acc']:.1f}%` | "
            f"job acc: `{metrics['job_acc']:.1f}%` | "
            f"cycle MAE: `{metrics['cyc_mae']:.4f}`"
        )
        train_losses.append(metrics["train_loss"])
        val_losses.append(metrics["val_loss"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, name="Train Loss", line=dict(color="#4e9af1")))
        fig.add_trace(go.Scatter(y=val_losses,   name="Val Loss",   line=dict(color="#f1914e")))
        fig.update_layout(
            title="Loss Curve", xaxis_title="Epoch", yaxis_title="Loss",
            template="plotly_dark", height=300, margin=dict(l=40, r=20, t=50, b=40),
        )
        loss_chart.plotly_chart(fig, width='stretch')

    with st.spinner("Training…"):
        history = ml_train(
            train_ds, val_ds, norm_stats, meta,
            epochs=epochs, batch_size=batch_size, lr=lr,
            alpha=alpha, beta=beta, gamma=gamma,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_model*2,
            checkpoint_path=CHECKPOINT_PATH,
            progress_cb=progress_cb,
        )

    progress_bar.progress(1.0)
    st.success(f"Training complete! Checkpoint saved to `{CHECKPOINT_PATH}`")

    final_acc = history["val_state_acc"][-1] * 100 if history["val_state_acc"] else 0
    final_job = history["val_job_acc"][-1] * 100   if history["val_job_acc"]   else 0
    final_mae = history["val_cyc_mae"][-1]          if history["val_cyc_mae"]   else 0
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Final State Acc",  f"{final_acc:.1f}%")
    mc2.metric("Final Job Acc",    f"{final_job:.1f}%")
    mc3.metric("Cycle Pos MAE",    f"{final_mae:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _tab_evaluate(df: pd.DataFrame | None):
    if df is None:
        _no_data_warning()
        return
    st.subheader("Model Evaluation on Robot Dataset")

    df_lab = _ss("df_labeled")
    if df_lab is None:
        st.warning("Generate labels in **Data & Labels** tab first.")
        return

    if not checkpoint_exists(CHECKPOINT_PATH):
        st.warning("No trained model found. Train the model first in **Train Model** tab.")
        return

    if not st.button("▶️ Run Evaluation", type="primary", key="robo_eval_btn"):
        return

    with st.spinner("Loading model and running inference…"):
        try:
            predictor = RoboPredictor.from_checkpoint(CHECKPOINT_PATH)
            metrics   = predictor.evaluate_on_labeled(df_lab)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            return

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Valid Samples",  f"{metrics['n_valid']:,}")
    mc2.metric("State Accuracy", f"{metrics['state_acc']*100:.2f}%")
    mc3.metric("Job Accuracy",   f"{metrics['job_acc']*100:.2f}%" if not np.isnan(metrics['job_acc']) else "N/A")
    mc4.metric("Cycle Pos MAE",  f"{metrics['cyc_mae']:.4f}" if not np.isnan(metrics['cyc_mae']) else "N/A")

    # ── State confusion matrix ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("State Confusion Matrix")
    cm    = metrics["state_cm"]
    names = ["STANDBY", "IDLE", "ACTIVE"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=names, y=names,
        colorscale="Blues", showscale=True,
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig_cm.update_layout(
        title="State Confusion Matrix (rows=true, cols=predicted)",
        xaxis_title="Predicted", yaxis_title="True",
        template="plotly_dark", height=380,
        margin=dict(l=80, r=20, t=60, b=60),
    )
    st.plotly_chart(fig_cm, width='stretch')

    # ── Per-class report ───────────────────────────────────────────────
    st.subheader("Per-Class Report")
    rep = metrics["state_report"]
    rows = []
    for cls in ["STANDBY", "IDLE", "ACTIVE"]:
        r = rep.get(cls, {})
        rows.append({
            "Class":     cls,
            "Precision": f"{r.get('precision',0)*100:.1f}%",
            "Recall":    f"{r.get('recall',0)*100:.1f}%",
            "F1":        f"{r.get('f1-score',0)*100:.1f}%",
            "Support":   int(r.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')

    # ── Job confusion matrix ───────────────────────────────────────────
    jcm = metrics["job_cm"]
    if jcm.sum() > 0:
        st.subheader("Operational Mode Confusion Matrix")
        jnames = [f"Mode {i}" for i in range(len(jcm))]
        fig_jcm = go.Figure(go.Heatmap(
            z=jcm, x=jnames, y=jnames,
            colorscale="Greens", showscale=True,
            text=[[str(v) for v in row] for row in jcm],
            texttemplate="%{text}",
        ))
        fig_jcm.update_layout(
            title="Job-Type Confusion Matrix",
            xaxis_title="Predicted", yaxis_title="True",
            template="plotly_dark", height=340,
            margin=dict(l=80, r=20, t=60, b=60),
        )
        st.plotly_chart(fig_jcm, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Inference View
# ══════════════════════════════════════════════════════════════════════════════

def _tab_inference(df: pd.DataFrame | None):
    if df is None:
        _no_data_warning()
        return
    st.subheader("Live Inference View")

    df_lab = _ss("df_labeled")
    if df_lab is None:
        st.warning("Generate labels in **Data & Labels** tab first.")
        return

    if not checkpoint_exists(CHECKPOINT_PATH):
        st.warning("No trained model found. Train the model first.")
        return

    n = len(df_lab)
    max_sec = n

    start_s, end_s = st.slider(
        "Time range (samples)", 0, max_sec - 1, (0, min(500, max_sec - 1)),
        key="robo_infer_range",
    )

    if st.button("🔮 Run Inference on Selection", key="robo_infer_btn"):
        lb = load_checkpoint(CHECKPOINT_PATH).get("look_back", 30)
        start_adj = max(0, start_s - lb)
        sub = df_lab.iloc[start_adj: end_s + 1].reset_index(drop=True)

        if len(sub) < lb:
            st.error(f"Selection too small (need at least {lb} samples).")
            return

        with st.spinner("Running inference…"):
            try:
                predictor = RoboPredictor.from_checkpoint(CHECKPOINT_PATH)
                result    = predictor.predict(sub)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        _ss_set("infer_result", result)

    result = _ss("infer_result")
    if result is None:
        st.info("Click **Run Inference on Selection** to see predictions.")
        return

    t = result["timestamp"] if "timestamp" in result.columns else pd.RangeIndex(len(result))

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Current Signals (I1 & I_Avg)",
            "State: Auto-label vs ML Prediction",
            "Cycle Position Regression",
            "Operational Mode (Job Type)",
        ),
        vertical_spacing=0.07,
    )

    # Row 1: currents (only I1 and I_Avg are meaningful)
    fig.add_trace(go.Scatter(x=t, y=result["i1"],    name="I1 (RMS)",
                             line=dict(color="#4e9af1", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=result["i_avg"], name="I_Avg",
                             line=dict(color="#f1f14e", width=1)), row=1, col=1)

    # Row 2: state comparison
    state_num = {"STANDBY": 0, "IDLE": 1, "ACTIVE": 2}
    true_num  = result["state_label"].map({0: 0, 1: 1, 2: 2})
    pred_num  = result["ml_state_name"].map(state_num)
    fig.add_trace(go.Scatter(x=t, y=true_num, name="True State",
                             line=dict(color="#888", width=2, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=pred_num, name="ML State",
                             line=dict(color="#e67e22", width=1.5)), row=2, col=1)

    # Row 3: cycle position
    cp_true = result["cycle_pos"].where(result["cycle_pos"].notna())
    cp_pred = result["ml_cycle_pos"]
    fig.add_trace(go.Scatter(x=t, y=cp_true, name="True Pos",
                             line=dict(color="#2ecc71", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=cp_pred, name="ML Pos",
                             line=dict(color="#e74c3c", width=1.5, dash="dash")), row=3, col=1)

    # Row 4: job type
    fig.add_trace(go.Scatter(
        x=t, y=result["ml_job_type"],
        mode="markers", name="ML Job",
        marker=dict(
            color=[_JOB_COLORS[j % len(_JOB_COLORS)] if j >= 0 else "#555"
                   for j in result["ml_job_type"]],
            size=4,
        ),
    ), row=4, col=1)

    fig.update_layout(
        height=800, template="plotly_dark",
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", y=-0.04),
    )
    fig.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="State (0-2)",  row=2, col=1,
                     tickvals=[0,1,2], ticktext=["STDBY","IDLE","ACTV"])
    fig.update_yaxes(title_text="Pos (0-1)",    row=3, col=1)
    fig.update_yaxes(title_text="Mode",          row=4, col=1)

    st.plotly_chart(fig, width='stretch')

    # ── Confidence histogram ───────────────────────────────────────────
    valid_conf = result["ml_state_conf"].dropna()
    if len(valid_conf):
        fig_conf = go.Figure(go.Histogram(
            x=valid_conf, nbinsx=30,
            marker_color="#9b59b6", opacity=0.8,
        ))
        fig_conf.update_layout(
            title="State Prediction Confidence Distribution",
            xaxis_title="Confidence", yaxis_title="Count",
            template="plotly_dark", height=260,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_conf, width='stretch')
