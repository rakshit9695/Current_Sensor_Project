"""
ui.py
=====
Streamlit UI for the iTransformer ML diagnostics page.
Imported and called from the main app.py as render_ml_page(df).

Four tabs:
  1. Data & Labels   – label distribution, cycle stats, job clusters
  2. Train Model     – hyperparams, train button, live loss curve
  3. Evaluation      – confusion matrices, accuracy, cycle-pos MAE
  4. Inference View  – overlay ML predictions on current signal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .label_generator import generate_labels, STATE_NAMES
from .trainer import (
    CHECKPOINT_PATH,
    checkpoint_exists,
    train as ml_train,
)
from .predictor import FurnacePredictor


# ── Colour palette ─────────────────────────────────────────────────────────────
STATE_COLORS = {"RED": "#e74c3c", "AMBER": "#f39c12", "GREEN": "#27ae60"}
JOB_PALETTE  = ["#4e9af1", "#f1914e", "#9b59b6", "#27ae60", "#e74c3c"]
PHASE_COLORS = dict(i1="#4e9af1", i2="#f1914e", i3="#4ef19a", i_avg="#f1f14e")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_ml_page(df: pd.DataFrame) -> None:
    """
    Render the full ML diagnostics page.

    Parameters
    ----------
    df : cleaned furnace DataFrame returned by data_loader.load_data
    """
    st.markdown("## 🤖 iTransformer — Machine Cycle Intelligence")
    st.caption(
        "Learns the repeating idle → no-load → load → no-load → idle cycle "
        "from raw current data.  Outputs current state, cycle position, and job type."
    )
    st.markdown("---")

    tab_data, tab_train, tab_eval, tab_infer = st.tabs([
        "📊 Data & Labels",
        "🏋️ Train Model",
        "📈 Evaluation",
        "🔍 Inference View",
    ])

    with tab_data:
        _tab_data(df)

    with tab_train:
        _tab_train(df)

    with tab_eval:
        _tab_eval(df)

    with tab_infer:
        _tab_infer(df)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Data & Labels
# ══════════════════════════════════════════════════════════════════════════════

def _tab_data(df: pd.DataFrame) -> None:
    st.subheader("Auto-Label Generation")
    st.markdown(
        "Labels are derived automatically from I_avg thresholds + cycle detection "
        "+ K-means clustering — no manual annotation required."
    )

    with st.expander("Label Generation Settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        red_max    = c1.number_input("RED max I_avg (A)",   min_value=0.1,  max_value=10.0, value=2.0,  step=0.1, key="lbl_red")
        green_min  = c2.number_input("GREEN min I_avg (A)", min_value=1.0,  max_value=30.0, value=12.0, step=0.5, key="lbl_grn")
        min_run    = c3.number_input("Min run (samples)",   min_value=2,    max_value=60,   value=10,   step=1,   key="lbl_mr")
        n_jobs     = c4.number_input("Job type clusters",   min_value=2,    max_value=6,    value=3,    step=1,   key="lbl_nj")

    run_btn = st.button("▶ Generate Labels", type="primary", key="btn_gen_labels")

    if run_btn or "ml_labeled_df" not in st.session_state:
        with st.spinner("Generating labels (cycle detection + K-means)…"):
            try:
                ldf, meta = generate_labels(
                    df,
                    red_max    = float(red_max),
                    green_min  = float(green_min),
                    min_run    = int(min_run),
                    n_job_types= int(n_jobs),
                )
                st.session_state["ml_labeled_df"] = ldf
                st.session_state["ml_label_meta"]  = meta
                st.success(f"Labels generated — {meta['n_cycles']} complete cycles found.")
            except Exception as e:
                st.error(f"Label generation failed: {e}")
                return

    ldf  = st.session_state.get("ml_labeled_df")
    meta = st.session_state.get("ml_label_meta")

    if ldf is None or meta is None:
        return

    # ── Summary metrics ────────────────────────────────────────────────
    st.markdown("### Label Summary")
    total = len(ldf)
    sc    = meta["state_counts"]
    cols  = st.columns(5)
    cols[0].metric("Total Samples",  f"{total:,}")
    cols[1].metric("🔴 RED",         f"{sc['RED']:,}",   f"{sc['RED']/total*100:.1f}%")
    cols[2].metric("🟡 AMBER",       f"{sc['AMBER']:,}", f"{sc['AMBER']/total*100:.1f}%")
    cols[3].metric("🟢 GREEN",       f"{sc['GREEN']:,}", f"{sc['GREEN']/total*100:.1f}%")
    cols[4].metric("Complete Cycles", f"{meta['n_cycles']:,}")

    # ── State distribution pie ─────────────────────────────────────────
    col_pie, col_dur = st.columns(2)
    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["RED (Idle)", "AMBER (No-Load)", "GREEN (Load)"],
            values=[sc["RED"], sc["AMBER"], sc["GREEN"]],
            marker=dict(colors=["#e74c3c", "#f39c12", "#27ae60"]),
            hole=0.4,
        ))
        fig_pie.update_layout(
            title="State Distribution", template="plotly_dark",
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_pie, width='stretch')

    # ── Cycle duration histogram ───────────────────────────────────────
    with col_dur:
        durs = meta["cycle_durations"]
        if durs:
            fig_dur = go.Figure(go.Histogram(
                x=durs, nbinsx=40,
                marker_color="#4e9af1",
                name="Cycle Duration",
            ))
            fig_dur.update_layout(
                title=f"Cycle Duration  (mean={meta['mean_cycle_sec']:.0f}s  "
                      f"std={meta['std_cycle_sec']:.0f}s)",
                xaxis_title="Duration (s)", yaxis_title="Count",
                template="plotly_dark", height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_dur, width='stretch')
        else:
            st.info("No complete cycles detected. Try adjusting thresholds.")

    # ── Job cluster scatter ─────────────────────────────────────────────
    if meta["n_cycles"] > 0 and meta.get("cycle_features") is not None:
        st.markdown("### Job Type Clusters (cycle-level)")
        feat = meta["cycle_features"]
        jlabels = meta.get("job_labels_per_cycle", [])
        if len(feat) == len(jlabels) and len(feat) > 0:
            feat_df = pd.DataFrame(feat, columns=[
                "log_duration", "mean_green_rms", "max_green_rms",
                "std_green_rms", "green_frac", "amber_frac",
            ])
            feat_df["job_type"] = jlabels
            feat_df["duration_s"] = np.expm1(feat[:, 0])

            n_clusters = meta["n_job_types"]
            job_labels_map = {
                0: "Light Load", 1: "Medium Load", 2: "Heavy Load",
                3: "Job D", 4: "Job E",
            }

            fig_scatter = go.Figure()
            for jt in range(n_clusters):
                mask = feat_df["job_type"] == jt
                fig_scatter.add_trace(go.Scatter(
                    x=feat_df.loc[mask, "duration_s"],
                    y=feat_df.loc[mask, "mean_green_rms"],
                    mode="markers",
                    name=job_labels_map.get(jt, f"Job {jt}"),
                    marker=dict(
                        color=JOB_PALETTE[jt % len(JOB_PALETTE)],
                        size=7, opacity=0.8,
                    ),
                    hovertemplate=(
                        "Duration: %{x:.0f}s<br>Mean GREEN I_avg: %{y:.2f}A"
                    ),
                ))
            fig_scatter.update_layout(
                title="Job Clusters  (x=cycle duration, y=mean load current)",
                xaxis_title="Cycle Duration (s)",
                yaxis_title="Mean GREEN I_avg (A)",
                template="plotly_dark", height=350,
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_scatter, width='stretch')

    # ── Raw label timeline preview ──────────────────────────────────────
    st.markdown("### Label Timeline Preview (first 2 hours)")
    preview = ldf.iloc[:7200].copy()
    if "timestamp" in preview.columns:
        t_axis = preview["timestamp"]
    else:
        t_axis = preview.index

    color_map = {0: "#e74c3c", 1: "#f39c12", 2: "#27ae60"}
    fig_tl = go.Figure()
    for state_int, color in color_map.items():
        mask = preview["state_label"] == state_int
        fig_tl.add_trace(go.Scatter(
            x=t_axis[mask], y=preview.loc[mask, "i_avg"],
            mode="markers",
            name=STATE_NAMES[state_int],
            marker=dict(color=color, size=2),
        ))
    fig_tl.update_layout(
        title="I_avg coloured by auto-label",
        xaxis_title="Time", yaxis_title="I_avg (A)",
        template="plotly_dark", height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_tl, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – Train Model
# ══════════════════════════════════════════════════════════════════════════════

def _tab_train(df: pd.DataFrame) -> None:
    st.subheader("Train iTransformer")

    ldf  = st.session_state.get("ml_labeled_df")
    meta = st.session_state.get("ml_label_meta")

    if ldf is None:
        st.warning("Go to **Data & Labels** first and click **Generate Labels**.")
        return

    st.markdown("#### Hyperparameters")
    c1, c2, c3 = st.columns(3)
    look_back  = c1.slider("Look-back window (s)",  30, 300, 120, 10, key="hp_lb")
    epochs     = c2.slider("Epochs",                 5,  50,  25,  5, key="hp_ep")
    batch_size = c3.selectbox("Batch size", [32, 64, 128, 256], index=1, key="hp_bs")

    c4, c5, c6 = st.columns(3)
    lr         = c4.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, key="hp_lr")
    d_model    = c5.selectbox("d_model",  [32, 64, 128], index=1, key="hp_dm")
    n_layers   = c6.selectbox("Layers",   [2, 3, 4],     index=1, key="hp_nl")

    c7, c8, c9 = st.columns(3)
    alpha      = c7.number_input("State loss weight α",    min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="hp_a")
    beta       = c8.number_input("CyclePos loss weight β", min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="hp_b")
    gamma      = c9.number_input("Job loss weight γ",      min_value=0.0, max_value=2.0, value=0.5, step=0.1, key="hp_g")

    if checkpoint_exists():
        st.info(f"A trained checkpoint exists at `{CHECKPOINT_PATH}`. "
                "Training again will overwrite it.")

    train_btn = st.button("🚀 Start Training", type="primary", key="btn_train")

    if train_btn:
        # ── Build datasets ───────────────────────────────────────────────
        from .dataset import build_datasets
        try:
            train_ds, val_ds, test_ds, norm_stats = build_datasets(
                ldf, look_back=int(look_back)
            )
        except ValueError as e:
            st.error(str(e))
            return

        st.markdown("#### Training Progress")
        progress_bar   = st.progress(0.0)
        status_text    = st.empty()
        loss_chart_ph  = st.empty()
        metrics_ph     = st.empty()

        history_accum: dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_state_acc": [], "val_state_acc": [],
            "val_cyc_mae": [], "val_job_acc": [],
        }

        def _progress_cb(epoch: int, total: int, m: dict) -> None:
            frac = epoch / total
            progress_bar.progress(frac)
            status_text.text(
                f"Epoch {epoch}/{total}  |  "
                f"train_loss={m['train_loss']:.4f}  val_loss={m['val_loss']:.4f}  |  "
                f"state_acc={m['state_acc']:.1f}%  job_acc={m['job_acc']:.1f}%  |  "
                f"cyc_mae={m['cyc_mae']:.4f}  ({m['elapsed_sec']:.1f}s)"
            )
            for k in history_accum:
                if k in m:
                    pass  # updated below after return
            # Redraw loss chart
            if epoch >= 2:
                fig = _loss_chart(
                    history_accum["train_loss"],
                    history_accum["val_loss"],
                )
                loss_chart_ph.plotly_chart(fig, width='stretch')

        # Patch progress_cb to collect history too
        hist_ref = [None]

        def _wrapped_cb(epoch, total, m):
            history_accum["train_loss"].append(m["train_loss"])
            history_accum["val_loss"].append(m["val_loss"])
            _progress_cb(epoch, total, m)

        try:
            with st.spinner("Training…"):
                history = ml_train(
                    train_ds   = train_ds,
                    val_ds     = val_ds,
                    norm_stats = norm_stats,
                    label_meta = meta,
                    epochs     = int(epochs),
                    batch_size = int(batch_size),
                    lr         = float(lr),
                    alpha      = float(alpha),
                    beta       = float(beta),
                    gamma      = float(gamma),
                    d_model    = int(d_model),
                    n_layers   = int(n_layers),
                    d_ff       = int(d_model) * 2,
                    progress_cb= _wrapped_cb,
                )

            progress_bar.progress(1.0)
            st.success(f"Training complete!  Checkpoint saved to `{CHECKPOINT_PATH}`")

            # Store test_ds for evaluation tab
            st.session_state["ml_test_ds"]  = test_ds
            st.session_state["ml_norm_stats"] = norm_stats

            # Final loss chart
            fig = _loss_chart(history["train_loss"], history["val_loss"])
            loss_chart_ph.plotly_chart(fig, width='stretch')

            # Final val metrics
            metrics_ph.markdown(
                f"**Best val state acc:** {max(history['val_state_acc'])*100:.1f}%  |  "
                f"**Best val job acc:** {max(history['val_job_acc'])*100:.1f}%  |  "
                f"**Best cyc MAE:** {min(history['val_cyc_mae']):.4f}"
            )

        except Exception as e:
            st.error(f"Training failed: {e}")
            raise


def _loss_chart(train_loss: list, val_loss: list) -> go.Figure:
    fig = go.Figure()
    epochs_range = list(range(1, len(train_loss) + 1))
    fig.add_trace(go.Scatter(x=epochs_range, y=train_loss,
                             name="Train Loss", line=dict(color="#4e9af1")))
    fig.add_trace(go.Scatter(x=epochs_range, y=val_loss,
                             name="Val Loss",   line=dict(color="#f1914e")))
    fig.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch", yaxis_title="Loss",
        template="plotly_dark", height=300,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 – Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _tab_eval(df: pd.DataFrame) -> None:
    st.subheader("Model Evaluation")

    if not checkpoint_exists():
        st.warning("No trained model found. Go to **Train Model** first.")
        return

    ldf = st.session_state.get("ml_labeled_df")
    if ldf is None:
        st.warning("Go to **Data & Labels** first and click **Generate Labels**.")
        return

    eval_btn = st.button("▶ Run Evaluation on Test Set", type="primary", key="btn_eval")

    if eval_btn or "ml_eval_results" not in st.session_state:
        try:
            predictor = FurnacePredictor.from_checkpoint()
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            return

        # Use last 15% as test set (matches temporal split in trainer)
        n     = len(ldf)
        t_idx = int(n * 0.85)
        test_df = ldf.iloc[t_idx:].copy()

        if len(test_df) < predictor.look_back:
            st.error("Test set is too small for the model's look_back window.")
            return

        with st.spinner("Running inference on test set…"):
            prog_bar = st.progress(0.0)
            results  = predictor.evaluate_on_labeled(test_df, batch_size=256)
            prog_bar.progress(1.0)

        st.session_state["ml_eval_results"] = results
        st.session_state["ml_predictor"]    = predictor

    results   = st.session_state.get("ml_eval_results", {})
    predictor = st.session_state.get("ml_predictor")

    if not results:
        return

    # ── Headline metrics ───────────────────────────────────────────────
    st.markdown("### Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("State Accuracy",      f"{results['state_acc']*100:.1f}%")
    c2.metric("Job Type Accuracy",   f"{results['job_acc']*100:.1f}%" if not np.isnan(results["job_acc"]) else "N/A")
    c3.metric("Cycle Pos MAE",       f"{results['cyc_mae']:.4f}" if not np.isnan(results["cyc_mae"]) else "N/A")
    c4.metric("Test Samples",        f"{results['n_valid']:,}")

    # ── Confusion matrices ─────────────────────────────────────────────
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.markdown("#### State Confusion Matrix")
        cm_state = results["state_cm"]
        labels   = ["RED", "AMBER", "GREEN"]
        fig_cm1  = _confusion_matrix_fig(cm_state, labels, "State Confusion Matrix")
        st.plotly_chart(fig_cm1, width='stretch')

    with col_cm2:
        st.markdown("#### Job Type Confusion Matrix")
        cm_job = results["job_cm"]
        n_jt   = cm_job.shape[0]
        job_lbls = [f"Job {i}" for i in range(n_jt)]
        fig_cm2  = _confusion_matrix_fig(cm_job, job_lbls, "Job Type Confusion Matrix")
        st.plotly_chart(fig_cm2, width='stretch')

    # ── Per-class report ───────────────────────────────────────────────
    report = results.get("state_report", {})
    if report:
        st.markdown("#### Per-Class Classification Report")
        rows = []
        for cls in ["RED", "AMBER", "GREEN"]:
            if cls in report:
                r = report[cls]
                rows.append(dict(
                    Class=cls,
                    Precision=f"{r['precision']:.3f}",
                    Recall   =f"{r['recall']:.3f}",
                    F1       =f"{r['f1-score']:.3f}",
                    Support  =int(r["support"]),
                ))
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Class"), width='stretch')


def _confusion_matrix_fig(cm: np.ndarray, labels: list[str], title: str) -> go.Figure:
    # Normalise rows for colour (show raw counts as text)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    fig = go.Figure(go.Heatmap(
        z        = cm_norm,
        x        = labels,
        y        = labels,
        colorscale = "Blues",
        showscale  = True,
        text     = cm,
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title       = title,
        xaxis_title = "Predicted",
        yaxis_title = "True",
        template    = "plotly_dark",
        height      = 320,
        margin      = dict(l=60, r=20, t=50, b=60),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 – Inference View
# ══════════════════════════════════════════════════════════════════════════════

def _tab_infer(df: pd.DataFrame) -> None:
    st.subheader("Live Inference Visualiser")

    if not checkpoint_exists():
        st.warning("No trained model found. Go to **Train Model** first.")
        return

    # ── Time range selector ───────────────────────────────────────────
    st.markdown("Select a time range from the dataset to visualise:")

    total_hours = len(df) / 3600.0
    c1, c2 = st.columns(2)
    start_h = c1.slider("Start (hours from beginning)", 0.0, max(0.0, total_hours - 0.5), 0.0, 0.1, key="inf_sh")
    dur_h   = c2.slider("Duration (hours)",              0.1, min(2.0, total_hours),       0.5, 0.1, key="inf_dh")

    start_row = int(start_h * 3600)
    end_row   = min(len(df), int((start_h + dur_h) * 3600))
    df_slice  = df.iloc[start_row:end_row].copy()

    run_btn = st.button("▶ Run Inference on Selection", type="primary", key="btn_infer")

    if run_btn or st.session_state.get("ml_infer_result_start") == start_row:
        try:
            predictor = FurnacePredictor.from_checkpoint()
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            return

        lb = predictor.look_back
        # We need look_back extra rows before the slice for context
        ctx_start = max(0, start_row - lb)
        df_ctx    = df.iloc[ctx_start:end_row].copy()

        if len(df_ctx) < lb:
            st.error(f"Selection too short (need at least {lb} rows for context).")
            return

        with st.spinner("Running inference…"):
            prog_ph = st.progress(0.0)
            result_ctx = predictor.predict(df_ctx, batch_size=256,
                                            progress_cb=lambda f: prog_ph.progress(f))
            prog_ph.progress(1.0)

        # Trim back to the requested slice (remove context rows)
        offset = start_row - ctx_start
        result = result_ctx.iloc[offset:].reset_index(drop=True)
        df_slice_reset = df_slice.reset_index(drop=True)

        st.session_state["ml_infer_result"]       = result
        st.session_state["ml_infer_df_slice"]     = df_slice_reset
        st.session_state["ml_infer_result_start"] = start_row

    result     = st.session_state.get("ml_infer_result")
    df_vis     = st.session_state.get("ml_infer_df_slice")

    if result is None or df_vis is None:
        st.info("Click **Run Inference on Selection** to see predictions.")
        return

    # ── Current signal + state band ────────────────────────────────────
    ldf  = st.session_state.get("ml_labeled_df")

    if "timestamp" in result.columns:
        t_axis = result["timestamp"]
    else:
        t_axis = pd.RangeIndex(len(result))

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Phase Currents",
            "ML State  vs  Auto-Label",
            "Cycle Position (ML prediction)",
            "Job Type (ML prediction)",
        ),
        vertical_spacing=0.06,
        row_heights=[0.35, 0.25, 0.20, 0.20],
    )

    # Row 1: phase currents
    for col, color in [("i1", PHASE_COLORS["i1"]), ("i2", PHASE_COLORS["i2"]),
                        ("i3", PHASE_COLORS["i3"]), ("i_avg", PHASE_COLORS["i_avg"])]:
        if col in result.columns:
            fig.add_trace(go.Scatter(
                x=t_axis, y=result[col], name=col.upper(),
                line=dict(color=color, width=1),
            ), row=1, col=1)

    # Row 2: ML state (numeric 0/1/2) vs auto-label
    ml_valid = result["ml_state"] >= 0
    fig.add_trace(go.Scatter(
        x=t_axis[ml_valid], y=result.loc[ml_valid, "ml_state"],
        name="ML State", mode="markers",
        marker=dict(
            color=[STATE_COLORS.get(n, "#888") for n in result.loc[ml_valid, "ml_state_name"]],
            size=4,
        ),
    ), row=2, col=1)

    # Auto-label (only if we have labeled df and the rows match up)
    slice_offset = int(start_h * 3600)
    if ldf is not None and slice_offset + len(result) <= len(ldf):
        auto_labels = ldf["state_label"].iloc[slice_offset : slice_offset + len(result)].values
        auto_names  = [STATE_NAMES.get(int(s), "?") for s in auto_labels]
        fig.add_trace(go.Scatter(
            x=t_axis, y=auto_labels + 0.15,     # slight offset to avoid overlap
            name="Auto-Label", mode="markers",
            marker=dict(
                color=[STATE_COLORS.get(n, "#888") for n in auto_names],
                size=3, symbol="circle-open",
            ),
        ), row=2, col=1)

    fig.update_yaxes(
        tickvals=[0, 1, 2], ticktext=["RED", "AMBER", "GREEN"],
        row=2, col=1,
    )

    # Row 3: cycle position
    fig.add_trace(go.Scatter(
        x=t_axis[ml_valid], y=result.loc[ml_valid, "ml_cycle_pos"],
        name="Cycle Position", line=dict(color="#9b59b6", width=1.5),
    ), row=3, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], title_text="0→1", row=3, col=1)

    # Row 4: job type
    job_valid = ml_valid & (result["ml_job_type"] >= 0)
    predictor = st.session_state.get("ml_predictor")
    n_jt = predictor.n_job_types if predictor is not None else 3
    palette_idx = result.loc[job_valid, "ml_job_type"].tolist()
    fig.add_trace(go.Scatter(
        x=t_axis[job_valid], y=result.loc[job_valid, "ml_job_type"],
        name="Job Type", mode="markers",
        marker=dict(
            color=[JOB_PALETTE[j % len(JOB_PALETTE)] for j in palette_idx],
            size=4,
        ),
    ), row=4, col=1)
    fig.update_yaxes(
        tickvals=list(range(n_jt)),
        ticktext=[f"Job {i}" for i in range(n_jt)],
        row=4, col=1,
    )

    fig.update_layout(
        height=700, template="plotly_dark",
        showlegend=True,
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Confidence histogram ───────────────────────────────────────────
    conf_valid = result.loc[ml_valid, "ml_state_conf"].dropna()
    if len(conf_valid) > 0:
        fig_conf = go.Figure(go.Histogram(
            x=conf_valid, nbinsx=40, marker_color="#4e9af1",
        ))
        fig_conf.update_layout(
            title="State Prediction Confidence Distribution",
            xaxis_title="Softmax Confidence", yaxis_title="Count",
            template="plotly_dark", height=250,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_conf, width='stretch')

    # ── Statistics table ───────────────────────────────────────────────
    st.markdown("#### Prediction Breakdown")
    vc = result.loc[ml_valid, "ml_state_name"].value_counts()
    rows = [
        dict(State=s, Count=int(vc.get(s, 0)), Fraction=f"{vc.get(s,0)/max(ml_valid.sum(),1)*100:.1f}%")
        for s in ["RED", "AMBER", "GREEN"]
    ]
    st.dataframe(pd.DataFrame(rows).set_index("State"), width='stretch')
