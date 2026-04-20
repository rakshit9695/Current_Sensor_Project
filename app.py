"""
app.py  –  Combined Current Analyser
======================================
Two machines, same pipeline, one app.

  Machine 1 : Wave Soldering Furnace   (wave_soldering/)
  Machine 2 : MV Conveyer              (mv_conveyer/)

Each machine has its own Page 1 (Validation) and Page 2 (RAG Visualisation).
Select the machine in the sidebar; all state is kept separately per machine.

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Wave Soldering imports ─────────────────────────────────────────────────────
from wave_soldering.data_loader        import load_data as ws_load_data, get_sample_rate as ws_sample_rate
from wave_soldering.windowing          import generate_windows as ws_gen_windows, get_window_timestamps
from wave_soldering.fft_analysis       import compute_window_fft, batch_compute_fft
from wave_soldering.validation         import validate_window as ws_val_win, validate_dataset as ws_val_ds
from wave_soldering.validation         import DEFAULT_THRESHOLDS as WS_VAL_DEFAULTS
from wave_soldering.feature_engineering import compute_features, WindowFeatures, features_to_dict
from wave_soldering.rag_classifier     import (
    classify as ws_classify,
    RED, AMBER, GREEN,
    STATE_COLORS as WS_STATE_COLORS,
    STATE_LABELS as WS_STATE_LABELS,
    DEFAULT_THRESHOLDS as WS_RAG_DEFAULTS,
)
from wave_soldering.state_manager      import StateManager, build_state_timeline
from wave_soldering.ml.ui              import render_ml_page
from mv_conveyer.ml.ui                 import render_ml_page as render_mvc_ml_page

# ── MV Conveyer imports ────────────────────────────────────────────────────────
from mv_conveyer.data_loader           import load_data as mvc_load_data, get_sample_rate as mvc_sample_rate
from mv_conveyer.windowing             import generate_windows as mvc_gen_windows
from mv_conveyer.fft_analysis          import compute_window_fft as mvc_fft_win, batch_compute_fft as mvc_batch_fft
from mv_conveyer.validation            import validate_window as mvc_val_win, validate_dataset as mvc_val_ds
from mv_conveyer.validation            import DEFAULT_THRESHOLDS as MVC_VAL_DEFAULTS
from mv_conveyer.feature_engineering   import compute_features as mvc_feat, features_to_dict as mvc_feat_dict
from mv_conveyer.rag_classifier        import (
    classify as mvc_classify,
    STATE_COLORS as MVC_STATE_COLORS,
    STATE_LABELS as MVC_STATE_LABELS,
    DEFAULT_THRESHOLDS as MVC_RAG_DEFAULTS,
)
from mv_conveyer.state_manager         import StateManager as mvc_StateManager


# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Current Analyser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.state-circle {
    width:130px; height:130px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; font-weight:bold; color:white;
    margin:auto; transition:all 0.3s ease;
}
.metric-box {
    background:#1e1e2e; border-radius:10px; padding:10px 14px;
    text-align:center; margin:3px;
}
.metric-label { font-size:0.72rem; color:#aaa; text-transform:uppercase; }
.metric-value { font-size:1.4rem; font-weight:bold; color:#fff; }
.machine-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-size:0.8rem; font-weight:bold; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants per machine
# ══════════════════════════════════════════════════════════════════════════════

MACHINE_META = {
    "furnace": {
        "label"        : "Wave Soldering Furnace",
        "icon"         : "🔥",
        "badge_color"  : "#e67e22",
        "default_ws"   : 60,
        "load_data"    : ws_load_data,
        "sample_rate"  : ws_sample_rate,
        "gen_windows"  : ws_gen_windows,
        "fft_win"      : compute_window_fft,
        "batch_fft"    : batch_compute_fft,
        "val_win"      : ws_val_win,
        "val_ds"       : ws_val_ds,
        "feat"         : compute_features,
        "feat_dict"    : features_to_dict,
        "classify"     : ws_classify,
        "StateManager" : StateManager,
        "val_defaults" : WS_VAL_DEFAULTS,
        "rag_defaults" : WS_RAG_DEFAULTS,
        "state_colors" : WS_STATE_COLORS,
        "state_labels" : WS_STATE_LABELS,
    },
    "conveyer": {
        "label"        : "MV Conveyer",
        "icon"         : "🏭",
        "badge_color"  : "#2980b9",
        "default_ws"   : 3,
        "load_data"    : mvc_load_data,
        "sample_rate"  : mvc_sample_rate,
        "gen_windows"  : mvc_gen_windows,
        "fft_win"      : mvc_fft_win,
        "batch_fft"    : mvc_batch_fft,
        "val_win"      : mvc_val_win,
        "val_ds"       : mvc_val_ds,
        "feat"         : mvc_feat,
        "feat_dict"    : mvc_feat_dict,
        "classify"     : mvc_classify,
        "StateManager" : mvc_StateManager,
        "val_defaults" : MVC_VAL_DEFAULTS,
        "rag_defaults" : MVC_RAG_DEFAULTS,
        "state_colors" : MVC_STATE_COLORS,
        "state_labels" : MVC_STATE_LABELS,
    },
}

PHASE_COLORS = dict(i1="#4e9af1", i2="#f1914e", i3="#4ef19a", i_avg="#f1f14e")
MAX_WINDOWS  = 1000   # sub-sample cap for large datasets


# ══════════════════════════════════════════════════════════════════════════════
# Session-state bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def _ss_init(**defaults):
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ss_init(
    current_machine = "furnace",
    current_page    = "validation",
    furnace  = dict(
        df=None, windows_proc=None, fft_cache=None,
        val_results=None, dataset_val=None,
        features_cache=None, raw_states=None,
        smoothed_states=None, win_centres=None,
        playing=False, play_idx=0,
        window_size_sec=60,
        rag_thr=dict(WS_RAG_DEFAULTS),
        val_thr=dict(WS_VAL_DEFAULTS),
        last_file=None,
    ),
    conveyer = dict(
        df=None, df_cyclic=None, windows_proc=None, fft_cache=None,
        val_results=None, dataset_val=None,
        features_cache=None, raw_states=None,
        smoothed_states=None, win_centres=None,
        playing=False, play_idx=0,
        window_size_sec=3,
        rag_thr=dict(MVC_RAG_DEFAULTS),
        val_thr=dict(MVC_VAL_DEFAULTS),
        last_file=None,
    ),
)


def ss(key=None):
    """Shorthand: return the current machine's state dict, or a key within it."""
    d = st.session_state[st.session_state.current_machine]
    return d if key is None else d[key]


def ss_set(key, value):
    st.session_state[st.session_state.current_machine][key] = value


def ss_invalidate():
    """Clear all computed results for current machine."""
    for k in ("windows_proc", "fft_cache", "val_results", "dataset_val",
              "features_cache", "raw_states", "smoothed_states", "win_centres"):
        ss_set(k, None)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    meta = MACHINE_META

    st.sidebar.title("⚡ Current Analyser")
    st.sidebar.markdown("---")

    # ── Machine selector ───────────────────────────────────────────────
    st.sidebar.subheader("🏭 Select Machine")
    machine_choice = st.sidebar.radio(
        "Machine",
        options=["furnace", "conveyer"],
        format_func=lambda m: f"{meta[m]['icon']}  {meta[m]['label']}",
        index=0 if st.session_state.current_machine == "furnace" else 1,
        label_visibility="collapsed",
    )
    if machine_choice != st.session_state.current_machine:
        st.session_state.current_machine = machine_choice
        st.rerun()

    st.sidebar.markdown("---")

    # ── Page selector ─────────────────────────────────────────────────
    st.sidebar.subheader("📄 Page")
    m = st.session_state.current_machine
    page_options = ["validation", "rag", "ml"]

    def _page_label(p):
        return {
            "validation": "📋  Data Validation",
            "rag":        "🚦  RAG Visualisation",
            "ml":         "🤖  ML Diagnostics",
        }[p]

    # Clamp current page if machine switched away from furnace
    if st.session_state.current_page not in page_options:
        st.session_state.current_page = "validation"

    page_choice = st.sidebar.radio(
        "Page",
        options=page_options,
        format_func=_page_label,
        index=page_options.index(st.session_state.current_page),
        label_visibility="collapsed",
    )
    st.session_state.current_page = page_choice

    st.sidebar.markdown("---")

    m = st.session_state.current_machine

    # ── Window size ───────────────────────────────────────────────────
    # Conveyer cycle: ~11s ON + ~4s OFF = ~15s period.
    # Window must be < 4s to sit inside a single IDLE gap without averaging it away.
    # Furnace state changes happen over minutes — larger windows are appropriate.
    st.sidebar.subheader("⚙️ Window Settings")
    cur_ws = ss("window_size_sec")
    if m == "conveyer":
        new_ws = st.sidebar.slider("Window size (seconds)", 2, 10, int(min(cur_ws, 10)), 1)
    else:
        new_ws = st.sidebar.slider("Window size (seconds)", 10, 120, int(max(cur_ws, 10)), 10)
    if new_ws != cur_ws:
        ss_set("window_size_sec", new_ws)
        ss_invalidate()

    # ── Validation thresholds ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Validation Thresholds")
    vt = ss("val_thr")
    vt["freq_tolerance_hz"]     = st.sidebar.number_input("Freq tolerance (Hz)",      0.1, 2.0, float(vt["freq_tolerance_hz"]),     0.1,  key=f"{m}_ft")
    vt["max_phase_imbalance"]   = st.sidebar.number_input("Max phase imbalance (CV)", 0.1, 3.0, float(vt["max_phase_imbalance"]),   0.05, key=f"{m}_pi")
    vt["valid_window_fraction"] = st.sidebar.number_input("Valid window fraction",    0.5, 1.0, float(vt["valid_window_fraction"]), 0.05, key=f"{m}_vwf")

    # ── RAG thresholds ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎚️ RAG Thresholds")
    rt = ss("rag_thr")
    rt["red_max_rms"]      = st.sidebar.number_input("RED max RMS (A)",   0.01, 10.0, float(rt["red_max_rms"]),   0.05, key=f"{m}_rmr")
    rt["green_min_rms"]    = st.sidebar.number_input("GREEN min RMS (A)", 0.01, 30.0, float(rt["green_min_rms"]), 0.05, key=f"{m}_gmr")
    rt["green_min_thd"]    = st.sidebar.number_input("GREEN min THD",     0.01, 1.0,  float(rt["green_min_thd"]), 0.01, key=f"{m}_gmt")
    rt["green_min_imbalance"] = st.sidebar.number_input("GREEN min imbalance", 0.01, 2.0, float(rt["green_min_imbalance"]), 0.01, key=f"{m}_gmi")


# ══════════════════════════════════════════════════════════════════════════════
# Data pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(df: pd.DataFrame):
    """Process windows → FFT → validation → features → classification for the active machine."""
    m    = st.session_state.current_machine
    meta = MACHINE_META[m]
    sr   = meta["sample_rate"](df)

    with st.spinner("Generating windows…"):
        windows = meta["gen_windows"](df, window_size_sec=ss("window_size_sec"), overlap=0.5, sample_rate=sr)

    # Sub-sample for very large datasets
    if len(windows) > MAX_WINDOWS:
        step = len(windows) // MAX_WINDOWS
        windows_proc = windows[::step]
    else:
        windows_proc = windows

    with st.spinner(f"Computing FFT for {len(windows_proc)} windows…"):
        fft_cache = meta["batch_fft"](windows_proc, sample_rate=sr)

    with st.spinner("Validating windows…"):
        val_results = [meta["val_win"](w.data, thresholds=ss("val_thr")) for w in windows_proc]
        dataset_val = meta["val_ds"](windows_proc, thresholds=ss("val_thr"))

    with st.spinner("Engineering features…"):
        features_cache = [meta["feat"](w.data, f) for w, f in zip(windows_proc, fft_cache)]

    with st.spinner("Classifying states…"):
        raw_states = [meta["classify"](f, thresholds=ss("rag_thr")).state for f in features_cache]
        raw_confs  = [meta["classify"](f, thresholds=ss("rag_thr")).confidence for f in features_cache]
        sm = meta["StateManager"](min_consecutive=3)
        smoothed = sm.run_batch(raw_states, raw_confs)

    ss_set("windows_proc",    windows_proc)
    ss_set("fft_cache",       fft_cache)
    ss_set("val_results",     val_results)
    ss_set("dataset_val",     dataset_val)
    ss_set("features_cache",  features_cache)
    ss_set("raw_states",      raw_states)
    ss_set("smoothed_states", smoothed)
    ss_set("win_centres",     get_window_timestamps(windows_proc))
    ss_set("play_idx",        0)
    ss_set("playing",         False)


# ══════════════════════════════════════════════════════════════════════════════
# Shared plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def fig_time_series(df, highlight_start=None, highlight_end=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Phase Currents  I1 / I2 / I3", "Average Current  I_avg"),
                        vertical_spacing=0.10)
    t = df["timestamp"]
    for col, color in [("i1", PHASE_COLORS["i1"]), ("i2", PHASE_COLORS["i2"]), ("i3", PHASE_COLORS["i3"])]:
        fig.add_trace(go.Scatter(x=t, y=df[col], name=col.upper(),
                                  line=dict(color=color, width=1),
                                  hovertemplate=f"{col.upper()}: %{{y:.3f}} A"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=df["i_avg"], name="I_AVG",
                              line=dict(color=PHASE_COLORS["i_avg"], width=1.5),
                              hovertemplate="I_avg: %{y:.3f} A"), row=2, col=1)
    if highlight_start and highlight_end:
        for row in (1, 2):
            fig.add_vrect(x0=highlight_start, x1=highlight_end,
                          fillcolor="rgba(255,255,0,0.12)", line_width=0, row=row, col=1)
    fig.update_layout(height=420, template="plotly_dark", showlegend=True,
                      margin=dict(l=40, r=20, t=40, b=30),
                      legend=dict(orientation="h", y=-0.12))
    fig.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=1)
    return fig


def fig_fft(fft_results, sample_rate=1.0):
    fig = go.Figure()
    for col in ("i1", "i2", "i3", "i_avg"):
        res = fft_results.get(col)
        if res is None or len(res.freqs) < 2:
            continue
        freqs_mhz = res.freqs * 1000.0
        fig.add_trace(go.Scatter(x=freqs_mhz[1:], y=res.magnitudes[1:],
                                  name=col.upper(), line=dict(color=PHASE_COLORS[col], width=1.5),
                                  hovertemplate="Freq: %{x:.1f} mHz<br>Mag: %{y:.4f} A"))
        fund_mhz = res.fundamental_freq * 1000.0
        if fund_mhz > 0:
            fig.add_vline(x=fund_mhz, line_dash="dash", line_color=PHASE_COLORS[col],
                          annotation_text=f"{fund_mhz:.0f} mHz", annotation_position="top")
    fig.update_layout(title="Operational Frequency Spectrum (FFT of RMS time-series)",
                      xaxis_title="Frequency (mHz)",
                      yaxis_title="Magnitude (A)",
                      template="plotly_dark", height=340,
                      margin=dict(l=40, r=20, t=50, b=40),
                      legend=dict(orientation="h"))
    return fig


def fig_grid_freq(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["frequency"],
                              mode="lines", line=dict(color="#9b59b6", width=1),
                              name="Grid Freq", hovertemplate="Freq: %{y:.2f} Hz"))
    fig.add_hline(y=50.0, line_dash="dash", line_color="#2ecc71",
                  annotation_text="50 Hz target")
    fig.add_hrect(y0=49.5, y1=50.5, fillcolor="rgba(46,204,113,0.08)", line_width=0)
    fig.update_layout(title="Sensor-Reported Grid Frequency (50 Hz validation)",
                      xaxis_title="Time", yaxis_title="Frequency (Hz)",
                      template="plotly_dark", height=240,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def fig_state_timeline(smoothed_states, win_centres, state_colors, state_labels, current_idx=None):
    state_num = {RED: 0, AMBER: 1, GREEN: 2}
    nums = [state_num.get(s, 1) for s in smoothed_states]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=win_centres, y=nums, mode="markers",
        marker=dict(color=[state_colors.get(s, "#888") for s in smoothed_states], size=6, symbol="square"),
        text=[state_labels.get(s, s) for s in smoothed_states],
        hovertemplate="%{text}<extra></extra>",
    ))
    if current_idx is not None and current_idx < len(win_centres):
        fig.add_vline(x=str(win_centres[current_idx]), line_color="white",
                      line_width=2, line_dash="dot")
    fig.update_layout(
        title="State Timeline",
        xaxis_title="Time",
        yaxis=dict(tickvals=[0, 1, 2],
                   ticktext=["🔴 " + state_labels.get(RED, "Idle"),
                             "🟡 " + state_labels.get(AMBER, "No Load"),
                             "🟢 " + state_labels.get(GREEN, "Load")]),
        template="plotly_dark", height=180,
        margin=dict(l=100, r=20, t=40, b=30), showlegend=False,
    )
    return fig


def _circle_html(state, active, state_colors, state_labels):
    color = state_colors.get(state, "#888")
    label = state_labels.get(state, state)
    glow  = f"box-shadow:0 0 40px 15px {color};" if active else ""
    opac  = "opacity:1.0;" if active else "opacity:0.18;"
    return (f'<div class="state-circle" style="background:{color};{glow}{opac}">'
            f'{label}</div><br/>')


def _machine_header(m: str):
    meta = MACHINE_META[m]
    color = meta["badge_color"]
    st.markdown(
        f'<span class="machine-badge" style="background:{color};color:white;">'
        f'{meta["icon"]}  {meta["label"]}</span>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Validation
# ══════════════════════════════════════════════════════════════════════════════

def page_validation():
    m    = st.session_state.current_machine
    meta = MACHINE_META[m]
    _machine_header(m)
    st.title("📋 Data Validation")

    # ── File upload ────────────────────────────────────────────────────
    if m == "conveyer":
        st.info(
            "Upload **both** MV Conveyer CSV files: "
            "`Stop_and_Run_MV_Conveyer.csv` (reference / lookback context) "
            "and `Cyclic_Run_MV_Conveyer.csv` (operational data)."
        )
        uploaded_files = st.file_uploader(
            "Upload CSV files (select both)", type=["csv"], accept_multiple_files=True
        )
        if not uploaded_files:
            return
        file_key = "|".join(sorted(f.name for f in uploaded_files))
        if ss("df") is None or ss("last_file") != file_key:
            with st.spinner("Loading data…"):
                # Combined df for validation + ML lookback context
                df_combined = meta["load_data"](uploaded_files)
                ss_set("df", df_combined)
                ss_set("last_file", file_key)
                ss_invalidate()
                # Cyclic-only df for RAG visualisation
                df_cyclic = None
                for f in uploaded_files:
                    if "cyclic" in f.name.lower():
                        f.seek(0)
                        df_cyclic = meta["load_data"](f)
                        break
                ss_set("df_cyclic", df_cyclic)
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is None:
            st.info(f"Upload the {meta['label']} CSV file to begin.")
            return
        if ss("df") is None or ss("last_file") != uploaded.name:
            with st.spinner("Loading data…"):
                df = meta["load_data"](uploaded)
            ss_set("df", df)
            ss_set("last_file", uploaded.name)
            ss_invalidate()

    df = ss("df")

    # Info banner
    sr = meta["sample_rate"](df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples",  f"{len(df):,}")
    duration_h = len(df) / (sr * 3600)
    c2.metric("Duration",       f"{duration_h:.1f} hrs")
    c3.metric("Sample Rate",    f"~{sr:.1f} Hz")
    c4.metric("I_avg range",    f"{df['i_avg'].min():.2f} – {df['i_avg'].max():.2f} A")

    # Time range selector
    st.subheader("Select Time Range")
    n_hours = max(1, int(len(df) / (sr * 3600)))
    default_h = min(4, n_hours)
    h_range = st.slider("Hours to analyse (from start)", 1, n_hours, default_h)
    df_view = df.iloc[: int(h_range * sr * 3600)]

    # Raw time-series
    st.subheader("Raw Time-Series")
    st.plotly_chart(fig_time_series(df_view), width='stretch')

    # Grid frequency
    st.subheader("Grid Frequency (50 Hz Validation)")
    st.plotly_chart(fig_grid_freq(df_view), width='stretch')

    # For conveyer: RAG pipeline runs on cyclic-only slice, not the combined df
    if m == "conveyer" and ss("df_cyclic") is not None:
        df_cyclic   = ss("df_cyclic")
        sr_cyc      = meta["sample_rate"](df_cyclic)
        n_hours_cyc = max(1, int(len(df_cyclic) / (sr_cyc * 3600)))
        h_cyc       = min(h_range, n_hours_cyc)
        rag_view    = df_cyclic.iloc[: int(h_cyc * sr_cyc * 3600)]
    else:
        rag_view = df_view

    # Run pipeline button
    if ss("windows_proc") is None:
        if st.button("🔍  Run Validation Analysis", type="primary"):
            run_pipeline(rag_view)
            st.rerun()
        return

    # Results
    windows_proc = ss("windows_proc")
    fft_cache    = ss("fft_cache")
    val_results  = ss("val_results")
    dataset_val  = ss("dataset_val")

    st.markdown("---")
    st.subheader("Validation Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Windows",     f"{dataset_val.total_windows:,}")
    m2.metric("Valid Windows",     f"{dataset_val.valid_windows:,}")
    m3.metric("Valid %",           f"{dataset_val.valid_pct:.1f}%")
    m4.metric("Avg Grid Freq",     f"{dataset_val.avg_freq:.2f} Hz")
    m5.metric("Avg Phase Imbal.",  f"{dataset_val.avg_phase_imbalance:.3f}")

    # FFT window inspector
    st.markdown("---")
    st.subheader("FFT Inspector — Select Window")

    # Find the last window with enough samples for a meaningful FFT.
    # A partial trailing window (e.g. 3 samples out of 60) produces a blank chart.
    MIN_FFT_SAMPLES = 8
    last_valid_idx = len(windows_proc) - 1
    for _i in range(len(windows_proc) - 1, -1, -1):
        if len(windows_proc[_i].data) >= MIN_FFT_SAMPLES:
            last_valid_idx = _i
            break

    sel_idx = st.slider("Window index", 0, last_valid_idx, 0)
    sel_win = windows_proc[sel_idx]
    sel_val = val_results[sel_idx]
    sel_fft = fft_cache[sel_idx]

    col_fft, col_meta = st.columns([2, 1])
    with col_fft:
        n_samp = len(sel_win.data)
        if n_samp < MIN_FFT_SAMPLES:
            st.warning(
                f"Window {sel_idx} only has {n_samp} sample(s) — too few for a meaningful FFT "
                f"(need ≥ {MIN_FFT_SAMPLES}). Select an earlier window."
            )
        else:
            st.plotly_chart(fig_fft(sel_fft, sample_rate=meta["sample_rate"](df_view)),
                            width='stretch')
    with col_meta:
        st.markdown(f"**Window {sel_idx}**")
        st.markdown(f"Start: `{sel_win.start_time.strftime('%H:%M:%S')}`")
        st.markdown(f"End:   `{sel_win.end_time.strftime('%H:%M:%S')}`")
        st.markdown(f"Samples: {len(sel_win.data)}")
        st.markdown("---")
        st.markdown(f"Grid freq OK: {'✅' if sel_val.freq_ok else '❌'}")
        st.markdown(f"Range OK:     {'✅' if sel_val.range_ok else '❌'}")
        st.markdown(f"Phase OK:     {'✅' if sel_val.phase_ok else '❌'}")
        st.markdown(f"Energy OK:    {'✅' if sel_val.energy_ok else '❌'}")
        st.markdown(f"Avg freq: **{sel_val.avg_freq:.2f} Hz**")
        st.markdown(f"Imbalance CV: **{sel_val.phase_imbalance:.3f}**")
        if not sel_val.is_valid:
            st.warning(sel_val.failure_reason)

    # Final verdict
    st.markdown("---")
    st.subheader("Final Verdict")
    if dataset_val.is_dataset_valid:
        st.success("✅  **VALID DATA** — sufficient windows pass all quality checks.")
    else:
        st.error(
            f"❌  **INVALID DATA** — only {dataset_val.valid_pct:.1f}% of windows are valid "
            f"(need {ss('val_thr')['valid_window_fraction']*100:.0f}%)"
        )

    st.markdown("---")
    if dataset_val.is_dataset_valid:
        if st.button("▶️  Proceed to RAG Analysis", type="primary"):
            st.session_state.current_page = "rag"
            st.rerun()
    else:
        st.button("▶️  Proceed to RAG Analysis", disabled=True,
                  help="Data must pass validation first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RAG State Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _page_rag_conveyer(df: pd.DataFrame, sc: dict, sl: dict, meta: dict):
    """
    RAG visualisation for MV Conveyer using per-sample classification.

    The conveyer cycles every ~15s (11s RUNNING + 4s IDLE). Any analysis
    window longer than ~3s averages across both states, making every window
    look GREEN. We classify each 1-second sample directly from its I_avg.
    Playback scrubs through samples (1 Hz) so the R/A/G circles animate live.
    """
    thr = ss("rag_thr")

    # Per-sample classification (all samples)
    sample_states = _per_sample_states(df, thr)
    n             = len(sample_states)
    states_arr    = np.array(sample_states)

    # ── Playback controls ──────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 2, 2])
    with ctrl1:
        if st.button("⏮ Reset", key="mvc_reset"):
            ss_set("play_idx", 0)
            ss_set("playing",  False)
            st.rerun()
        play_label = "⏸ Pause" if ss("playing") else "▶ Play"
        if st.button(play_label, key="mvc_play"):
            ss_set("playing", not ss("playing"))
            st.rerun()
    with ctrl2:
        speed = st.slider("Speed (samples/sec)", 1, 100, 10, key="mvc_rag_speed")
    with ctrl3:
        manual_idx = st.slider(
            "Jump to sample", 0, n - 1,
            min(ss("play_idx"), n - 1), key="mvc_rag_jump",
        )
        if manual_idx != ss("play_idx") and not ss("playing"):
            ss_set("play_idx", manual_idx)

    idx      = max(0, min(ss("play_idx"), n - 1))
    cur_state = sample_states[idx]
    cur_iavg  = float(df["i_avg"].iloc[idx])
    cur_ts    = df["timestamp"].iloc[idx]

    # ── State circles + stats + thresholds ────────────────────────────
    col_lights, col_stats, col_info = st.columns([2, 2, 2])

    with col_lights:
        st.markdown("### Current State")
        html = ""
        for state in (GREEN, AMBER, RED):
            html += _circle_html(state, state == cur_state, sc, sl)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;font-size:1.2rem;margin-top:8px;">'
            f'<b style="color:{sc.get(cur_state,"#fff")};">{sl.get(cur_state, cur_state)}</b>'
            f' — I_avg = {cur_iavg:.3f} A</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Sample {idx + 1} / {n}  ·  {cur_ts.strftime('%H:%M:%S')}")

    with col_stats:
        st.markdown("### Per-Sample State Counts")
        counts = {RED:   int((states_arr == RED).sum()),
                  AMBER: int((states_arr == AMBER).sum()),
                  GREEN: int((states_arr == GREEN).sum())}
        tot = max(n, 1)
        st.metric(f"🔴 {sl.get(RED,'HALT')}",      f"{counts[RED]:,}",   f"{counts[RED]/tot*100:.1f}%")
        st.metric(f"🟡 {sl.get(AMBER,'IDLE')}",    f"{counts[AMBER]:,}", f"{counts[AMBER]/tot*100:.1f}%")
        st.metric(f"🟢 {sl.get(GREEN,'RUNNING')}",  f"{counts[GREEN]:,}", f"{counts[GREEN]/tot*100:.1f}%")

    with col_info:
        st.markdown("### Thresholds in Use")
        st.markdown(f"- **HALT (RED):** I_avg < **{thr['red_max_rms']:.2f} A**")
        st.markdown(f"- **IDLE (AMBER):** {thr['red_max_rms']:.2f} – {thr['green_min_rms']:.2f} A")
        st.markdown(f"- **RUNNING (GREEN):** I_avg ≥ **{thr['green_min_rms']:.2f} A**")
        st.markdown(f"- **Total samples:** {n:,}")
        st.caption("Classification is per 1-second sample — no window averaging.")

    # ── Live cycle view (±60 s centred on current sample) ─────────────
    st.markdown("---")
    st.markdown("### Live View — Cycle Detail (±60 s around current position)")

    ctx_sec  = 60
    t_start  = cur_ts - pd.Timedelta(seconds=ctx_sec)
    t_end    = cur_ts + pd.Timedelta(seconds=ctx_sec)
    view_mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
    df_view   = df[view_mask]
    st_view   = states_arr[df_view.index.to_numpy()]
    t_np      = df_view["timestamp"].to_numpy()
    iavg_np   = df_view["i_avg"].to_numpy()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Phase Currents  I1 / I2 / I3",
            "I_avg — coloured by state",
            "State Indicator  (RUNNING=1 / IDLE=0)",
        ),
        row_heights=[0.4, 0.35, 0.25],
        vertical_spacing=0.08,
    )

    # Row 1: Phase currents
    for col_name, color in [("i1", PHASE_COLORS["i1"]),
                             ("i2", PHASE_COLORS["i2"]),
                             ("i3", PHASE_COLORS["i3"])]:
        fig.add_trace(go.Scatter(
            x=t_np, y=df_view[col_name].to_numpy(), name=col_name.upper(),
            line=dict(color=color, width=1.5),
        ), row=1, col=1)

    # Row 2: I_avg coloured per-sample state
    for state, color in sc.items():
        mask = st_view == state
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=t_np[mask], y=iavg_np[mask],
            mode="markers", name=sl.get(state, state),
            marker=dict(color=color, size=5, opacity=0.9),
            hovertemplate=f"{sl.get(state, state)}: %{{y:.3f}} A<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=thr["green_min_rms"], line_dash="dash", line_color=sc[GREEN],
                  annotation_text=f"RUN ≥ {thr['green_min_rms']} A",
                  annotation_position="top left", row=2, col=1)
    fig.add_hline(y=thr["red_max_rms"], line_dash="dash", line_color=sc[RED],
                  annotation_text=f"HALT < {thr['red_max_rms']} A",
                  annotation_position="bottom left", row=2, col=1)

    # Row 3: Binary step indicator
    state_bin = np.where(st_view == GREEN, 1.0, 0.0)
    fig.add_trace(go.Scatter(
        x=t_np, y=state_bin,
        mode="lines", name="State",
        line=dict(color="#aaaaaa", width=1, shape="hv"),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.15)",
        showlegend=False,
    ), row=3, col=1)

    # Vertical cursor at current playback position (across all rows)
    fig.add_vline(x=cur_ts, line_dash="dot", line_color="white", line_width=2.0)

    fig.update_layout(
        height=580, template="plotly_dark", showlegend=True,
        margin=dict(l=50, r=20, t=60, b=30),
        legend=dict(orientation="h", y=-0.06),
    )
    fig.update_yaxes(title_text="Current (A)", row=1, col=1)
    fig.update_yaxes(title_text="I_avg (A)",   row=2, col=1)
    fig.update_yaxes(title_text="State",       row=3, col=1,
                     tickvals=[0, 1], ticktext=["IDLE/HALT", "RUNNING"])
    st.plotly_chart(fig, width='stretch')

    # ── Full-dataset state overview (with current-position marker) ─────
    st.markdown("---")
    st.markdown("### Full Dataset — State Overview")

    step      = max(1, n // 2000)
    t_ds      = df["timestamp"].iloc[::step].to_numpy()
    iavg_ds   = df["i_avg"].to_numpy()[::step]
    states_ds = states_arr[::step]

    fig_ov = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("I_avg (downsampled overview)",
                                           "State Indicator over full session"),
                           vertical_spacing=0.12)

    fig_ov.add_trace(go.Scatter(
        x=t_ds, y=iavg_ds, name="I_avg",
        line=dict(color=PHASE_COLORS["i_avg"], width=1),
    ), row=1, col=1)
    fig_ov.add_hline(y=thr["green_min_rms"], line_dash="dash", line_color=sc[GREEN], row=1, col=1)
    fig_ov.add_hline(y=thr["red_max_rms"],  line_dash="dash", line_color=sc[RED],   row=1, col=1)

    state_bin_ds = np.where(states_ds == GREEN, 1.0, 0.0)
    fig_ov.add_trace(go.Scatter(
        x=t_ds, y=state_bin_ds,
        mode="lines", name="Running=1 / Idle=0",
        line=dict(color=sc[GREEN], width=1, shape="hv"),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.20)",
    ), row=2, col=1)

    # Current position marker on overview
    fig_ov.add_vline(x=cur_ts, line_dash="dot", line_color="white", line_width=2.0)

    fig_ov.update_layout(
        height=380, template="plotly_dark", showlegend=False,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    fig_ov.update_yaxes(title_text="I_avg (A)", row=1, col=1)
    fig_ov.update_yaxes(title_text="State", row=2, col=1,
                        tickvals=[0, 1], ticktext=["IDLE", "RUN"])
    st.plotly_chart(fig_ov, width='stretch')

    # ── Playback advance ───────────────────────────────────────────────
    if ss("playing"):
        if idx < n - 1:
            ss_set("play_idx", idx + 1)
            time.sleep(max(0.02, 1.0 / speed))
            st.rerun()
        else:
            ss_set("playing", False)
            st.success("✅ Playback complete!")


def _per_sample_states(df: pd.DataFrame, thresholds: dict) -> list[str]:
    """
    Classify every sample directly from its I_avg value.
    Correct for rapid-cycling machines (conveyer) where window averaging
    smears short IDLE gaps into GREEN.
    """
    halt_max  = thresholds.get("red_max_rms",   0.10)
    green_min = thresholds.get("green_min_rms",  0.35)
    iavg      = df["i_avg"].to_numpy()
    states    = []
    for v in iavg:
        if v < halt_max:
            states.append(RED)
        elif v >= green_min:
            states.append(GREEN)
        else:
            states.append(AMBER)
    return states


def page_rag():
    m    = st.session_state.current_machine
    meta = MACHINE_META[m]
    sc   = meta["state_colors"]
    sl   = meta["state_labels"]

    _machine_header(m)
    st.title("🚦 RAG State Visualisation")

    # ── Conveyer: per-sample classification (no window pipeline needed) ──
    # The conveyer cycle is ~15s (11s ON + 4s OFF). Any window > ~3s
    # averages across ON and OFF, pulling RMS above the GREEN threshold
    # and hiding all IDLE periods. We classify each sample directly.
    if m == "conveyer":
        df_cyc = ss("df_cyclic")
        if df_cyc is None:
            st.warning(
                "No cyclic data loaded. Go to **Data Validation** and upload "
                "both CSV files (make sure one filename contains 'cyclic')."
            )
            if st.button("← Back to Data Validation"):
                st.session_state.current_page = "validation"
                st.rerun()
            return
        _page_rag_conveyer(df_cyc, sc, sl, meta)
        return

    # ── Furnace: window-based pipeline required ────────────────────────
    if ss("smoothed_states") is None:
        st.warning("No processed data found. Please complete Page 1 first.")
        if st.button("← Back to Data Validation"):
            st.session_state.current_page = "validation"
            st.rerun()
        return

    df = ss("df")
    windows_proc    = ss("windows_proc")
    smoothed_states = ss("smoothed_states")
    raw_states      = ss("raw_states")
    features_cache  = ss("features_cache")
    val_results     = ss("val_results")
    fft_cache       = ss("fft_cache")
    win_centres     = ss("win_centres")
    n_windows       = len(windows_proc)

    # ── Playback controls ──────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 2, 2])
    with ctrl1:
        if st.button("⏮ Reset"):
            ss_set("play_idx", 0)
            ss_set("playing",  False)
            st.rerun()
        play_label = "⏸ Pause" if ss("playing") else "▶ Play"
        if st.button(play_label):
            ss_set("playing", not ss("playing"))
            st.rerun()
    with ctrl2:
        speed = st.slider("Speed (steps/sec)", 1, 50, 10)
    with ctrl3:
        # Exclude trailing partial windows (too few samples for FFT/features)
        _min_samp = 8
        _last_full = n_windows - 1
        for _j in range(n_windows - 1, -1, -1):
            if len(windows_proc[_j].data) >= _min_samp:
                _last_full = _j
                break
        manual_idx = st.slider("Jump to window", 0, _last_full, min(ss("play_idx"), _last_full))
        if manual_idx != ss("play_idx") and not ss("playing"):
            ss_set("play_idx", manual_idx)

    idx = max(0, min(ss("play_idx"), _last_full))
    cur_state    = smoothed_states[idx]
    cur_features = features_cache[idx]
    cur_val      = val_results[idx]
    cur_window   = windows_proc[idx]
    cur_fft      = fft_cache[idx]

    # ── Row 1: State circles | Features | Window info ──────────────────
    col_lights, col_feat, col_info = st.columns([2, 2, 2])

    with col_lights:
        st.markdown("### Current State")
        html = ""
        for state in (GREEN, AMBER, RED):
            html += _circle_html(state, state == cur_state, sc, sl)
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;font-size:1.2rem;margin-top:8px;">'
            f'<b style="color:{sc.get(cur_state,"#fff")};">{sl.get(cur_state, cur_state)}</b>'
            f'</div>', unsafe_allow_html=True,
        )

    with col_feat:
        st.markdown("### Feature Panel")
        fd = meta["feat_dict"](cur_features)

        def _mb(label, value, unit=""):
            return (f'<div class="metric-box">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}{unit}</div>'
                    f'</div>')

        st.markdown(
            _mb("RMS I_avg",       f"{fd['rms_i_avg']:.3f}", " A") +
            _mb("RMS I1",          f"{fd['rms_i1']:.3f}",    " A") +
            _mb("RMS I2",          f"{fd['rms_i2']:.3f}",    " A") +
            _mb("RMS I3",          f"{fd['rms_i3']:.3f}",    " A"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _mb("Phase Imbalance", f"{fd['phase_imbalance']:.3f}") +
            _mb("THD (proxy)",     f"{fd['thd']:.3f}") +
            _mb("Variance I_avg",  f"{fd['variance_i_avg']:.5f}", " A²"),
            unsafe_allow_html=True,
        )

    with col_info:
        st.markdown("### Window Status")
        badge = "✅  VALID" if cur_val.is_valid else "❌  INVALID"
        st.markdown(f'<div style="font-size:2rem;text-align:center;">{badge}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Window:** {idx + 1} / {n_windows}")
        st.markdown(f"**Start:** {cur_window.start_time.strftime('%H:%M:%S')}")
        st.markdown(f"**End:**   {cur_window.end_time.strftime('%H:%M:%S')}")
        st.markdown(f"**Samples:** {len(cur_window.data)}")
        st.markdown(f"**Raw → Smoothed:** {raw_states[idx]} → **{cur_state}**")
        if not cur_val.is_valid:
            st.caption(f"⚠️ {cur_val.failure_reason}")
        fft_ref = cur_fft.get("i_avg") or cur_fft.get("i1")
        if fft_ref:
            st.markdown(f"**Cycle freq:** {fft_ref.fundamental_freq * 1000:.1f} mHz")

    # ── Row 2: Live time-series + FFT ─────────────────────────────────
    st.markdown("---")
    col_ts, col_fft_panel = st.columns([3, 2])
    with col_ts:
        st.markdown("### Live Time-Series (current window highlighted)")
        ctx_sec = ss("window_size_sec") * 3
        df_zoom = df[(df["timestamp"] >= cur_window.start_time - pd.Timedelta(seconds=ctx_sec)) &
                     (df["timestamp"] <= cur_window.end_time   + pd.Timedelta(seconds=ctx_sec))]
        st.plotly_chart(
            fig_time_series(df_zoom, cur_window.start_time, cur_window.end_time),
            width='stretch',
        )
    with col_fft_panel:
        st.markdown("### Operational Frequency Spectrum")
        st.plotly_chart(
            fig_fft(cur_fft, sample_rate=meta["sample_rate"](df)),
            width='stretch',
        )

    # ── Row 3: State timeline + summary ───────────────────────────────
    st.markdown("---")
    st.markdown("### State Timeline")
    st.plotly_chart(
        fig_state_timeline(smoothed_states, win_centres, sc, sl, current_idx=idx),
        width='stretch',
    )

    counts = {RED: 0, AMBER: 0, GREEN: 0}
    for s in smoothed_states:
        counts[s] += 1
    tot = max(1, len(smoothed_states))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"🔴 {sl.get(RED,  'Idle')} windows",     f"{counts[RED]:,}",   f"{counts[RED]/tot*100:.1f}%")
    c2.metric(f"🟡 {sl.get(AMBER,'No Load')} windows",  f"{counts[AMBER]:,}", f"{counts[AMBER]/tot*100:.1f}%")
    c3.metric(f"🟢 {sl.get(GREEN,'Load')} windows",     f"{counts[GREEN]:,}", f"{counts[GREEN]/tot*100:.1f}%")

    # ── Playback advance ───────────────────────────────────────────────
    if ss("playing"):
        if idx < n_windows - 1:
            ss_set("play_idx", idx + 1)
            time.sleep(max(0.02, 1.0 / speed))
            st.rerun()
        else:
            ss_set("playing", False)
            st.success("✅ Playback complete!")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def page_ml():
    """ML Diagnostics page — routes to furnace or conveyer ML based on machine."""
    m = st.session_state.current_machine

    if m == "conveyer":
        _machine_header("conveyer")
        df = ss("df")
        if df is None:
            st.warning(
                "No data loaded yet. Go to **Data Validation** first, "
                "upload both conveyer CSV files, then return here."
            )
            if st.button("← Go to Data Validation"):
                st.session_state.current_page = "validation"
                st.rerun()
            return
        df_cyclic = ss("df_cyclic")
        # Pass combined df (for lookback context) + cyclic df (labels/training target)
        render_mvc_ml_page(df, cyclic_df=df_cyclic)
    else:
        _machine_header("furnace")
        df = ss("df")
        if df is None:
            st.warning(
                "No data loaded yet. Go to **Data Validation** first, "
                "upload the furnace CSV, then return here."
            )
            if st.button("← Go to Data Validation"):
                st.session_state.current_page = "validation"
                st.rerun()
            return
        render_ml_page(df)


def main():
    render_sidebar()
    if st.session_state.current_page == "validation":
        page_validation()
    elif st.session_state.current_page == "ml":
        page_ml()
    else:
        page_rag()


if __name__ == "__main__":
    main()
