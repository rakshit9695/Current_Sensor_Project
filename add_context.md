# add_context.md — EPSON Robot (`/robo`) Pipeline Addition

## Changelog & Technical Documentation

---

## 1. Executive Summary

The Current Sensor Project has been extended from a **two-machine** platform (Wave Soldering Furnace + MV Conveyer) to a **three-machine** platform with the addition of the **EPSON Robot** pipeline. The robot pipeline lives in the `robo/` package and follows the identical module structure established by `wave_soldering/` and `mv_conveyer/`, but is adapted throughout for single-phase current data.

### What changed

| Area | Before | After |
|------|--------|-------|
| Machine count | 2 (furnace, conveyer) | 3 (furnace, conveyer, **robo**) |
| `app.py` sidebar | 2-option radio selector | 3-option radio selector |
| New package | — | `robo/` (8 core modules + 6 ML modules) |
| Data committed | `data/` had furnace + conveyer CSVs | Added `data/robo/` CSV (14,861 rows) |
| `.gitignore` | Did not exist | Added — ignores `__pycache__/`, `*.pyc`, `*.pyo` |

### Key architectural difference

The EPSON Robot is a **single-phase** device. Only phase I1 carries current; I2 and I3 are always 0. The sensor also provides an `I_Avg` column that is a **separate sensor measurement** (not a computed `(I1+I2+I3)/3`). This single-phase nature propagates through every module in the pipeline — FFT skips I2/I3, feature engineering sets phase imbalance to 0, the iTransformer uses `n_vars=2` instead of 4, etc.

---

## 2. Source Data Characteristics

**CSV file:** `data/robo/selec_em4m_datalog_robo_20260416.csv`

| Property | Value |
|----------|-------|
| Rows | 14,861 |
| Sampling rate | 1 Hz (one reading per second) |
| Duration | ~4.1 hours of continuous operation |
| Columns | `Timestamp`, `I1`, `I2`, `I3`, `I_Avg`, `Frequency` |

### Column semantics

| Column | Description | Range |
|--------|-------------|-------|
| `Timestamp` | ISO 8601 datetime — proper datetimes, no wrapping issues (unlike furnace MM:SS.f format) | `2026-04-16 ...` |
| `I1` | **RMS current** — the only active phase | 0.22 – 0.42 A |
| `I2` | Always `0.0` — no current on phase 2 | 0.0 |
| `I3` | Always `0.0` — no current on phase 3 | 0.0 |
| `I_Avg` | **Sensor-reported average** (NOT `(I1+I2+I3)/3`) — a separate non-RMS measurement from the Selec EM4M meter | 0.073 – 0.141 A |
| `Frequency` | Grid frequency | 49.9 – 50.2 Hz (mostly 50.0) |

> **Critical design decision:** `I_Avg` is used as-read from the CSV. It is **not recomputed** as an arithmetic mean of the three phases. Numerically, `I_Avg ≈ I1 / 3`, but it represents a fundamentally different measurement (sensor-internal average vs. RMS).

### Comparison with existing machines

| Property | Furnace | MV Conveyer | EPSON Robot |
|----------|---------|-------------|-------------|
| Phases active | 3 (I1, I2, I3 all non-zero) | 3 (I1, I2, I3 all non-zero) | **1 (I1 only; I2=I3=0)** |
| I_Avg meaning | `(I1+I2+I3)/3` recomputed | `(I1+I2+I3)/3` recomputed | **Sensor-reported** (kept as-is) |
| Timestamp format | `MM:SS.f` (wraps, needs reconstruction) | ISO datetime | ISO datetime |
| CSV files per session | 1 | **2** (stop_and_run + cyclic) | 1 |
| Current range | ~5–25 A | ~0.5–8 A | **0.22–0.42 A** (very low draw) |
| Typical sample rate | ~1 Hz | ~200 Hz | ~1 Hz |
| Operational pattern | Gradual thermal changes | Rapid ON/OFF cycling (~15s periods) | **Gradual state transitions** |

---

## 3. Data Analysis & State Discovery

Since the user had no prior knowledge of the robot's operational states, we performed **unsupervised analysis** to determine RAG thresholds from the data itself.

### 3.1 K-Means Clustering

Ran K-means with K=2 and K=3 on the feature space `[I1, I_Avg]`:

**K=3 results:**

| Cluster | I1 Range | I1 Mean | I_Avg Range | Samples |
|---------|----------|---------|-------------|---------|
| 0 | 0.22 – 0.34 A | 0.32 A | < 0.085 A | ~69 |
| 1 | 0.34 – 0.38 A | 0.36 A | 0.085 – 0.115 A | ~5,200 |
| 2 | 0.38 – 0.42 A | 0.39 A | ≥ 0.115 A | ~9,500 |

### 3.2 Histogram Gap Analysis

The I1 histogram revealed a **natural gap between 0.24 A and 0.30 A**, corresponding to the STANDBY/IDLE boundary. This gap validates the cluster-derived thresholds.

### 3.3 Temporal Analysis

- 30-minute block analysis showed consistent operational behaviour throughout the 4.1-hour session
- 69 low-current samples concentrated at the end of the recording (likely robot shutdown)
- No rapid cycling detected — transitions are gradual (suitable for window-based RAG, not per-sample)

### 3.4 Final RAG Thresholds

Based on the analysis above, thresholds were set on **I_Avg** (the more stable discriminator):

| RAG State | Colour | Label | I_Avg Condition | I1 Equivalent |
|-----------|--------|-------|-----------------|---------------|
| RED | `#e74c3c` | **STANDBY** | I_Avg < 0.085 A | I1 < ~0.25 A |
| AMBER | `#f39c12` | **IDLE** | 0.085 ≤ I_Avg < 0.115 A | I1 ~0.30 – 0.35 A |
| GREEN | `#27ae60` | **ACTIVE** | I_Avg ≥ 0.115 A | I1 ≥ ~0.35 A |

> **Note:** Unlike the furnace (HALT/IDLE/RUN) and conveyer (OFF/IDLE/ON), the robot uses **STANDBY/IDLE/ACTIVE** terminology to reflect its operational semantics.

---

## 4. New Package Structure — `robo/`

```
robo/
├── __init__.py                  # Package marker
├── data_loader.py               # CSV loading + column normalisation
├── windowing.py                 # Time-based sliding window generator
├── fft_analysis.py              # FFT on I1 and I_Avg (skips I2, I3)
├── feature_engineering.py       # Time + frequency domain features
├── validation.py                # Per-window + dataset-level quality checks
├── rag_classifier.py            # Rule-based RED/AMBER/GREEN classifier
├── state_manager.py             # Anti-flicker hysteresis state machine
└── ml/
    ├── __init__.py              # Sub-package marker
    ├── dataset.py               # PyTorch Dataset (n_vars=2: I1, I_Avg)
    ├── itransformer.py          # iTransformer architecture (n_vars=2)
    ├── label_generator.py       # Auto-labeling: states → cycles → job clusters
    ├── predictor.py             # Inference wrapper (sliding-window prediction)
    ├── trainer.py               # Training loop + checkpoint management
    └── ui.py                    # Streamlit ML page (4 tabs)
```

This mirrors the existing `wave_soldering/` and `mv_conveyer/` structures exactly.

---

## 5. Module-by-Module Technical Details

### 5.1 `robo/data_loader.py` (116 lines)

**Purpose:** Load a single EPSON Robot CSV file and produce a standardised DataFrame.

**Key functions:**
- `load_data(source) → pd.DataFrame` — Main entry point. Accepts file path, `Path`, or file-like (Streamlit `UploadedFile`).
- `get_sample_rate(df) → float` — Estimates Hz from median timestamp delta.

**Single-phase adaptations:**
- Column `I_Avg` from CSV is **preserved as-is** (mapped to internal column `i_avg`)
- I2 and I3 are loaded but will always be 0.0
- Timestamps are parsed as ISO 8601 (no reconstruction needed)
- Fallback: if I_Avg column is missing, falls back to `(I1+I2+I3)/3`

**Output columns:** `timestamp`, `timestamp_str`, `i1`, `i2`, `i3`, `i_avg`, `frequency`

**Difference from furnace/conveyer:** Single file input (no multi-file merge), I_Avg not recomputed.

---

### 5.2 `robo/windowing.py` (97 lines)

**Purpose:** Generate time-based sliding windows over the DataFrame.

**Key function:**
- `generate_windows(df, window_size_sec=30.0, overlap=0.50, sample_rate=1.0) → list[Window]`

**Window dataclass:** `Window(index, start_time, end_time, data, start_row, end_row)`

**Configuration:**
- Default window size: **30 seconds** (vs. furnace's 30s and conveyer's 3s)
- Default overlap: 50%
- Minimum rows per window: `max(2, window_size_sec × sample_rate × 0.5)`

**For the robot CSV (14,861 rows at 1 Hz):** This produces ~990 windows with default settings.

**Utility:** `get_window_timestamps(windows) → np.ndarray` — Centre timestamps for timeline plots.

---

### 5.3 `robo/fft_analysis.py` (145 lines)

**Purpose:** Compute FFT on each window's current signals to extract operational-frequency features.

**Important caveat (documented in module docstring):**
> The CSV contains pre-processed current readings at ~1 Hz — NOT raw AC waveforms. At 1 Hz the Nyquist limit is 0.5 Hz, so true 50 Hz power-line harmonics cannot be observed. What the FFT reveals is the **operational cycling frequency** of the robot (e.g., pick-and-place rhythm).

**Key functions:**
- `compute_fft(signal, sample_rate) → FFTResult` — Core FFT with Hann window, mean removal
- `compute_window_fft(window_data, sample_rate) → dict[str, FFTResult]` — Runs FFT on **I1 and I_Avg only** (I2/I3 skipped because they're always 0)
- `batch_compute_fft(windows, sample_rate) → list[dict[str, FFTResult]]` — Batch version

**FFTResult fields:** `freqs`, `magnitudes`, `fundamental_freq`, `fundamental_mag`, `harmonic2_mag`, `harmonic3_mag`, `total_energy`, `high_freq_energy`, `thd`

**Single-phase adaptation:** Only iterates over `("i1", "i_avg")` instead of `("i1", "i2", "i3", "i_avg")`.

---

### 5.4 `robo/feature_engineering.py` (157 lines)

**Purpose:** Compute comprehensive time-domain and frequency-domain features for each window.

**Key function:**
- `compute_features(window_df, fft_results) → WindowFeatures`
- `features_to_dict(f: WindowFeatures) → dict` — Flatten for tabular display

**WindowFeatures dataclass (20 fields):**

| Category | Fields | Notes |
|----------|--------|-------|
| Time-domain RMS | `rms_i1`, `rms_i2`, `rms_i3`, `rms_i_avg` | I2/I3 always 0 |
| Time-domain variance | `variance_i1`, `variance_i2`, `variance_i3`, `variance_i_avg` | I2/I3 always 0 |
| Peak-to-peak | `peak_to_peak_i1`, `peak_to_peak_i2`, `peak_to_peak_i3` | I2/I3 always 0 |
| Phase balance | `phase_imbalance` | **Always 0.0** for single-phase |
| Frequency domain | `fundamental_freq`, `fundamental_energy`, `harmonic2_energy`, `harmonic3_energy`, `total_energy`, `high_freq_energy` | From I_Avg FFT |
| Derived | `thd`, `high_freq_ratio` | THD = harmonic distortion proxy |

**Single-phase adaptation:**
- `phase_imbalance` is **hardcoded to 0.0** (meaningless for single-phase)
- FFT reference signal is I_Avg (primary) with I1 as fallback
- I2/I3 fields are retained for interface compatibility but always 0

---

### 5.5 `robo/validation.py` (169 lines)

**Purpose:** Data quality checks at both per-window and dataset levels.

**Key functions:**
- `validate_window(window_df, thresholds) → WindowValidation`
- `validate_dataset(windows, thresholds) → DatasetValidation`

**Validation checks per window:**

| Check | What it validates | Single-phase note |
|-------|-------------------|-------------------|
| `freq_ok` | Grid frequency within ±0.5 Hz of 50 Hz, ≥90% of samples | Same as other machines |
| `range_ok` | All current values finite, non-negative, at least one > 0.01 A | Same |
| `phase_ok` | Phase imbalance CV ≤ threshold | **Always passes** — `max_phase_imbalance=2.0` |
| `energy_ok` | Mean I_Avg ≥ `min_current_floor` | Floor set to **0.01 A** (robot baseline ~0.07 A) |

**Default thresholds (relaxed for single-phase):**

```python
DEFAULT_THRESHOLDS = dict(
    freq_target_hz        = 50.0,
    freq_tolerance_hz     = 0.5,
    freq_valid_fraction   = 0.90,
    min_current_floor     = 0.01,   # Very low — robot I_Avg baseline ~0.07 A
    max_phase_imbalance   = 2.00,   # Meaningless for single-phase; never fails
    valid_window_fraction = 0.80,
)
```

**Comparison:** Furnace/conveyer use `min_current_floor ≈ 0.1–0.5 A` and `max_phase_imbalance ≈ 0.15–0.30`. The robot's thresholds are deliberately relaxed because it's single-phase with very low current draw.

---

### 5.6 `robo/rag_classifier.py` (140 lines)

**Purpose:** Rule-based classification of each window into RED/AMBER/GREEN states.

**Key function:**
- `classify(features: WindowFeatures, thresholds) → ClassificationResult`

**Classification logic (decision order):**

1. **RED (STANDBY):** `rms_i_avg ≤ 0.085 A`
2. **GREEN (ACTIVE):** `rms_i_avg ≥ 0.115 A` AND secondary quality indicator (THD or variance) ≥ 0.3
3. **GREEN (ACTIVE, strong):** `rms_i_avg ≥ 0.115 × 1.10 A` (bypasses quality check)
4. **AMBER (IDLE):** Everything else (the intermediate band)

**Scoring:** Each classification returns normalised sub-scores for `rms`, `variance`, and `thd` (all in 0–1 range), plus a confidence score and human-readable reason string.

**Default thresholds:**

```python
DEFAULT_THRESHOLDS = dict(
    red_max_rms        = 0.085,    # A — I_Avg below this → STANDBY
    green_min_rms      = 0.115,    # A — I_Avg above this → candidate ACTIVE
    green_min_thd      = 0.05,     # — — operational THD threshold
    green_min_variance = 0.0005,   # A² — within-window variance threshold
)
```

**Single-phase adaptation:**
- **No `green_min_imbalance` threshold** — phase imbalance is meaningless for single-phase
- This required a conditional change in `app.py`'s sidebar (only show imbalance slider when the machine's RAG defaults include it)

**State labels and colours:**

```python
STATE_COLORS = {RED: "#e74c3c", AMBER: "#f39c12", GREEN: "#27ae60"}
STATE_LABELS = {RED: "STANDBY", AMBER: "IDLE", GREEN: "ACTIVE"}
```

---

### 5.7 `robo/state_manager.py` (132 lines)

**Purpose:** Anti-flicker hysteresis state machine to smooth raw per-window classifications.

**Class:** `StateManager(min_consecutive=3, initial_state=RED)`

**How it works:**
- Tracks a `_candidate` state and a `_streak` counter
- A state change is only accepted when the candidate has appeared in **N consecutive windows** (default N=3)
- This prevents rapid toggling between IDLE and ACTIVE when the signal hovers near 0.115 A

**Key methods:**
- `update(new_state, window_index, confidence) → str` — Feed one window, get smoothed state
- `run_batch(raw_states, confidences) → list[str]` — Process entire timeline at once
- `reset(initial_state)` — Clear internal state

**Utility function:** `build_state_timeline(smoothed_states, window_centres) → list[dict]` — Run-length encodes the timeline for state bar charts.

---

### 5.8 `robo/ml/dataset.py` (130 lines)

**Purpose:** PyTorch Dataset for the iTransformer, with z-score normalisation.

**Feature columns:** `["i1", "i_avg"]` — only 2 variables (vs. 4 for furnace/conveyer which include I2, I3).

**Classes:**
- `NormStats` — Stores per-channel mean/std from training data; provides `normalise(x)` and serialisation
- `RoboDataset(Dataset)` — Sliding look-back window dataset
  - Input tensor shape: `(look_back, 2)` — [I1, I_Avg] normalised
  - Target: `(state_label, cycle_pos, job_type)` at the window's last position

**Key function:** `build_datasets(df_labeled, look_back=30) → (train_ds, val_ds, test_ds, norm_stats)`
- Uses temporal split: 70% train / 15% val / 15% test (no shuffle — time series)

---

### 5.9 `robo/ml/itransformer.py` (241 lines)

**Purpose:** iTransformer model architecture adapted for 2-variate single-phase input.

**Architecture:**

```
Input: (B, T, 2)  — look-back window of [I1, I_Avg]
    ↓ transpose → (B, 2, T)
    ↓ VariateEmbedding → (B, 2, d_model)    # each variate gets its own embedding
    ↓ N × iTransformerEncoderLayer           # self-attention over variate axis
    ↓ LayerNorm
    ↓ flatten → (B, 2 × d_model)
    ↓ SharedProjection → (B, d_model)
    ├── state_head → (B, 3)      # STANDBY/IDLE/ACTIVE logits
    ├── cycle_pos_head → (B,)    # 0–1 sigmoid regression
    └── job_head → (B, K)        # job-type logits
```

**Default hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `seq_len` | 120 | Look-back window (configurable) |
| `n_vars` | **2** | I1, I_Avg (hardcoded in `build_model()`) |
| `d_model` | 64 | Embedding dimension |
| `n_heads` | 4 | Multi-head attention heads |
| `n_layers` | 3 | Encoder depth |
| `d_ff` | 128 | Feed-forward inner dimension |
| `dropout` | 0.1 | |
| `n_states` | 3 | STANDBY / IDLE / ACTIVE |
| `n_job_types` | 3 | Configurable via label generator |

**Key difference from furnace/conveyer:** `n_vars=2` instead of 4. The shared projection layer is `2 × d_model → d_model` instead of `4 × d_model → 2 × d_model`, resulting in a significantly smaller model.

---

### 5.10 `robo/ml/label_generator.py` (308 lines)

**Purpose:** Fully automatic label generation pipeline — no manual annotation required.

**Pipeline stages:**

```
Raw DataFrame
    ↓ label_states()         — Per-sample state from I_Avg thresholds
    ↓ smooth_labels()        — RLE smoothing to remove glitches (min_run=3)
    ↓ detect_cycles()        — Identify contiguous ACTIVE segments as cycles
    ↓ assign_cycle_labels()  — Add cycle_id + cycle_pos (0.0→1.0) columns
    ↓ compute_cycle_features() — Extract per-cycle feature vectors
    ↓ cluster_job_types()    — K-means on cycle features → job_type labels
```

**Thresholds (matching RAG classifier):**
- `DEFAULT_STANDBY_MAX = 0.085 A` — I_Avg below this → state 0 (STANDBY)
- `DEFAULT_ACTIVE_MIN = 0.115 A` — I_Avg at/above this → state 2 (ACTIVE)
- Between → state 1 (IDLE)

**Per-cycle features used for K-means clustering:**

| Feature | Description |
|---------|-------------|
| `log_duration` | log10(segment length + 1) |
| `mean_iavg` | Mean I_Avg during ACTIVE segment |
| `std_iavg` | Std deviation of I_Avg |
| `mean_i1` | Mean I1 during segment |

**Single-phase adaptation:** Uses `mean_i1` instead of `mean_i23` (which furnace/conveyer use since they have active I2/I3 phases). Does not include phase imbalance in cycle features.

**Output:** Adds columns `state_label`, `cycle_id`, `cycle_pos`, `job_type` to the DataFrame, plus a `meta` dict with pipeline parameters, cycle statistics, and serialised scaler/kmeans objects.

---

### 5.11 `robo/ml/predictor.py` (191 lines)

**Purpose:** Inference wrapper that slides the trained model over a DataFrame.

**Class:** `RoboPredictor`
- `from_checkpoint(path) → RoboPredictor` — Load from saved checkpoint
- `predict(df, batch_size=256) → pd.DataFrame` — Sliding-window inference
- `evaluate_on_labeled(df_labeled) → dict` — Classification metrics on labeled data

**Columns added by `predict()`:**

| Column | Type | Description |
|--------|------|-------------|
| `ml_state` | int | 0=STANDBY, 1=IDLE, 2=ACTIVE |
| `ml_state_name` | str | Human-readable state name |
| `ml_state_conf` | float | Softmax probability (0–1) |
| `ml_cycle_pos` | float | Predicted position in ACTIVE segment (0–1) |
| `ml_job_type` | int | Predicted job cluster (0..K-1) |
| `ml_job_name` | str | "Light Load" / "Heavy Load" / ... |

**Implementation detail:** Uses `np.lib.stride_tricks.sliding_window_view` for efficient batched inference without Python loops over windows.

---

### 5.12 `robo/ml/trainer.py` (299 lines)

**Purpose:** Training loop with multi-task loss, early stopping, and checkpoint I/O.

**Multi-task loss:**

```
L = α · CE(state_logits, state_true)
  + β · MSE(cycle_pos_pred, cycle_pos_true)     [masked: only ACTIVE samples]
  + γ · CE(job_logits, job_true)                 [masked: only samples with job_type ≥ 0]
```

Default weights: α=1.0, β=0.5, γ=0.5

**Training configuration:**

| Parameter | Default |
|-----------|---------|
| Epochs | 25 (max) |
| Batch size | 64 |
| Learning rate | 1e-3 (Adam) |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=7 epochs |
| Gradient clipping | max_norm=1.0 |
| Weight decay | 1e-5 |

**Checkpoint path:** `robo/ml/checkpoints/model.pt`

**Checkpoint contents:**
```python
{
    "model_state_dict": ...,
    "hparams": {seq_len, n_vars, d_model, n_heads, n_layers, d_ff, n_states, n_job_types},
    "look_back": int,
    "norm_stats": {"mean": [...], "std": [...]},
    "label_meta": {...},  # pipeline parameters and cycle statistics
    "train_history": {"train_loss": [...], "val_loss": [...], ...}
}
```

---

### 5.13 `robo/ml/ui.py` (594 lines)

**Purpose:** Full Streamlit ML diagnostics page with 4 interactive tabs.

**Entry point:** `render_ml_page(df: pd.DataFrame)`

**Tab 1 — Data & Labels:**
- Configurable thresholds: STANDBY max, ACTIVE min, min run length, K clusters
- "Generate Labels" button → runs full auto-labeling pipeline
- Visualisation: time-series with state colour overlay, cycle ID markers, job-type distribution, per-cycle feature table

**Tab 2 — Train Model:**
- Hyperparameter controls: look-back, epochs, batch size, LR, d_model, n_heads, n_layers, loss weights
- Live training progress bar with epoch-by-epoch metrics
- Loss/accuracy curves plotted after training completes

**Tab 3 — Evaluation:**
- Loads latest checkpoint
- Runs inference on labeled data
- Displays: state confusion matrix, per-class precision/recall/F1, cycle position MAE, job-type accuracy

**Tab 4 — Inference View:**
- Loads trained model
- Overlays ML predictions on raw current time-series
- State confidence heatmap, cycle position tracker, job-type timeline

**Session state keys:** All prefixed with `robo_ml_` to avoid collisions with furnace/conveyer ML state.

---

## 6. Changes to `app.py`

The main Streamlit application (`app.py`) was modified to integrate the robot as a third machine. All changes are additive — no existing furnace or conveyer logic was altered.

### 6.1 Import block (lines 59–73)

Added 14 import statements for all robo modules, aliased with `robo_` prefix to avoid namespace collisions:

```python
from robo.data_loader         import load_data as robo_load_data, get_sample_rate as robo_sample_rate
from robo.windowing           import generate_windows as robo_gen_windows
from robo.fft_analysis        import compute_window_fft as robo_fft_win, batch_compute_fft as robo_batch_fft
from robo.validation          import validate_window as robo_val_win, validate_dataset as robo_val_ds
from robo.validation          import DEFAULT_THRESHOLDS as ROBO_VAL_DEFAULTS
from robo.feature_engineering import compute_features as robo_feat, features_to_dict as robo_feat_dict
from robo.rag_classifier      import (classify as robo_classify, STATE_COLORS as ROBO_STATE_COLORS,
                                       STATE_LABELS as ROBO_STATE_LABELS, DEFAULT_THRESHOLDS as ROBO_RAG_DEFAULTS)
from robo.state_manager       import StateManager as robo_StateManager
from robo.ml.ui               import render_ml_page as render_robo_ml_page
```

### 6.2 `MACHINE_META` dictionary (lines 180–200)

Added `"robo"` entry with all function references:

```python
"robo": {
    "label"        : "EPSON Robot",
    "icon"         : "🤖",
    "badge_color"  : "#8e44ad",
    "default_ws"   : 30,
    "load_data"    : robo_load_data,
    "sample_rate"  : robo_sample_rate,
    "gen_windows"  : robo_gen_windows,
    "fft_win"      : robo_fft_win,
    "batch_fft"    : robo_batch_fft,
    "val_win"      : robo_val_win,
    "val_ds"       : robo_val_ds,
    "feat"         : robo_feat,
    "feat_dict"    : robo_feat_dict,
    "classify"     : robo_classify,
    "StateManager" : robo_StateManager,
    "val_defaults" : ROBO_VAL_DEFAULTS,
    "rag_defaults" : ROBO_RAG_DEFAULTS,
    "state_colors" : ROBO_STATE_COLORS,
    "state_labels" : ROBO_STATE_LABELS,
},
```

### 6.3 Session state initialisation

Added `robo` dict inside `_ss_init()` with the same structure as furnace:

```python
robo = dict(
    df=None, windows_proc=None, fft_cache=None,
    val_results=None, dataset_val=None,
    features_cache=None, raw_states=None,
    smoothed_states=None, win_centres=None,
    playing=False, play_idx=0,
    window_size_sec=30,
    rag_thr=dict(ROBO_RAG_DEFAULTS),
    val_thr=dict(ROBO_VAL_DEFAULTS),
    last_file=None,
),
```

Note: No `df_cyclic` field (that's conveyer-only for its dual-CSV workflow).

### 6.4 Sidebar machine radio

Changed from 2-option to 3-option with list-based index lookup:

```python
# Before
options=["furnace", "conveyer"]
index=0 if st.session_state.current_machine == "furnace" else 1

# After
options=["furnace", "conveyer", "robo"]
index=["furnace", "conveyer", "robo"].index(st.session_state.current_machine)
```

### 6.5 Sidebar RAG thresholds (conditional imbalance)

The `green_min_imbalance` slider is now **conditionally rendered** — only shown when the machine's RAG defaults include that key. The robot has no `green_min_imbalance` threshold:

```python
# Before (always shown)
rt["green_min_imbalance"] = st.sidebar.number_input(...)

# After (conditional)
if "green_min_imbalance" in rt:
    rt["green_min_imbalance"] = st.sidebar.number_input(...)
```

### 6.6 Window size slider

No code change needed — the robot falls into the existing `else` branch which uses range 10–120s (same as furnace). The conveyer-specific branch uses 2–10s.

### 6.7 File upload (page_validation)

No code change needed — the robot uses single-file upload, which is handled by the existing `else` branch (furnace also uses single file). Only conveyer requires the `if m == "conveyer"` multi-file branch.

### 6.8 RAG page routing (page_rag)

No code change needed — the robot uses **window-based classification** (like furnace), which is the `else` branch. Only conveyer uses per-sample classification (the `if m == "conveyer"` branch).

### 6.9 ML page routing (page_ml)

Added explicit `elif m == "robo"` case:

```python
elif m == "robo":
    _machine_header("robo")
    df = ss("df")
    if df is None:
        st.warning("No data loaded yet...")
        return
    render_robo_ml_page(df)
```

### 6.10 Docstring update

```python
# Before
"""Two machines, same pipeline, one app."""

# After
"""Three machines, same pipeline, one app.
  Machine 1 : Wave Soldering Furnace   (wave_soldering/)
  Machine 2 : MV Conveyer              (mv_conveyer/)
  Machine 3 : EPSON Robot              (robo/)"""
```

---

## 7. Pipeline Flow — End to End

The data pipeline for the robot follows this exact sequence (same as furnace/conveyer, routed via `MACHINE_META`):

```
CSV Upload
    ↓ robo.data_loader.load_data()           → pd.DataFrame
    ↓ robo.windowing.generate_windows()      → list[Window]
    ↓ robo.fft_analysis.batch_compute_fft()  → list[dict[str, FFTResult]]
    ↓ robo.validation.validate_window()      → WindowValidation (per window)
    ↓ robo.validation.validate_dataset()     → DatasetValidation (aggregate)
    ↓ robo.feature_engineering.compute_features() → WindowFeatures (per window)
    ↓ robo.rag_classifier.classify()         → ClassificationResult (per window)
    ↓ robo.state_manager.StateManager.run_batch() → list[str] (smoothed states)
```

For the ML pipeline:

```
pd.DataFrame
    ↓ robo.ml.label_generator.generate_labels()  → (df_labeled, meta)
    ↓ robo.ml.dataset.build_datasets()            → (train_ds, val_ds, test_ds, norm_stats)
    ↓ robo.ml.trainer.train()                     → history dict + checkpoint saved
    ↓ robo.ml.predictor.RoboPredictor.predict()   → df with ml_state, ml_cycle_pos, ml_job_type
```

---

## 8. Verification & Testing

### 8.1 Import verification

All 15 robo module imports resolve correctly from `app.py`:

```
from robo.data_loader         → load_data, get_sample_rate     ✓
from robo.windowing           → generate_windows               ✓
from robo.fft_analysis        → compute_window_fft, batch_compute_fft  ✓
from robo.validation          → validate_window, validate_dataset, DEFAULT_THRESHOLDS  ✓
from robo.feature_engineering → compute_features, features_to_dict  ✓
from robo.rag_classifier      → classify, STATE_COLORS, STATE_LABELS, DEFAULT_THRESHOLDS  ✓
from robo.state_manager       → StateManager                   ✓
from robo.ml.ui               → render_ml_page                 ✓
```

### 8.2 End-to-end pipeline test

Ran the full pipeline on the actual CSV:

```
✓ Loaded 14,861 rows at 1.0 Hz
✓ Generated 990 windows (30s, 50% overlap)
✓ Computed FFT — keys: ['i1', 'i_avg']
✓ Engineered features — type: WindowFeatures
✓ Classified windows — mix of AMBER and GREEN states
✓ Validation — all windows pass quality checks
✓ Dataset validation — summary generated
```

### 8.3 app.py syntax and import validation

```
✓ AST parse — no syntax errors
✓ Top-level execution — all imports resolve
✓ MACHINE_META["robo"] — all 16 keys present
```

---

## 9. Git Commits

### Commit 1: `a74babb`
```
feat: add EPSON Robot (robo) analysis pipeline and integrate into app

- Add robo/ package with full pipeline: data_loader, windowing, fft_analysis,
  feature_engineering, validation, rag_classifier, state_manager
- Add robo/ml/ sub-package: dataset, itransformer, label_generator, predictor,
  trainer, ui (iTransformer with n_vars=2 for single-phase I1 + I_Avg)
- Single-phase design: I2/I3 always 0, uses CSV I_Avg directly (not recomputed)
- RAG states: STANDBY (RED, I_Avg<0.085), IDLE (AMBER), ACTIVE (GREEN, >=0.115)
- Data-driven thresholds from K=3 clustering on 14,861 samples
- Update app.py: add robo to machine selector, session state, sidebar controls,
  validation/RAG/ML page routing; conditionally hide phase imbalance control

16 files changed, 2,789 insertions(+), 5 deletions(-)
```

### Commit 2: `014f767`
```
chore: add .gitignore for pycache and commit robo CSV data

2 files changed, 14,865 insertions(+)
  - .gitignore (new: __pycache__/, *.pyc, *.pyo)
  - data/robo/selec_em4m_datalog_robo_20260416.csv (14,861 data rows)
```

---

## 10. Summary of All Files Added / Modified

### New files (16)

| File | Lines | Purpose |
|------|-------|---------|
| `robo/__init__.py` | 1 | Package marker |
| `robo/data_loader.py` | 116 | CSV loading, I_Avg preservation |
| `robo/windowing.py` | 97 | Sliding window generator |
| `robo/fft_analysis.py` | 145 | FFT on I1 + I_Avg only |
| `robo/feature_engineering.py` | 157 | 20 time+freq domain features |
| `robo/validation.py` | 169 | Relaxed single-phase validation |
| `robo/rag_classifier.py` | 140 | STANDBY/IDLE/ACTIVE classifier |
| `robo/state_manager.py` | 132 | Hysteresis anti-flicker |
| `robo/ml/__init__.py` | 1 | ML sub-package marker |
| `robo/ml/dataset.py` | 130 | PyTorch Dataset (n_vars=2) |
| `robo/ml/itransformer.py` | 241 | iTransformer (n_vars=2) |
| `robo/ml/label_generator.py` | 308 | Auto-labeling + K-means clustering |
| `robo/ml/predictor.py` | 191 | Sliding-window inference |
| `robo/ml/trainer.py` | 299 | Training loop + checkpoints |
| `robo/ml/ui.py` | 594 | 4-tab Streamlit ML page |
| `.gitignore` | 3 | Ignore pycache |
| **Total new** | **2,724** | |

### Modified files (1)

| File | Insertions | Deletions | Changes |
|------|------------|-----------|---------|
| `app.py` | +68 | -5 | Imports, MACHINE_META, session state, sidebar, page routing |

### Data committed (1)

| File | Rows | Size |
|------|------|------|
| `data/robo/selec_em4m_datalog_robo_20260416.csv` | 14,861 | ~750 KB |
