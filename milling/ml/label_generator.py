"""
milling/ml/label_generator.py
=============================
Auto-labeling pipeline for Milling Machine data. No manual annotation required.

Three states
------------
  0 = HALT    (RED   / "Idle")    : machine off / control only, I_Avg < halt_max
  1 = IDLE    (AMBER / "No-Load") : spindle/feed energised, not cutting
                                    I_Avg in [halt_max, run_min)
  2 = RUNNING (GREEN / "Cutting") : active material removal, all 3 phases loaded
                                    I_Avg >= run_min

(The integer aliases HALT/IDLE/RUNNING are kept for code symmetry with the
other machine pipelines; their human-readable labels are Idle / No-Load /
Cutting — see STATE_NAMES.)

Cycle definition
----------------
A "cycle" is one contiguous CUTTING segment (minimum min_run samples).
  - cycle_id  : integer index of the cutting segment (0-based)
  - cycle_pos : 0.0 → 1.0 fractional position within the cutting segment

The Idle/No-Load gaps between cutting segments do NOT receive a cycle_id
(they are set to -1 / NaN), matching the other pipelines' convention.

Job clustering
--------------
K-means on per-cycle features separates distinct machining operations
(e.g. light finishing passes vs heavy roughing cuts) by duration and load.
"""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ── State constants ────────────────────────────────────────────────────────────
HALT    = 0   # RED   – Idle
IDLE    = 1   # AMBER – No-Load
RUNNING = 2   # GREEN – Cutting

STATE_NAMES = {HALT: "Idle", IDLE: "No-Load", RUNNING: "Cutting"}

# ── Default thresholds (calibrated from actual data) ──────────────────────────
DEFAULT_HALT_MAX  = 1.0    # A  I_Avg < 1.0 → Idle
DEFAULT_RUN_MIN   = 3.0    # A  I_Avg >= 3.0 → Cutting
DEFAULT_MIN_RUN   = 5      # samples  minimum cutting segment length
DEFAULT_N_JOBS    = 3      # K-means clusters (machining operation types)


# ── Step 1: Per-sample state labeling ─────────────────────────────────────────

def label_states(
    df: pd.DataFrame,
    halt_max: float = DEFAULT_HALT_MAX,
    run_min:  float = DEFAULT_RUN_MIN,
) -> pd.Series:
    """
    Assign raw per-sample state labels based on I_Avg thresholds.

    Returns a Series of ints: 0=HALT, 1=IDLE, 2=RUNNING.
    """
    iavg = df["i_avg"].to_numpy(dtype=np.float32)
    labels = np.ones(len(iavg), dtype=np.int32)   # default IDLE
    labels[iavg < halt_max]  = HALT
    labels[iavg >= run_min]  = RUNNING
    return pd.Series(labels, index=df.index, name="state_label")


# ── Step 2: RLE smoothing to remove glitches ──────────────────────────────────

def smooth_labels(
    labels: pd.Series,
    min_run: int = DEFAULT_MIN_RUN,
) -> pd.Series:
    """
    Run-length encode the label sequence and remove runs shorter than min_run
    by replacing them with the neighbouring (majority) label.

    Parameters
    ----------
    min_run : minimum number of consecutive samples before a state is accepted.
              Runs shorter than this are merged into the surrounding state.
    """
    arr = labels.to_numpy().copy()
    n   = len(arr)

    # Build RLE: list of (value, start, end_exclusive)
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        v = arr[i]
        j = i
        while j < n and arr[j] == v:
            j += 1
        runs.append((v, i, j))
        i = j

    # Remove short runs (merge with previous or next)
    changed = True
    while changed:
        changed = False
        new_runs = []
        k = 0
        while k < len(runs):
            v, s, e = runs[k]
            if (e - s) < min_run and len(runs) > 1:
                # Replace this run with the label of the longer neighbour
                prev_len = runs[k-1][2] - runs[k-1][1] if k > 0 else -1
                next_len = runs[k+1][2] - runs[k+1][1] if k < len(runs)-1 else -1
                replacement = runs[k-1][0] if prev_len >= next_len else runs[k+1][0]
                arr[s:e] = replacement
                changed = True
                # Rebuild runs from scratch after modification
                new_runs = []
                i = 0
                while i < n:
                    vv = arr[i]
                    j  = i
                    while j < n and arr[j] == vv:
                        j += 1
                    new_runs.append((vv, i, j))
                    i = j
                runs = new_runs
                break
            else:
                new_runs.append((v, s, e))
                k += 1

    return pd.Series(arr, index=labels.index, name="state_label")


# ── Step 3: Detect RUNNING segments (cycles) ──────────────────────────────────

def detect_cycles(
    labels: pd.Series,
    min_run: int = DEFAULT_MIN_RUN,
) -> pd.Series:
    """
    Identify contiguous RUNNING segments as individual cycles.

    Returns a Series of int cycle_id:
      >= 0  : sample belongs to cycle cycle_id
        -1  : sample is HALT or IDLE (not in a cycle)
    """
    arr     = labels.to_numpy()
    cyc_ids = np.full(len(arr), -1, dtype=np.int32)
    cycle   = 0
    i       = 0

    while i < len(arr):
        if arr[i] == RUNNING:
            j = i
            while j < len(arr) and arr[j] == RUNNING:
                j += 1
            seg_len = j - i
            if seg_len >= min_run:
                cyc_ids[i:j] = cycle
                cycle += 1
            i = j
        else:
            i += 1

    return pd.Series(cyc_ids, index=labels.index, name="cycle_id")


# ── Step 4: Assign cycle positions ────────────────────────────────────────────

def assign_cycle_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    cycle_ids: pd.Series,
) -> pd.DataFrame:
    """
    Add cycle_id and cycle_pos columns to df.

    cycle_pos is 0.0 → 1.0 within each RUNNING segment.
    Samples outside cycles (HALT/IDLE gaps) get cycle_id=-1 and cycle_pos=NaN.
    """
    out = df.copy()
    out["state_label"] = labels.values
    out["cycle_id"]    = cycle_ids.values
    out["cycle_pos"]   = np.nan

    for cid in cycle_ids[cycle_ids >= 0].unique():
        mask = cycle_ids == cid
        n    = mask.sum()
        out.loc[mask, "cycle_pos"] = np.linspace(0.0, 1.0, n)

    return out


# ── Step 5: Compute per-cycle features for clustering ─────────────────────────

def compute_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one feature vector per detected RUNNING cycle.

    Features
    --------
    cycle_id       : integer cycle index
    duration_s     : length of the RUNNING segment (seconds / samples at 1 Hz)
    mean_iavg      : mean I_Avg during the segment
    max_iavg       : peak I_Avg
    std_iavg       : standard deviation of I_Avg during the segment
    mean_i1        : mean I1 during segment
    mean_i23       : mean of (I2+I3)/2 during segment (phase load indicator)
    log_duration   : log10(duration_s + 1) – linearises the large range
    """
    rows = []
    for cid, grp in df[df["cycle_id"] >= 0].groupby("cycle_id"):
        dur  = len(grp)
        iavg = grp["i_avg"].to_numpy(dtype=np.float32)
        i1   = grp["i1"].to_numpy(dtype=np.float32)
        i2   = grp["i2"].to_numpy(dtype=np.float32)
        i3   = grp["i3"].to_numpy(dtype=np.float32)
        rows.append(dict(
            cycle_id     = int(cid),
            duration_s   = float(dur),
            mean_iavg    = float(iavg.mean()),
            max_iavg     = float(iavg.max()),
            std_iavg     = float(iavg.std()),
            mean_i1      = float(i1.mean()),
            mean_i23     = float(((i2 + i3) / 2).mean()),
            log_duration = float(np.log10(dur + 1)),
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Step 6: K-means job clustering ────────────────────────────────────────────

def cluster_job_types(
    cycle_features: pd.DataFrame,
    n_job_types: int = DEFAULT_N_JOBS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """
    Cluster RUNNING cycles by their feature profiles.

    Returns
    -------
    cycle_features : DataFrame with a new 'job_type' column (0-based int)
    scaler         : fitted StandardScaler
    kmeans         : fitted KMeans model
    """
    if cycle_features.empty:
        return cycle_features, StandardScaler(), KMeans(n_clusters=n_job_types)

    FEAT_COLS = ["log_duration", "mean_iavg", "std_iavg", "mean_i23"]
    X = cycle_features[FEAT_COLS].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_job_types, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X_sc)

    # Stable ordering: sort clusters by mean log_duration (ascending)
    # → cluster 0 = shortest cycles (cyclic mode)
    # → cluster 1 = longest cycles (continuous mode)
    cluster_order = np.argsort(
        [X[raw_labels == c, 0].mean() for c in range(n_job_types)]
    )
    label_map = {old: new for new, old in enumerate(cluster_order)}
    stable = np.array([label_map[l] for l in raw_labels], dtype=np.int32)

    cf = cycle_features.copy()
    cf["job_type"] = stable
    return cf, scaler, km


# ── Master function ────────────────────────────────────────────────────────────

def generate_labels(
    df: pd.DataFrame,
    halt_max:    float = DEFAULT_HALT_MAX,
    run_min:     float = DEFAULT_RUN_MIN,
    min_run:     int   = DEFAULT_MIN_RUN,
    n_job_types: int   = DEFAULT_N_JOBS,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full auto-labeling pipeline on a raw DataFrame.

    Parameters
    ----------
    df          : output of data_loader.load_data()
    halt_max    : I_Avg threshold (A) below which state = HALT
    run_min     : I_Avg threshold (A) at/above which state = RUNNING
    min_run     : minimum consecutive RUNNING samples to form a cycle
    n_job_types : number of K-means clusters

    Returns
    -------
    df_labeled  : input df extended with columns:
                    state_label, cycle_id, cycle_pos, job_type
    meta        : dict with pipeline parameters and summary statistics
    """
    # Step 1 – raw state labels
    raw_labels = label_states(df, halt_max=halt_max, run_min=run_min)

    # Step 2 – smooth (remove glitch runs)
    smooth = smooth_labels(raw_labels, min_run=min_run)

    # Step 3 – cycle detection
    cyc_ids = detect_cycles(smooth, min_run=min_run)

    # Step 4 – positional labels
    df_lab = assign_cycle_labels(df, smooth, cyc_ids)

    # Step 5 – per-cycle features
    cyc_feats = compute_cycle_features(df_lab)

    # Step 6 – job clustering (only if we have cycles)
    if not cyc_feats.empty:
        n_clusters = min(n_job_types, len(cyc_feats))
        cyc_feats, scaler, kmeans = cluster_job_types(
            cyc_feats, n_job_types=n_clusters
        )
        # Map cycle job_type back to per-sample level
        cid_to_job = dict(zip(cyc_feats["cycle_id"], cyc_feats["job_type"]))
        df_lab["job_type"] = df_lab["cycle_id"].map(cid_to_job).fillna(-1).astype(int)
    else:
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=n_job_types)
        cyc_feats["job_type"] = []
        df_lab["job_type"] = -1

    n_cycles = int(cyc_ids.max() + 1) if (cyc_ids >= 0).any() else 0
    dur_arr  = cyc_feats["duration_s"].to_numpy() if not cyc_feats.empty else np.array([])

    # Serialise sklearn objects for checkpoint storage
    scaler_b64 = base64.b64encode(pickle.dumps(scaler)).decode()
    kmeans_b64 = base64.b64encode(pickle.dumps(kmeans)).decode()

    meta = dict(
        halt_max         = halt_max,
        run_min          = run_min,
        min_run          = min_run,
        n_job_types      = n_job_types,
        n_cycles         = n_cycles,
        cycle_durations  = dur_arr.tolist(),
        cycle_mean_dur   = float(dur_arr.mean()) if len(dur_arr) else 0.0,
        cycle_median_dur = float(np.median(dur_arr)) if len(dur_arr) else 0.0,
        scaler_b64       = scaler_b64,
        kmeans_b64       = kmeans_b64,
        cycle_features   = cyc_feats.to_dict(orient="list"),
    )

    return df_lab, meta
