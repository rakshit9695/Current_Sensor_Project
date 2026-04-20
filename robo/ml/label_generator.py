"""
robo/ml/label_generator.py
Auto-labeling pipeline for EPSON Robot data. No manual annotation required.

Three states
------------
  0 = STANDBY (RED)   : robot in standby, I_Avg < standby_max
  1 = IDLE    (AMBER) : robot energised but not at full load
                        I_Avg in [standby_max, active_min)
  2 = ACTIVE  (GREEN) : robot actively operating
                        I_Avg >= active_min

Cycle definition
----------------
A "cycle" is one contiguous ACTIVE segment (minimum min_run samples).
  - cycle_id  : integer index of the active segment (0-based)
  - cycle_pos : 0.0 → 1.0 fractional position within the ACTIVE segment

The IDLE/STANDBY gaps between active segments do NOT receive a cycle_id
(they are set to -1 / NaN).

Job clustering
--------------
K-means on per-cycle features separates distinct operational modes.
For the robot this could be different tasks (pick-and-place patterns,
welding cycles, etc.) differentiated by duration and current profile.
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
STANDBY = 0   # RED
IDLE    = 1   # AMBER
ACTIVE  = 2   # GREEN

STATE_NAMES = {STANDBY: "STANDBY", IDLE: "IDLE", ACTIVE: "ACTIVE"}

# ── Default thresholds (calibrated from K-means clustering) ───────────────────
DEFAULT_STANDBY_MAX = 0.085   # A  I_Avg < 0.085 → STANDBY
DEFAULT_ACTIVE_MIN  = 0.115   # A  I_Avg >= 0.115 → ACTIVE
DEFAULT_MIN_RUN     = 3       # samples  minimum ACTIVE segment length
DEFAULT_N_JOBS      = 2       # K-means clusters


# ── Step 1: Per-sample state labeling ─────────────────────────────────────────

def label_states(
    df: pd.DataFrame,
    standby_max: float = DEFAULT_STANDBY_MAX,
    active_min:  float = DEFAULT_ACTIVE_MIN,
) -> pd.Series:
    """
    Assign raw per-sample state labels based on I_Avg thresholds.
    Returns a Series of ints: 0=STANDBY, 1=IDLE, 2=ACTIVE.
    """
    iavg = df["i_avg"].to_numpy(dtype=np.float32)
    labels = np.ones(len(iavg), dtype=np.int32)   # default IDLE
    labels[iavg < standby_max]  = STANDBY
    labels[iavg >= active_min]  = ACTIVE
    return pd.Series(labels, index=df.index, name="state_label")


# ── Step 2: RLE smoothing to remove glitches ──────────────────────────────────

def smooth_labels(
    labels: pd.Series,
    min_run: int = DEFAULT_MIN_RUN,
) -> pd.Series:
    """
    Run-length encode the label sequence and remove runs shorter than min_run.
    """
    arr = labels.to_numpy().copy()
    n   = len(arr)

    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        v = arr[i]
        j = i
        while j < n and arr[j] == v:
            j += 1
        runs.append((v, i, j))
        i = j

    changed = True
    while changed:
        changed = False
        new_runs = []
        k = 0
        while k < len(runs):
            v, s, e = runs[k]
            if (e - s) < min_run and len(runs) > 1:
                prev_len = runs[k-1][2] - runs[k-1][1] if k > 0 else -1
                next_len = runs[k+1][2] - runs[k+1][1] if k < len(runs)-1 else -1
                replacement = runs[k-1][0] if prev_len >= next_len else runs[k+1][0]
                arr[s:e] = replacement
                changed = True
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


# ── Step 3: Detect ACTIVE segments (cycles) ──────────────────────────────────

def detect_cycles(
    labels: pd.Series,
    min_run: int = DEFAULT_MIN_RUN,
) -> pd.Series:
    """
    Identify contiguous ACTIVE segments as individual cycles.
    """
    arr     = labels.to_numpy()
    cyc_ids = np.full(len(arr), -1, dtype=np.int32)
    cycle   = 0
    i       = 0

    while i < len(arr):
        if arr[i] == ACTIVE:
            j = i
            while j < len(arr) and arr[j] == ACTIVE:
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
    """Add cycle_id and cycle_pos columns to df."""
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
    Extract one feature vector per detected ACTIVE cycle.

    Features
    --------
    cycle_id       : integer cycle index
    duration_s     : length of the ACTIVE segment (seconds at 1 Hz)
    mean_iavg      : mean I_Avg during the segment
    max_iavg       : peak I_Avg
    std_iavg       : standard deviation of I_Avg
    mean_i1        : mean I1 during segment
    log_duration   : log10(duration_s + 1)
    """
    rows = []
    for cid, grp in df[df["cycle_id"] >= 0].groupby("cycle_id"):
        dur  = len(grp)
        iavg = grp["i_avg"].to_numpy(dtype=np.float32)
        i1   = grp["i1"].to_numpy(dtype=np.float32)
        rows.append(dict(
            cycle_id     = int(cid),
            duration_s   = float(dur),
            mean_iavg    = float(iavg.mean()),
            max_iavg     = float(iavg.max()),
            std_iavg     = float(iavg.std()),
            mean_i1      = float(i1.mean()),
            log_duration = float(np.log10(dur + 1)),
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Step 6: K-means job clustering ────────────────────────────────────────────

def cluster_job_types(
    cycle_features: pd.DataFrame,
    n_job_types: int = DEFAULT_N_JOBS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """Cluster ACTIVE cycles by their feature profiles."""
    if cycle_features.empty:
        return cycle_features, StandardScaler(), KMeans(n_clusters=n_job_types)

    FEAT_COLS = ["log_duration", "mean_iavg", "std_iavg", "mean_i1"]
    X = cycle_features[FEAT_COLS].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_job_types, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(X_sc)

    # Stable ordering: sort clusters by mean log_duration (ascending)
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
    standby_max: float = DEFAULT_STANDBY_MAX,
    active_min:  float = DEFAULT_ACTIVE_MIN,
    min_run:     int   = DEFAULT_MIN_RUN,
    n_job_types: int   = DEFAULT_N_JOBS,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full auto-labeling pipeline on a raw DataFrame.

    Parameters
    ----------
    df          : output of data_loader.load_data()
    standby_max : I_Avg threshold (A) below which state = STANDBY
    active_min  : I_Avg threshold (A) at/above which state = ACTIVE
    min_run     : minimum consecutive ACTIVE samples to form a cycle
    n_job_types : number of K-means clusters

    Returns
    -------
    df_labeled  : input df extended with columns:
                    state_label, cycle_id, cycle_pos, job_type
    meta        : dict with pipeline parameters and summary statistics
    """
    raw_labels = label_states(df, standby_max=standby_max, active_min=active_min)
    smooth = smooth_labels(raw_labels, min_run=min_run)
    cyc_ids = detect_cycles(smooth, min_run=min_run)
    df_lab = assign_cycle_labels(df, smooth, cyc_ids)

    cyc_feats = compute_cycle_features(df_lab)

    if not cyc_feats.empty:
        n_clusters = min(n_job_types, len(cyc_feats))
        cyc_feats, scaler, kmeans = cluster_job_types(
            cyc_feats, n_job_types=n_clusters
        )
        cid_to_job = dict(zip(cyc_feats["cycle_id"], cyc_feats["job_type"]))
        df_lab["job_type"] = df_lab["cycle_id"].map(cid_to_job).fillna(-1).astype(int)
    else:
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=n_job_types)
        cyc_feats["job_type"] = []
        df_lab["job_type"] = -1

    n_cycles = int(cyc_ids.max() + 1) if (cyc_ids >= 0).any() else 0
    dur_arr  = cyc_feats["duration_s"].to_numpy() if not cyc_feats.empty else np.array([])

    scaler_b64 = base64.b64encode(pickle.dumps(scaler)).decode()
    kmeans_b64 = base64.b64encode(pickle.dumps(kmeans)).decode()

    meta = dict(
        standby_max      = standby_max,
        active_min       = active_min,
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
