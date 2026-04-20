"""
label_generator.py
==================
Auto-generates ML training labels from raw 1 Hz RMS current data.

No manual annotation required.  Four label columns are added to the DataFrame:

  state_label  (int)   : 0=RED, 1=AMBER, 2=GREEN  (per-sample)
  cycle_id     (int)   : sequential cycle index; -1 for idle (RED) periods
  cycle_pos    (float) : 0.0–1.0 position within the cycle; NaN for idle
  job_type     (int)   : K-means cluster label; -1 for idle

Completely independent of the rule-based RAG/FFT code.
"""

from __future__ import annotations

import warnings
from itertools import groupby

import numpy as np
import pandas as pd

# ── State constants ────────────────────────────────────────────────────────────
STATE_RED   = 0
STATE_AMBER = 1
STATE_GREEN = 2
STATE_NAMES = {STATE_RED: "RED", STATE_AMBER: "AMBER", STATE_GREEN: "GREEN"}


# ── Step 1 – per-sample state labels ──────────────────────────────────────────

def label_states(
    df: pd.DataFrame,
    red_max: float = 2.0,
    green_min: float = 12.0,
) -> np.ndarray:
    """
    Assign RED / AMBER / GREEN to every sample using simple I_avg thresholds.

    Parameters
    ----------
    df        : DataFrame with an 'i_avg' column
    red_max   : I_avg at or below this → RED
    green_min : I_avg at or above this → GREEN
    """
    i = df["i_avg"].to_numpy(dtype=np.float64)
    labels = np.full(len(i), STATE_AMBER, dtype=np.int32)
    labels[i <= red_max]   = STATE_RED
    labels[i >= green_min] = STATE_GREEN
    return labels


# ── Step 2 – smooth short noise bursts ────────────────────────────────────────

def _rle(labels: np.ndarray):
    """Run-length encode an integer array.  Returns list of (value, start, end)."""
    runs = []
    pos = 0
    for val, grp in groupby(labels.tolist()):
        n = sum(1 for _ in grp)
        runs.append([int(val), pos, pos + n])
        pos += n
    return runs


def smooth_labels(labels: np.ndarray, min_run: int = 10) -> np.ndarray:
    """
    Eliminate state runs shorter than *min_run* samples by absorbing them
    into the surrounding state.  Iterates until stable (max 10 passes).
    """
    result = labels.copy()
    for _ in range(10):
        runs = _rle(result)
        changed = False
        for i, (val, start, end) in enumerate(runs):
            if (end - start) < min_run:
                # Pick neighbour with the longer run (prefer left on tie)
                left_len  = (runs[i-1][2] - runs[i-1][1]) if i > 0             else 0
                right_len = (runs[i+1][2] - runs[i+1][1]) if i < len(runs)-1   else 0
                if left_len == 0 and right_len == 0:
                    continue
                new_val = runs[i-1][0] if left_len >= right_len else runs[i+1][0]
                result[start:end] = new_val
                changed = True
        if not changed:
            break
    return result


# ── Step 3 – detect complete machine cycles ───────────────────────────────────

def detect_cycles(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Find complete machine cycles in the smoothed state sequence.

    A **cycle** is a contiguous non-RED segment that contains at least one
    GREEN sample.  It starts at the first sample after a RED run ends and
    finishes at the last sample before the next RED run begins.

    Returns
    -------
    List of (start_idx, end_idx) tuples – both indices are inclusive.
    """
    runs = _rle(labels)
    cycles: list[tuple[int, int]] = []

    i = 0
    while i < len(runs):
        val, start, end = runs[i]

        # Entry point: RED run followed by non-RED content
        if val == STATE_RED and i + 1 < len(runs):
            cycle_entry = end          # first sample after this RED run
            # Scan forward collecting non-RED runs until the next RED
            j = i + 1
            has_green = False
            while j < len(runs) and runs[j][0] != STATE_RED:
                if runs[j][0] == STATE_GREEN:
                    has_green = True
                j += 1

            if has_green:
                # cycle_exit = start of closing RED run (exclusive)
                cycle_exit = runs[j][1] if j < len(runs) else len(labels)
                if cycle_exit > cycle_entry:
                    cycles.append((cycle_entry, cycle_exit - 1))
            # Advance to the closing RED run (or end)
            i = j
        else:
            i += 1

    return cycles


# ── Step 4 – assign cycle_id and cycle_pos ────────────────────────────────────

def assign_cycle_labels(
    n_samples: int,
    cycles: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a list of (start, end) cycle boundaries, produce:
      cycle_id  : int array,   -1 outside cycles, 0/1/2/... inside
      cycle_pos : float array, NaN outside cycles, 0.0–1.0 inside
    """
    cycle_id  = np.full(n_samples, -1, dtype=np.int32)
    cycle_pos = np.full(n_samples, np.nan, dtype=np.float32)

    for cid, (cstart, cend) in enumerate(cycles):
        length = cend - cstart          # total samples in this cycle
        if length < 1:
            continue
        cycle_id[cstart : cend + 1] = cid
        cycle_pos[cstart : cend + 1] = np.linspace(0.0, 1.0, length + 1)

    return cycle_id, cycle_pos


# ── Step 5 – extract cycle-level features for clustering ─────────────────────

def compute_cycle_features(
    df: pd.DataFrame,
    cycles: list[tuple[int, int]],
    labels: np.ndarray,
) -> np.ndarray:
    """
    Extract a feature vector for each cycle used for K-means job clustering.

    Features (all normalised later by the caller):
      0  log1p(duration_sec)
      1  mean I_avg during GREEN samples
      2  max  I_avg during GREEN samples
      3  std  I_avg during GREEN samples
      4  green_fraction (proportion of cycle in GREEN)
      5  amber_fraction (proportion of cycle in AMBER)
    """
    i_avg = df["i_avg"].to_numpy(dtype=np.float64)
    feats = []

    for cstart, cend in cycles:
        seg_labels = labels[cstart : cend + 1]
        seg_iavg   = i_avg[cstart : cend + 1]
        duration   = cend - cstart + 1

        green_mask = seg_labels == STATE_GREEN
        amber_mask = seg_labels == STATE_AMBER

        green_iavg = seg_iavg[green_mask]
        mean_g = float(green_iavg.mean()) if len(green_iavg) > 0 else 0.0
        max_g  = float(green_iavg.max())  if len(green_iavg) > 0 else 0.0
        std_g  = float(green_iavg.std())  if len(green_iavg) > 1 else 0.0

        green_frac = float(green_mask.sum()) / max(duration, 1)
        amber_frac = float(amber_mask.sum()) / max(duration, 1)

        feats.append([
            float(np.log1p(duration)),
            mean_g,
            max_g,
            std_g,
            green_frac,
            amber_frac,
        ])

    return np.array(feats, dtype=np.float32)


# ── Step 6 – K-means job type clustering ─────────────────────────────────────

def cluster_job_types(
    cycle_features: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, object, object]:
    """
    Cluster cycles into *n_clusters* job types using K-means.

    Returns
    -------
    labels    : int array of shape (n_cycles,) with cluster indices
    scaler    : fitted StandardScaler  (save with checkpoint)
    kmeans    : fitted KMeans          (save with checkpoint)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    if len(cycle_features) < n_clusters:
        # Not enough cycles – assign all to cluster 0
        dummy_labels = np.zeros(len(cycle_features), dtype=np.int32)
        scaler = StandardScaler().fit(cycle_features)
        # Build a minimal KMeans with 1 cluster so it can still predict
        km = KMeans(n_clusters=1, random_state=random_state, n_init=10)
        km.fit(scaler.transform(cycle_features))
        return dummy_labels, scaler, km

    scaler = StandardScaler()
    X = scaler.fit_transform(cycle_features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km.fit(X)

    # Stable labelling: sort cluster centres by mean I_avg (feature index 1)
    # so that 0 = lowest load, n-1 = highest load
    centres_raw = scaler.inverse_transform(km.cluster_centers_)
    order = np.argsort(centres_raw[:, 1])          # sort by mean GREEN I_avg
    remap = {old: new for new, old in enumerate(order)}
    raw_labels = km.labels_
    sorted_labels = np.array([remap[l] for l in raw_labels], dtype=np.int32)

    # Reorder KMeans centres so predict() returns consistent labels
    km.cluster_centers_ = km.cluster_centers_[order]
    # Remap stored labels_ too
    km.labels_ = sorted_labels

    return sorted_labels, scaler, km


# ── Master function ───────────────────────────────────────────────────────────

def generate_labels(
    df: pd.DataFrame,
    red_max:    float = 2.0,
    green_min:  float = 12.0,
    min_run:    int   = 10,
    n_job_types: int  = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Full label generation pipeline.

    Parameters
    ----------
    df           : cleaned furnace DataFrame (from data_loader.load_data)
    red_max      : I_avg threshold for RED state
    green_min    : I_avg threshold for GREEN state
    min_run      : minimum run-length (samples) below which a state is absorbed
    n_job_types  : number of job clusters

    Returns
    -------
    df_labeled   : input DataFrame with four new columns added
    meta         : dict with clustering artefacts, thresholds, cycle stats
    """
    df = df.copy()

    # 1. Raw per-sample labels
    raw = label_states(df, red_max=red_max, green_min=green_min)

    # 2. Smooth
    smooth = smooth_labels(raw, min_run=min_run)

    # 3. Cycle detection
    cycles = detect_cycles(smooth)

    # 4. Cycle position
    cycle_id, cycle_pos = assign_cycle_labels(len(smooth), cycles)

    # 5. Cycle features for clustering
    cycle_feats = compute_cycle_features(df, cycles, smooth)

    # 6. Cluster job types
    if len(cycles) > 0:
        job_cluster_labels, scaler, kmeans = cluster_job_types(
            cycle_feats, n_clusters=min(n_job_types, len(cycles))
        )
    else:
        job_cluster_labels = np.array([], dtype=np.int32)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        kmeans = None

    # 7. Map cluster labels back to per-sample level
    job_type = np.full(len(smooth), -1, dtype=np.int32)
    for cid, (cstart, cend) in enumerate(cycles):
        if cid < len(job_cluster_labels):
            job_type[cstart : cend + 1] = job_cluster_labels[cid]

    # 8. Attach to DataFrame
    df["state_label"] = smooth
    df["cycle_id"]    = cycle_id
    df["cycle_pos"]   = cycle_pos
    df["job_type"]    = job_type

    # 9. Build metadata
    n_valid_cycles = len(cycles)
    durations = [c[1] - c[0] + 1 for c in cycles]
    meta = dict(
        red_max        = red_max,
        green_min      = green_min,
        min_run        = min_run,
        n_job_types    = int(min(n_job_types, max(1, n_valid_cycles))),
        n_cycles       = n_valid_cycles,
        cycle_durations= durations,
        mean_cycle_sec = float(np.mean(durations)) if durations else 0.0,
        std_cycle_sec  = float(np.std(durations))  if durations else 0.0,
        cycles         = cycles,
        scaler         = scaler,
        kmeans         = kmeans,
        cycle_features = cycle_feats,
        job_labels_per_cycle = job_cluster_labels.tolist() if len(job_cluster_labels) > 0 else [],
        state_counts   = {
            "RED"  : int((smooth == STATE_RED  ).sum()),
            "AMBER": int((smooth == STATE_AMBER).sum()),
            "GREEN": int((smooth == STATE_GREEN).sum()),
        },
    )

    return df, meta
