"""
validation.py
Data-quality checks for the soldering furnace current data.

Validation strategy (adapted for 1 Hz RMS data):
-------------------------------------------------
1. Grid-frequency check  – the sensor's 'frequency' column should read
   ~50 Hz (±0.5 Hz) for every sample in the window.
2. Current range check   – each phase current must be finite and
   non-negative; at least one phase must be non-zero.
3. Phase consistency     – std(i1_rms, i2_rms, i3_rms) / mean must be
   below a configured threshold (catches dead phases / sensor faults).
4. Signal energy check   – the window's average current must exceed a
   minimum floor (avoids classifying noise as real data).

A window passes validation when ALL four checks pass.
Dataset-level result: fraction of valid windows >= valid_fraction_threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Default thresholds (tuned for soldering furnace data) ─────────────────────

DEFAULT_THRESHOLDS = dict(
    freq_target_hz        = 50.0,   # expected grid frequency
    freq_tolerance_hz     = 0.5,    # acceptable deviation
    freq_valid_fraction   = 0.90,   # fraction of samples in window that must pass
    min_current_floor     = 0.05,   # A  – window mean i_avg must exceed this
    max_phase_imbalance   = 0.80,   # CV (std/mean) across phases – 0 = perfect balance
    valid_window_fraction = 0.80,   # fraction of windows needed for VALID dataset
)


# ── Per-window validation ─────────────────────────────────────────────────────

@dataclass
class WindowValidation:
    is_valid: bool
    freq_ok: bool
    range_ok: bool
    phase_ok: bool
    energy_ok: bool
    freq_valid_pct: float     # % of samples with freq in tolerance
    avg_freq: float           # mean measured frequency in window
    phase_imbalance: float    # CV of phase RMS values
    avg_i_avg: float          # mean i_avg in window
    failure_reason: str       # human-readable reason if invalid


def validate_window(
    window_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> WindowValidation:
    """
    Validate a single window DataFrame.

    Parameters
    ----------
    window_df   : rows for this window (i1, i2, i3, i_avg, frequency)
    thresholds  : override dict (uses DEFAULT_THRESHOLDS for missing keys)
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    reasons: list[str] = []

    # ── 1. Grid frequency check ───────────────────────────────────────
    if "frequency" in window_df.columns:
        freqs = window_df["frequency"].dropna()
        avg_freq = float(freqs.mean()) if len(freqs) else 50.0
        in_band = np.abs(freqs - thr["freq_target_hz"]) <= thr["freq_tolerance_hz"]
        freq_valid_pct = float(in_band.mean() * 100.0)
        freq_ok = (freq_valid_pct / 100.0) >= thr["freq_valid_fraction"]
    else:
        avg_freq = 50.0
        freq_valid_pct = 100.0
        freq_ok = True

    if not freq_ok:
        reasons.append(f"Grid freq {avg_freq:.2f} Hz out of band")

    # ── 2. Current range check ────────────────────────────────────────
    all_finite = True
    for col in ("i1", "i2", "i3"):
        if col in window_df.columns:
            vals = window_df[col].dropna()
            if not np.all(np.isfinite(vals)) or (vals < 0).any():
                all_finite = False
                break

    # At least one phase must carry meaningful current
    any_current = False
    for col in ("i1", "i2", "i3"):
        if col in window_df.columns:
            if window_df[col].max() > 0.01:
                any_current = True
                break

    range_ok = all_finite and any_current
    if not range_ok:
        reasons.append("Current values out of valid range or all-zero")

    # ── 3. Phase consistency check ────────────────────────────────────
    phase_rms_vals = []
    for col in ("i1", "i2", "i3"):
        if col in window_df.columns:
            phase_rms_vals.append(float(window_df[col].mean()))

    if len(phase_rms_vals) >= 2:
        p_mean = np.mean(phase_rms_vals)
        p_std  = np.std(phase_rms_vals)
        if p_mean > 0.01:
            phase_imbalance = p_std / p_mean
        else:
            phase_imbalance = 0.0
        phase_ok = phase_imbalance <= thr["max_phase_imbalance"]
    else:
        phase_imbalance = 0.0
        phase_ok = True

    if not phase_ok:
        reasons.append(
            f"Phase imbalance CV={phase_imbalance:.2f} > {thr['max_phase_imbalance']:.2f}"
        )

    # ── 4. Energy / signal floor check ───────────────────────────────
    if "i_avg" in window_df.columns:
        avg_i_avg = float(window_df["i_avg"].mean())
    else:
        avg_i_avg = np.mean(phase_rms_vals) if phase_rms_vals else 0.0

    energy_ok = avg_i_avg >= thr["min_current_floor"]
    if not energy_ok:
        reasons.append(
            f"Mean i_avg {avg_i_avg:.3f} A < floor {thr['min_current_floor']:.3f} A"
        )

    is_valid = freq_ok and range_ok and phase_ok and energy_ok
    failure_reason = "; ".join(reasons) if reasons else "OK"

    return WindowValidation(
        is_valid       = is_valid,
        freq_ok        = freq_ok,
        range_ok       = range_ok,
        phase_ok       = phase_ok,
        energy_ok      = energy_ok,
        freq_valid_pct = freq_valid_pct,
        avg_freq       = avg_freq,
        phase_imbalance= phase_imbalance,
        avg_i_avg      = avg_i_avg,
        failure_reason = failure_reason,
    )


# ── Dataset-level validation ──────────────────────────────────────────────────

@dataclass
class DatasetValidation:
    is_dataset_valid: bool
    total_windows: int
    valid_windows: int
    valid_pct: float
    avg_freq: float
    avg_phase_imbalance: float
    avg_i_avg: float
    window_results: list[WindowValidation]


def validate_dataset(
    windows: list,
    thresholds: dict | None = None,
) -> DatasetValidation:
    """
    Run per-window validation over all windows and aggregate results.

    Parameters
    ----------
    windows    : list of Window objects (must have a .data DataFrame)
    thresholds : optional override dict
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    results: list[WindowValidation] = []
    for w in windows:
        df = w.data if hasattr(w, "data") else w
        results.append(validate_window(df, thresholds=thresholds))

    n_total = len(results)
    n_valid = sum(1 for r in results if r.is_valid)
    valid_pct = (n_valid / n_total * 100.0) if n_total else 0.0

    avg_freq      = float(np.mean([r.avg_freq for r in results])) if results else 50.0
    avg_imbalance = float(np.mean([r.phase_imbalance for r in results])) if results else 0.0
    avg_i_avg     = float(np.mean([r.avg_i_avg for r in results])) if results else 0.0

    is_valid = (valid_pct / 100.0) >= thr["valid_window_fraction"]

    return DatasetValidation(
        is_dataset_valid    = is_valid,
        total_windows       = n_total,
        valid_windows       = n_valid,
        valid_pct           = valid_pct,
        avg_freq            = avg_freq,
        avg_phase_imbalance = avg_imbalance,
        avg_i_avg           = avg_i_avg,
        window_results      = results,
    )
