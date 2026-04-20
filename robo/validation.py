"""
robo/validation.py
Data-quality checks for the EPSON Robot current data.

Key differences vs mv_conveyer:
- Single-phase: I2 = I3 = 0 always, so phase imbalance is meaningless.
  max_phase_imbalance set very high (2.0) to never fail.
- min_current_floor is very low (0.01 A) — robot baseline is 0.07+ A
- All frequency thresholds are the same (grid is still 50 Hz)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ── Default thresholds ────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = dict(
    freq_target_hz        = 50.0,
    freq_tolerance_hz     = 0.5,
    freq_valid_fraction   = 0.90,
    min_current_floor     = 0.01,   # A  — robot I_Avg baseline ~0.07 A
    max_phase_imbalance   = 2.00,   # CV — meaningless for single-phase; always passes
    valid_window_fraction = 0.80,
)


# ── Per-window validation ─────────────────────────────────────────────────────

@dataclass
class WindowValidation:
    is_valid: bool
    freq_ok: bool
    range_ok: bool
    phase_ok: bool
    energy_ok: bool
    freq_valid_pct: float
    avg_freq: float
    phase_imbalance: float
    avg_i_avg: float
    failure_reason: str


def validate_window(
    window_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> WindowValidation:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []

    # 1. Grid frequency check
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

    # 2. Current range check
    all_finite = True
    for col in ("i1", "i2", "i3"):
        if col in window_df.columns:
            vals = window_df[col].dropna()
            if not np.all(np.isfinite(vals)) or (vals < 0).any():
                all_finite = False
                break

    any_current = any(
        window_df[c].max() > 0.01
        for c in ("i1", "i2", "i3")
        if c in window_df.columns
    )
    range_ok = all_finite and any_current
    if not range_ok:
        reasons.append("Current values out of valid range or all-zero")

    # 3. Phase consistency check (always passes for single-phase robot)
    phase_rms_vals = [
        float(window_df[c].mean())
        for c in ("i1", "i2", "i3")
        if c in window_df.columns
    ]
    if len(phase_rms_vals) >= 2:
        p_mean = np.mean(phase_rms_vals)
        p_std  = np.std(phase_rms_vals)
        phase_imbalance = p_std / p_mean if p_mean > 0.01 else 0.0
        phase_ok = phase_imbalance <= thr["max_phase_imbalance"]
    else:
        phase_imbalance = 0.0
        phase_ok = True

    if not phase_ok:
        reasons.append(
            f"Phase imbalance CV={phase_imbalance:.2f} > {thr['max_phase_imbalance']:.2f}"
        )

    # 4. Energy floor check
    avg_i_avg = (
        float(window_df["i_avg"].mean())
        if "i_avg" in window_df.columns
        else np.mean(phase_rms_vals) if phase_rms_vals else 0.0
    )
    energy_ok = avg_i_avg >= thr["min_current_floor"]
    if not energy_ok:
        reasons.append(
            f"Mean i_avg {avg_i_avg:.3f} A < floor {thr['min_current_floor']:.3f} A"
        )

    return WindowValidation(
        is_valid        = all([freq_ok, range_ok, phase_ok, energy_ok]),
        freq_ok         = freq_ok,
        range_ok        = range_ok,
        phase_ok        = phase_ok,
        energy_ok       = energy_ok,
        freq_valid_pct  = freq_valid_pct,
        avg_freq        = avg_freq,
        phase_imbalance = phase_imbalance,
        avg_i_avg       = avg_i_avg,
        failure_reason  = "; ".join(reasons) if reasons else "OK",
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
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    results = [
        validate_window(w.data if hasattr(w, "data") else w, thresholds=thresholds)
        for w in windows
    ]
    n_total = len(results)
    n_valid = sum(1 for r in results if r.is_valid)
    valid_pct = (n_valid / n_total * 100.0) if n_total else 0.0

    return DatasetValidation(
        is_dataset_valid    = (valid_pct / 100.0) >= thr["valid_window_fraction"],
        total_windows       = n_total,
        valid_windows       = n_valid,
        valid_pct           = valid_pct,
        avg_freq            = float(np.mean([r.avg_freq for r in results])) if results else 50.0,
        avg_phase_imbalance = float(np.mean([r.phase_imbalance for r in results])) if results else 0.0,
        avg_i_avg           = float(np.mean([r.avg_i_avg for r in results])) if results else 0.0,
        window_results      = results,
    )
