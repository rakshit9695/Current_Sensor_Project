"""
feature_engineering.py
Compute all time-domain and frequency-domain features for a window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .fft_analysis import FFTResult


@dataclass
class WindowFeatures:
    # ── Time-domain ───────────────────────────────────────────────────
    rms_i1: float
    rms_i2: float
    rms_i3: float
    rms_i_avg: float
    variance_i1: float
    variance_i2: float
    variance_i3: float
    variance_i_avg: float
    peak_to_peak_i1: float
    peak_to_peak_i2: float
    peak_to_peak_i3: float
    phase_imbalance: float     # std(rms_i1, rms_i2, rms_i3) / mean

    # ── Frequency-domain ─────────────────────────────────────────────
    fundamental_freq: float    # Hz – dominant operational cycle frequency
    fundamental_energy: float  # magnitude² at fundamental
    harmonic2_energy: float    # 2nd harmonic
    harmonic3_energy: float    # 3rd harmonic
    total_energy: float        # total spectral energy
    high_freq_energy: float    # energy above sample_rate / 4

    # ── Derived ───────────────────────────────────────────────────────
    thd: float                 # harmonic distortion proxy
    high_freq_ratio: float     # high_freq_energy / total_energy

    # ── Per-phase FFT (optional, for deeper inspection) ───────────────
    fft_results: dict[str, FFTResult] = field(default_factory=dict, repr=False)


def compute_features(
    window_df: pd.DataFrame,
    fft_results: dict[str, FFTResult],
) -> WindowFeatures:
    """
    Compute all features for a single window.

    Parameters
    ----------
    window_df   : DataFrame for the window (i1, i2, i3, i_avg columns)
    fft_results : dict returned by fft_analysis.compute_window_fft
    """
    def _safe_get(col: str) -> np.ndarray:
        if col in window_df.columns:
            return window_df[col].to_numpy(dtype=np.float64)
        return np.zeros(max(1, len(window_df)))

    i1   = _safe_get("i1")
    i2   = _safe_get("i2")
    i3   = _safe_get("i3")
    i_avg= _safe_get("i_avg")

    # ── RMS (for RMS-sampled data, this is RMS of the window samples) ──
    def rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0

    rms_i1   = rms(i1)
    rms_i2   = rms(i2)
    rms_i3   = rms(i3)
    rms_i_avg= rms(i_avg)

    # ── Variance ───────────────────────────────────────────────────────
    var_i1   = float(np.var(i1))
    var_i2   = float(np.var(i2))
    var_i3   = float(np.var(i3))
    var_i_avg= float(np.var(i_avg))

    # ── Peak-to-peak ──────────────────────────────────────────────────
    def p2p(arr: np.ndarray) -> float:
        return float(np.ptp(arr)) if len(arr) > 1 else 0.0

    ptp_i1 = p2p(i1)
    ptp_i2 = p2p(i2)
    ptp_i3 = p2p(i3)

    # ── Phase imbalance ───────────────────────────────────────────────
    rms_phases = np.array([rms_i1, rms_i2, rms_i3])
    phase_mean = rms_phases.mean()
    phase_imbalance = float(rms_phases.std() / phase_mean) if phase_mean > 0.01 else 0.0

    # ── Frequency-domain from i_avg FFT (most representative) ─────────
    fft_ref = fft_results.get("i_avg") or fft_results.get("i1")

    if fft_ref is not None:
        fund_freq    = fft_ref.fundamental_freq
        fund_energy  = fft_ref.fundamental_mag ** 2
        harm2_energy = fft_ref.harmonic2_mag ** 2
        harm3_energy = fft_ref.harmonic3_mag ** 2
        total_energy = fft_ref.total_energy
        hf_energy    = fft_ref.high_freq_energy
        thd_val      = fft_ref.thd
    else:
        fund_freq = fund_energy = harm2_energy = harm3_energy = 0.0
        total_energy = hf_energy = thd_val = 0.0

    hf_ratio = float(hf_energy / total_energy) if total_energy > 1e-9 else 0.0

    return WindowFeatures(
        rms_i1           = rms_i1,
        rms_i2           = rms_i2,
        rms_i3           = rms_i3,
        rms_i_avg        = rms_i_avg,
        variance_i1      = var_i1,
        variance_i2      = var_i2,
        variance_i3      = var_i3,
        variance_i_avg   = var_i_avg,
        peak_to_peak_i1  = ptp_i1,
        peak_to_peak_i2  = ptp_i2,
        peak_to_peak_i3  = ptp_i3,
        phase_imbalance  = phase_imbalance,
        fundamental_freq = fund_freq,
        fundamental_energy = fund_energy,
        harmonic2_energy = harm2_energy,
        harmonic3_energy = harm3_energy,
        total_energy     = total_energy,
        high_freq_energy = hf_energy,
        thd              = thd_val,
        high_freq_ratio  = hf_ratio,
        fft_results      = fft_results,
    )


def features_to_dict(f: WindowFeatures) -> dict:
    """Flatten a WindowFeatures dataclass to a plain dict (no nested objects)."""
    return {
        "rms_i1"           : f.rms_i1,
        "rms_i2"           : f.rms_i2,
        "rms_i3"           : f.rms_i3,
        "rms_i_avg"        : f.rms_i_avg,
        "variance_i1"      : f.variance_i1,
        "variance_i2"      : f.variance_i2,
        "variance_i3"      : f.variance_i3,
        "variance_i_avg"   : f.variance_i_avg,
        "peak_to_peak_i1"  : f.peak_to_peak_i1,
        "peak_to_peak_i2"  : f.peak_to_peak_i2,
        "peak_to_peak_i3"  : f.peak_to_peak_i3,
        "phase_imbalance"  : f.phase_imbalance,
        "fundamental_freq" : f.fundamental_freq,
        "fundamental_energy": f.fundamental_energy,
        "harmonic2_energy" : f.harmonic2_energy,
        "harmonic3_energy" : f.harmonic3_energy,
        "total_energy"     : f.total_energy,
        "high_freq_energy" : f.high_freq_energy,
        "thd"              : f.thd,
        "high_freq_ratio"  : f.high_freq_ratio,
    }
