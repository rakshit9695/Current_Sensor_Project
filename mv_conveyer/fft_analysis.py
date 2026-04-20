"""
fft_analysis.py
FFT and frequency-domain feature extraction.

IMPORTANT NOTE ON DATA FORMAT
------------------------------
The CSV contains pre-processed RMS current readings sampled at ~1 Hz —
NOT raw AC waveforms.  At 1 Hz the Nyquist limit is 0.5 Hz, so true
50 Hz power-line content cannot be observed in the current time-series
FFT.

What we DO here:
  1. FFT of the RMS time-series inside each window
     → reveals the *operational cycling frequency* of the furnace
       (typically 0.05–0.4 Hz for a soldering furnace heater cycle).
  2. Grid-frequency validation is done separately in validation.py
     using the 'frequency' column that the sensor reports directly.

Frequency-band mapping (relative to sample rate / 2 = 0.5 Hz):
  "fundamental"  → dominant peak in 0.05 – 0.45 Hz
  "2nd harmonic" → 2× fundamental
  "3rd harmonic" → 3× fundamental
  "high-freq"    → above (sample_rate / 4) = 0.25 Hz
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows as sig_windows
from dataclasses import dataclass


@dataclass
class FFTResult:
    """Holds FFT output for a single signal / window."""
    freqs: np.ndarray          # frequency axis [Hz]
    magnitudes: np.ndarray     # one-sided magnitude spectrum
    sample_rate: float         # Hz
    n_samples: int

    # ── Extracted scalar features ──────────────────────────────────────
    fundamental_freq: float    # Hz  – dominant operational cycle frequency
    fundamental_mag: float     # magnitude at fundamental
    harmonic2_mag: float       # 2nd harmonic magnitude
    harmonic3_mag: float       # 3rd harmonic magnitude
    total_energy: float        # sum of all magnitudes²
    high_freq_energy: float    # energy above (sample_rate / 4)
    thd: float                 # harmonic distortion proxy


# ── Core FFT routine ──────────────────────────────────────────────────────────

def compute_fft(
    signal: np.ndarray,
    sample_rate: float = 1.0,
) -> FFTResult:
    """
    Compute one-sided magnitude spectrum of *signal* and extract
    operational-frequency features.

    Parameters
    ----------
    signal      : 1-D array of RMS current values
    sample_rate : sampling rate in Hz (typically 1.0 for this dataset)
    """
    n = len(signal)
    if n < 4:
        empty = np.zeros(1)
        return FFTResult(
            freqs=empty, magnitudes=empty,
            sample_rate=sample_rate, n_samples=n,
            fundamental_freq=0.0, fundamental_mag=0.0,
            harmonic2_mag=0.0, harmonic3_mag=0.0,
            total_energy=0.0, high_freq_energy=0.0,
            thd=0.0,
        )

    # Apply Hann window to reduce spectral leakage
    win = sig_windows.hann(n)
    windowed = (signal - np.mean(signal)) * win   # remove DC, apply window

    # FFT
    fft_vals = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_vals) * (2.0 / n)    # one-sided, amplitude-scaled

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # ── Feature extraction ─────────────────────────────────────────────
    # Exclude DC (index 0) when looking for peaks
    search_start = 1  # skip DC bin

    if len(magnitudes) <= search_start:
        fund_idx = 0
    else:
        fund_idx = int(np.argmax(magnitudes[search_start:])) + search_start

    fundamental_freq = float(freqs[fund_idx]) if fund_idx < len(freqs) else 0.0
    fundamental_mag  = float(magnitudes[fund_idx]) if fund_idx < len(magnitudes) else 0.0

    def _mag_at_freq(target_hz: float, tol_hz: float = 0.05) -> float:
        """Return magnitude of bin closest to target_hz (within tolerance)."""
        if len(freqs) == 0 or target_hz <= 0:
            return 0.0
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        if abs(freqs[idx] - target_hz) <= tol_hz:
            return float(magnitudes[idx])
        return 0.0

    harmonic2_mag = _mag_at_freq(fundamental_freq * 2)
    harmonic3_mag = _mag_at_freq(fundamental_freq * 3)

    total_energy    = float(np.sum(magnitudes ** 2))
    hf_cutoff       = sample_rate / 4.0   # > quarter of Nyquist
    hf_mask         = freqs > hf_cutoff
    high_freq_energy = float(np.sum(magnitudes[hf_mask] ** 2)) if hf_mask.any() else 0.0

    # THD proxy: energy in harmonics 2 & 3 relative to fundamental
    thd = 0.0
    if fundamental_mag > 1e-9:
        thd = np.sqrt(harmonic2_mag ** 2 + harmonic3_mag ** 2) / fundamental_mag

    return FFTResult(
        freqs=freqs,
        magnitudes=magnitudes,
        sample_rate=sample_rate,
        n_samples=n,
        fundamental_freq=fundamental_freq,
        fundamental_mag=fundamental_mag,
        harmonic2_mag=harmonic2_mag,
        harmonic3_mag=harmonic3_mag,
        total_energy=total_energy,
        high_freq_energy=high_freq_energy,
        thd=thd,
    )


def compute_window_fft(
    window_data,        # Window object or DataFrame
    sample_rate: float = 1.0,
) -> dict[str, FFTResult]:
    """
    Compute FFT for each phase (i1, i2, i3) and i_avg in the window.

    Returns a dict keyed by channel name.
    """
    # Accept both Window objects and raw DataFrames
    df = window_data.data if hasattr(window_data, "data") else window_data

    results: dict[str, FFTResult] = {}
    for col in ("i1", "i2", "i3", "i_avg"):
        if col in df.columns:
            sig = df[col].to_numpy(dtype=np.float64)
            results[col] = compute_fft(sig, sample_rate=sample_rate)
    return results


# ── Batch processing ──────────────────────────────────────────────────────────

def batch_compute_fft(
    windows: list,
    sample_rate: float = 1.0,
) -> list[dict[str, FFTResult]]:
    """Run compute_window_fft for every window. Returns list aligned with windows."""
    return [compute_window_fft(w, sample_rate=sample_rate) for w in windows]
