"""
data_loader.py
Load and preprocess the soldering furnace CSV file.
- Handles the MM:SS.f wrapping timestamp format
- Computes i_avg from phase currents
- Standardises column names
"""

import pandas as pd
import numpy as np
from pathlib import Path


def _parse_timestamp_str_to_seconds(ts_str: str) -> float:
    """
    Convert a single 'MM:SS.f' string to total seconds (float).
    e.g. '32:32.1' -> 32*60 + 32.1 = 1952.1
    """
    parts = str(ts_str).strip().split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    # Fallback – try direct float conversion
    return float(ts_str)


def _build_sequential_timestamps(raw_ts_series: pd.Series) -> pd.DatetimeIndex:
    """
    Reconstruct monotonic timestamps from wrapping MM:SS.f strings.

    The sensor resets its MM counter every 60 minutes, so we detect
    each wrap-around and add an hour offset accordingly.
    """
    raw_secs = raw_ts_series.map(_parse_timestamp_str_to_seconds).values  # seconds within current hour

    total_secs = np.empty(len(raw_secs), dtype=np.float64)
    hour_offset = 0.0
    prev = raw_secs[0]
    total_secs[0] = hour_offset + prev

    for i in range(1, len(raw_secs)):
        cur = raw_secs[i]
        # Detect wrap: current value is much smaller than previous
        if cur < prev - 30 * 60:       # dropped by more than 30 minutes → wrap
            hour_offset += 3600.0
        total_secs[i] = hour_offset + cur
        prev = cur

    base = pd.Timestamp("2024-01-01 00:00:00")
    return pd.to_datetime(base.value + (total_secs * 1e9).astype(np.int64))


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the soldering furnace CSV and return a clean DataFrame with columns:
        timestamp   – pandas Timestamp (monotonic, 1 s spacing)
        timestamp_str – original text label for display
        i1, i2, i3  – phase RMS currents [A]
        i_avg       – mean of i1/i2/i3 [A]
        frequency   – measured grid frequency [Hz]
    """
    df = pd.read_csv(filepath, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # ── Column name normalisation ──────────────────────────────────────
    rename = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "").replace("_", "")
        if "time" in cl or "stamp" in cl:
            rename[col] = "timestamp_str"
        elif cl in ("i1",):
            rename[col] = "i1"
        elif cl in ("i2",):
            rename[col] = "i2"
        elif cl in ("i3",):
            rename[col] = "i3"
        elif cl in ("freq", "frequency", "hz"):
            rename[col] = "frequency"
    df = df.rename(columns=rename)

    # ── Timestamp reconstruction ───────────────────────────────────────
    if "timestamp_str" in df.columns:
        df["timestamp"] = _build_sequential_timestamps(df["timestamp_str"])
    else:
        # No timestamp column – synthesise from row index
        df["timestamp"] = pd.date_range(
            start="2024-01-01", periods=len(df), freq="1s"
        )
        df["timestamp_str"] = df["timestamp"].dt.strftime("%H:%M:%S")

    # ── Numeric conversion ─────────────────────────────────────────────
    for col in ("i1", "i2", "i3"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            df[col] = 0.0

    if "frequency" in df.columns:
        df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(50.0)
    else:
        df["frequency"] = 50.0

    # ── Derived column ─────────────────────────────────────────────────
    df["i_avg"] = (df["i1"] + df["i2"] + df["i3"]) / 3.0

    # ── Final clean-up ─────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    # Keep only needed columns in a fixed order
    cols = ["timestamp", "timestamp_str", "i1", "i2", "i3", "i_avg", "frequency"]
    df = df[[c for c in cols if c in df.columns]]

    return df


def get_sample_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate in Hz from the median timestamp difference."""
    if len(df) < 2:
        return 1.0
    diffs = df["timestamp"].diff().dropna().dt.total_seconds()
    median_dt = diffs.median()
    return 1.0 / median_dt if median_dt > 0 else 1.0
