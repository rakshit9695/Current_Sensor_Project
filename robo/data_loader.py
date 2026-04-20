"""
robo/data_loader.py
Load and preprocess EPSON Robot CSV data.

Key facts about the source data:
  - Single-phase: only I1 carries current; I2 = I3 = 0 always
  - I1 is the RMS current reading
  - I_Avg in the CSV is a sensor-reported average (NOT (I1+I2+I3)/3)
    → we keep it as-is rather than recomputing
  - Timestamps are proper ISO datetimes (no wrapping)
  - Current range: I1 ≈ 0.22–0.42 A, I_Avg ≈ 0.07–0.14 A
  - Single CSV file per session
"""

import pandas as pd
import numpy as np
from pathlib import Path


def _load_single(src) -> pd.DataFrame:
    """
    Load one CSV (file path string, Path, or file-like object) and return
    a normalised DataFrame with standardised column names.
    """
    df = pd.read_csv(src, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # ── Column name normalisation ──────────────────────────────────────
    rename = {}
    for col in df.columns:
        cl = col.lower().replace(" ", "").replace("_", "")
        if cl in ("timestamp", "time", "datetime"):
            rename[col] = "timestamp_str"
        elif cl == "i1":
            rename[col] = "i1"
        elif cl == "i2":
            rename[col] = "i2"
        elif cl == "i3":
            rename[col] = "i3"
        elif cl in ("iavg", "i_avg", "iaverage"):
            rename[col] = "i_avg_src"
        elif cl in ("freq", "frequency", "hz"):
            rename[col] = "frequency"
    df = df.rename(columns=rename)

    # ── Parse ISO datetime timestamps ──────────────────────────────────
    if "timestamp_str" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_str"], errors="coerce")
        if df["timestamp"].isna().any():
            n_bad = df["timestamp"].isna().sum()
            print(f"[data_loader] Warning: {n_bad} timestamp failures; using index fallback.")
            base = df["timestamp"].dropna().iloc[0] if not df["timestamp"].dropna().empty \
                   else pd.Timestamp("2026-01-01")
            fallback = pd.Series([base + pd.Timedelta(seconds=i) for i in range(len(df))])
            df["timestamp"] = df["timestamp"].fillna(fallback)
    else:
        df["timestamp"] = pd.date_range(start="2026-01-01", periods=len(df), freq="1s")
        df["timestamp_str"] = df["timestamp"].dt.strftime("%H:%M:%S")

    if "timestamp_str" not in df.columns:
        df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

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

    # ── Use the CSV's own I_Avg (sensor-reported, not recomputed) ──────
    if "i_avg_src" in df.columns:
        df["i_avg"] = pd.to_numeric(df["i_avg_src"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["i_avg"] = (df["i1"] + df["i2"] + df["i3"]) / 3.0

    return df


def load_data(source) -> pd.DataFrame:
    """
    Load EPSON Robot data from a single CSV source, return a clean DataFrame.

    Parameters
    ----------
    source : str | Path | file-like
        Single CSV path or file-like object.

    Returns
    -------
    DataFrame with columns:
        timestamp, timestamp_str, i1, i2, i3, i_avg, frequency
    """
    df = _load_single(source)

    # ── Sort by timestamp ──────────────────────────────────────────────
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Final column selection ─────────────────────────────────────────
    cols = ["timestamp", "timestamp_str", "i1", "i2", "i3", "i_avg", "frequency"]
    df = df[[c for c in cols if c in df.columns]]

    return df


def get_sample_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate in Hz from median timestamp difference."""
    if len(df) < 2:
        return 1.0
    diffs = df["timestamp"].diff().dropna().dt.total_seconds()
    median_dt = diffs.median()
    return 1.0 / median_dt if median_dt > 0 else 1.0
