"""
windowing.py
Time-based sliding window generator for the EPSON Robot current sensor data.

Since the data is sampled at 1 Hz (one reading per second), a
"window" is a contiguous slice of N rows where N = window_size_sec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Window:
    """One sliding window's worth of data."""
    index: int                       # sequential window number
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    data: pd.DataFrame               # rows for this window
    start_row: int                   # global row index of first row
    end_row: int                     # global row index of last row (inclusive)


def generate_windows(
    df: pd.DataFrame,
    window_size_sec: float = 30.0,
    overlap: float = 0.50,
    sample_rate: float = 1.0,
) -> list[Window]:
    """
    Produce a list of sliding Window objects.

    Parameters
    ----------
    df              : full dataframe with a 'timestamp' column
    window_size_sec : duration of each window in seconds
    overlap         : fractional overlap in [0, 1)
    sample_rate     : samples per second (used to convert sec → rows)
    """
    if df.empty:
        return []

    window_size_sec = float(window_size_sec)
    step_sec = window_size_sec * (1.0 - overlap)
    min_rows = max(2, int(window_size_sec * sample_rate * 0.5))

    t_start = df["timestamp"].iloc[0]
    t_end   = df["timestamp"].iloc[-1]

    windows: list[Window] = []
    win_idx = 0
    cur_start_sec = 0.0
    t0 = t_start

    while True:
        win_start = t0 + pd.Timedelta(seconds=cur_start_sec)
        win_end   = win_start + pd.Timedelta(seconds=window_size_sec)

        if win_start > t_end:
            break

        mask = (df["timestamp"] >= win_start) & (df["timestamp"] < win_end)
        sub  = df.loc[mask].copy()

        if len(sub) >= min_rows:
            rows = sub.index.tolist()
            windows.append(
                Window(
                    index=win_idx,
                    start_time=win_start,
                    end_time=win_end,
                    data=sub.reset_index(drop=True),
                    start_row=rows[0],
                    end_row=rows[-1],
                )
            )
            win_idx += 1

        cur_start_sec += step_sec

        if win_end >= t_end + pd.Timedelta(seconds=step_sec):
            break

    return windows


def get_window_timestamps(windows: list[Window]) -> np.ndarray:
    """Return array of window centre timestamps (for timeline plots)."""
    centres = [
        w.start_time + (w.end_time - w.start_time) / 2
        for w in windows
    ]
    return np.array(centres, dtype="datetime64[ms]")
