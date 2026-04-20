"""
dataset.py
==========
PyTorch Dataset for the iTransformer.

Each sample returns a look-back window of normalised current readings and
the corresponding multi-task labels.

Input tensor  : (look_back, 4)   – I1, I2, I3, I_avg (z-score normalised)
State label   : int   0/1/2
Cycle pos     : float 0.0–1.0  (or -1.0 → masked in loss)
Job type      : int   0..K-1   (or -1   → masked in loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── Normalisation stats ───────────────────────────────────────────────────────

@dataclass
class NormStats:
    """Per-channel mean and std computed on the training split."""
    mean: np.ndarray   # shape (4,)  – I1, I2, I3, I_avg
    std:  np.ndarray   # shape (4,)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """x shape: (T, 4) → normalised (T, 4)."""
        return (x - self.mean) / np.where(self.std < 1e-8, 1.0, self.std)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(mean=np.array(d["mean"], dtype=np.float32),
                   std =np.array(d["std"],  dtype=np.float32))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "NormStats":
        cols = ["i1", "i2", "i3", "i_avg"]
        arr = df[cols].to_numpy(dtype=np.float32)
        return cls(mean=arr.mean(axis=0), std=arr.std(axis=0))


# ── Dataset ───────────────────────────────────────────────────────────────────

class FurnaceDataset(Dataset):
    """
    Sliding look-back window dataset.

    Each index *i* produces the window  df[i : i+look_back]  as input
    and the labels AT position  i+look_back-1  as targets.

    Samples where the look-back window straddles a data discontinuity
    (cycle_id changes) are NOT excluded — the model is expected to learn
    from these transition windows too.
    """

    FEATURE_COLS = ["i1", "i2", "i3", "i_avg"]

    def __init__(
        self,
        df: pd.DataFrame,
        look_back: int = 120,
        norm_stats: Optional[NormStats] = None,
    ) -> None:
        """
        Parameters
        ----------
        df         : labeled DataFrame (output of label_generator.generate_labels)
        look_back  : window length in samples (seconds at 1 Hz)
        norm_stats : normalisation stats; if None, computed from *this* df
        """
        if len(df) < look_back:
            raise ValueError(
                f"DataFrame has {len(df)} rows but look_back={look_back}. "
                "Need at least look_back rows."
            )

        self.look_back = look_back
        self.norm_stats = norm_stats or NormStats.from_dataframe(df)

        # Build raw arrays once – faster than per-sample DataFrame slicing
        raw = df[self.FEATURE_COLS].to_numpy(dtype=np.float32)          # (N, 4)
        self._x = self.norm_stats.normalise(raw)                         # (N, 4)

        self._state   = df["state_label"].to_numpy(dtype=np.int64)       # (N,)
        self._cyc_pos = df["cycle_pos"].to_numpy(dtype=np.float32).copy() # (N,)  NaN → -1
        self._job     = df["job_type"].to_numpy(dtype=np.int64)          # (N,)

        # Replace NaN cycle positions with -1 sentinel
        nan_mask = np.isnan(self._cyc_pos)
        self._cyc_pos[nan_mask] = -1.0

        # Number of valid windows
        self._n = len(df) - look_back + 1

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        x          : FloatTensor (look_back, 4)
        state      : LongTensor  ()
        cycle_pos  : FloatTensor ()  – -1 if not in an active cycle
        job_type   : LongTensor  ()  – -1 if not in an active cycle
        """
        end = idx + self.look_back
        x   = torch.from_numpy(self._x[idx:end])        # (look_back, 4)
        tgt = end - 1                                     # label at last step

        state     = torch.tensor(self._state[tgt],   dtype=torch.long)
        cycle_pos = torch.tensor(self._cyc_pos[tgt], dtype=torch.float32)
        job_type  = torch.tensor(self._job[tgt],     dtype=torch.long)

        return x, state, cycle_pos, job_type


# ── Temporal train / val / test split ─────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a labeled DataFrame preserving temporal order.
    Returns (train_df, val_df, test_df).
    """
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def build_datasets(
    df_labeled: pd.DataFrame,
    look_back: int = 120,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple[FurnaceDataset, FurnaceDataset, FurnaceDataset, NormStats]:
    """
    Convenience wrapper: split → compute norm stats on train → build datasets.

    Returns (train_ds, val_ds, test_ds, norm_stats)
    """
    train_df, val_df, test_df = temporal_split(df_labeled, train_frac, val_frac)

    # Norm stats computed on training data only (prevent leakage)
    norm_stats = NormStats.from_dataframe(train_df)

    train_ds = FurnaceDataset(train_df, look_back=look_back, norm_stats=norm_stats)
    val_ds   = FurnaceDataset(val_df,   look_back=look_back, norm_stats=norm_stats)
    test_ds  = FurnaceDataset(test_df,  look_back=look_back, norm_stats=norm_stats)

    return train_ds, val_ds, test_ds, norm_stats
