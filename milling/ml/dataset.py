"""
milling/ml/dataset.py
=====================
PyTorch Dataset for the Milling Machine iTransformer.

Each sample returns a look-back window of normalised current readings and
the corresponding multi-task labels (same structure as the other pipelines).

Input tensor  : (look_back, 4)   – I1, I2, I3, I_avg (z-score normalised)
State label   : int  0=Idle / 1=No-Load / 2=Cutting
Cycle pos     : float 0.0–1.0  (or -1.0 → masked in loss)
Job type      : int  0..K-1   (or -1   → masked in loss)
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

class MillingDataset(Dataset):
    """
    Sliding look-back window dataset for the Milling Machine.

    Each index *i* produces window df[i : i+look_back] as input and
    the labels AT position i+look_back-1 as targets.
    """

    FEATURE_COLS = ["i1", "i2", "i3", "i_avg"]

    def __init__(
        self,
        df: pd.DataFrame,
        look_back: int = 30,
        norm_stats: Optional[NormStats] = None,
    ) -> None:
        if len(df) < look_back:
            raise ValueError(
                f"DataFrame has {len(df)} rows but look_back={look_back}. "
                "Need at least look_back rows."
            )

        self.look_back  = look_back
        self.norm_stats = norm_stats or NormStats.from_dataframe(df)

        raw      = df[self.FEATURE_COLS].to_numpy(dtype=np.float32)
        self._x  = self.norm_stats.normalise(raw)

        self._state   = df["state_label"].to_numpy(dtype=np.int64)
        self._cyc_pos = df["cycle_pos"].to_numpy(dtype=np.float32).copy()
        self._job     = df["job_type"].to_numpy(dtype=np.int64)

        nan_mask = np.isnan(self._cyc_pos)
        self._cyc_pos[nan_mask] = -1.0

        self._n = len(df) - look_back + 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        end = idx + self.look_back
        x   = torch.from_numpy(self._x[idx:end])        # (look_back, 4)
        tgt = end - 1

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
    n  = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def build_datasets(
    df_labeled:  pd.DataFrame,
    look_back:   int   = 30,
    train_frac:  float = 0.70,
    val_frac:    float = 0.15,
) -> tuple[MillingDataset, MillingDataset, MillingDataset, NormStats]:
    train_df, val_df, test_df = temporal_split(df_labeled, train_frac, val_frac)
    norm_stats = NormStats.from_dataframe(train_df)
    train_ds = MillingDataset(train_df, look_back=look_back, norm_stats=norm_stats)
    val_ds   = MillingDataset(val_df,   look_back=look_back, norm_stats=norm_stats)
    test_ds  = MillingDataset(test_df,  look_back=look_back, norm_stats=norm_stats)
    return train_ds, val_ds, test_ds, norm_stats
