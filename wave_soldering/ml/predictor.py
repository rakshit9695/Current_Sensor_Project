"""
predictor.py
============
Inference wrapper: slide the trained iTransformer over a DataFrame and
return per-sample predictions.

Returned columns added to a copy of the input DataFrame:
  ml_state        : int   0/1/2   (predicted RAG state)
  ml_state_name   : str   "RED" / "AMBER" / "GREEN"
  ml_state_conf   : float 0-1     (softmax probability of predicted class)
  ml_cycle_pos    : float 0-1     (predicted position in machine cycle)
  ml_job_type     : int   0..K-1  (predicted job cluster)
  ml_job_name     : str   "Job A" / "Job B" / ...
"""

from __future__ import annotations

import base64
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .itransformer import ITransformerModel
from .trainer import CHECKPOINT_PATH, load_model_from_checkpoint
from .dataset import NormStats

# State name mapping (mirrors label_generator constants)
_STATE_NAMES = {0: "RED", 1: "AMBER", 2: "GREEN"}


def _job_names(n: int) -> dict[int, str]:
    """Map cluster index → human label (Low / Medium / High load by convention)."""
    if n == 1:
        return {0: "Standard"}
    if n == 2:
        return {0: "Light Load", 1: "Heavy Load"}
    labels = ["Light Load", "Medium Load", "Heavy Load"]
    extra  = [f"Job {chr(65+i)}" for i in range(n)]
    names  = (labels + extra)[:n]
    return {i: names[i] for i in range(n)}


class FurnacePredictor:
    """
    Wraps a trained ITransformerModel for sliding-window inference.

    Usage
    -----
        predictor = FurnacePredictor.from_checkpoint()
        result_df = predictor.predict(df)
    """

    def __init__(
        self,
        model:      ITransformerModel,
        norm_stats: NormStats,
        look_back:  int,
        n_job_types: int,
    ) -> None:
        self.model       = model.eval()
        self.norm_stats  = norm_stats
        self.look_back   = look_back
        self.n_job_types = n_job_types
        self._job_names  = _job_names(n_job_types)

    @classmethod
    def from_checkpoint(cls, path: Path = CHECKPOINT_PATH) -> "FurnacePredictor":
        model, ckpt = load_model_from_checkpoint(path)
        norm_stats  = NormStats.from_dict(ckpt["norm_stats"])
        look_back   = ckpt["look_back"]
        n_job_types = ckpt["hparams"]["n_job_types"]
        return cls(model, norm_stats, look_back, n_job_types)

    # ── Core inference ─────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        progress_cb=None,  # optional callable(frac) for UI progress
    ) -> pd.DataFrame:
        """
        Run sliding-window inference over *df* and return a new DataFrame
        with prediction columns appended.

        The first (look_back - 1) rows will have NaN predictions since
        there is not yet a full window available.
        """
        FEATURE_COLS = ["i1", "i2", "i3", "i_avg"]

        raw   = df[FEATURE_COLS].to_numpy(dtype=np.float32)     # (N, 4)
        normed = self.norm_stats.normalise(raw)                  # (N, 4)
        N     = len(normed)
        lb    = self.look_back

        if N < lb:
            raise ValueError(
                f"DataFrame has {N} rows but model requires at least "
                f"look_back={lb} rows."
            )

        # Pre-allocate outputs
        n_valid = N - lb + 1
        pred_state     = np.full(N, -1, dtype=np.int32)
        pred_state_conf= np.full(N, np.nan, dtype=np.float32)
        pred_cycle_pos = np.full(N, np.nan, dtype=np.float32)
        pred_job_type  = np.full(N, -1, dtype=np.int32)

        # Build all windows as a single (n_valid, lb, 4) tensor
        # Use stride tricks for memory efficiency on large datasets
        windows = np.ascontiguousarray(
            np.lib.stride_tricks.sliding_window_view(
                normed, window_shape=(lb, normed.shape[1])
            ).squeeze(axis=1)
        )    # (n_valid, lb, 4)  — contiguous, writable copy

        device = torch.device("cpu")

        # Batch inference
        with torch.no_grad():
            for start in range(0, n_valid, batch_size):
                end  = min(start + batch_size, n_valid)
                x    = torch.from_numpy(windows[start:end]).to(device)  # (B, lb, 4)

                sl, cp, jl = self.model(x)

                probs      = F.softmax(sl, dim=-1).numpy()   # (B, 3)
                states_arr = probs.argmax(axis=-1)            # (B,)
                confs      = probs.max(axis=-1)               # (B,)
                cyc_pos    = cp.numpy()                       # (B,)
                jobs       = jl.argmax(dim=-1).numpy()        # (B,)

                # Map back: window ending at (start+lb-1) to (end+lb-2)
                idx = slice(start + lb - 1, end + lb - 1)
                pred_state[idx]      = states_arr
                pred_state_conf[idx] = confs
                pred_cycle_pos[idx]  = cyc_pos
                pred_job_type[idx]   = jobs

                if progress_cb is not None:
                    progress_cb(end / n_valid)

        out = df.copy()
        out["ml_state"]      = pred_state
        out["ml_state_name"] = [
            _STATE_NAMES.get(int(s), "–") if s >= 0 else "–"
            for s in pred_state
        ]
        out["ml_state_conf"] = pred_state_conf
        out["ml_cycle_pos"]  = pred_cycle_pos
        out["ml_job_type"]   = pred_job_type
        out["ml_job_name"]   = [
            self._job_names.get(int(j), "–") if j >= 0 else "–"
            for j in pred_job_type
        ]
        return out

    # ── Evaluation helpers ─────────────────────────────────────────────────

    def evaluate_on_labeled(self, df_labeled: pd.DataFrame, batch_size: int = 256) -> dict:
        """
        Run inference on a labeled DataFrame and return classification metrics.

        Requires 'state_label', 'cycle_pos', 'job_type' columns (from label_generator).
        """
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix
        )

        result = self.predict(df_labeled, batch_size=batch_size)

        # Mask out the first look_back-1 rows (no predictions)
        valid = result["ml_state"] >= 0

        true_state = df_labeled.loc[valid, "state_label"].to_numpy()
        pred_state = result.loc[valid, "ml_state"].to_numpy()

        # State metrics
        state_acc = accuracy_score(true_state, pred_state)
        state_cm  = confusion_matrix(true_state, pred_state, labels=[0, 1, 2])
        state_report = classification_report(
            true_state, pred_state,
            labels=[0, 1, 2],
            target_names=["RED", "AMBER", "GREEN"],
            output_dict=True,
            zero_division=0,
        )

        # Cycle position MAE (valid cycle samples only)
        cyc_true = df_labeled.loc[valid, "cycle_pos"].to_numpy(dtype=np.float32)
        cyc_pred = result.loc[valid, "ml_cycle_pos"].to_numpy(dtype=np.float32)
        cyc_mask = np.isfinite(cyc_true) & (cyc_true >= 0)
        cyc_mae  = float(np.abs(cyc_pred[cyc_mask] - cyc_true[cyc_mask]).mean()) if cyc_mask.any() else float("nan")

        # Job type accuracy (valid samples only)
        job_true = df_labeled.loc[valid, "job_type"].to_numpy()
        job_pred = result.loc[valid, "ml_job_type"].to_numpy()
        job_mask = job_true >= 0
        if job_mask.any():
            job_acc = accuracy_score(job_true[job_mask], job_pred[job_mask])
            job_cm  = confusion_matrix(
                job_true[job_mask], job_pred[job_mask],
                labels=list(range(self.n_job_types))
            )
        else:
            job_acc = float("nan")
            job_cm  = np.zeros((self.n_job_types, self.n_job_types), dtype=int)

        return dict(
            state_acc    = state_acc,
            state_cm     = state_cm,
            state_report = state_report,
            cyc_mae      = cyc_mae,
            job_acc      = job_acc,
            job_cm       = job_cm,
            n_valid      = int(valid.sum()),
        )
