"""
trainer.py
==========
Training loop, multi-task loss, and checkpoint management.

Multi-task loss
---------------
  L = α·CE(state) + β·MSE(cycle_pos, masked) + γ·CE(job_type, masked)

Samples with cycle_id = -1 (idle / between cycles) have cycle_pos=-1 and
job_type=-1.  These are masked out of the cycle-position and job-type losses
so the model is not penalised for those outputs during idle periods.

Checkpoint format
-----------------
  wave_soldering/ml/checkpoints/model.pt  →  torch.save dict with keys:
    model_state_dict, hparams, norm_stats, label_meta, train_history
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .itransformer import ITransformerModel, build_model
from .dataset import FurnaceDataset, NormStats

CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"


# ── Loss ──────────────────────────────────────────────────────────────────────

def multi_task_loss(
    state_logits: torch.Tensor,   # (B, 3)
    cycle_pos_pred: torch.Tensor, # (B,)
    job_logits: torch.Tensor,     # (B, K)
    state_true: torch.Tensor,     # (B,)  long
    cycle_pos_true: torch.Tensor, # (B,)  float  (-1 = masked)
    job_true: torch.Tensor,       # (B,)  long   (-1 = masked)
    alpha: float = 1.0,
    beta:  float = 0.5,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """
    Returns (total_loss, component_dict_for_logging).
    """
    # State classification (all samples)
    loss_state = F.cross_entropy(state_logits, state_true)

    # Cycle position regression (only valid samples)
    cyc_mask = cycle_pos_true >= 0.0
    if cyc_mask.any():
        loss_cyc = F.mse_loss(cycle_pos_pred[cyc_mask], cycle_pos_true[cyc_mask])
    else:
        loss_cyc = torch.tensor(0.0, device=state_logits.device)

    # Job type classification (only valid samples)
    job_mask = job_true >= 0
    if job_mask.any():
        loss_job = F.cross_entropy(job_logits[job_mask], job_true[job_mask])
    else:
        loss_job = torch.tensor(0.0, device=state_logits.device)

    total = alpha * loss_state + beta * loss_cyc + gamma * loss_job

    return total, dict(
        total  = total.item(),
        state  = loss_state.item(),
        cyc    = loss_cyc.item(),
        job    = loss_job.item(),
    )


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    preds = logits.argmax(dim=-1)
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        return (preds[mask] == targets[mask]).float().mean().item()
    return (preds == targets).float().mean().item()


def evaluate(
    model: ITransformerModel,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 1.0,
    beta:  float = 0.5,
    gamma: float = 0.5,
) -> dict:
    """Run one full evaluation pass and return averaged metrics."""
    model.eval()
    accum = dict(total=0.0, state=0.0, cyc=0.0, job=0.0,
                 state_acc=0.0, job_acc=0.0, cyc_mae=0.0,
                 n_batches=0, n_cyc=0, n_job=0)

    with torch.no_grad():
        for x, st, cp, jt in loader:
            x, st, cp, jt = x.to(device), st.to(device), cp.to(device), jt.to(device)
            sl, cpp, jl = model(x)

            _, comp = multi_task_loss(sl, cpp, jl, st, cp, jt, alpha, beta, gamma)
            for k in ("total", "state", "cyc", "job"):
                accum[k] += comp[k]

            accum["state_acc"] += _accuracy(sl, st)

            cyc_mask = cp >= 0.0
            if cyc_mask.any():
                accum["cyc_mae"] += (cpp[cyc_mask] - cp[cyc_mask]).abs().mean().item()
                accum["n_cyc"] += 1

            job_mask = jt >= 0
            if job_mask.any():
                accum["job_acc"] += _accuracy(jl, jt, mask=job_mask)
                accum["n_job"] += 1

            accum["n_batches"] += 1

    nb = max(accum["n_batches"], 1)
    nc = max(accum["n_cyc"], 1)
    nj = max(accum["n_job"], 1)
    return dict(
        loss       = accum["total"]     / nb,
        loss_state = accum["state"]     / nb,
        loss_cyc   = accum["cyc"]       / nb,
        loss_job   = accum["job"]       / nb,
        state_acc  = accum["state_acc"] / nb,
        cyc_mae    = accum["cyc_mae"]   / nc,
        job_acc    = accum["job_acc"]   / nj,
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    train_ds: FurnaceDataset,
    val_ds:   FurnaceDataset,
    norm_stats: NormStats,
    label_meta: dict,
    *,
    epochs:     int   = 25,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    alpha:      float = 1.0,
    beta:       float = 0.5,
    gamma:      float = 0.5,
    d_model:    int   = 64,
    n_heads:    int   = 4,
    n_layers:   int   = 3,
    d_ff:       int   = 128,
    dropout:    float = 0.1,
    patience:   int   = 7,
    checkpoint_path: Path = CHECKPOINT_PATH,
    progress_cb: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """
    Train the iTransformer model.

    Parameters
    ----------
    train_ds / val_ds  : FurnaceDataset instances
    norm_stats         : normalisation stats (saved in checkpoint)
    label_meta         : dict from label_generator.generate_labels (saved in checkpoint)
    progress_cb        : optional callback(epoch, total_epochs, metrics_dict) for UI updates
                         (called after each epoch)

    Returns
    -------
    history : dict with lists of per-epoch metrics
    """
    device = torch.device("cpu")   # CPU-only; deterministic

    n_job_types = label_meta.get("n_job_types", 3)
    look_back   = train_ds.look_back

    # ── Build model ────────────────────────────────────────────────────
    model = build_model(
        seq_len     = look_back,
        n_job_types = n_job_types,
        d_model     = d_model,
        n_heads     = n_heads,
        n_layers    = n_layers,
        d_ff        = d_ff,
        dropout     = dropout,
    ).to(device)

    # ── Data loaders ───────────────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Optimiser + scheduler ──────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # ── History ────────────────────────────────────────────────────────
    history: dict[str, list] = {
        k: [] for k in (
            "train_loss", "val_loss", "train_state_acc", "val_state_acc",
            "val_cyc_mae", "val_job_acc", "lr"
        )
    }

    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_imp = 0
    best_state_dict = None

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Epoch loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_state_acc = 0.0
        n_batches = 0

        for x, st, cp, jt in train_loader:
            x, st, cp, jt = x.to(device), st.to(device), cp.to(device), jt.to(device)
            optimizer.zero_grad()
            sl, cpp, jl = model(x)
            loss, comp = multi_task_loss(sl, cpp, jl, st, cp, jt, alpha, beta, gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += comp["total"]
            epoch_state_acc += _accuracy(sl, st)
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        train_acc  = epoch_state_acc / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device, alpha, beta, gamma)
        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_state_acc"].append(train_acc)
        history["val_state_acc"].append(val_metrics["state_acc"])
        history["val_cyc_mae"].append(val_metrics["cyc_mae"])
        history["val_job_acc"].append(val_metrics["job_acc"])
        history["lr"].append(current_lr)

        metrics_snapshot = dict(
            epoch       = epoch,
            train_loss  = round(train_loss, 4),
            val_loss    = round(val_metrics["loss"], 4),
            state_acc   = round(val_metrics["state_acc"] * 100, 1),
            job_acc     = round(val_metrics["job_acc"] * 100, 1),
            cyc_mae     = round(val_metrics["cyc_mae"], 4),
            elapsed_sec = round(elapsed, 1),
        )

        if progress_cb is not None:
            progress_cb(epoch, epochs, metrics_snapshot)

        # ── Save best ──────────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss   = val_metrics["loss"]
            best_epoch      = epoch
            epochs_no_imp   = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= patience:
            break

    # ── Restore best weights ───────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── Save checkpoint ────────────────────────────────────────────────
    _save_checkpoint(model, norm_stats, label_meta, history, checkpoint_path)

    return history


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def _save_checkpoint(
    model:     ITransformerModel,
    norm_stats: NormStats,
    label_meta: dict,
    history:   dict,
    path:      Path = CHECKPOINT_PATH,
) -> None:
    # Strip non-serialisable objects from label_meta
    serialisable_meta = {
        k: v for k, v in label_meta.items()
        if k not in ("scaler", "kmeans", "cycle_features", "cycles")
    }
    # Store scaler / kmeans separately as their own dict via pickle-safe repr
    import pickle, base64

    def _pkl_b64(obj) -> str:
        return base64.b64encode(pickle.dumps(obj)).decode("ascii")

    serialisable_meta["scaler_pkl"]  = _pkl_b64(label_meta.get("scaler"))
    serialisable_meta["kmeans_pkl"]  = _pkl_b64(label_meta.get("kmeans"))

    torch.save(
        dict(
            model_state_dict = model.state_dict(),
            hparams          = model.hparams(),
            look_back        = model.seq_len,
            norm_stats       = norm_stats.to_dict(),
            label_meta       = serialisable_meta,
            train_history    = history,
        ),
        path,
    )


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> dict:
    """Load a saved checkpoint and return the raw dict."""
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_from_checkpoint(path: Path = CHECKPOINT_PATH) -> tuple[ITransformerModel, dict]:
    """
    Restore model weights and return (model, checkpoint_dict).
    The checkpoint dict also contains norm_stats and label_meta.
    """
    ckpt = load_checkpoint(path)
    hp   = ckpt["hparams"]

    model = ITransformerModel(
        seq_len    = hp["seq_len"],
        n_vars     = hp.get("n_vars", 4),
        d_model    = hp["d_model"],
        n_heads    = hp["n_heads"],
        n_layers   = hp["n_layers"],
        d_ff       = hp.get("d_ff", hp["d_model"] * 2),
        n_states   = hp.get("n_states", 3),
        n_job_types= hp["n_job_types"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def checkpoint_exists(path: Path = CHECKPOINT_PATH) -> bool:
    return path.exists()
