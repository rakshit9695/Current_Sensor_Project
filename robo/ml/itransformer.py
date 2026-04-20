"""
robo/ml/itransformer.py
iTransformer architecture for the EPSON Robot.

Identical architecture to the conveyer/furnace version, but with n_vars=2
(I1, I_Avg) since I2 and I3 are always zero for the single-phase robot.

Tasks
-----
  1. state_logits  : (B, 3)  – STANDBY / IDLE / ACTIVE classification
  2. cycle_pos     : (B,)    – 0-1 regression (position in active segment)
  3. job_logits    : (B, K)  – job-type classification
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class VariateEmbedding(nn.Module):
    """
    Project each variate's time series from R^T → R^d_model.

    Input  : (B, N, T)
    Output : (B, N, d_model)
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj    = nn.Linear(seq_len, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        out = self.norm(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class iTransformerEncoderLayer(nn.Module):
    """
    One iTransformer encoder layer.
    Attention is over the variate axis (N tokens).
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        d_ff:     int,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────────

class ITransformerModel(nn.Module):
    """
    iTransformer with three task heads.

    Parameters
    ----------
    seq_len      : look-back window length (T)
    n_vars       : number of input variates — 2 for robot (I1, I_Avg)
    d_model      : embedding / hidden dimension
    n_heads      : number of attention heads
    n_layers     : number of encoder layers
    d_ff         : feed-forward inner dimension
    dropout      : dropout probability
    n_states     : output classes for state head (3: STANDBY/IDLE/ACTIVE)
    n_job_types  : output classes for job head (K)
    """

    def __init__(
        self,
        seq_len:    int   = 120,
        n_vars:     int   = 2,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 3,
        d_ff:       int   = 128,
        dropout:    float = 0.1,
        n_states:   int   = 3,
        n_job_types: int  = 3,
    ) -> None:
        super().__init__()

        self.seq_len    = seq_len
        self.n_vars     = n_vars
        self.d_model    = d_model
        self.d_ff       = d_ff
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.n_states   = n_states
        self.n_job_types= n_job_types

        self.embedding = VariateEmbedding(seq_len, d_model, dropout)

        self.encoder = nn.ModuleList([
            iTransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        shared_dim = n_vars * d_model
        self.shared_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        proj_out = shared_dim // 2

        self.state_head = nn.Sequential(
            nn.Linear(proj_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_states),
        )

        self.cycle_pos_head = nn.Sequential(
            nn.Linear(proj_out, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.job_head = nn.Sequential(
            nn.Linear(proj_out, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_job_types),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, N) – look-back window, normalised

        Returns
        -------
        state_logits : (B, n_states)
        cycle_pos    : (B,)
        job_logits   : (B, n_job_types)
        """
        B, T, N = x.shape
        x = x.transpose(1, 2)                  # (B, N, T)
        x = self.embedding(x)                   # (B, N, d_model)

        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)

        x = x.reshape(B, -1)                    # (B, N*d_model)
        shared = self.shared_proj(x)

        state_logits = self.state_head(shared)
        cycle_pos    = self.cycle_pos_head(shared).squeeze(-1)
        job_logits   = self.job_head(shared)

        return state_logits, cycle_pos, job_logits

    def hparams(self) -> dict:
        return dict(
            seq_len    = self.seq_len,
            n_vars     = self.n_vars,
            d_model    = self.d_model,
            n_heads    = self.n_heads,
            n_layers   = self.n_layers,
            d_ff       = self.d_ff,
            n_states   = self.n_states,
            n_job_types= self.n_job_types,
        )


# ── Convenience factory ────────────────────────────────────────────────────────

def build_model(
    seq_len:     int   = 120,
    n_job_types: int   = 3,
    d_model:     int   = 64,
    n_heads:     int   = 4,
    n_layers:    int   = 3,
    d_ff:        int   = 128,
    dropout:     float = 0.1,
) -> ITransformerModel:
    return ITransformerModel(
        seq_len    = seq_len,
        n_vars     = 2,          # I1, I_Avg only (single-phase robot)
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = n_layers,
        d_ff       = d_ff,
        dropout    = dropout,
        n_states   = 3,
        n_job_types= n_job_types,
    )
