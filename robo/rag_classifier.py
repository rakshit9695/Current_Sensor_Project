"""
robo/rag_classifier.py
Rule-based RAG classifier for the EPSON Robot.

States (calibrated from K-means clustering on actual data)
-----------------------------------------------------------
RED   – STANDBY : Robot in standby / minimal draw.
                  I_Avg < 0.085 A  (I1 < ~0.25 A — gap in histogram)
AMBER – IDLE    : Robot energised, not at full operation.
                  0.085 <= I_Avg < 0.115 A  (I1 ~0.30–0.35 A)
GREEN – ACTIVE  : Robot actively running operations.
                  I_Avg >= 0.115 A  (I1 >= ~0.35 A)

Observed current ranges
-----------------------
  I1  : 0.22 – 0.42 A   (RMS current)
  I_Avg: 0.073 – 0.141 A (sensor-reported average, NOT I1/3)
  I2 = I3 = 0 always (single-phase)

Key discriminators
------------------
  1. I_Avg level – primary separator for all three states
  2. Variance    – ACTIVE windows show higher variance from task dynamics
  3. THD         – operational cycle signatures in FFT

Note: Phase imbalance is meaningless (single-phase), not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from .feature_engineering import WindowFeatures

RED   = "RED"
AMBER = "AMBER"
GREEN = "GREEN"

STATE_COLORS = {RED: "#e74c3c", AMBER: "#f39c12", GREEN: "#27ae60"}
STATE_LABELS = {RED: "STANDBY", AMBER: "IDLE", GREEN: "ACTIVE"}

# ── Defaults tuned to observed data ───────────────────────────────────────────
DEFAULT_THRESHOLDS = dict(
    # RED (STANDBY) ──────────────────────────────────────────────────────
    red_max_rms      = 0.085,   # A – I_Avg RMS below this → STANDBY

    # GREEN (ACTIVE) ─────────────────────────────────────────────────────
    green_min_rms    = 0.115,   # A – I_Avg RMS above this → candidate ACTIVE
    green_min_thd    = 0.05,    # – – any THD from task-switching
    green_min_variance = 0.0005,# A² – within-window variance from operations
)


@dataclass
class ClassificationResult:
    state: str
    confidence: float
    reason: str
    scores: dict[str, float]


def classify(
    features: WindowFeatures,
    thresholds: dict | None = None,
) -> ClassificationResult:
    """
    Classify one window's features into RED / AMBER / GREEN.

    Decision logic (in order):
      1. RMS < red_max_rms                        → RED  (STANDBY)
      2. RMS >= green_min_rms AND quality OK       → GREEN (ACTIVE)
      3. RMS >= green_min_rms * 1.10 (strong)      → GREEN (ACTIVE)
      4. Everything else                           → AMBER (IDLE)
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    rms   = features.rms_i_avg
    var   = features.variance_i_avg
    thd   = features.thd

    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    # ── Normalised sub-scores (all in 0–1) ────────────────────────────────
    rms_score  = clamp(rms / max(thr["green_min_rms"],     1e-9))
    var_score  = clamp(var / max(thr["green_min_variance"], 1e-9))
    thd_score  = clamp(thd / max(thr["green_min_thd"],      1e-9))

    scores = dict(
        rms      = round(rms_score, 3),
        variance = round(var_score, 3),
        thd      = round(thd_score, 3),
    )

    # ── RED: very low RMS (STANDBY) ──────────────────────────────────────
    if rms <= thr["red_max_rms"]:
        confidence = clamp(1.0 - rms / max(thr["red_max_rms"], 1e-9))
        return ClassificationResult(
            state      = RED,
            confidence = round(confidence, 3),
            reason     = f"RMS={rms:.4f} A <= {thr['red_max_rms']} A (STANDBY threshold)",
            scores     = scores,
        )

    # ── GREEN: RMS above active threshold + quality indicator ────────────
    if rms >= thr["green_min_rms"]:
        intensity = max(thd_score, var_score)

        if intensity >= 0.3:
            confidence = clamp((rms_score + intensity) / 2.0)
            parts = [f"RMS={rms:.4f} A >= {thr['green_min_rms']} A"]
            if thd >= thr["green_min_thd"]:      parts.append(f"THD={thd:.3f}")
            if var >= thr["green_min_variance"]:  parts.append(f"Var={var:.5f}")
            return ClassificationResult(
                state      = GREEN,
                confidence = round(confidence, 3),
                reason     = "; ".join(parts),
                scores     = scores,
            )

        # RMS clearly above threshold even without secondary indicators
        if rms >= thr["green_min_rms"] * 1.10:
            return ClassificationResult(
                state      = GREEN,
                confidence = round(clamp(rms_score * 0.85), 3),
                reason     = f"RMS={rms:.4f} A >> threshold (ACTIVE dominant)",
                scores     = scores,
            )

    # ── AMBER: intermediate zone (IDLE) ──────────────────────────────────
    span = max(thr["green_min_rms"] - thr["red_max_rms"], 1e-9)
    confidence = clamp((rms - thr["red_max_rms"]) / span)
    return ClassificationResult(
        state      = AMBER,
        confidence = round(confidence, 3),
        reason     = (
            f"RMS={rms:.4f} A in IDLE band "
            f"[{thr['red_max_rms']:.3f}, {thr['green_min_rms']:.3f}] A"
        ),
        scores     = scores,
    )
