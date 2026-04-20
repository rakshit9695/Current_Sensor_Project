"""
mv_conveyer/rag_classifier.py
Rule-based RAG classifier for the MV Conveyer.

States (re-calibrated against actual data)
------------------------------------------
RED   – HALT   : Machine completely stopped. I_Avg < 0.10 A.
                 In practice: I1 also near zero, I2 = I3 = 0.
AMBER – IDLE   : Other equipment energised, conveyor motor OFF.
                 I_Avg 0.10–0.35 A, I2 ≈ 0, I3 ≈ 0 (single-phase residual).
GREEN – RUNNING: Conveyor motor active, all three phases engaged.
                 I_Avg ≥ 0.35 A, I2 > 0.05 A, I3 > 0.05 A.

Observed current ranges from two data files
-------------------------------------------
  HALT    :  I_Avg < 0.10  (not in current data, but logically defined)
  IDLE    :  I_Avg 0.12–0.30 A  (I2 ≈ 0, I3 ≈ 0.05 A, I1 ≈ 0.38 A)
  RUNNING :  I_Avg 0.38–0.79 A  (I1 ≈ 0.96–1.48 A, I2 ≈ 0.28–0.59 A,
                                  I3 ≈ 0.25–0.61 A)

Key discriminators
------------------
  1. I_Avg level   – primary separator for all three states
  2. Phase balance – IDLE: I2 + I3 ≈ 0  vs RUNNING: I2 + I3 > 0.1 A
  3. Variance      – RUNNING windows show higher within-window variance
                     due to the 3-phase motor start/stop dynamics
"""

from __future__ import annotations

from dataclasses import dataclass
from .feature_engineering import WindowFeatures

RED   = "RED"
AMBER = "AMBER"
GREEN = "GREEN"

STATE_COLORS = {RED: "#e74c3c", AMBER: "#f39c12", GREEN: "#27ae60"}
STATE_LABELS = {RED: "HALT", AMBER: "IDLE", GREEN: "RUNNING"}

# ── Defaults tuned to observed data ───────────────────────────────────────────
DEFAULT_THRESHOLDS = dict(
    # RED (HALT) ─────────────────────────────────────────────────────────
    red_max_rms      = 0.10,    # A  – I_Avg RMS below this → HALT

    # GREEN (RUNNING) ────────────────────────────────────────────────────
    green_min_rms    = 0.35,    # A  – I_Avg RMS above this → candidate RUNNING
    green_min_thd    = 0.05,    # –  – any THD detected from motor switching
    green_min_variance = 0.002, # A² – within-window variance from motor dynamics
    green_min_imbalance = 0.15, # CV – phase imbalance drops when all 3 phases load
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
      1. RMS < red_max_rms                           → RED  (HALT)
      2. RMS >= green_min_rms  AND phase balance OK  → GREEN (RUNNING)
      3. RMS >= green_min_rms  AND RMS >> threshold  → GREEN (dominant signal)
      4. Everything else                             → AMBER (IDLE)
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    rms   = features.rms_i_avg
    var   = features.variance_i_avg
    thd   = features.thd
    imbal = features.phase_imbalance

    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    # ── Normalised sub-scores (all in 0–1) ────────────────────────────────
    rms_score  = clamp(rms   / max(thr["green_min_rms"],       1e-9))
    var_score  = clamp(var   / max(thr["green_min_variance"],   1e-9))
    thd_score  = clamp(thd   / max(thr["green_min_thd"],        1e-9))
    imb_score  = clamp(imbal / max(thr["green_min_imbalance"],  1e-9))

    scores = dict(
        rms      = round(rms_score, 3),
        variance = round(var_score, 3),
        thd      = round(thd_score, 3),
        imbalance= round(imb_score, 3),
    )

    # ── RED: very low RMS ─────────────────────────────────────────────────
    if rms <= thr["red_max_rms"]:
        confidence = clamp(1.0 - rms / max(thr["red_max_rms"], 1e-9))
        return ClassificationResult(
            state      = RED,
            confidence = round(confidence, 3),
            reason     = f"RMS={rms:.3f} A <= {thr['red_max_rms']} A (HALT threshold)",
            scores     = scores,
        )

    # ── GREEN: RMS above running threshold + at least one quality indicator ──
    if rms >= thr["green_min_rms"]:
        intensity = max(thd_score, var_score, imb_score)

        if intensity >= 0.3:
            confidence = clamp((rms_score + intensity) / 2.0)
            parts = [f"RMS={rms:.3f} A >= {thr['green_min_rms']} A"]
            if thd   >= thr["green_min_thd"]:        parts.append(f"THD={thd:.3f}")
            if var   >= thr["green_min_variance"]:   parts.append(f"Var={var:.4f}")
            if imbal >= thr["green_min_imbalance"]:  parts.append(f"Imb={imbal:.2f}")
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
                reason     = f"RMS={rms:.3f} A >> threshold (3-phase motor dominant)",
                scores     = scores,
            )

    # ── AMBER: intermediate zone (IDLE – other equipment only) ────────────
    span = max(thr["green_min_rms"] - thr["red_max_rms"], 1e-9)
    confidence = clamp((rms - thr["red_max_rms"]) / span)
    return ClassificationResult(
        state      = AMBER,
        confidence = round(confidence, 3),
        reason     = (
            f"RMS={rms:.3f} A in IDLE band "
            f"[{thr['red_max_rms']:.2f}, {thr['green_min_rms']:.2f}] A"
        ),
        scores     = scores,
    )
