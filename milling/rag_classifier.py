"""
milling/rag_classifier.py
Rule-based RAG classifier for the Milling Machine.

States (re-calibrated against actual data)
------------------------------------------
RED   – Idle    : Machine off / control circuitry only. I_Avg < ~1.0 A.
AMBER – No Load : Spindle/feed energised but not cutting.
                  I_Avg ~1.0 – 3.0 A.
GREEN – Cutting : Active material removal, all three phases loaded.
                  I_Avg ≥ ~3.0 A (peaks to ~34 A on heavy cuts).

Observed current ranges (single milling data file)
--------------------------------------------------
  Idle     :  I_Avg < 1.0 A   (~3.5% of samples)
  No Load  :  I_Avg 1.0 – 3.0 A
  Cutting  :  I_Avg 3.0 – 34 A (median running draw ~3.2 A)

Key discriminators
------------------
  1. I_Avg level   – primary separator for all three states
  2. Variance      – cutting windows show higher within-window variance
  3. Phase balance – three balanced phases when running (CV ~0.13)
"""

from __future__ import annotations

from dataclasses import dataclass
from .feature_engineering import WindowFeatures

RED   = "RED"
AMBER = "AMBER"
GREEN = "GREEN"

STATE_COLORS = {RED: "#e74c3c", AMBER: "#f39c12", GREEN: "#27ae60"}
STATE_LABELS = {RED: "Idle", AMBER: "No Load", GREEN: "Cutting"}

# ── Defaults tuned to observed data ───────────────────────────────────────────
DEFAULT_THRESHOLDS = dict(
    # RED (Idle) ─────────────────────────────────────────────────────────
    red_max_rms      = 1.0,     # A  – I_Avg RMS below this → Idle

    # GREEN (Cutting) ────────────────────────────────────────────────────
    green_min_rms    = 3.0,     # A  – I_Avg RMS above this → candidate Cutting
    green_min_thd    = 0.05,    # –  – any THD detected from cutting load
    green_min_variance = 0.05,  # A² – within-window variance from cutting dynamics
    green_min_imbalance = 0.15, # CV – phase imbalance indicator
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
            reason     = f"RMS={rms:.3f} A <= {thr['red_max_rms']} A (Idle threshold)",
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
            f"RMS={rms:.3f} A in No-Load band "
            f"[{thr['red_max_rms']:.2f}, {thr['green_min_rms']:.2f}] A"
        ),
        scores     = scores,
    )
