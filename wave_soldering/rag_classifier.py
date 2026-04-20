"""
rag_classifier.py
Rule-based (deterministic) RAG state classifier.

States
------
RED   – Idle      : furnace off / no power draw
AMBER – No Load   : furnace powered, no product / low load
GREEN – Load      : furnace running with full production load

All thresholds are configurable.  No machine learning is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from .feature_engineering import WindowFeatures

# ── State constants ────────────────────────────────────────────────────────────
RED   = "RED"
AMBER = "AMBER"
GREEN = "GREEN"

STATE_COLORS = {
    RED  : "#e74c3c",
    AMBER: "#f39c12",
    GREEN: "#27ae60",
}

STATE_LABELS = {
    RED  : "Idle",
    AMBER: "No Load",
    GREEN: "Load",
}

# ── Default thresholds ────────────────────────────────────────────────────────
#
# Tuned for the soldering furnace data where:
#   - Idle:    i_avg ≈ 0.2 – 1.5 A  (heaters off, only control circuitry)
#   - No Load: i_avg ≈ 2 – 12 A     (heaters cycling without product)
#   - Load:    i_avg ≈ 12 – 24 A    (heaters running at full duty cycle)
#
# All thresholds exposed so the Streamlit UI can override them.

DEFAULT_THRESHOLDS = dict(
    # ── RED (Idle) ──────────────────────────────────────────────────────
    red_max_rms          = 2.0,    # A  – mean RMS below this → RED
    red_max_variance     = 1.0,    # A² – variance below this → favour RED

    # ── GREEN (Load) ────────────────────────────────────────────────────
    green_min_rms        = 12.0,   # A  – mean RMS above this → GREEN candidate
    green_min_thd        = 0.15,   # THD above → stronger GREEN signal
    green_min_variance   = 5.0,    # A² – variance above → stronger GREEN signal
    green_min_imbalance  = 0.20,   # phase CV above → stronger GREEN signal

    # ── AMBER sits between RED and GREEN ───────────────────────────────
    # (no extra thresholds needed – AMBER is the fallthrough state)
)


@dataclass
class ClassificationResult:
    state: str                # RED | AMBER | GREEN
    confidence: float         # 0–1 heuristic confidence
    reason: str               # human-readable explanation
    scores: dict[str, float]  # component scores used


def classify(
    features: WindowFeatures,
    thresholds: dict | None = None,
) -> ClassificationResult:
    """
    Deterministic rule-based RAG classification.

    Logic (priority order):
      1. If RMS is very low  → RED
      2. If RMS is high AND at least one intensity indicator is high → GREEN
      3. Otherwise           → AMBER
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    rms   = features.rms_i_avg
    var   = features.variance_i_avg
    thd   = features.thd
    imbal = features.phase_imbalance

    # ── Component scores (0 = no signal, 1 = strong signal) ───────────
    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    rms_score  = clamp(rms / max(thr["green_min_rms"], 1e-9))
    var_score  = clamp(var / max(thr["green_min_variance"], 1e-9))
    thd_score  = clamp(thd / max(thr["green_min_thd"], 1e-9))
    imb_score  = clamp(imbal / max(thr["green_min_imbalance"], 1e-9))

    scores = dict(
        rms=round(rms_score, 3),
        variance=round(var_score, 3),
        thd=round(thd_score, 3),
        imbalance=round(imb_score, 3),
    )

    # ── Rule 1: RED ────────────────────────────────────────────────────
    if rms <= thr["red_max_rms"] and var <= thr["red_max_variance"]:
        confidence = clamp(1.0 - rms / max(thr["red_max_rms"], 1e-9))
        return ClassificationResult(
            state=RED,
            confidence=round(confidence, 3),
            reason=f"RMS={rms:.2f} A <= {thr['red_max_rms']} A, Var={var:.2f} A2 <= {thr['red_max_variance']} A2",
            scores=scores,
        )

    # ── Rule 2: GREEN ──────────────────────────────────────────────────
    rms_green = rms >= thr["green_min_rms"]
    intensity = max(thd_score, var_score, imb_score)   # at least one indicator high

    if rms_green and intensity >= 0.5:
        confidence = clamp((rms_score + intensity) / 2.0)
        reason_parts = [f"RMS={rms:.2f} A >= {thr['green_min_rms']} A"]
        if thd >= thr["green_min_thd"]:
            reason_parts.append(f"THD={thd:.3f}")
        if var >= thr["green_min_variance"]:
            reason_parts.append(f"Var={var:.2f} A²")
        if imbal >= thr["green_min_imbalance"]:
            reason_parts.append(f"Imbalance={imbal:.2f}")
        return ClassificationResult(
            state=GREEN,
            confidence=round(confidence, 3),
            reason="; ".join(reason_parts),
            scores=scores,
        )

    # ── Rule 2b: GREEN with very high RMS even without other indicators ─
    if rms >= thr["green_min_rms"] * 1.2:
        confidence = clamp(rms_score * 0.8)
        return ClassificationResult(
            state=GREEN,
            confidence=round(confidence, 3),
            reason=f"RMS={rms:.2f} A >> {thr['green_min_rms']} A (dominant)",
            scores=scores,
        )

    # ── Rule 3: AMBER (fallthrough) ────────────────────────────────────
    # Confidence based on how far RMS is from the RED boundary
    confidence = clamp((rms - thr["red_max_rms"]) / max(thr["green_min_rms"] - thr["red_max_rms"], 1e-9))
    return ClassificationResult(
        state=AMBER,
        confidence=round(confidence, 3),
        reason=(
            f"RMS={rms:.2f} A between thresholds "
            f"[{thr['red_max_rms']}, {thr['green_min_rms']}]"
        ),
        scores=scores,
    )
