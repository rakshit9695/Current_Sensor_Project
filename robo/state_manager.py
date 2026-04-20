"""
robo/state_manager.py
Hysteresis / anti-flicker state machine for the EPSON Robot.

A new state is only accepted once it has appeared in N consecutive
windows.  This prevents rapid toggling between states when the signal
is near a classification boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from .rag_classifier import RED, AMBER, GREEN


@dataclass
class StateEntry:
    """Record of one confirmed state change."""
    window_index: int
    state: str
    confidence: float


class StateManager:
    """
    Anti-flicker state machine with hysteresis.

    Parameters
    ----------
    min_consecutive : number of consecutive windows a candidate state
                      must appear before it replaces the current state
    initial_state   : starting state (defaults to RED)
    """

    def __init__(
        self,
        min_consecutive: int = 3,
        initial_state: str = RED,
    ) -> None:
        self.min_consecutive = max(1, min_consecutive)
        self._current_state  = initial_state
        self._candidate      = initial_state
        self._streak         = 0
        self._history: list[StateEntry] = []

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def history(self) -> list[StateEntry]:
        return self._history

    def update(
        self,
        new_state: str,
        window_index: int = 0,
        confidence: float = 1.0,
    ) -> str:
        """
        Feed the raw per-window classification and return the smoothed state.
        """
        if new_state == self._candidate:
            self._streak += 1
        else:
            self._candidate = new_state
            self._streak    = 1

        if self._streak >= self.min_consecutive and new_state != self._current_state:
            self._current_state = new_state
            self._history.append(
                StateEntry(
                    window_index=window_index,
                    state=new_state,
                    confidence=confidence,
                )
            )

        return self._current_state

    def reset(self, initial_state: str = RED) -> None:
        self._current_state = initial_state
        self._candidate     = initial_state
        self._streak        = 0
        self._history.clear()

    def run_batch(
        self,
        raw_states: list[str],
        confidences: list[float] | None = None,
    ) -> list[str]:
        """
        Process a list of raw window states and return the smoothed
        state for every window position.
        """
        if confidences is None:
            confidences = [1.0] * len(raw_states)

        self.reset(initial_state=raw_states[0] if raw_states else RED)
        smoothed: list[str] = []
        for i, (s, c) in enumerate(zip(raw_states, confidences)):
            smoothed.append(self.update(s, window_index=i, confidence=c))
        return smoothed


# ── Utility ───────────────────────────────────────────────────────────────────

def build_state_timeline(
    smoothed_states: list[str],
    window_centres,
) -> list[dict]:
    """
    Build a compact run-length encoded timeline for the state bar chart.
    """
    if not smoothed_states:
        return []

    timeline = []
    current  = smoothed_states[0]
    start_t  = window_centres[0]

    for i in range(1, len(smoothed_states)):
        if smoothed_states[i] != current:
            timeline.append(dict(start=start_t, end=window_centres[i - 1], state=current))
            current = smoothed_states[i]
            start_t = window_centres[i]

    timeline.append(dict(start=start_t, end=window_centres[-1], state=current))
    return timeline
