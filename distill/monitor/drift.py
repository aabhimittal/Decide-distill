"""M4: drift alarms on live traffic.

- PSI over the student's predicted-class distribution vs a reference window: the inputs moved (covariate shift).
  It needs no labels, so it sees every request.
- Agreement with the teacher, the right answer moving (concept drift), measured on two streams, each against
  its own offline expectation because they are different populations:
  * the random shadow sample of rows the student answered (expected: calib agreement on accepted rows);
  * every fallback, the student's own guess vs the teacher's answer (expected: calib agreement on rejected
    rows). Fallbacks are labeled anyway, and when the gate rejects most traffic they are most of the signal.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def psi(expected, actual, eps: float = 1e-4) -> float:
    """Population Stability Index, Σ (a_i − e_i)·ln(a_i / e_i), over proportions; empty bins floored at `eps`."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e = np.maximum(e / e.sum(), eps)
    a = np.maximum(a / a.sum(), eps)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    def __init__(self, reference_counts, window: int = 500, psi_alarm: float = 0.2,
                 agreement_floor: float | None = None, fallback_agreement_floor: float | None = None,
                 min_n: int = 100, min_shadow: int = 30):
        """`reference_counts`: predicted-class counts on pre-deployment traffic. Alarms stay off until a window
        holds `min_n` predictions / `min_shadow` labels: PSI and agreement on a handful of rows are noise."""
        self.ref = np.asarray(reference_counts, dtype=float)
        self.preds: deque[int] = deque(maxlen=window)
        self.shadow: deque[bool] = deque(maxlen=window)
        self.fallback: deque[bool] = deque(maxlen=window)
        self.psi_alarm, self.min_n, self.min_shadow = psi_alarm, min_n, min_shadow
        self.floors = {"agreement": agreement_floor, "fallback_agreement": fallback_agreement_floor}

    def observe_prediction(self, k: int) -> None:
        self.preds.append(k)

    def observe_shadow(self, agrees: bool) -> None:
        self.shadow.append(bool(agrees))

    def observe_fallback(self, agrees: bool) -> None:
        self.fallback.append(bool(agrees))

    def status(self) -> dict:
        n = len(self.preds)
        p = psi(self.ref, np.bincount(self.preds, minlength=len(self.ref))) if n else None
        out = {"n": n, "psi": p, "psi_alarm": bool(n >= self.min_n and p > self.psi_alarm)}
        for name, rows, count in [("agreement", self.shadow, "n_shadow"),
                                  ("fallback_agreement", self.fallback, "n_fallback")]:
            rate, floor = (float(np.mean(rows)) if rows else None), self.floors[name]
            out |= {count: len(rows), name: rate, f"{name}_alarm": bool(
                floor is not None and len(rows) >= self.min_shadow and rate < floor)}
        return out
