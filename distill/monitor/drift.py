"""M4: drift alarms on live traffic.

Two signals, because they catch different failures:
- PSI over the student's predicted-class distribution vs a reference window: the inputs moved (covariate shift).
  It needs no labels, so it sees every request.
- Rolling agreement with the teacher on a random shadow sample: the right answer moved (concept drift), or the
  student is wrong in a way its own output distribution doesn't show.
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
                 agreement_floor: float | None = None, min_n: int = 100, min_shadow: int = 30):
        """`reference_counts`: predicted-class counts on pre-deployment traffic. Alarms stay off until a window
        holds `min_n` predictions / `min_shadow` shadow labels: PSI and agreement on a handful of rows are noise."""
        self.ref = np.asarray(reference_counts, dtype=float)
        self.preds: deque[int] = deque(maxlen=window)
        self.shadow: deque[bool] = deque(maxlen=window)
        self.psi_alarm, self.agreement_floor = psi_alarm, agreement_floor
        self.min_n, self.min_shadow = min_n, min_shadow

    def observe_prediction(self, k: int) -> None:
        self.preds.append(k)

    def observe_shadow(self, agrees: bool) -> None:
        self.shadow.append(bool(agrees))

    def status(self) -> dict:
        n, m = len(self.preds), len(self.shadow)
        p = psi(self.ref, np.bincount(self.preds, minlength=len(self.ref))) if n else None
        agree = float(np.mean(self.shadow)) if m else None
        return {
            "n": n, "psi": p, "psi_alarm": bool(n >= self.min_n and p > self.psi_alarm),
            "n_shadow": m, "agreement": agree,
            "agreement_alarm": bool(self.agreement_floor is not None and m >= self.min_shadow
                                    and agree < self.agreement_floor),
        }
