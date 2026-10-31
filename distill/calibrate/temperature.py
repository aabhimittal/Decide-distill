"""Post-hoc temperature scaling and expected calibration error."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """T minimizing held-out NLL of softmax(logits / T).

    NLL is convex in 1/T, so a golden-section search over 1/T finds the global minimum in the bracket.
    """
    a, b = 1 / hi, 1 / lo
    g = (np.sqrt(5) - 1) / 2
    for _ in range(100):
        c, d = b - g * (b - a), a + g * (b - a)
        if nll(logits, y, 1 / c) < nll(logits, y, 1 / d):
            b = d
        else:
            a = c
    return float(2 / (a + b))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: |accuracy − confidence| per confidence bin, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
