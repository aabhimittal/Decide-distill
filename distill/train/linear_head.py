"""M1: a multinomial logistic-regression head on frozen features."""

from __future__ import annotations

import numpy as np

from distill.calibrate.temperature import softmax


class LinearHead:
    def __init__(self, l2: float = 1e-3, steps: int = 500):
        self.l2, self.steps = l2, steps

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "LinearHead":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # constant features would otherwise divide by zero
        Z = np.hstack([(X - self.mean) / self.std, np.ones((len(X), 1))])
        Y = np.eye(n_classes)[y]
        n = len(Z)
        # Step 1/L, with L bounding the softmax-CE Hessian: 0.5·λmax(ZᵀZ/n) + l2. λmax by power iteration.
        v = np.random.default_rng(0).standard_normal(Z.shape[1])
        for _ in range(50):
            v = Z.T @ (Z @ v) / n
            v /= np.linalg.norm(v)
        lr = 1.0 / (0.5 * float(v @ (Z.T @ (Z @ v)) / n) + self.l2)
        W = prev = np.zeros((Z.shape[1], n_classes))
        for t in range(1, self.steps + 1):  # Nesterov-accelerated gradient descent; the loss is convex
            look = W + (t - 2) / (t + 1) * (W - prev)
            grad = Z.T @ (softmax(Z @ look) - Y) / n + self.l2 * look
            prev, W = W, look - lr * grad
        self.W = W
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.std
        return Z @ self.W[:-1] + self.W[-1]
