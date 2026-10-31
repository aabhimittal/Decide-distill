"""M4: which unlabeled requests to send to the teacher for labels.

Score = normalized entropy of the student's answer (it is unsure) + `weight` × under-representation of the
request's cluster in training (it is somewhere the student has barely been). Entropy alone keeps picking the
same ambiguous boundary cases; the cluster term pulls labels toward regions that drift has opened up.

Not for the shadow sample: agreement monitoring needs a random sample, and this one is biased on purpose.
"""

from __future__ import annotations

import numpy as np


def entropy(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(axis=1)


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm; returns centroids. Empty clusters keep their previous centroid."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    for _ in range(iters):
        assign = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(C)):
            if (assign == j).any():
                C[j] = X[assign == j].mean(0)
    return C


def prioritize(P_pool: np.ndarray, X_pool: np.ndarray, X_train: np.ndarray, budget: int,
               k: int = 8, weight: float = 1.0, seed: int = 0) -> np.ndarray:
    """Indices into the pool, best first, at most `budget` of them."""
    X = np.vstack([X_pool, X_train]).astype(np.float64)
    # Cosine k-means (centre, then unit rows). Per-dimension z-scores were worse: on sparse features a token
    # seen once gets a huge z and clusters form around noise.
    Z = X - X.mean(0)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    C = kmeans(Z, k, seed=seed)
    assign = ((Z[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    a_pool, a_train = assign[: len(X_pool)], assign[len(X_pool):]
    pool_share = np.bincount(a_pool, minlength=len(C)) / len(a_pool)
    train_share = np.bincount(a_train, minlength=len(C)) / max(len(a_train), 1)
    under = pool_share / np.maximum(pool_share + train_share, 1e-12)  # 1: absent from training, 0.5: balanced
    score = entropy(P_pool) / np.log(P_pool.shape[1]) + weight * under[a_pool]
    return np.argsort(-score, kind="stable")[:budget]
