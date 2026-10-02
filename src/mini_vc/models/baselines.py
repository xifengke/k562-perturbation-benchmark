"""Transparent response-prediction baselines implemented with NumPy."""

from __future__ import annotations

import numpy as np


class MeanResponseBaseline:
    """Predict the mean training perturbation delta for every condition."""

    def __init__(self) -> None:
        self.mean_delta: np.ndarray | None = None

    def fit(self, delta: np.ndarray) -> "MeanResponseBaseline":
        if len(delta) == 0:
            raise ValueError("cannot fit mean response on zero samples")
        self.mean_delta = np.asarray(delta, dtype=np.float64).mean(axis=0)
        return self

    def predict_delta(self, n_samples: int) -> np.ndarray:
        if self.mean_delta is None:
            raise RuntimeError("model is not fitted")
        return np.repeat(self.mean_delta[None, :], n_samples, axis=0).astype(np.float32)


class RidgeBaseline:
    """Multi-output ridge regression from multi-hot gene IDs to expression delta."""

    def __init__(self, alpha: float) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None

    @staticmethod
    def _with_intercept(features: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.asarray(features, dtype=np.float64), np.ones(len(features), dtype=np.float64)]
        )

    def fit(self, features: np.ndarray, delta: np.ndarray) -> "RidgeBaseline":
        design = self._with_intercept(features)
        target = np.asarray(delta, dtype=np.float64)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        penalty[-1, -1] = 0.0
        self.coef_ = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
        return self

    def predict_delta(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        return (self._with_intercept(features) @ self.coef_).astype(np.float32)

