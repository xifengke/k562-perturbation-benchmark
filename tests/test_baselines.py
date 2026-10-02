"""Unit tests for NumPy baselines and metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.models import MeanResponseBaseline, RidgeBaseline  # noqa: E402


class BaselineTests(unittest.TestCase):
    def test_mean_response(self) -> None:
        target = np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
        model = MeanResponseBaseline().fit(target)
        prediction = model.predict_delta(3)
        np.testing.assert_allclose(prediction, [[2.0, 4.0]] * 3)

    def test_ridge_recovers_additive_response(self) -> None:
        features = np.asarray(
            [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32
        )
        target = np.asarray(
            [[0, 0], [1, 2], [3, 4], [4, 6]], dtype=np.float32
        )
        model = RidgeBaseline(alpha=1e-8).fit(features, target)
        np.testing.assert_allclose(model.predict_delta(features), target, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

