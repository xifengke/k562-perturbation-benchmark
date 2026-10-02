"""Tests for condition aggregation and prediction alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.evaluation.analysis import condition_centroids  # noqa: E402


class AnalysisTests(unittest.TestCase):
    def test_condition_centroids_are_sorted_and_aligned(self) -> None:
        artifact = {
            "perturbation": np.asarray(["B", "A", "B"]),
            "true_expression": np.asarray([[2, 4], [1, 3], [4, 6]], dtype=float),
            "predicted_expression": np.asarray(
                [[1, 3], [2, 4], [3, 5]], dtype=float
            ),
            "reference": np.asarray([[0, 1], [0, 1], [0, 1]], dtype=float),
        }
        centroids = condition_centroids(artifact)
        np.testing.assert_array_equal(centroids["condition"], ["A", "B"])
        np.testing.assert_allclose(centroids["true_expression"], [[1, 3], [3, 5]])
        np.testing.assert_allclose(
            centroids["predicted_expression"], [[2, 4], [2, 4]]
        )
        np.testing.assert_allclose(centroids["reference"], [[0, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
