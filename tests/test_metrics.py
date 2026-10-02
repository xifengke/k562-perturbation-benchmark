from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.evaluation import evaluate_predictions  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_no_change_has_no_de_ranking(self) -> None:
        samples = pd.DataFrame(
            {
                "split": ["test", "test"],
                "perturbation": ["A", "A"],
            }
        )
        reference = np.zeros((2, 4), dtype=np.float32)
        truth = np.asarray(
            [[2.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        macro, per_condition = evaluate_predictions(
            samples,
            truth,
            reference,
            reference,
            split="test",
            top_ks=[2],
        )
        self.assertIsNone(macro["top_2_recovery"])
        self.assertTrue(np.isnan(per_condition.loc[0, "top_2_recovery"]))

    def test_de_ranking_recovery(self) -> None:
        samples = pd.DataFrame(
            {"split": ["test"], "perturbation": ["A"]}
        )
        reference = np.zeros((1, 4), dtype=np.float32)
        truth = np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
        prediction = np.asarray([[5.0, 2.0, 0.5, 0.1]], dtype=np.float32)
        macro, _ = evaluate_predictions(
            samples,
            truth,
            prediction,
            reference,
            split="test",
            top_ks=[2],
        )
        self.assertEqual(macro["top_2_recovery"], 1.0)


if __name__ == "__main__":
    unittest.main()
