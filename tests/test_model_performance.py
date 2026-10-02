"""Check diagnostic error decomposition and saved prediction alignment."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_model_performance import (  # noqa: E402
    FOLDS,
    MODEL_BASES,
    metric_diagnostics,
    prediction_geometry,
)


class ModelPerformanceTests(unittest.TestCase):
    def test_error_decomposition_and_condition_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results/scgpt_folds"
            output.mkdir(parents=True)
            rows = []
            summary = []
            for index, model in enumerate(MODEL_BASES):
                summary.append({"model": model, "mse_all": 0.002, "mse_de": 0.02})
                labels = list(range(131))
                if index % 2:
                    labels.reverse()
                for label in labels:
                    rows.append({
                        "model": model,
                        "perturbation": f"condition_{label}",
                        "pearson_delta": 0.5 + label / 1000 - index / 100,
                        "mse_all": 0.002 + index / 1000,
                        "top_100_recovery": 0.7,
                    })
            pd.DataFrame(summary).to_csv(output / "model_comparison_131_conditions.csv", index=False)
            pd.DataFrame(rows).to_csv(output / "all_models_per_condition.csv", index=False)
            tradeoffs, paired = metric_diagnostics(root)
            np.testing.assert_allclose(tradeoffs.mse_other_1900, 2 / 1900)
            for index, row in enumerate(paired.itertuples()):
                self.assertAlmostEqual(row.pearson_difference_vs_ridge, -index / 100)
                self.assertEqual(row.pearson_wins_vs_ridge, 0)

    @staticmethod
    def _save_predictions(root: Path):
        for fold_index, scheme in enumerate(FOLDS):
            count = 27 if fold_index == 0 else 26
            ids = np.asarray([f"{scheme}_{index}" for index in range(count)])
            reference = np.full((count, 3), 2.0)
            true = reference + np.asarray([-1.0, 0.0, 1.0])
            for model, base in MODEL_BASES.items():
                output = root / "results/scgpt_folds" / base / scheme / model
                output.mkdir(parents=True)
                # The constant delta offset disappears when centering over genes.
                predicted = reference + np.asarray([-0.5, 0.0, 0.5]) + 3.0
                np.savez_compressed(
                    output / "predictions_test.npz",
                    sample_id=ids,
                    perturbation=ids,
                    reference=reference,
                    true_expression=true,
                    predicted_expression=predicted,
                )

    def test_centered_geometry_is_invariant_to_constant_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._save_predictions(root)
            geometry = prediction_geometry(root)
            self.assertEqual(len(geometry), 131 * len(MODEL_BASES))
            np.testing.assert_allclose(geometry.centered_response_norm_ratio, 0.5)
            np.testing.assert_allclose(geometry.response_projection_slope, 0.5)

    def test_misaligned_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._save_predictions(root)
            path = root / "results/scgpt_folds/baselines" / FOLDS[0] / "mlp/predictions_test.npz"
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
            arrays["sample_id"] = arrays["sample_id"][::-1]
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "alignment differs"):
                prediction_geometry(root)


if __name__ == "__main__":
    unittest.main()
