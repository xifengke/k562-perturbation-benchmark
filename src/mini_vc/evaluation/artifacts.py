"""Save predictions and metric artifacts in one consistent format."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .metrics import evaluate_predictions


def save_evaluation(
    output_dir: Path,
    model_name: str,
    scheme: str,
    dataset,
    predicted_delta: np.ndarray,
    top_ks: list[int],
    extra: dict | None = None,
) -> dict:
    model_dir = output_dir / scheme / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    predicted_expression = dataset.reference + predicted_delta
    macro, per_condition = evaluate_predictions(
        dataset.samples,
        dataset.expression,
        predicted_expression,
        dataset.reference,
        "test",
        top_ks,
    )
    payload = {"model": model_name, "scheme": scheme, **macro, **(extra or {})}
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    per_condition.to_csv(model_dir / "metrics_per_condition.csv", index=False)
    test_mask = dataset.split_mask("test", include_control=False)
    np.savez_compressed(
        model_dir / "predictions_test.npz",
        sample_id=dataset.samples.loc[test_mask, "sample_id"].to_numpy(dtype=str),
        perturbation=dataset.samples.loc[test_mask, "perturbation"].to_numpy(dtype=str),
        true_expression=dataset.expression[test_mask],
        reference=dataset.reference[test_mask],
        predicted_expression=predicted_expression[test_mask],
    )
    return payload

