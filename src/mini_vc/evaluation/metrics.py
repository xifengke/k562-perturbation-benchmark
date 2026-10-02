"""Evaluate expression and response vectors with condition-macro metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(
        np.sqrt(np.square(first_centered).sum() * np.square(second_centered).sum())
    )
    if denominator == 0:
        return float("nan")
    return float(first_centered @ second_centered / denominator)


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return _correlation(rankdata(first), rankdata(second))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return float("nan")
    return float(first @ second / denominator)


def _vector_metrics(
    true_expression: np.ndarray,
    predicted_expression: np.ndarray,
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
    de_k: int,
    top_ks: list[int],
) -> dict[str, float]:
    error = predicted_expression - true_expression
    true_de = np.argsort(np.abs(true_delta))[::-1][: min(de_k, len(true_delta))]
    de_error = predicted_delta[true_de] - true_delta[true_de]
    result = {
        "mse_all": float(np.square(error).mean()),
        "rmse_all": float(np.sqrt(np.square(error).mean())),
        "pearson_all": _correlation(true_expression, predicted_expression),
        "spearman_all": _spearman(true_expression, predicted_expression),
        "pearson_delta": _correlation(true_delta, predicted_delta),
        "spearman_delta": _spearman(true_delta, predicted_delta),
        "cosine_delta": _cosine(true_delta, predicted_delta),
        "mse_de": float(np.square(de_error).mean()),
        "pearson_de": _correlation(true_delta[true_de], predicted_delta[true_de]),
        "spearman_de": _spearman(true_delta[true_de], predicted_delta[true_de]),
    }
    true_ranking = np.argsort(np.abs(true_delta))[::-1]
    predicted_magnitudes = np.abs(predicted_delta)
    has_predicted_ranking = not np.allclose(
        predicted_magnitudes,
        predicted_magnitudes[0],
        rtol=0.0,
        atol=1e-12,
    )
    predicted_ranking = np.argsort(predicted_magnitudes)[::-1]
    for k in top_ks:
        effective_k = min(int(k), len(true_delta))
        if not has_predicted_ranking:
            # A constant score contains no ordering information.  Returning an
            # arbitrary overlap (from argsort's tie order) would make the
            # no-change baseline look as though it recovered DE genes.
            result[f"top_{k}_recovery"] = float("nan")
        else:
            overlap = len(
                set(true_ranking[:effective_k].tolist())
                & set(predicted_ranking[:effective_k].tolist())
            )
            result[f"top_{k}_recovery"] = overlap / effective_k
    return result


def evaluate_predictions(
    samples: pd.DataFrame,
    true_expression: np.ndarray,
    predicted_expression: np.ndarray,
    reference: np.ndarray,
    split: str,
    top_ks: list[int],
    de_k: int = 100,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Evaluate condition centroids and macro-average over non-control labels."""
    mask = (samples["split"].to_numpy() == split) & (
        samples["perturbation"].to_numpy() != "control"
    )
    selected = samples.loc[mask].copy()
    true_selected = true_expression[mask]
    predicted_selected = predicted_expression[mask]
    reference_selected = reference[mask]
    rows = []
    for perturbation, indices in selected.groupby("perturbation", observed=True).indices.items():
        condition_true = true_selected[indices].mean(axis=0)
        condition_predicted = predicted_selected[indices].mean(axis=0)
        condition_reference = reference_selected[indices].mean(axis=0)
        metrics = _vector_metrics(
            condition_true,
            condition_predicted,
            condition_true - condition_reference,
            condition_predicted - condition_reference,
            de_k,
            top_ks,
        )
        rows.append(
            {
                "perturbation": perturbation,
                "n_pseudobulk_replicates": len(indices),
                **metrics,
            }
        )
    per_condition = pd.DataFrame(rows).sort_values("perturbation").reset_index(drop=True)
    numeric = per_condition.select_dtypes(include=[np.number]).drop(
        columns=["n_pseudobulk_replicates"], errors="ignore"
    )
    macro: dict[str, float | int | None] = {
        "split": split,
        "n_conditions": int(len(per_condition)),
    }
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        macro[column] = float(finite.mean()) if len(finite) else None
    return macro, per_condition
