"""Diagnose saved five-fold predictions and logs without retraining models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_BASES = {
    "ridge": "baselines",
    "mlp": "baselines",
    "mini_vc": "mini_vc",
    "scgpt_frozen": "scgpt",
    "scgpt_finetuned": "scgpt",
}
FOLDS = [f"unseen_combo_fold{index}" for index in range(5)]


def metric_diagnostics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = root / "results/scgpt_folds"
    summary = pd.read_csv(directory / "model_comparison_131_conditions.csv")
    conditions = pd.read_csv(directory / "all_models_per_condition.csv")
    reference = conditions.loc[conditions.model == "ridge"].set_index("perturbation")
    if len(reference) != 131 or not reference.index.is_unique:
        raise ValueError("expected 131 unique Ridge test conditions")
    rows = []
    rng = np.random.default_rng(42)
    for model in MODEL_BASES:
        selected = conditions.loc[conditions.model == model].set_index("perturbation")
        if not selected.index.is_unique or set(selected.index) != set(reference.index):
            raise ValueError(f"condition mismatch for {model}")
        selected = selected.loc[reference.index]
        differences = (selected.pearson_delta - reference.pearson_delta).to_numpy()
        resampled = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
        low, high = np.quantile(resampled, [0.025, 0.975])
        rows.append({
            "model": model,
            "n_conditions": len(selected),
            "pearson_difference_vs_ridge": differences.mean(),
            "ci_low": low,
            "ci_high": high,
            "pearson_wins_vs_ridge": int((differences > 0).sum()),
            "mse_wins_vs_ridge": int((selected.mse_all < reference.mse_all).sum()),
            "top_100_wins_vs_ridge": int((selected.top_100_recovery > reference.top_100_recovery).sum()),
        })
    # Every fold evaluates 2,000 genes, with DE metrics on the top 100.
    summary["mse_other_1900"] = (2000 * summary.mse_all - 100 * summary.mse_de) / 1900
    return summary, pd.DataFrame(rows)


def training_diagnostics(root: Path) -> pd.DataFrame:
    rows = []
    for scheme in FOLDS:
        samples = pd.read_csv(root / "data/processed/norman_scgpt_folds" / scheme / "samples.csv")
        train = samples.loc[(samples.split == "train") & (samples.perturbation != "control")]
        for model, base in MODEL_BASES.items():
            if model == "ridge":
                continue
            directory = root / "results/scgpt_folds" / base / scheme / model
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            history = pd.read_csv(directory / "training_log.csv")
            selected = history.loc[history.epoch == metrics["best_epoch"]]
            if len(selected) != 1:
                raise ValueError(f"best epoch missing or duplicated: {scheme}/{model}")
            best = selected.iloc[0]
            train_column, validation_column = {
                "mini_vc": ("train_response", "validation_response"),
                "mlp": ("train_loss", "validation_loss"),
                "scgpt_frozen": ("train_response_mse", "validation_response_mse"),
                "scgpt_finetuned": ("train_response_mse", "validation_response_mse"),
            }[model]
            rows.append({
                "scheme": scheme,
                "model": model,
                "n_training_pseudobulks_noncontrol": len(train),
                "n_training_conditions_noncontrol": train.perturbation.nunique(),
                "n_training_gemgroups": train.gemgroup.nunique(),
                "best_epoch": int(metrics["best_epoch"]),
                "epochs_trained": len(history),
                "train_response_mse_at_best_epoch": float(best[train_column]),
                "validation_response_mse_at_best_epoch": float(best[validation_column]),
                "validation_reference_reconstruction_mse": (
                    float(best["validation_reference_reconstruction"]) if model == "mini_vc" else np.nan
                ),
            })
    return pd.DataFrame(rows)


def prediction_geometry(root: Path) -> pd.DataFrame:
    rows = []
    for scheme in FOLDS:
        expected_ids = None
        expected_true = None
        expected_reference = None
        for model, base in MODEL_BASES.items():
            path = root / "results/scgpt_folds" / base / scheme / model / "predictions_test.npz"
            with np.load(path, allow_pickle=False) as saved:
                ids = saved["sample_id"].astype(str)
                labels = saved["perturbation"].astype(str)
                true = saved["true_expression"].astype(np.float64)
                reference = saved["reference"].astype(np.float64)
                predicted = saved["predicted_expression"].astype(np.float64)
            if expected_ids is None:
                expected_ids, expected_true, expected_reference = ids, true, reference
            elif not (
                np.array_equal(ids, expected_ids)
                and np.array_equal(true, expected_true)
                and np.array_equal(reference, expected_reference)
            ):
                raise ValueError(f"saved prediction alignment differs: {scheme}/{model}")
            if not (np.isfinite(true).all() and np.isfinite(predicted).all()):
                raise ValueError(f"nonfinite expression: {scheme}/{model}")
            for label in np.unique(labels):
                mask = labels == label
                true_delta = (true[mask] - reference[mask]).mean(axis=0)
                predicted_delta = (predicted[mask] - reference[mask]).mean(axis=0)
                centered_true = true_delta - true_delta.mean()
                centered_predicted = predicted_delta - predicted_delta.mean()
                true_norm = np.linalg.norm(centered_true)
                if true_norm == 0:
                    raise ValueError(f"zero response variance: {scheme}/{label}")
                rows.append({
                    "scheme": scheme,
                    "model": model,
                    "perturbation": label,
                    "centered_response_norm_ratio": np.linalg.norm(centered_predicted) / true_norm,
                    "response_projection_slope": centered_predicted @ centered_true / true_norm**2,
                })
    result = pd.DataFrame(rows)
    for model, frame in result.groupby("model"):
        if len(frame) != 131 or frame.perturbation.nunique() != 131:
            raise ValueError(f"incomplete prediction geometry: {model}")
    return result


def main() -> None:
    tradeoffs, paired = metric_diagnostics(ROOT)
    training = training_diagnostics(ROOT)
    geometry = prediction_geometry(ROOT)
    output = ROOT / "results/model_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "metric_tradeoffs": tradeoffs,
        "paired_vs_ridge": paired,
        "training_summary": training,
        "prediction_geometry": geometry,
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    print(tradeoffs[["model", "mse_all", "mse_de", "mse_other_1900"]].to_string(index=False))
    print(paired.to_string(index=False))
    print(geometry.groupby("model")["centered_response_norm_ratio"].mean().to_string())
    print(f"Wrote diagnostics to {output}")


if __name__ == "__main__":
    main()
