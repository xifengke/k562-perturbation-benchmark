"""Analyze saved test results and generate comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


MODEL_COLORS = {
    "ridge": "#4C78A8",
    "mlp": "#F58518",
    "mini_vc": "#54A24B",
}
MODEL_LABELS = {"ridge": "Ridge", "mlp": "MLP", "mini_vc": "Mini-VC"}
SCHEMES = ["seen", "unseen_combo", "unseen_gene"]
MODELS = ["ridge", "mlp", "mini_vc"]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _prediction_path(repo_root: Path, scheme: str, model: str) -> Path:
    if model == "mini_vc":
        return (
            repo_root
            / "results"
            / "mini_vc"
            / scheme
            / model
            / "predictions_test.npz"
        )
    return (
        repo_root
        / "results"
        / "baselines"
        / scheme
        / model
        / "predictions_test.npz"
    )


def _condition_metrics_path(repo_root: Path, scheme: str, model: str) -> Path:
    base = "mini_vc" if model == "mini_vc" else "baselines"
    return (
        repo_root
        / "results"
        / base
        / scheme
        / model
        / "metrics_per_condition.csv"
    )


def load_prediction_artifacts(
    repo_root: Path, scheme: str
) -> dict[str, dict[str, np.ndarray]]:
    artifacts = {
        model: dict(np.load(_prediction_path(repo_root, scheme, model)))
        for model in MODELS
    }
    reference = artifacts[MODELS[0]]
    for model in MODELS[1:]:
        current = artifacts[model]
        np.testing.assert_array_equal(current["sample_id"], reference["sample_id"])
        np.testing.assert_array_equal(
            current["perturbation"], reference["perturbation"]
        )
        np.testing.assert_allclose(
            current["true_expression"], reference["true_expression"]
        )
        np.testing.assert_allclose(current["reference"], reference["reference"])
    return artifacts


def condition_centroids(artifact: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    labels = artifact["perturbation"].astype(str)
    conditions = np.sort(np.unique(labels))
    true_rows = []
    predicted_rows = []
    reference_rows = []
    for condition in conditions:
        mask = labels == condition
        true_rows.append(artifact["true_expression"][mask].mean(axis=0))
        predicted_rows.append(artifact["predicted_expression"][mask].mean(axis=0))
        reference_rows.append(artifact["reference"][mask].mean(axis=0))
    return {
        "condition": conditions,
        "true_expression": np.stack(true_rows),
        "predicted_expression": np.stack(predicted_rows),
        "reference": np.stack(reference_rows),
    }


def plot_training_curves(repo_root: Path, output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for axis, scheme in zip(axes, SCHEMES):
        mlp = pd.read_csv(
            repo_root
            / "results"
            / "baselines"
            / scheme
            / "mlp"
            / "training_log.csv"
        )
        mini = pd.read_csv(
            repo_root
            / "results"
            / "mini_vc"
            / scheme
            / "mini_vc"
            / "training_log.csv"
        )
        axis.plot(
            mlp["epoch"],
            mlp["validation_loss"],
            color=MODEL_COLORS["mlp"],
            label="MLP validation",
        )
        axis.plot(
            mlp["epoch"],
            mlp["train_loss"],
            color=MODEL_COLORS["mlp"],
            alpha=0.35,
            linestyle="--",
            label="MLP train",
        )
        axis.plot(
            mini["epoch"],
            mini["validation_response"],
            color=MODEL_COLORS["mini_vc"],
            label="Mini-VC validation",
        )
        axis.plot(
            mini["epoch"],
            mini["train_response"],
            color=MODEL_COLORS["mini_vc"],
            alpha=0.35,
            linestyle="--",
            label="Mini-VC train",
        )
        axis.set_title(scheme.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Response MSE")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Training and validation response loss", fontweight="bold")
    path = output_dir / "training_curves.png"
    _save_figure(figure, path)
    return path


def plot_model_comparison(
    comparison: pd.DataFrame, output_dir: Path
) -> Path:
    selected = comparison[comparison["model"].isin(MODELS)].copy()
    metrics = [
        ("mse_all", "All-gene MSE", "lower is better"),
        ("pearson_delta", "Response Pearson", "higher is better"),
        ("top_100_recovery", "Top-100 DE recovery", "higher is better"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(SCHEMES))
    width = 0.24
    for axis, (metric, label, direction) in zip(axes, metrics):
        for model_index, model in enumerate(MODELS):
            values = [
                selected[
                    (selected["scheme"] == scheme)
                    & (selected["model"] == model)
                ][metric].iloc[0]
                for scheme in SCHEMES
            ]
            axis.bar(
                x + (model_index - 1) * width,
                values,
                width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        axis.set_xticks(x, [name.replace("_", "\n") for name in SCHEMES])
        axis.set_ylabel(label)
        axis.set_title(direction, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        if metric != "mse_all":
            axis.set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=9)
    figure.suptitle("Frozen-test model comparison", fontweight="bold")
    path = output_dir / "model_comparison.png"
    _save_figure(figure, path)
    return path


def load_condition_metrics(repo_root: Path, scheme: str) -> pd.DataFrame:
    merged = None
    for model in MODELS:
        frame = pd.read_csv(_condition_metrics_path(repo_root, scheme, model))
        frame = frame[["perturbation", "pearson_delta", "top_20_recovery"]].rename(
            columns={
                "pearson_delta": f"pearson_delta_{model}",
                "top_20_recovery": f"top_20_recovery_{model}",
            }
        )
        merged = frame if merged is None else merged.merge(frame, on="perturbation")
    assert merged is not None
    return merged.sort_values("perturbation").reset_index(drop=True)


def plot_per_condition(condition_metrics: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(condition_metrics))
    offsets = {"ridge": -0.2, "mlp": 0.0, "mini_vc": 0.2}
    for model in MODELS:
        axis.scatter(
            condition_metrics[f"pearson_delta_{model}"],
            y + offsets[model],
            s=28,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_yticks(y, condition_metrics["perturbation"])
    axis.set_xlabel("Condition-level response Pearson")
    axis.set_xlim(-0.05, 1.02)
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, ncol=1, loc="upper left")
    axis.set_title("All unseen-combination test conditions", fontweight="bold")
    path = output_dir / "unseen_combo_per_condition.png"
    _save_figure(figure, path)
    return path


def plot_delta_scatter(
    artifacts: dict[str, dict[str, np.ndarray]], output_dir: Path
) -> Path:
    centroids = {
        model: condition_centroids(artifact) for model, artifact in artifacts.items()
    }
    values = []
    for model in MODELS:
        centroid = centroids[model]
        values.extend(
            [
                centroid["true_expression"] - centroid["reference"],
                centroid["predicted_expression"] - centroid["reference"],
            ]
        )
    limit = float(np.quantile(np.abs(np.concatenate([v.ravel() for v in values])), 0.995))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        centroid = centroids[model]
        true_delta = centroid["true_expression"] - centroid["reference"]
        predicted_delta = centroid["predicted_expression"] - centroid["reference"]
        axis.hexbin(
            true_delta.ravel(),
            predicted_delta.ravel(),
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        axis.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
        correlation = np.corrcoef(true_delta.ravel(), predicted_delta.ravel())[0, 1]
        axis.set_title(f"{MODEL_LABELS[model]}\nr = {correlation:.3f}")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_xlabel("True delta")
        axis.grid(alpha=0.1)
    axes[0].set_ylabel("Predicted delta")
    figure.subplots_adjust(top=0.76, wspace=0.2)
    figure.suptitle(
        "Unseen-combination condition centroids (99.5% display range)",
        fontweight="bold",
        y=0.98,
    )
    path = output_dir / "unseen_combo_delta_scatter.png"
    _save_figure(figure, path)
    return path


def plot_pca(
    artifacts: dict[str, dict[str, np.ndarray]], output_dir: Path
) -> tuple[Path, np.ndarray]:
    shared = artifacts["ridge"]
    blocks = [shared["reference"], shared["true_expression"]]
    blocks.extend(artifacts[model]["predicted_expression"] for model in MODELS)
    pca = PCA(n_components=2, random_state=42)
    transformed = pca.fit_transform(np.concatenate(blocks))
    block_size = len(shared["reference"])
    coordinates = [
        transformed[index * block_size : (index + 1) * block_size]
        for index in range(len(blocks))
    ]
    reference_coordinates, true_coordinates = coordinates[:2]
    predicted_coordinates = dict(zip(MODELS, coordinates[2:]))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, model in zip(axes, MODELS):
        axis.scatter(
            reference_coordinates[:, 0],
            reference_coordinates[:, 1],
            s=14,
            color="#B8B8B8",
            alpha=0.45,
            label="Reference",
        )
        axis.scatter(
            true_coordinates[:, 0],
            true_coordinates[:, 1],
            s=18,
            color="#222222",
            marker="x",
            alpha=0.7,
            label="True",
        )
        axis.scatter(
            predicted_coordinates[model][:, 0],
            predicted_coordinates[model][:, 1],
            s=18,
            color=MODEL_COLORS[model],
            alpha=0.7,
            label="Predicted",
        )
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle("Unseen-combination pseudobulk PCA", fontweight="bold")
    path = output_dir / "unseen_combo_pca.png"
    _save_figure(figure, path)
    return path, pca.explained_variance_ratio_


def plot_median_condition_de(
    repo_root: Path,
    artifacts: dict[str, dict[str, np.ndarray]],
    condition_metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, str, float]:
    ordered = condition_metrics.sort_values("pearson_delta_mini_vc").reset_index(drop=True)
    selected = ordered.iloc[len(ordered) // 2]
    condition = str(selected["perturbation"])
    genes = pd.read_csv(
        repo_root / "data" / "processed" / "norman_v0_1" / "unseen_combo" / "genes.csv"
    )["gene"].astype(str).to_numpy()

    rows = []
    row_labels = ["True", "Ridge", "MLP", "Mini-VC"]
    for model_index, model in enumerate(MODELS):
        artifact = artifacts[model]
        mask = artifact["perturbation"].astype(str) == condition
        reference = artifact["reference"][mask].mean(axis=0)
        true_delta = artifact["true_expression"][mask].mean(axis=0) - reference
        predicted_delta = (
            artifact["predicted_expression"][mask].mean(axis=0) - reference
        )
        if model_index == 0:
            top_indices = np.argsort(np.abs(true_delta))[::-1][:20]
            rows.append(true_delta[top_indices])
        rows.append(predicted_delta[top_indices])
    matrix = np.stack(rows)
    limit = float(np.abs(matrix).max())
    figure, axis = plt.subplots(figsize=(12, 3.8))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(top_indices)), genes[top_indices], rotation=60, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(
        f"Median Mini-VC condition: {condition} — top 20 true-response genes",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Expression delta")
    path = output_dir / "unseen_combo_de_heatmap_median.png"
    _save_figure(figure, path)
    return path, condition, float(selected["pearson_delta_mini_vc"])


def analyze_results(repo_root: Path) -> dict:
    output_dir = repo_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(
        repo_root / "results" / "mini_vc" / "combined_model_comparison.csv"
    )
    artifacts = load_prediction_artifacts(repo_root, "unseen_combo")
    condition_metrics = load_condition_metrics(repo_root, "unseen_combo")

    figures = {
        "training_curves": plot_training_curves(repo_root, output_dir),
        "model_comparison": plot_model_comparison(comparison, output_dir),
        "per_condition": plot_per_condition(condition_metrics, output_dir),
        "delta_scatter": plot_delta_scatter(artifacts, output_dir),
    }
    pca_path, explained_variance = plot_pca(artifacts, output_dir)
    figures["pca"] = pca_path
    de_path, median_condition, median_condition_pearson = plot_median_condition_de(
        repo_root, artifacts, condition_metrics, output_dir
    )
    figures["de_heatmap"] = de_path

    wins = {}
    ties = {}
    for metric in ["pearson_delta", "top_20_recovery"]:
        columns = [f"{metric}_{model}" for model in MODELS]
        values = condition_metrics[columns].to_numpy(dtype=float)
        maximum = values.max(axis=1, keepdims=True)
        is_winner = np.isclose(values, maximum, rtol=1e-9, atol=1e-12)
        wins[metric] = {
            model: int(is_winner[:, model_index].sum())
            for model_index, model in enumerate(MODELS)
        }
        ties[metric] = int((is_winner.sum(axis=1) > 1).sum())
    summary = {
        "primary_scheme": "unseen_combo",
        "n_test_conditions": int(len(condition_metrics)),
        "median_condition": median_condition,
        "median_condition_mini_vc_pearson_delta": median_condition_pearson,
        "pca_explained_variance_ratio": explained_variance.tolist(),
        "condition_wins": wins,
        "condition_win_ties": ties,
        "figures": {
            name: path.relative_to(repo_root).as_posix()
            for name, path in figures.items()
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
