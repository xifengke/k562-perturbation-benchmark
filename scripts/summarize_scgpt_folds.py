"""Summarize five-fold condition-level predictions without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODELS = {
    "ridge": ROOT / "results/scgpt_folds/baselines",
    "mlp": ROOT / "results/scgpt_folds/baselines",
    "mini_vc": ROOT / "results/scgpt_folds/mini_vc",
    "scgpt_frozen": ROOT / "results/scgpt_folds/scgpt",
    "scgpt_finetuned": ROOT / "results/scgpt_folds/scgpt",
}
FOLDS = [f"unseen_combo_fold{i}" for i in range(5)]
METRICS = ["pearson_delta", "mse_all", "mse_de", "top_20_recovery", "top_100_recovery"]


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        cells = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = np.nanmean(values[indices], axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def load_condition_results() -> pd.DataFrame:
    rows = []
    expected = None
    for model, base in MODELS.items():
        frames = []
        for fold in FOLDS:
            path = base / fold / model / "metrics_per_condition.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame.insert(0, "fold", fold)
            frames.append(frame)
        all_conditions = pd.concat(frames, ignore_index=True)
        if all_conditions.perturbation.duplicated().any():
            raise ValueError(f"a perturbation appears in multiple test folds for {model}")
        labels = set(all_conditions.perturbation)
        if expected is None:
            expected = labels
        elif labels != expected:
            raise ValueError(f"{model} was evaluated on a different condition set")
        all_conditions.insert(0, "model", model)
        rows.append(all_conditions)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    output = ROOT / "results/scgpt_folds"
    output.mkdir(parents=True, exist_ok=True)
    data = load_condition_results()
    data.to_csv(output / "all_models_per_condition.csv", index=False)
    n_conditions = data.loc[data.model == "ridge", "perturbation"].nunique()
    if n_conditions != 131:
        raise ValueError(f"expected 131 tested double conditions, got {n_conditions}")

    rng = np.random.default_rng(42)
    summary = []
    for model in MODELS:
        frame = data.loc[data.model == model]
        row = {"model": model, "n_conditions": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_mean_ci(
                values, rng
            )
        summary.append(row)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output / "model_comparison_131_conditions.csv", index=False)

    comparisons = [
        ("scgpt_frozen", "mlp"),
        ("scgpt_finetuned", "scgpt_frozen"),
        ("scgpt_finetuned", "mlp"),
        ("scgpt_finetuned", "ridge"),
        ("scgpt_finetuned", "mini_vc"),
    ]
    paired = []
    for left, right in comparisons:
        left_data = data.loc[data.model == left].set_index("perturbation")
        right_data = data.loc[data.model == right].set_index("perturbation")
        if set(left_data.index) != set(right_data.index):
            raise ValueError("paired condition labels differ")
        right_data = right_data.loc[left_data.index]
        for metric in METRICS:
            differences = (
                left_data[metric].to_numpy(dtype=float)
                - right_data[metric].to_numpy(dtype=float)
            )
            low, high = bootstrap_mean_ci(differences, rng)
            paired.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "metric": metric,
                    "mean_a_minus_b": float(np.nanmean(differences)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_conditions": len(differences),
                }
            )
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(output / "paired_differences.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary_df))
    means = summary_df.pearson_delta.to_numpy()
    lower = means - summary_df.pearson_delta_ci_low.to_numpy()
    upper = summary_df.pearson_delta_ci_high.to_numpy() - means
    colors = ["#6b7280", "#1976d2", "#7c3aed", "#f59e0b", "#d97706"]
    ax.bar(x, means, color=colors)
    ax.errorbar(x, means, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=4)
    ax.set_xticks(x, summary_df.model, rotation=20, ha="right")
    ax.set_ylabel("Response Pearson (condition macro)")
    ax.set_title("Five-fold held-out double perturbations (131 conditions)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/scgpt_fivefold_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    x_data = data.loc[data.model == "mlp"].set_index("perturbation")["pearson_delta"]
    y_data = data.loc[data.model == "scgpt_finetuned"].set_index("perturbation")["pearson_delta"]
    y_data = y_data.loc[x_data.index]
    ax.scatter(x_data, y_data, alpha=0.65, s=18)
    ax.plot([-1, 1], [-1, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(-0.2, 1.02)
    ax.set_ylim(-0.2, 1.02)
    ax.set_xlabel("MLP response Pearson")
    ax.set_ylabel("Fine-tuned scGPT response Pearson")
    ax.set_title("Paired test conditions")
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/scgpt_vs_mlp_conditions.png", dpi=180)
    plt.close(fig)

    audit = json.loads(
        (ROOT / "results/scgpt_folds/scgpt/unseen_combo_fold0/scgpt_frozen/input_audit.json").read_text(
            encoding="utf-8"
        )
    )
    oov_genes = set(audit["perturbation_coverage"]["oov_genes"])
    mlp_conditions = data.loc[data.model == "mlp"].set_index("perturbation")
    tuned_conditions = data.loc[data.model == "scgpt_finetuned"].set_index("perturbation")
    no_oov = [
        condition
        for condition in mlp_conditions.index
        if not set(condition.split("_")).intersection(oov_genes)
    ]
    no_oov_gap = (
        tuned_conditions.loc[no_oov, "pearson_delta"]
        - mlp_conditions.loc[no_oov, "pearson_delta"]
    ).mean()
    n_oov_test = n_conditions - len(no_oov)
    summary_text = markdown_table(summary_df[["model", "pearson_delta", "mse_all", "top_20_recovery", "top_100_recovery"]])
    primary_paired = markdown_table(paired_df.loc[paired_df.metric == "pearson_delta"])
    report = f"""# scGPT five-fold K562 perturbation benchmark

## Protocol

- Norman 2019 K562 CRISPRa data; the same 2,000-gene pseudobulk target as V0.1.
- Five complete-condition folds: 26 or 27 unseen double perturbations per fold; every one of the 131 doubles is tested once. All component singles remain in training.
- Fold-specific training cells select HVGs. Validation conditions select checkpoints. Test conditions never select hyperparameters.
- Frozen scGPT and partial fine-tuning use the same control cells, gene mapping, action adapter, prediction head, and optimizer settings for the head. Fine-tuning updates the last two transformer layers at a lower learning rate.
- This is a transfer experiment using the official pretrained scGPT encoder with a new condition-level perturbation head. It is not a claim to reproduce scGPT's published perturbation benchmark end to end.
- Each scGPT context uses four real control cells from the matching split and gemgroup; 512 train-selected genes are quantile binned for the 51-bin checkpoint. The response target is still the average of separate perturbed cells, not a paired-cell trajectory.
- The released FlashMHA Wqkv weights are mapped to PyTorch MultiheadAttention in_proj weights. All {audit['checkpoint']['loaded_encoder_tensors']} of {audit['checkpoint']['encoder_parameter_tensors']} encoder parameter tensors were loaded. Checkpoint SHA256: `{audit['checkpoint']['checkpoint_sha256']}`.
- Perturbation gene coverage: {audit['perturbation_coverage']['mapped_perturbation_genes']} of {audit['perturbation_coverage']['total_perturbation_genes']}. ELMSAN1 maps to its approved symbol MIDEAS. OOV genes {', '.join(audit['perturbation_coverage']['oov_genes'])} use separately trainable embeddings in both scGPT arms.

## Results across all 131 held-out combinations

The table is a macro average over all 131 test conditions. Intervals in the CSV are condition-level bootstrap intervals; they describe variation across this dataset and are not external-cohort validation.

{summary_text}

![Five-fold comparison](../results/figures/scgpt_fivefold_comparison.png)

## Paired response Pearson differences

Positive values favor model A. Each difference uses the same held-out condition in both models.

{primary_paired}

![Per-condition scGPT versus MLP](../results/figures/scgpt_vs_mlp_conditions.png)

## Main finding

The original 13-combination test favored MLP. With every double combination tested once, Ridge has the highest mean response Pearson, MLP has the highest top-100 recovery, and Mini-VC has the lowest all-gene MSE. Neither scGPT transfer arm consistently exceeds those simpler models. Partial fine-tuning changes mean response Pearson by only +0.000116 relative to the frozen encoder; its paired condition-bootstrap interval spans zero (-0.003107 to +0.003351). The fine-tuned arm trails MLP by -0.009551 in mean response Pearson (paired interval -0.016493 to -0.003959). These comparisons support reporting the negative result rather than selecting one metric or one favorable fold.

## Interpretation limits

The [model performance analysis](model_performance_analysis.md) examines training size,
metric tradeoffs, response amplitude, and the transfer architecture using saved results.
It separates measured observations from explanations that were not tested by ablation.

- V0.1 test conditions had already been inspected before this extension. Five-fold coverage reduces split sensitivity but does not create an untouched external test cohort.
- The three OOV perturbation genes lack scGPT pretrained gene semantics; their trainable fallbacks are reported explicitly.
- Only {n_oov_test} of 131 held-out double conditions involve an OOV gene. Excluding them, fine-tuned scGPT minus MLP mean response Pearson is {no_oov_gap:+.6f}; OOV coverage alone does not explain the overall gap.
- K562 is a malignant human cell line, whereas the primary scGPT whole-human checkpoint was pretrained on normal human cells.
- The encoder sees only 512 selected genes from four control cells per split and gemgroup. This input budget and the newly initialized perturbation head may limit transfer; the result does not rule out stronger scGPT-based architectures.
- The paired intervals resample conditions within this dataset. Conditions share cells, component genes, and experimental batches, so the intervals are descriptive rather than formal population-level significance claims.
- This experiment predicts condition-level expression means for one CRISPRa dataset. It does not demonstrate single-cell trajectories, other cell lines, or other perturbation technologies.
"""
    (ROOT / "reports/scgpt_fivefold_results.md").write_text(report, encoding="utf-8")
    print(summary_df[["model", "pearson_delta", "mse_all", "top_100_recovery"]].to_string(index=False))
    print(f"Report: {ROOT / 'reports/scgpt_fivefold_results.md'}")


if __name__ == "__main__":
    main()
