"""Create paired five-fold results and a coverage-aware Reactome report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.data.splits import perturbation_genes  # noqa: E402


FOLDS = [f"unseen_combo_fold{i}" for i in range(5)]
METRICS = ["pearson_delta", "mse_all", "top_20_recovery", "top_100_recovery"]
BASE = ROOT / "results/scgpt_folds/baselines"
OUTPUT = ROOT / "results/pathway_knowledge"
SHUFFLE_SEEDS = [1001, 1002, 1003]


def load_conditions(base: Path, model: str) -> pd.DataFrame:
    frames = []
    for fold in FOLDS:
        path = base / fold / model / "metrics_per_condition.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "fold", fold)
        frames.append(frame)
    joined = pd.concat(frames, ignore_index=True)
    if len(joined) != 131 or joined.perturbation.nunique() != 131:
        raise ValueError(f"expected 131 unique double conditions for {base}/{model}")
    return joined.set_index("perturbation").sort_index()


def paired_interval(differences: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10000, len(differences)))
    return tuple(np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]))


def format_table(frame: pd.DataFrame) -> str:
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(rows)


def main() -> None:
    datasets = {
        "ridge": load_conditions(BASE, "ridge"),
        "mlp": load_conditions(BASE, "mlp"),
        "ridge_reactome": load_conditions(OUTPUT / "reactome", "ridge"),
        "mlp_reactome": load_conditions(OUTPUT / "reactome", "mlp"),
    }
    for seed in SHUFFLE_SEEDS:
        for model in ("ridge", "mlp"):
            datasets[f"{model}_shuffle_{seed}"] = load_conditions(
                OUTPUT / f"shuffle_{seed}", model
            )
    reference = datasets["ridge"]["fold"]
    for name, frame in datasets.items():
        if not frame["fold"].equals(reference):
            raise ValueError(f"condition-to-fold assignment differs for {name}")

    for model in ("ridge", "mlp"):
        mean = datasets[model].copy()
        for metric in METRICS:
            mean[metric] = np.mean(
                [datasets[f"{model}_shuffle_{seed}"][metric].to_numpy()
                 for seed in SHUFFLE_SEEDS],
                axis=0,
            )
        datasets[f"{model}_shuffle_mean"] = mean

    all_conditions = pd.concat(
        [frame.assign(model=name).reset_index() for name, frame in datasets.items()],
        ignore_index=True,
    )
    all_conditions.to_csv(OUTPUT / "all_models_per_condition.csv", index=False)

    summary_rows = []
    for name, frame in datasets.items():
        summary_rows.append(
            {"model": name, "n_conditions": len(frame),
             **{metric: float(frame[metric].mean()) for metric in METRICS}}
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "model_comparison_131_conditions.csv", index=False)

    pairs = [
        ("ridge_reactome", "ridge"),
        ("mlp_reactome", "mlp"),
        ("ridge_reactome", "ridge_shuffle_mean"),
        ("mlp_reactome", "mlp_shuffle_mean"),
    ]
    paired_rows = []
    for left, right in pairs:
        for metric in METRICS:
            differences = (
                datasets[left][metric].to_numpy(dtype=float)
                - datasets[right][metric].to_numpy(dtype=float)
            )
            low, high = paired_interval(differences)
            paired_rows.append(
                {"model_a": left, "model_b": right, "metric": metric,
                 "mean_a_minus_b": float(differences.mean()),
                 "ci_low": float(low), "ci_high": float(high),
                 "n_conditions": len(differences)}
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUTPUT / "paired_differences.csv", index=False)

    audit = json.loads(
        (OUTPUT / "reactome/unseen_combo_fold0/pathway_audit.json").read_text(
            encoding="utf-8"
        )
    )
    patterns = [set(row["members_among_perturbations"]) for row in audit["pathway_patterns"]]
    annotated = set().union(*patterns)
    coverage_rows = []
    for condition in reference.index:
        genes = set(perturbation_genes(condition))
        if len(genes) != 2:
            raise ValueError(f"expected double perturbation: {condition}")
        count = len(genes & annotated)
        shared = any(genes <= pattern for pattern in patterns)
        coverage_rows.append(
            {"perturbation": condition, "n_annotated_genes": count,
             "coverage": ["none", "one", "both"][count],
             "shared_pathway": shared}
        )
    coverage = pd.DataFrame(coverage_rows).set_index("perturbation")
    coverage.reset_index().to_csv(OUTPUT / "condition_pathway_coverage.csv", index=False)

    subgroup_rows = []
    for model in ("ridge", "mlp"):
        differences = datasets[f"{model}_reactome"]["pearson_delta"] - datasets[model]["pearson_delta"]
        for group, mask in [
            ("none", coverage.coverage == "none"),
            ("one", coverage.coverage == "one"),
            ("both", coverage.coverage == "both"),
            ("shared", coverage.shared_pathway),
            ("both_no_shared", (coverage.coverage == "both") & ~coverage.shared_pathway),
        ]:
            values = differences[mask]
            subgroup_rows.append(
                {"model": model, "group": group, "n_conditions": len(values),
                 "mean_pearson_change": float(values.mean()),
                 "fraction_improved": float((values > 0).mean())}
            )
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups.to_csv(OUTPUT / "coverage_subgroups.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, model in zip(axes, ("ridge", "mlp")):
        x = datasets[model]["pearson_delta"]
        y = datasets[f"{model}_reactome"]["pearson_delta"]
        colors = coverage["n_annotated_genes"].map(
            {0: "#9ca3af", 1: "#60a5fa", 2: "#1d4ed8"}
        )
        ax.scatter(x, y, c=colors, s=16, alpha=0.7)
        ax.plot([-0.2, 1.02], [-0.2, 1.02], "--", color="black", linewidth=1)
        ax.set_title(model.upper())
        ax.set_xlabel("Original response Pearson")
        ax.set_xlim(-0.2, 1.02)
        ax.set_ylim(-0.2, 1.02)
    axes[0].set_ylabel("Reactome response Pearson")
    fig.suptitle("Paired held-out double perturbations (131 conditions)")
    fig.tight_layout()
    figure_path = ROOT / "results/figures/reactome_vs_original_conditions.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    display_names = ["ridge", "ridge_reactome", "ridge_shuffle_mean",
                     "mlp", "mlp_reactome", "mlp_shuffle_mean"]
    display = summary.set_index("model").loc[display_names].reset_index()
    for metric in ("pearson_delta", "mse_all", "top_100_recovery"):
        display[metric] = display[metric].map(lambda value: f"{value:.4f}")
    summary_table = format_table(
        display[["model", "pearson_delta", "mse_all", "top_100_recovery"]]
    )
    pearson_paired = paired[paired.metric == "pearson_delta"].copy()
    for metric in ("mean_a_minus_b", "ci_low", "ci_high"):
        pearson_paired[metric] = pearson_paired[metric].map(lambda value: f"{value:+.4f}")
    paired_table = format_table(
        pearson_paired[["model_a", "model_b", "mean_a_minus_b", "ci_low", "ci_high"]]
    )
    subgroup_display = subgroups.copy()
    subgroup_display["mean_pearson_change"] = subgroup_display["mean_pearson_change"].map(
        lambda value: f"{value:+.4f}"
    )
    subgroup_display["fraction_improved"] = subgroup_display["fraction_improved"].map(
        lambda value: f"{value:.1%}"
    )
    subgroup_table = format_table(subgroup_display)
    report = f"""# Reactome pathway knowledge in the K562 perturbation benchmark

## 中文速览

Reactome 通路成员关系作为额外扰动特征，沿用原来的五折、
131 个双基因测试组合。它是静态知识，**没有测量 K562 细胞内的蛋白含量**。
通路表覆盖 105 个扰动基因中的 66 个。加入通路后，Ridge 的响应 Pearson 从
{summary.set_index('model').loc['ridge', 'pearson_delta']:.4f} 降至
{summary.set_index('model').loc['ridge_reactome', 'pearson_delta']:.4f}；MLP 从
{summary.set_index('model').loc['mlp', 'pearson_delta']:.4f} 降至
{summary.set_index('model').loc['mlp_reactome', 'pearson_delta']:.4f}。
三个随机打乱通路对应关系的对照也没有支持“真实通路关系能带来增益”。
两个模型的平均表现均低于身份特征基线。配置为 `configs/reactome_pathway.json`，
逐条件指标、配对差异和覆盖统计保存在 `results/pathway_knowledge/`。

## What was added

This is **static protein/pathway knowledge**, not measured protein abundance or
activity in the Norman K562 cells. The source is the [official Reactome human
pathway GMT](https://reactome.org/download/current/ReactomePathways.gmt.zip),
downloaded once and identified by SHA256 `{audit['archive_sha256']}`.

Each perturbation retains its original 105-dimensional gene-identity vector.
For each retained pathway, we add (1) a binary feature that either perturbed
gene belongs to it and (2) a binary feature that both genes belong to it.
Pathways with identical membership among the 105 target genes are collapsed.
The fixed filter retains pathways with 2–30 represented target genes. This
produces {audit['distinct_pathway_patterns']} distinct pathway patterns and
{audit['n_features']} total input features. All features use public annotations
and gene names only; no RNA target or test metric is used to select them.

Reactome annotates {audit['genes_in_selected_patterns']} of {audit['perturbation_genes']}
perturbation genes under this filter. The other
{audit['perturbation_genes'] - audit['genes_in_selected_patterns']} keep only their
identity features. Among 131 tested double combinations,
{int((coverage.coverage == 'none').sum())} have neither gene annotated,
{int((coverage.coverage == 'one').sum())} have one,
{int((coverage.coverage == 'both').sum())} have both, and
{int(coverage.shared_pathway.sum())} share a retained pathway.

## Protocol

- Reuse the original five folds and 2,000-gene pseudobulk targets. Each of the
  131 double combinations is tested exactly once. The previous identity-only
  Ridge and MLP runs are reused without retraining.
- Keep the same Ridge alpha candidates, validation selection, train-plus-validation
  refit, MLP architecture, optimizer, seed, and early stopping as the original runs.
- Fit two added-feature models: Ridge + Reactome and MLP + Reactome.
- Train three negative controls per model. Each permutes the Reactome annotation
  rows across the 105 genes with a fixed seed (1001, 1002, 1003), preserving
  feature dimensions and pathway frequencies while breaking the correct mapping.
- Report test metrics as macro averages over the same 131 conditions. Paired
  bootstrap intervals resample these conditions; shared component genes and
  batches mean the intervals are descriptive, not an independent-cohort test.

## Results

Lower MSE is better; higher Pearson and top-100 recovery are better.
`shuffle_mean` averages the three negative-control predictions' **metrics**
per condition. It is not an ensembled prediction.

{summary_table}

The differences below are `model_a - model_b`; positive response Pearson favors
`model_a`.

{paired_table}

![Reactome versus original condition performance](../results/figures/reactome_vs_original_conditions.png)

## Annotation coverage

The table shows the change in response Pearson relative to each identity-only
model. Coverage groups are descriptive and are not separately tuned models.

{subgroup_table}

## Interpretation

In this implementation, adding Reactome pathway membership **did not improve**
the primary five-fold benchmark. Both models are worse than their identity-only
versions on average. Correct annotations also do not outperform the three
shuffled mappings on average. The result is evidence against this particular
feature design as an improvement for this small K562 pseudobulk task, not
evidence that protein biology is uninformative.

The prior is sparse for these target genes, pathway membership is broad and
context independent, and it does not encode whether a protein is expressed,
phosphorylated, activated, or inhibited in K562. Shared pathway membership
also does not specify the direction of a double-perturbation interaction.
Additional input columns can make fitting harder even when they are biologically
plausible. The 131 test conditions were already used in the earlier V0.2
benchmark; this is a secondary comparison, not a new untouched external test.

## Reproduce

Run from the repository root after installing the dependencies in the README:

```powershell
conda activate mini_vc
python scripts/download_reactome.py
python scripts/train_reactome.py
python scripts/summarize_reactome.py
```

The archive is kept in `data/external/reactome/` and ignored by Git. Final CSV
tables are included in `results/pathway_knowledge/`; pathway audits and model
checkpoints are generated locally by the commands above.
"""
    report_path = ROOT / "reports/reactome_pathway_results.md"
    report_path.write_text(report, encoding="utf-8")
    print(summary[["model", *METRICS]].to_string(index=False))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
