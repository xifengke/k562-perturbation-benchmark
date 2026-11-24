"""Compare matched single-gene sums with the saved five-fold Mini-VC predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.data.dataset import PseudobulkDataset, load_pseudobulk_dataset  # noqa: E402
from mini_vc.data.splits import perturbation_genes  # noqa: E402
from mini_vc.evaluation import evaluate_predictions  # noqa: E402


PRIMARY_METRICS = {
    "mse_all": "全基因 MSE ↓",
    "pearson_delta": "响应 Pearson ↑",
    "top_100_recovery": "Top-100 recovery ↑",
}
MODELS = {"additive": "单基因相加", "mini_vc": "Mini-VC"}
FLOAT32_CORRELATIONS = ["pearson_all", "pearson_delta", "cosine_delta", "pearson_de"]
CLOSE_THRESHOLDS = {"mse_all": 0.05, "pearson_delta": 0.01, "top_100_recovery": 0.02}


def matched_additive(
    dataset: PseudobulkDataset, test: pd.DataFrame, fold: str
) -> tuple[np.ndarray, pd.DataFrame]:
    samples = dataset.samples
    singles = samples.loc[(samples.split == "train") & (samples.nperts == 1)]
    singles = singles.set_index(["perturbation", "gemgroup"], verify_integrity=True)
    controls = samples.loc[samples.perturbation == "control"].set_index(
        ["split", "gemgroup"], verify_integrity=True
    )["sample_id"]
    positions = pd.Series(samples.index, index=samples.sample_id)
    matched = []
    for row in test.itertuples():
        gene_a, gene_b = perturbation_genes(row.perturbation)
        try:
            a = singles.at[(gene_a, row.gemgroup), "sample_id"]
            b = singles.at[(gene_b, row.gemgroup), "sample_id"]
        except KeyError as error:
            raise ValueError(f"missing matched training single for {fold}/{row.sample_id}") from error
        matched.append({
            "fold": fold,
            "perturbation": row.perturbation,
            "gemgroup": row.gemgroup,
            "test_sample_id": row.sample_id,
            "test_control_sample_id": controls.at[("test", row.gemgroup)],
            "gene_a": gene_a,
            "train_a_sample_id": a,
            "gene_b": gene_b,
            "train_b_sample_id": b,
            "train_control_sample_id": controls.at[("train", row.gemgroup)],
        })
    trace = pd.DataFrame(matched)
    delta_a = dataset.delta[positions.loc[trace.train_a_sample_id]]
    delta_b = dataset.delta[positions.loc[trace.train_b_sample_id]]
    return dataset.reference[test.index] + (delta_a + delta_b), trace


def load_mini_vc_predictions(
    directory: Path, fold: str, dataset: PseudobulkDataset, test: pd.DataFrame
) -> np.ndarray:
    # The NPZ has no gene names. The checkpoint carries the training axis.
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["scheme"] != fold
        or checkpoint["model_name"] != "mini_vc"
        or checkpoint["n_genes"] != len(dataset.genes)
        or checkpoint["genes"] != dataset.genes.gene.astype(str).tolist()
    ):
        raise ValueError(f"Mini-VC checkpoint gene axis or fold mismatch: {fold}")

    with np.load(directory / "predictions_test.npz", allow_pickle=False) as saved:
        expected = {
            "sample_id": test.sample_id.to_numpy(dtype=str),
            "perturbation": test.perturbation.to_numpy(dtype=str),
            "true_expression": dataset.expression[test.index],
            "reference": dataset.reference[test.index],
        }
        for name, values in expected.items():
            np.testing.assert_array_equal(saved[name], values, err_msg=f"{fold}: {name}")
        predicted = saved["predicted_expression"]
    if predicted.shape != expected["true_expression"].shape or predicted.dtype != np.float32:
        raise ValueError(f"unexpected saved prediction shape or dtype: {fold}")
    if not np.isfinite(predicted).all():
        raise ValueError(f"nonfinite Mini-VC predictions: {fold}")
    return predicted


def evaluate_fold(
    config: dict, fold: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    data_dir = ROOT / config["data_dir"] / fold
    directory = ROOT / config["output_dir"] / fold / "mini_vc"
    dataset = load_pseudobulk_dataset(data_dir)
    test = dataset.samples.loc[dataset.split_mask("test", include_control=False)]
    split = json.loads((data_dir / "split_details.json").read_text(encoding="utf-8"))
    other_conditions = set(dataset.samples.loc[dataset.samples.split != "test", "perturbation"])
    if (
        not test.nperts.eq(2).all()
        or test.duplicated(["perturbation", "gemgroup"]).any()
        or set(test.perturbation) != set(split["test_double_conditions"])
        or other_conditions.intersection(test.perturbation)
        or len(dataset.genes) != 2000
    ):
        raise ValueError(f"unexpected held-out split or gene count: {fold}")

    additive, trace = matched_additive(dataset, test, fold)
    predictions = {
        "additive": additive,
        "mini_vc": load_mini_vc_predictions(directory, fold, dataset, test),
    }
    fold_rows = []
    conditions = []
    keys = ["perturbation", "n_pseudobulk_replicates"]
    for model, predicted in predictions.items():
        macro, scores = evaluate_predictions(
            test, dataset.expression[test.index], predicted, dataset.reference[test.index],
            split="test", top_ks=config["de_top_k"],
        )
        if model == "mini_vc":
            saved_macro = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            saved_conditions = pd.read_csv(directory / "metrics_per_condition.csv")
            # A different NumPy/BLAS build can round float32 dot products differently.
            np.testing.assert_allclose(
                scores[FLOAT32_CORRELATIONS], saved_conditions[FLOAT32_CORRELATIONS],
                rtol=0, atol=3e-7, err_msg=f"{fold} historical correlations",
            )
            pd.testing.assert_frame_equal(
                scores.drop(columns=FLOAT32_CORRELATIONS),
                saved_conditions.drop(columns=FLOAT32_CORRELATIONS),
                rtol=1e-7, atol=1e-9, obj=f"{fold} historical conditions",
            )
            for metric, value in macro.items():
                if isinstance(value, float):
                    np.testing.assert_allclose(value, saved_macro[metric], rtol=1e-7, atol=1e-9)
                elif value != saved_macro[metric]:
                    raise ValueError(f"historical fold metric mismatch: {fold}/{metric}")
        conditions.append(scores.set_index(keys).add_prefix(f"{model}_"))
        fold_rows.append({"fold": fold, "model": model, "n_test_records": len(test), **macro})

    pd.testing.assert_index_equal(conditions[0].index, conditions[1].index)
    paired = pd.concat(conditions, axis=1).reset_index()
    paired.insert(0, "fold", fold)
    return paired, trace, fold_rows


def compare_conditions(paired: pd.DataFrame) -> None:
    for metric in PRIMARY_METRICS:
        direction = -1 if metric == "mse_all" else 1
        advantage = direction * (paired[f"mini_vc_{metric}"] - paired[f"additive_{metric}"])
        if not np.isfinite(advantage).all():
            raise ValueError(f"nonfinite paired values for {metric}")
        paired[f"mini_vc_advantage_{metric}"] = advantage
    paired["mini_vc_relative_mse_reduction"] = (
        paired.mini_vc_advantage_mse_all / paired.additive_mse_all
    )
    close_columns = []
    for metric, threshold in CLOSE_THRESHOLDS.items():
        difference = (
            paired.mini_vc_relative_mse_reduction
            if metric == "mse_all" else paired[f"mini_vc_advantage_{metric}"]
        )
        column = f"close_{metric}"
        paired[column] = difference.abs().le(threshold + 1e-12)
        close_columns.append(column)
    paired["close_all_three"] = paired[close_columns].all(axis=1)


def make_report(paired: pd.DataFrame, overall: pd.DataFrame) -> str:
    lines = [
        "# 五折单基因响应相加基线",
        "",
        "## 评估口径",
        "",
        "每条测试记录的预测表达为 `test_control + (train_A - train_control) + (train_B - train_control)`。"
        "两个单基因和训练对照都取自当前折训练集的同一 GEM group，最后加回该记录所在 GEM group 的测试对照。"
        "使用每折原有的 2,000 个基因和原评估函数，按组合等权汇总各 GEM group 的表达，再评分。"
        "总分对 131 个组合做宏平均。Top-100 recovery 按响应绝对值排序。",
        "",
        "逐折核验了测试样本 ID、扰动标签、真实表达和对照矩阵。训练 checkpoint 的基因名与 `genes.csv` 顺序一致。"
        "从 Mini-VC 保存预测复算的逐条件和逐折指标与历史文件一致，逐条件浮点相关性容差为 3×10⁻⁷。",
        "",
        "## 131 个组合的配对比较",
        "",
        "| 指标 | 单基因相加 | Mini-VC | 相加胜 | Mini-VC 胜 | 持平 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    overall = overall.set_index("model")
    for metric, title in PRIMARY_METRICS.items():
        advantage = paired[f"mini_vc_advantage_{metric}"]
        lines.append(
            f"| {title} | {overall.at['additive', metric]:.6f} | {overall.at['mini_vc', metric]:.6f} "
            f"| {(advantage < 0).sum()} | {(advantage > 0).sum()} | {(advantage == 0).sum()} |"
        )

    advantages = paired[[f"mini_vc_advantage_{metric}" for metric in PRIMARY_METRICS]]
    close = paired.loc[paired.close_all_three, "perturbation"]
    mse_reduction = 1 - overall.at["mini_vc", "mse_all"] / overall.at["additive", "mse_all"]
    pearson_gap = overall.at["additive", "pearson_delta"] - overall.at["mini_vc", "pearson_delta"]
    recovery_gap = overall.at["mini_vc", "top_100_recovery"] - overall.at["additive", "top_100_recovery"]
    lines.extend([
        "",
        f"Mini-VC 的宏平均全基因 MSE 下降 {mse_reduction:.1%}，Top-100 recovery 增加 {recovery_gap:.4f}。"
        f"相加基线的宏平均响应 Pearson 高 {pearson_gap:.4f}。",
    ])
    lines.extend(["", "## 组合差异", ""])
    for model, mask in [("additive", (advantages < 0).all(axis=1)), ("mini_vc", (advantages > 0).all(axis=1))]:
        lines.append(f"- {MODELS[model]}在 {mask.sum()} 个组合上三项均胜。")
    lines.append(f"- 按下列描述性阈值，三项同时接近的组合有 {len(close)} 个。")
    for metric, title in PRIMARY_METRICS.items():
        mask = paired[f"close_{metric}"]
        gap = (
            "相对差距 ≤5%" if metric == "mse_all"
            else f"差距 ≤{CLOSE_THRESHOLDS[metric]:.2f}"
        )
        examples = "、".join(paired.loc[mask, "perturbation"].sort_values().head(3))
        name = title.rsplit(" ", 1)[0]
        lines.append(f"- {name} {gap}的组合有 {mask.sum()} 个，例如 {examples}。")
    lines.extend([
        "",
        "下表列出每项指标上各模型优势最大的三个组合。MSE 使用绝对下降量，其他指标使用绝对增量。"
        "逐组合 CSV 中 `mini_vc_advantage_*` 的正值支持 Mini-VC，负值支持相加基线。",
        "",
        "| 指标 | 单基因相加优势最大 | Mini-VC 改善最大 |",
        "|---|---|---|",
    ])
    for metric, title in PRIMARY_METRICS.items():
        column = f"mini_vc_advantage_{metric}"
        values = paired.set_index("perturbation")[column]
        additive = "、".join(f"{label}（{-value:.6f}）" for label, value in values[values.lt(0)].nsmallest(3).items())
        mini_vc = "、".join(f"{label}（{value:.6f}）" for label, value in values[values.gt(0)].nlargest(3).items())
        lines.append(f"| {title} | {additive} | {mini_vc} |")
    lines.extend([
        "",
        "## 复算与边界",
        "",
        "在安装仓库依赖的 Python 环境中，从仓库根目录运行以下命令。",
        "",
        "```powershell",
        "python -B reports/additive_single_gene_baseline/evaluate_additive.py",
        "```",
        "",
        "`per_condition.csv` 保存完整配对分数，`close_*` 列标记接近的组合。"
        "`sample_matches.csv` 记录每条测试记录及其两个训练单基因和对照的样本 ID。"
        "样本 ID 在折内解释。`per_fold.csv` 和 `overall.csv` 保存全部原评估指标。",
        "",
        "结果适用于本次 Norman K562 CRISPRa 五折测试中，同批次两个单基因响应均可用的未见组合。"
        "这里的相加是在 log-normalized pseudobulk 表达空间中进行，性能差异只提供该任务中的预测比较。"
        "组合的相加拟合优劣本身不足以推断生物互作。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    config = json.loads((ROOT / "configs/mini_vc_scgpt_folds.json").read_text(encoding="utf-8"))
    folds = [evaluate_fold(config, fold) for fold in config["schemes"]]
    paired = pd.concat([fold[0] for fold in folds], ignore_index=True)
    trace = pd.concat([fold[1] for fold in folds], ignore_index=True)
    per_fold = pd.DataFrame([row for fold in folds for row in fold[2]])
    if len(paired) != 131 or not paired.perturbation.is_unique or len(trace) != 1048:
        raise ValueError("expected 131 unique test combinations and 1,048 matched test records")
    compare_conditions(paired)

    metrics = per_fold.columns.drop(["fold", "model", "n_test_records", "split", "n_conditions"])
    overall = pd.DataFrame([
        {"model": model, "n_conditions": len(paired), "n_test_records": len(trace),
         **{metric: paired[f"{model}_{metric}"].mean() for metric in metrics}}
        for model in MODELS
    ])
    historical = pd.read_csv(ROOT / "results/scgpt_folds/model_comparison_131_conditions.csv")
    saved = historical.set_index("model").loc["mini_vc"]
    recomputed = overall.set_index("model").loc["mini_vc"]
    for metric in ["n_conditions", *PRIMARY_METRICS, "mse_de", "top_20_recovery"]:
        np.testing.assert_allclose(recomputed[metric], saved[metric], rtol=1e-7, atol=1e-9)

    for name, frame in [("per_condition", paired), ("sample_matches", trace),
                        ("per_fold", per_fold), ("overall", overall)]:
        frame.to_csv(OUTPUT / f"{name}.csv", index=False, lineterminator="\n")
    with (OUTPUT / "conclusion.md").open("w", encoding="utf-8", newline="\n") as report:
        report.write(make_report(paired, overall))
    print(overall[["model", "n_conditions", "n_test_records", *PRIMARY_METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
