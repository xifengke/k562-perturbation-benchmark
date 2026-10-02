"""Generate the Mini-VC report from saved comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _format(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.4f}"


def _markdown_table(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, table.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def build_report(comparison_path: Path) -> str:
    results = pd.read_csv(comparison_path)
    selected = results[
        results["model"].isin(["ridge", "mlp", "mini_vc"])
    ].copy()
    columns = [
        "scheme",
        "model",
        "n_conditions",
        "mse_all",
        "pearson_delta",
        "pearson_de",
        "top_100_recovery",
    ]
    table = selected[columns].copy()
    for column in columns[3:]:
        table[column] = table[column].map(_format)
    table.columns = [
        "拆分",
        "模型",
        "测试条件数",
        "全基因 MSE↓",
        "响应 Pearson↑",
        "Top-100 DE Pearson↑",
        "Top-100 恢复率↑",
    ]
    lines = [
        "# Mini-VC 训练与评价",
        "",
        "本报告由 `scripts/summarize_mini_vc.py` 直接读取 "
        "`results/mini_vc/combined_model_comparison.csv` 生成。",
        "",
        _markdown_table(table),
        "",
        "## 按拆分解读",
        "",
    ]
    for scheme in selected["scheme"].unique():
        subset = selected[selected["scheme"] == scheme].set_index("model")
        mini = subset.loc["mini_vc"]
        baselines = subset.drop(index="mini_vc")
        best_baseline_name = baselines["mse_all"].idxmin()
        best_baseline = baselines.loc[best_baseline_name]
        mse_change = 100.0 * (
            mini["mse_all"] / best_baseline["mse_all"] - 1.0
        )
        mse_comparison = (
            f"高 {mse_change:.1f}%"
            if mse_change >= 0
            else f"低 {-mse_change:.1f}%"
        )
        best_delta_name = subset["pearson_delta"].idxmax()
        lines.append(
            f"- `{scheme}`：Mini-VC MSE 比最佳基线 `{best_baseline_name}` "
            f"{mse_comparison}；响应 Pearson 最高模型为 `{best_delta_name}`。"
        )
    lines.extend(
        [
            "",
            "## Mini-VC checkpoint",
            "",
        ]
    )
    mini_rows = selected[selected["model"] == "mini_vc"].copy()
    checkpoint_table = mini_rows[
        [
            "scheme",
            "best_epoch",
            "epochs_trained",
            "best_validation_response_mse",
            "device",
        ]
    ].copy()
    checkpoint_table["best_epoch"] = checkpoint_table["best_epoch"].astype(int)
    checkpoint_table["epochs_trained"] = checkpoint_table["epochs_trained"].astype(int)
    checkpoint_table["best_validation_response_mse"] = checkpoint_table[
        "best_validation_response_mse"
    ].map(_format)
    checkpoint_table.columns = [
        "拆分",
        "最佳 epoch",
        "训练 epoch",
        "最佳 validation response MSE",
        "设备",
    ]
    lines.extend(
        [
            _markdown_table(checkpoint_table),
            "",
            "每个 checkpoint 均由 validation response MSE 选择；test 不参与 early "
            "stopping 或学习率调整。",
            "",
            "## 结论边界",
            "",
            "Mini-VC 学会了可用的 latent transition，但 V0.1 中并未稳定超越更简单的 "
            "Ridge/MLP。unseen-gene 尤其受 identity-only perturbation 表示限制。"
            "这说明增加 Encoder–Transition–Decoder 结构本身不保证更好的泛化。",
            "",
            "三种拆分均保存 checkpoint、training log、逐条件指标和测试预测。"
            "两次完整训练的日志哈希一致；checkpoint 重载预测也通过数值核对。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "mini_vc"
        / "combined_model_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "stage8_training_summary.md",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.comparison), encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
