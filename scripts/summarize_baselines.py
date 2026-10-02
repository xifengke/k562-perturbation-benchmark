"""Generate the baseline report from saved evaluation outputs."""

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
    columns = [
        "scheme",
        "model",
        "n_conditions",
        "mse_all",
        "pearson_delta",
        "pearson_de",
        "top_100_recovery",
    ]
    table = results[columns].copy()
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
        "# 基础模型比较",
        "",
        "本报告由 `scripts/summarize_baselines.py` 直接读取 "
        "`results/baselines/model_comparison.csv` 生成。",
        "",
        _markdown_table(table),
        "",
        "## 结果解读",
        "",
    ]
    for scheme in results["scheme"].unique():
        subset = results[results["scheme"] == scheme].set_index("model")
        ridge = subset.loc["ridge"]
        mlp = subset.loc["mlp"]
        no_change = subset.loc["no_change"]
        best_model = subset["mse_all"].idxmin()
        reduction = 100.0 * (1.0 - mlp["mse_all"] / no_change["mse_all"])
        versus_ridge = 100.0 * (1.0 - mlp["mse_all"] / ridge["mse_all"])
        ridge_comparison = (
            f"比 ridge 低 {versus_ridge:.1f}%"
            if versus_ridge >= 0
            else f"比 ridge 高 {-versus_ridge:.1f}%"
        )
        lines.append(
            f"- `{scheme}`：MLP 相对 no-change 将全基因 MSE 降低 "
            f"{reduction:.1f}%，MSE {ridge_comparison}；"
            f"该拆分最低 MSE 模型是 `{best_model}`。"
        )
    lines.extend(
        [
            "- no-change 的预测响应恒为零，因此响应相关性和 DE 排名没有数学定义，"
            "报告为 `—`，而不是让排序函数任意打破并列。",
            "- `unseen_combo` 检验已见单基因效应能否组合到未见双扰动；加性 ridge "
            "正好是这个假设的透明基线。",
            "- MLP 使用 perturbation multi-hot 预测 delta，并通过 "
            "`predicted expression = matched control + delta` 形成最终输出。"
            "模型只用 validation early stopping；测试集不参与训练。",
            "- `unseen_gene` 中测试基因从训练条件完全移除。当前 identity-only 多热表示"
            "没有这些基因的先验信息，因此该拆分主要量化这种表示的泛化上限。",
            "",
            "## 数据流与关键代码",
            "",
            "1. `src/mini_vc/data/dataset.py`：读取 pseudobulk，按 split 和 gemgroup "
            "找到对应 control，并计算 `delta = perturbed - control`。",
            "2. `src/mini_vc/models/baselines.py`：总体平均响应，以及带不惩罚截距的"
            "多输出 ridge 闭式解。",
            "3. `src/mini_vc/models/mlp.py`：multi-hot 扰动响应网络；未知训练基因被"
            "显式 mask，control/完全未知操作输出 zero delta。",
            "4. `src/mini_vc/training/torch_runner.py`：固定 seed、GPU/CPU 自动选择、"
            "early stopping、checkpoint 和训练日志。",
            "5. `src/mini_vc/training/baseline_runner.py`：只用 validation 选择 alpha，"
            "随后用 train+validation 重拟合，最终只在 test 报告指标。",
            "6. `src/mini_vc/evaluation/metrics.py`：先求每个 perturbation 的测试质心，"
            "再按条件宏平均，避免细胞数多的条件支配分数。",
            "",
            "## 训练产物",
            "",
            "no-change、mean-response、ridge 和 MLP 均在三种拆分上完成训练和评价。"
            "MLP checkpoint、逐 epoch 日志、测试预测和逐条件指标均已保存。"
            "初始 state-concat MLP 的结果及输入修改记录在 "
            "`reports/mlp_revision_log.md`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=REPO_ROOT / "results" / "baselines" / "model_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "baseline_results.md",
    )
    args = parser.parse_args()
    report = build_report(args.comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
