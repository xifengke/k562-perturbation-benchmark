"""Generate figures and a report from saved test predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.evaluation.analysis import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    analyze_results,
)


def build_report(summary: dict) -> str:
    comparison = pd.read_csv(
        REPO_ROOT / "results" / "mini_vc" / "combined_model_comparison.csv"
    )
    primary = comparison[
        (comparison["scheme"] == summary["primary_scheme"])
        & comparison["model"].isin(MODELS)
    ].set_index("model")
    win_delta = summary["condition_wins"]["pearson_delta"]
    win_de = summary["condition_wins"]["top_20_recovery"]
    tie_delta = summary["condition_win_ties"]["pearson_delta"]
    tie_de = summary["condition_win_ties"]["top_20_recovery"]
    pc1, pc2 = summary["pca_explained_variance_ratio"]
    figure = summary["figures"]
    lines = [
        "# 基础实验结果分析",
        "",
        "本报告由保存的 test 预测和逐条件指标生成，分析过程不重新训练模型。",
        "",
        "## 总体模型比较",
        "",
        f"![模型比较](../{figure['model_comparison']})",
        "",
        "Mini-VC 在三个拆分上均未稳定超过最佳简单基线。seen 中 MLP 更好；"
        "unseen-gene 中 Ridge 更好。Ridge 的表现与加性近似适合本任务的解释相容，"
        "但模型排名不能单独确定生物互作机制。",
        "",
        "## 训练过程",
        "",
        f"![训练曲线](../{figure['training_curves']})",
        "",
        "seen 和 unseen-combination 的 validation loss 持续下降后进入平台期；"
        "unseen-gene 很早停止，符合 identity-only 表示无法学习全新基因语义的限制。",
        "",
        "## Unseen-combination：主要泛化任务",
        "",
        f"测试集包含 {summary['n_test_conditions']} 个完整未见组合。按逐条件 response "
        f"Pearson 计，获胜条件数为（{tie_delta} 个条件并列）："
        + "、".join(f"{MODEL_LABELS[m]} {win_delta[m]}" for m in MODELS)
        + f"。按 top-20 DE recovery 计（{tie_de} 个条件并列，并列模型均计入）："
        + "、".join(f"{MODEL_LABELS[m]} {win_de[m]}" for m in MODELS)
        + "。",
        "",
        f"![逐条件比较](../{figure['per_condition']})",
        "",
        "逐条件图展示全部测试组合，按名称排序。",
        "",
        f"![响应散点](../{figure['delta_scatter']})",
        "",
        "散点图使用 condition-centroid expression delta，比较相对 control 的响应。",
        "图标题中的 `r` 是把所有 condition×gene delta 合并后的 pooled correlation；"
        "主结果表使用逐 condition Pearson 的宏平均，所以两者数值不会完全相同。",
        "",
        "## PCA",
        "",
        f"![PCA](../{figure['pca']})",
        "",
        f"共同 PCA 的 PC1/PC2 分别解释 {pc1 * 100:.1f}% 和 {pc2 * 100:.1f}% 方差。"
        "PCA 展示整体表达空间的位置关系，不作为性能指标。",
        "",
        "## 中位数条件的 DE 响应",
        "",
        f"选择规则是 Mini-VC response Pearson 排序后的中位数，而非最佳案例。"
        f"选中条件为 `{summary['median_condition']}`，Mini-VC Pearson 为 "
        f"{summary['median_condition_mini_vc_pearson_delta']:.3f}。",
        "",
        f"![DE heatmap](../{figure['de_heatmap']})",
        "",
        "## 结论",
        "",
        f"在 unseen-combination 上，Mini-VC MSE 为 {primary.loc['mini_vc', 'mse_all']:.4f}，"
        f"response Pearson 为 {primary.loc['mini_vc', 'pearson_delta']:.3f}；MLP 分别为 "
        f"{primary.loc['mlp', 'mse_all']:.4f} 和 {primary.loc['mlp', 'pearson_delta']:.3f}。"
        "两者接近，Mini-VC 没有取得整体优势。这张表仅包含最初的 13 个测试组合；"
        "全部 131 个组合的结果见 [五折比较](scgpt_fivefold_results.md)。",
        "",
        "基础测试集已在五折和 Reactome 实验前观察过，后两组比较属于同一数据集上的扩展分析。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    summary = analyze_results(REPO_ROOT)
    report = build_report(summary)
    output = REPO_ROOT / "reports" / "stage9_results_analysis.md"
    output.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
