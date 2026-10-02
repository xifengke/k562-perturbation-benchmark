"""Inspect the Norman 2019 AnnData file without preprocessing it.

The script keeps the expression matrix in backed/read-only mode. Matrix shape,
sparsity, metadata, perturbation balance, batches, and count-scale heuristics
are written to a Chinese Markdown report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import anndata as ad
import h5py
import numpy as np
import pandas as pd


CONTROL_NAMES = {"control", "ctrl", "vehicle", "untreated", "unperturbed"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Inspect a perturbation H5AD file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "raw" / "NormanWeissman2019_filtered.h5ad",
        help="input H5AD file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "reports" / "data_summary.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="number of most abundant perturbations to report",
    )
    return parser.parse_args()


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_float(value: float, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    header_list = [str(item) for item in headers]
    lines = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join("---" for _ in header_list) + " |",
    ]
    for row in rows:
        values = [str(item).replace("|", "\\|") for item in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def sample_nonzero_values(dataset: h5py.Dataset) -> np.ndarray:
    """Read small, evenly spaced blocks from HDF5 X/data."""
    total = len(dataset)
    if total == 0:
        return np.array([], dtype=float)

    block_size = min(100_000, total)
    block_count = min(9, max(1, total // block_size))
    starts = np.linspace(0, max(0, total - block_size), block_count, dtype=np.int64)
    blocks = [np.asarray(dataset[int(start) : int(start) + block_size]) for start in starts]
    return np.concatenate(blocks)


def find_controls(labels: pd.Index) -> list[str]:
    controls: list[str] = []
    for label in labels.astype(str):
        normalized = label.strip().lower()
        if normalized in CONTROL_NAMES:
            controls.append(label)
    return controls


def describe_numeric(series: pd.Series) -> list[tuple[str, str]]:
    quantiles = series.astype(float).quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    labels = {
        0.0: "最小值",
        0.01: "1%",
        0.25: "25%",
        0.5: "中位数",
        0.75: "75%",
        0.99: "99%",
        1.0: "最大值",
    }
    return [(labels[index], fmt_float(value, 1)) for index, value in quantiles.items()]


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    report_path = args.report.resolve()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {input_path}\n"
            "Run `python scripts/download_data.py` first."
        )
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        obs = adata.obs.copy()
        var = adata.var.copy()
        n_cells, n_genes = adata.shape
        obs_columns = [(name, str(dtype)) for name, dtype in obs.dtypes.items()]
        var_columns = [(name, str(dtype)) for name, dtype in var.dtypes.items()]
        layer_names = list(adata.layers.keys())
        uns_keys = list(adata.uns.keys())
        has_raw = adata.raw is not None

        if "perturbation" not in obs:
            raise KeyError("Required metadata column `perturbation` is missing")

        perturb_counts = obs["perturbation"].astype(str).value_counts()
        perturb_counts = perturb_counts[perturb_counts > 0]
        controls = find_controls(perturb_counts.index)
        control_count = int(perturb_counts.loc[controls].sum()) if controls else 0

        if "nperts" in obs:
            cells_by_nperts = obs["nperts"].value_counts().sort_index()
            label_nperts = obs[["perturbation", "nperts"]].drop_duplicates()
            ambiguous_labels = int(
                (label_nperts.groupby("perturbation", observed=True)["nperts"].nunique() > 1).sum()
            )
            conditions_by_nperts = label_nperts["nperts"].value_counts().sort_index()
        else:
            cells_by_nperts = pd.Series(dtype=int)
            conditions_by_nperts = pd.Series(dtype=int)
            ambiguous_labels = 0

        batch_column = "gemgroup" if "gemgroup" in obs else None
        batch_counts = (
            obs[batch_column].value_counts().sort_index()
            if batch_column is not None
            else pd.Series(dtype=int)
        )

        condition_min = int(perturb_counts.min())
        condition_median = float(perturb_counts.median())
        condition_max = int(perturb_counts.max())
        imbalance_ratio = condition_max / condition_min
        noncontrol_counts = perturb_counts.drop(index=controls, errors="ignore")
        noncontrol_ratio = float(noncontrol_counts.max() / noncontrol_counts.min())

        count_cv = (
            float(obs["ncounts"].std() / obs["ncounts"].mean())
            if "ncounts" in obs
            else float("nan")
        )
        ncount_stats = describe_numeric(obs["ncounts"]) if "ncounts" in obs else []
        ngene_stats = describe_numeric(obs["ngenes"]) if "ngenes" in obs else []
        good_coverage_counts = (
            obs["good_coverage"].value_counts().sort_index(ascending=False)
            if "good_coverage" in obs
            else pd.Series(dtype=int)
        )
        unique_guides = int(obs["guide_id"].nunique()) if "guide_id" in obs else 0
        duplicate_genes = int(var.index.duplicated().sum())
        duplicate_cells = int(obs.index.duplicated().sum())
    finally:
        adata.file.close()

    with h5py.File(input_path, "r") as handle:
        matrix = handle["X"]
        encoding = matrix.attrs.get("encoding-type", "unknown")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        matrix_shape = tuple(int(item) for item in matrix.attrs.get("shape", (n_cells, n_genes)))
        values = matrix["data"]
        nnz = int(len(values))
        stored_dtype = str(values.dtype)
        sampled_values = sample_nonzero_values(values)

    total_entries = int(n_cells) * int(n_genes)
    sparsity = 1.0 - nnz / total_entries
    if sampled_values.size:
        sampled_min = float(sampled_values.min())
        sampled_max = float(sampled_values.max())
        integer_fraction = float(
            np.mean(np.isclose(sampled_values, np.round(sampled_values), atol=1e-6))
        )
        negative_fraction = float(np.mean(sampled_values < 0))
    else:
        sampled_min = sampled_max = integer_fraction = negative_fraction = float("nan")

    has_log1p = "log1p" in uns_keys
    raw_counts_likely = bool(
        sampled_values.size
        and sampled_min >= 0
        and negative_fraction == 0
        and integer_fraction > 0.99999
        and not has_log1p
        and not layer_names
        and (np.isnan(count_cv) or count_cv > 0.05)
    )
    scale_conclusion = (
        "高度符合未归一化 UMI counts：非零值为非负整数，细胞总 counts 明显不恒定，"
        "且对象没有 `log1p` 标记或额外 layer。虽然矩阵以 float32 存储，数值仍是整数 counts。"
        if raw_counts_likely
        else "无法仅凭启发式检查确认原始 counts；预处理前需要人工复核。"
    )

    multiplicity_names = {0: "control", 1: "single", 2: "double"}
    multiplicity_rows = []
    for nperts in sorted(set(cells_by_nperts.index) | set(conditions_by_nperts.index)):
        multiplicity_rows.append(
            (
                multiplicity_names.get(int(nperts), str(nperts)),
                fmt_int(int(conditions_by_nperts.get(nperts, 0))),
                fmt_int(int(cells_by_nperts.get(nperts, 0))),
                f"{int(cells_by_nperts.get(nperts, 0)) / n_cells:.2%}",
            )
        )

    top_rows = [
        (label, fmt_int(count), f"{count / n_cells:.2%}")
        for label, count in perturb_counts.head(args.top_n).items()
    ]
    batch_rows = [
        (str(label), fmt_int(count), f"{count / n_cells:.2%}")
        for label, count in batch_counts.items()
    ]
    coverage_rows = [
        (str(label), fmt_int(count), f"{count / n_cells:.2%}")
        for label, count in good_coverage_counts.items()
    ]

    report = f"""# Norman 2019 数据探索报告

本报告由 `scripts/inspect_data.py` 从真实 H5AD 文件自动生成。检查过程使用只读
backed 模式，没有 normalization、HVG selection 或 train/test split。

## 一眼结论

- 矩阵：**{fmt_int(n_cells)} cells × {fmt_int(n_genes)} genes**。
- 扰动标签：**{fmt_int(len(perturb_counts))}** 个，其中 control {fmt_int(control_count)} 个细胞。
- 条件组成：1 个 control、105 个单基因条件、131 个双基因条件。
- 数据包含 **{fmt_int(len(batch_counts))} 个 `{batch_column or '未发现'}` batches**。
- 矩阵稀疏度约 **{sparsity:.2%}**。
- 数值尺度判断：**{'raw counts' if raw_counts_likely else '未确认'}**。
- 类别明显不均衡：最大条件 / 最小条件为 **{imbalance_ratio:.1f}×**；去掉 control 后仍为 **{noncontrol_ratio:.1f}×**。

## 基本结构

| 项目 | 值 |
|---|---:|
| AnnData shape | `{fmt_int(n_cells)} × {fmt_int(n_genes)}` |
| HDF5 matrix shape | `{fmt_int(matrix_shape[0])} × {fmt_int(matrix_shape[1])}` |
| 稀疏格式 | `{encoding}` |
| 非零元素 nnz | {fmt_int(nnz)} |
| 总矩阵元素 | {fmt_int(total_entries)} |
| 稀疏度 | {sparsity:.4%} |
| 存储 dtype | `{stored_dtype}` |
| cell index 重复数 | {fmt_int(duplicate_cells)} |
| gene index 重复数 | {fmt_int(duplicate_genes)} |
| guide-level labels | {fmt_int(unique_guides)} |
| harmonized perturbation labels | {fmt_int(len(perturb_counts))} |
| `.raw` | `{has_raw}` |
| layers | `{layer_names}` |
| `uns` keys | `{uns_keys}` |

## 一行、一个样本和一个基因分别是什么？

- **一行**是一个经过 GEO/scPerturb 过滤、并成功获得转录组测量的 K562 细胞。
- **一个观测样本**在 cell-level 数据中就是一个细胞；但它不是某个 control cell 的“扰动后版本”。
- **一列**是一个基因。矩阵元素是该细胞中检测到的该基因 UMI count。
- 建模时会从互不重叠的细胞集合构建 condition-level/pseudobulk 样本，减少 dropout，且不制造假配对。

## Perturbation label 与 control

`perturbation` 是 scPerturb 整理后的基因级扰动标签。单基因标签类似 `KLF1`，
双基因标签类似 `CEBPE_RUNX1T1`。`guide_id` 保留更细的 guide/cassette 身份，
因此本文件有 {fmt_int(unique_guides)} 个 guide-level labels，但只有
{fmt_int(len(perturb_counts))} 个 gene-level perturbation labels。

control 标签为 `{', '.join(controls) if controls else '未自动识别'}`，表示两个位置均使用
non-targeting guide 的细胞。单基因条件由目标 guide 加 negative-control guide 构成，
双基因条件的两个位置均为目标基因 guide。

{markdown_table(['类型', '条件数', '细胞数', '细胞占比'], multiplicity_rows)}

同一 `perturbation` 同时映射到多个 `nperts` 值的异常标签数：**{ambiguous_labels}**。

## Metadata columns

### Cell metadata (`adata.obs`)

{markdown_table(['字段', 'dtype'], obs_columns)}

### Gene metadata (`adata.var`)

{markdown_table(['字段', 'dtype'], var_columns)}

## Counts、normalization 与 log1p 检查

{scale_conclusion}

检查证据：

- 从 `X/data` 均匀抽取 {fmt_int(sampled_values.size)} 个非零值；
- sampled nonzero min/max：{fmt_float(sampled_min, 1)} / {fmt_float(sampled_max, 1)}；
- integer-like fraction：{integer_fraction:.6%}；
- negative fraction：{negative_fraction:.6%}；
- `uns['log1p']` 是否存在：`{has_log1p}`；
- `layers`：`{layer_names}`；
- cell total-count coefficient of variation：{count_cv:.3f}。

注意：“raw counts”在这里指 Cell Ranger 之后的 UMI count matrix，不是 FASTQ，也不是完全未经
cell filtering 的测序数据。后续仍需做 QC、library-size normalization 和 `log1p`。

### 每个细胞的 total counts

{markdown_table(['分位点', 'ncounts'], ncount_stats)}

### 每个细胞检测到的 genes

{markdown_table(['分位点', 'ngenes'], ngene_stats)}

## Batch 信息

`gemgroup` 对应合并数据中的 8 个 10x Chromium lane/gem groups。各组规模相近，但不能因此
假定没有 batch effect；后续 split 和可视化都应保留该字段。

{markdown_table([batch_column or 'batch', '细胞数', '占比'], batch_rows)}

## Guide coverage

`good_coverage=False` 的细胞仍存在于这份 filtered H5AD 中，共
{fmt_int(int(good_coverage_counts.get(False, 0)))} 个。该字段与 `number_of_cells` 的含义和是否用于
额外 QC，应结合原始处理说明决定，不能看到 “filtered” 就默认全部通过 guide QC。

{markdown_table(['good_coverage', '细胞数', '占比'], coverage_rows)}

## 类别不均衡

- 每个 perturbation 的细胞数：最少 {fmt_int(condition_min)}，中位数 {fmt_float(condition_median, 1)}，最多 {fmt_int(condition_max)}。
- 最大类是 control，{fmt_int(control_count)} cells，占 {control_count / n_cells:.2%}。
- 最大/最小条件比例为 {imbalance_ratio:.1f}×；排除 control 后为 {noncontrol_ratio:.1f}×。
- 因此训练 loss 不能让大条件无限主导，评价也必须先按 perturbation 分别计算再做 macro average。

细胞数最多的 {args.top_n} 个条件：

{markdown_table(['perturbation', '细胞数', '占全部细胞'], top_rows)}

## 预测任务

给定 reference 和扰动身份，预测条件平均响应：

> 给定独立估计的 K562 control/reference state 与一个单基因或双基因 CRISPRa identity，
> 预测该 perturbation 条件下的平均表达和相对 control 的 `delta expression`。

训练和测试细胞互不重叠。unseen-combination 留出完整双基因条件，
同时将对应单基因条件保留在训练集，用于评价组合泛化。

## 预处理选择

1. 稀疏读取表达矩阵，按块处理，避免将整个矩阵转为 dense。
2. 使用原始 counts 逐细胞进行 library-size normalization 和 `log1p`。
3. 保留 `gemgroup`，在同批次内构建 reference 和 target pseudobulk。
4. 每个划分用训练细胞选择 2,000 个 HVG，并保留可匹配的扰动目标。
5. 记录 cell ID 和 condition 的 split assignment；HVG 由训练细胞拟合，测试 DE 排名由测试 delta 构成。
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Shape: {fmt_int(n_cells)} cells x {fmt_int(n_genes)} genes")
    print(f"Perturbations: {fmt_int(len(perturb_counts))}")
    print(f"Control cells: {fmt_int(control_count)}")
    print(
        "Multiplicity: "
        + ", ".join(
            f"nperts={int(key)}: {fmt_int(int(value))} cells"
            for key, value in cells_by_nperts.items()
        )
    )
    print(f"Sparsity: {sparsity:.4%} ({fmt_int(nnz)} nonzero values)")
    print(f"Scale: {'likely raw UMI counts' if raw_counts_likely else 'uncertain'}")
    print(f"Batches: {fmt_int(len(batch_counts))} ({batch_column or 'none detected'})")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
