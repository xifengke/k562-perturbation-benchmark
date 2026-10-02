"""Memory-safe preprocessing for the Norman 2019 CSC count matrix.

The two important transformations are intentionally explicit:

1. each nonzero count x_ij becomes log(1 + target_sum * x_ij / library_i);
2. pseudobulk expression is the arithmetic mean of those transformed cell
   vectors within (split, perturbation, gemgroup).

HVG statistics are fitted on training cells only for each evaluation scheme.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from .splits import make_splits, perturbation_genes


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    return config


def make_qc_mask(obs: pd.DataFrame, config: dict) -> tuple[np.ndarray, dict[str, int]]:
    """Apply transparent metadata-based QC and return failure counts."""
    checks: dict[str, np.ndarray] = {}
    if config.get("require_good_coverage", True):
        checks["bad_guide_coverage"] = ~obs["good_coverage"].astype(bool).to_numpy()
    if config.get("require_single_cell", True):
        checks["not_single_cell"] = obs["number_of_cells"].to_numpy() != 1
    checks["low_counts"] = obs["ncounts"].to_numpy() < float(config["min_counts"])
    checks["low_genes"] = obs["ngenes"].to_numpy() < int(config["min_genes"])
    checks["high_mito"] = (
        obs["percent_mito"].to_numpy() > float(config["max_percent_mito"])
    )

    failed = np.zeros(len(obs), dtype=bool)
    for values in checks.values():
        failed |= values
    summary = {name: int(values.sum()) for name, values in checks.items()}
    summary["removed_union"] = int(failed.sum())
    summary["retained"] = int((~failed).sum())
    return ~failed, summary


def _score_hvgs(
    mean: np.ndarray, variance: np.ndarray, detected: np.ndarray, mean_bins: int
) -> np.ndarray:
    """Return mean-bin standardized log dispersion scores."""
    score = np.full(mean.shape, -np.inf, dtype=np.float64)
    valid = (mean > 0) & (detected > 0) & np.isfinite(variance)
    dispersion = np.full(mean.shape, np.nan, dtype=np.float64)
    dispersion[valid] = np.log(variance[valid] / mean[valid] + 1e-12)

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return score
    bin_count = min(mean_bins, len(valid_indices))
    bins = pd.qcut(
        pd.Series(mean[valid_indices]), q=bin_count, labels=False, duplicates="drop"
    ).to_numpy()
    for bin_id in np.unique(bins):
        indices = valid_indices[bins == bin_id]
        values = dispersion[indices]
        std = float(np.nanstd(values))
        if std > 0:
            score[indices] = (values - float(np.nanmean(values))) / std
        else:
            score[indices] = 0.0
    return score


def _choose_hvgs(
    var_names: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    detected: np.ndarray,
    n_top_genes: int,
    min_train_cells: int,
    mean_bins: int,
    forced_genes: set[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    score = _score_hvgs(mean, variance, detected, mean_bins)
    eligible = detected >= min_train_cells
    forced_indices = np.asarray(
        [index for index, gene in enumerate(var_names) if gene in forced_genes], dtype=int
    )
    if len(forced_indices) > n_top_genes:
        raise ValueError("number of forced perturbation genes exceeds n_top_genes")

    ranking = np.argsort(np.where(eligible, score, -np.inf))[::-1]
    selected: list[int] = forced_indices.tolist()
    selected_set = set(selected)
    for index in ranking:
        index = int(index)
        if len(selected) >= n_top_genes:
            break
        if index not in selected_set and np.isfinite(score[index]):
            selected.append(index)
            selected_set.add(index)
    if len(selected) != n_top_genes:
        raise ValueError(
            f"only {len(selected)} genes passed HVG requirements; requested {n_top_genes}"
        )

    selected_array = np.asarray(sorted(selected), dtype=int)
    table = pd.DataFrame(
        {
            "gene": var_names[selected_array],
            "gene_index": selected_array,
            "train_mean_log1p": mean[selected_array],
            "train_variance_log1p": variance[selected_array],
            "train_detected_cells": detected[selected_array],
            "hvg_score": score[selected_array],
            "forced_perturbation_gene": [
                gene in forced_genes for gene in var_names[selected_array]
            ],
        }
    )
    return selected_array, table


def _compute_train_gene_statistics(
    input_path: Path,
    library_sizes: np.ndarray,
    train_masks: dict[str, np.ndarray],
    target_sum: float,
    block_genes: int = 256,
) -> dict[str, dict[str, np.ndarray]]:
    """Scan the CSC matrix once and fit log-normalized gene moments on train only."""
    with h5py.File(input_path, "r") as handle:
        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") not in ("csc_matrix", b"csc_matrix"):
            raise ValueError("streaming preprocessor currently requires CSC-formatted X")
        n_cells, n_genes = (int(value) for value in matrix.attrs["shape"])
        indptr = np.asarray(matrix["indptr"])
        data_dataset = matrix["data"]
        indices_dataset = matrix["indices"]

        stats = {
            name: {
                "sum": np.zeros(n_genes, dtype=np.float64),
                "sumsq": np.zeros(n_genes, dtype=np.float64),
                "detected": np.zeros(n_genes, dtype=np.int64),
                "n_train": np.asarray(int(mask.sum()), dtype=np.int64),
            }
            for name, mask in train_masks.items()
        }
        if any(len(mask) != n_cells for mask in train_masks.values()):
            raise ValueError("train mask length does not match matrix rows")

        for gene_start in range(0, n_genes, block_genes):
            if gene_start % 4096 == 0:
                print(
                    f"Fitting train-only gene statistics: {gene_start:,}/{n_genes:,} genes",
                    flush=True,
                )
            gene_end = min(gene_start + block_genes, n_genes)
            value_start = int(indptr[gene_start])
            value_end = int(indptr[gene_end])
            block_values = np.asarray(data_dataset[value_start:value_end], dtype=np.float64)
            block_rows = np.asarray(indices_dataset[value_start:value_end], dtype=np.int64)
            local_ptr = indptr[gene_start : gene_end + 1] - value_start

            for offset, gene_index in enumerate(range(gene_start, gene_end)):
                lo = int(local_ptr[offset])
                hi = int(local_ptr[offset + 1])
                rows = block_rows[lo:hi]
                counts = block_values[lo:hi]
                for name, train_mask in train_masks.items():
                    keep = train_mask[rows]
                    if not np.any(keep):
                        continue
                    selected_rows = rows[keep]
                    normalized = np.log1p(
                        counts[keep] * target_sum / library_sizes[selected_rows]
                    )
                    stats[name]["sum"][gene_index] = normalized.sum()
                    stats[name]["sumsq"][gene_index] = np.square(normalized).sum()
                    stats[name]["detected"][gene_index] = int(keep.sum())

    for values in stats.values():
        n_train = int(values["n_train"])
        mean = values.pop("sum") / n_train
        sumsq = values.pop("sumsq")
        variance = np.maximum((sumsq - n_train * np.square(mean)) / (n_train - 1), 0.0)
        values["mean"] = mean
        values["variance"] = variance
    return stats


def _group_metadata(
    obs: pd.DataFrame, assignments: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "split": assignments,
            "perturbation": obs["perturbation"].astype(str).to_numpy(),
            "nperts": obs["nperts"].to_numpy(dtype=int),
            "gemgroup": obs["gemgroup"].to_numpy(dtype=int),
        },
        index=obs.index,
    )
    metadata = (
        frame.groupby(["split", "perturbation", "nperts", "gemgroup"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["split", "perturbation", "gemgroup"])
        .reset_index(drop=True)
    )
    keys = pd.MultiIndex.from_frame(metadata[["split", "perturbation", "nperts", "gemgroup"]])
    cell_keys = pd.MultiIndex.from_frame(frame[["split", "perturbation", "nperts", "gemgroup"]])
    group_ids = keys.get_indexer(cell_keys)
    if np.any(group_ids < 0):
        raise AssertionError("failed to assign one or more cells to a pseudobulk group")
    metadata.insert(0, "sample_id", [f"pb_{index:05d}" for index in range(len(metadata))])
    return metadata, group_ids


def _make_pseudobulk(
    input_path: Path,
    qc_positions: np.ndarray,
    library_sizes_all: np.ndarray,
    assignments: np.ndarray,
    obs_qc: pd.DataFrame,
    gene_indices: np.ndarray,
    target_sum: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    metadata, group_ids_qc = _group_metadata(obs_qc, assignments)
    group_ids_all = np.full(len(library_sizes_all), -1, dtype=np.int64)
    group_ids_all[qc_positions] = group_ids_qc
    group_sizes = metadata["n_cells"].to_numpy(dtype=np.float64)
    expression = np.zeros((len(metadata), len(gene_indices)), dtype=np.float32)

    with h5py.File(input_path, "r") as handle:
        matrix = handle["X"]
        indptr = np.asarray(matrix["indptr"])
        data = matrix["data"]
        indices = matrix["indices"]
        for output_column, gene_index in enumerate(gene_indices):
            if output_column % 500 == 0:
                print(
                    f"Building pseudobulk matrix: {output_column:,}/{len(gene_indices):,} genes",
                    flush=True,
                )
            lo = int(indptr[gene_index])
            hi = int(indptr[gene_index + 1])
            rows = np.asarray(indices[lo:hi], dtype=np.int64)
            groups = group_ids_all[rows]
            keep = groups >= 0
            if not np.any(keep):
                continue
            selected_rows = rows[keep]
            normalized = np.log1p(
                np.asarray(data[lo:hi], dtype=np.float64)[keep]
                * target_sum
                / library_sizes_all[selected_rows]
            )
            sums = np.bincount(
                groups[keep], weights=normalized, minlength=len(metadata)
            )
            expression[:, output_column] = (sums / group_sizes).astype(np.float32)
    return expression, metadata


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_preprocessing(config_path: Path, repo_root: Path) -> dict:
    config = load_config(config_path)
    input_path = (repo_root / config["data"]["input_path"]).resolve()
    output_dir = (repo_root / config["data"]["output_dir"]).resolve()
    report_path = (repo_root / config["data"]["report_path"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        obs_all = adata.obs.copy()
        var_names = adata.var_names.astype(str).to_numpy()
    finally:
        adata.file.close()

    qc_mask, qc_summary = make_qc_mask(obs_all, config["qc"])
    qc_positions = np.flatnonzero(qc_mask)
    obs_qc = obs_all.iloc[qc_positions].copy()
    splits = make_splits(
        obs_qc,
        list(config["splits"]["schemes"]),
        int(config["seed"]),
        float(config["splits"]["validation_fraction"]),
        float(config["splits"]["test_fraction"]),
    )

    library_sizes = obs_all["ncounts"].to_numpy(dtype=np.float64)
    train_masks: dict[str, np.ndarray] = {}
    for name, result in splits.items():
        full_mask = np.zeros(len(obs_all), dtype=bool)
        full_mask[qc_positions] = result.assignments == "train"
        train_masks[name] = full_mask

    stats = _compute_train_gene_statistics(
        input_path,
        library_sizes,
        train_masks,
        float(config["normalization"]["target_sum"]),
    )

    perturbation_gene_set = {
        gene
        for label in obs_qc["perturbation"].astype(str).unique()
        for gene in perturbation_genes(label)
    }
    available_gene_set = set(var_names.tolist())
    missing_perturbation_genes = sorted(perturbation_gene_set - available_gene_set)
    selected_by_scheme: dict[str, np.ndarray] = {}
    scheme_summaries: dict[str, dict] = {}
    cell_splits = pd.DataFrame(index=obs_all.index)
    cell_splits.index.name = "cell_id"
    cell_splits["qc_pass"] = qc_mask

    for name, result in splits.items():
        full_assignments = np.full(len(obs_all), "excluded_qc", dtype="U11")
        full_assignments[qc_positions] = result.assignments
        cell_splits[f"split_{name}"] = full_assignments

        selected, gene_table = _choose_hvgs(
            var_names,
            stats[name]["mean"],
            stats[name]["variance"],
            stats[name]["detected"],
            int(config["hvg"]["n_top_genes"]),
            int(config["hvg"]["min_train_cells"]),
            int(config["hvg"]["mean_bins"]),
            perturbation_gene_set if config["hvg"]["include_perturbation_genes"] else set(),
        )
        selected_by_scheme[name] = selected
        expression, metadata = _make_pseudobulk(
            input_path,
            qc_positions,
            library_sizes,
            result.assignments,
            obs_qc,
            selected,
            float(config["normalization"]["target_sum"]),
        )

        scheme_dir = output_dir / name
        scheme_dir.mkdir(parents=True, exist_ok=True)
        np.save(scheme_dir / "expression.npy", expression)
        metadata.to_csv(scheme_dir / "samples.csv", index=False)
        gene_table.to_csv(scheme_dir / "genes.csv", index=False)
        with (scheme_dir / "split_details.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(result.details), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        split_cell_counts = pd.Series(result.assignments).value_counts().to_dict()
        split_sample_counts = metadata["split"].value_counts().to_dict()
        scheme_summaries[name] = {
            "n_hvg": int(expression.shape[1]),
            "n_pseudobulk_samples": int(expression.shape[0]),
            "cell_counts": {key: int(value) for key, value in split_cell_counts.items()},
            "sample_counts": {key: int(value) for key, value in split_sample_counts.items()},
            "forced_perturbation_genes": int(
                gene_table["forced_perturbation_gene"].sum()
            ),
        }

    cell_splits.to_csv(output_dir / "cell_splits.csv.gz", compression="gzip")
    manifest = {
        "input": str(input_path),
        "config": str(config_path.resolve()),
        "seed": int(config["seed"]),
        "qc": qc_summary,
        "normalization": config["normalization"],
        "hvg": {
            **config["hvg"],
            "perturbation_genes_total": len(perturbation_gene_set),
            "perturbation_genes_present": len(
                perturbation_gene_set - set(missing_perturbation_genes)
            ),
            "missing_perturbation_genes": missing_perturbation_genes,
        },
        "schemes": scheme_summaries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    overlap = None
    if len(selected_by_scheme) > 1:
        sets = [set(indices.tolist()) for indices in selected_by_scheme.values()]
        overlap = len(set.intersection(*sets))

    summary_rows = []
    for name, values in scheme_summaries.items():
        summary_rows.append(
            "| {name} | {cells} | {samples} | {hvg} | {forced} |".format(
                name=name,
                cells=" / ".join(
                    str(values["cell_counts"].get(split, 0))
                    for split in ("train", "validation", "test")
                ),
                samples=" / ".join(
                    str(values["sample_counts"].get(split, 0))
                    for split in ("train", "validation", "test")
                ),
                hvg=values["n_hvg"],
                forced=values["forced_perturbation_genes"],
            )
        )

    report = f"""# Preprocessing 与 split 报告

## QC

- 输入细胞：{len(obs_all):,}
- 保留细胞：{qc_summary['retained']:,}
- 去除细胞（条件并集）：{qc_summary['removed_union']:,}
- `good_coverage=False`：{qc_summary.get('bad_guide_coverage', 0):,}
- `number_of_cells != 1`：{qc_summary.get('not_single_cell', 0):,}
- counts < {config['qc']['min_counts']}：{qc_summary['low_counts']:,}
- genes < {config['qc']['min_genes']}：{qc_summary['low_genes']:,}
- mitochondrial percentage > {config['qc']['max_percent_mito']}：{qc_summary['high_mito']:,}

这些失败原因可能重叠，所以各项不能直接相加。`number_of_cells > 1` 被当作 multiplet，
`number_of_cells == 0` 与低 guide coverage 失败相符。

## Normalization 与 HVG

- 每个细胞先按 library size 缩放至 {config['normalization']['target_sum']:.0f} counts；
- 对每个非零值执行 `log1p`；
- 每个 split scheme 只用其 train cells 拟合 gene mean/variance/HVG；
- 选择 {config['hvg']['n_top_genes']} genes，并强制保留实验涉及且存在于矩阵中的 perturbation genes；
- {len(perturbation_gene_set)} 个 perturbation genes 中有 {len(perturbation_gene_set) - len(missing_perturbation_genes)} 个能按 gene symbol 在矩阵中匹配；未匹配：`{', '.join(missing_perturbation_genes)}`；
- {len(selected_by_scheme)} 套 train-only HVG 集合的共同基因数：{overlap if overlap is not None else 'N/A'}。

## Split 与 pseudobulk 数量

单元格顺序均为 train / validation / test。

| scheme | cells | pseudobulk samples | genes | forced perturbation genes |
|---|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

每个 pseudobulk 是同一 `(split, perturbation, gemgroup)` 中所有 QC-passing cells 的
log-normalized expression 均值。不存在 control–perturbed 单细胞伪配对。

## 输出文件

- `cell_splits.csv.gz`：每个原始 cell 的 QC 与各划分的 split assignment；
- `<scheme>/expression.npy`：`n_pseudobulk × 2000` float32 expression；
- `<scheme>/samples.csv`：行 metadata；
- `<scheme>/genes.csv`：列 metadata 和 train-only HVG statistics；
- `<scheme>/split_details.json`：held-out batches/conditions/genes；
- `manifest.json`：配置和尺寸汇总。
"""
    report_path.write_text(report, encoding="utf-8")
    return manifest
