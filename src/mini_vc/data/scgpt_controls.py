"""Prepare real control cells for the scGPT transfer encoder.

Only control cells from the corresponding split and gemgroup are encoded.
No perturbed target cell is ever used as a model input. Gene selection uses
the fold's training-only HVG scores; the benchmark target remains the original
2,000-gene pseudobulk defined by the existing preprocessing pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


@dataclass
class ControlTokenBank:
    gene_ids: np.ndarray  # [n_groups, n_cells, n_tokens]
    values: np.ndarray  # [n_groups, n_cells, n_tokens]
    keys: list[tuple[str, int]]
    input_genes: list[str]

    def get(self, split: str, gemgroup: int) -> tuple[np.ndarray, np.ndarray]:
        index = self.keys.index((split, int(gemgroup)))
        return self.gene_ids[index], self.values[index]


def _bin_row(row: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """Deterministic tie handling for scGPT's nonzero quantile binning."""
    output = np.zeros(len(row), dtype=np.float32)
    nonzero = row > 0
    if np.any(nonzero):
        edges = np.quantile(row[nonzero], np.linspace(0, 1, n_bins - 1))
        output[nonzero] = np.digitize(row[nonzero], edges).astype(np.float32)
    return output


def _stable_cell_order(cell_id: str, scheme: str) -> bytes:
    return hashlib.blake2b(f"{scheme}|{cell_id}".encode(), digest_size=8).digest()


def build_control_token_bank(
    raw_path: Path,
    split_path: Path,
    dataset,
    scheme: str,
    vocab: dict[str, int],
    cache_path: Path,
    n_cells_per_group: int = 4,
    max_tokens: int = 512,
) -> ControlTokenBank:
    genes = dataset.genes.copy()
    genes = genes.loc[genes["gene"].astype(str).isin(vocab)]
    genes = genes.sort_values(["hvg_score", "gene"], ascending=[False, True])
    genes = genes.head(max_tokens)
    if len(genes) != max_tokens:
        raise ValueError(f"only {len(genes)} scGPT input genes; need {max_tokens}")
    input_genes = genes["gene"].astype(str).tolist()
    gene_indices = genes["gene_index"].to_numpy(dtype=np.int64)
    token_ids = np.asarray([vocab[gene] for gene in input_genes], dtype=np.int64)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            cached_genes = saved["input_genes"].astype(str).tolist()
            cached_ids = saved["gene_ids"]
            cached_values = saved["values"]
            if (
                cached_genes != input_genes
                or cached_ids.shape[1:] != (n_cells_per_group, max_tokens)
                or cached_values.shape != cached_ids.shape
                or not np.array_equal(cached_ids[0, 0], token_ids)
            ):
                raise ValueError(
                    f"scGPT control cache does not match current input settings: {cache_path}"
                )
            return ControlTokenBank(
                gene_ids=cached_ids,
                values=cached_values,
                keys=list(zip(saved["splits"].astype(str), saved["gemgroups"].tolist())),
                input_genes=cached_genes,
            )

    splits = pd.read_csv(split_path, index_col="cell_id", usecols=["cell_id", f"split_{scheme}"])
    adata = ad.read_h5ad(raw_path, backed="r")
    try:
        obs = adata.obs[["perturbation", "gemgroup", "ncounts"]].copy()
    finally:
        adata.file.close()
    obs = obs.join(splits, how="left")
    if obs[f"split_{scheme}"].isna().any():
        raise ValueError("raw cells and saved split assignments do not align")
    controls = obs.loc[obs["perturbation"].astype(str) == "control"]
    keys = [
        (str(row.split), int(row.gemgroup))
        for row in dataset.samples.loc[dataset.samples.perturbation == "control"]
        .drop_duplicates(["split", "gemgroup"])
        .itertuples()
    ]
    selected_ids: list[str] = []
    group_ids: list[list[str]] = []
    for split, gemgroup in keys:
        members = controls.loc[
            (controls[f"split_{scheme}"] == split)
            & (controls["gemgroup"].to_numpy() == gemgroup)
        ]
        ordered = sorted(members.index.astype(str), key=lambda x: _stable_cell_order(x, scheme))
        if len(ordered) < n_cells_per_group:
            raise ValueError(f"too few control cells in {(split, gemgroup)}")
        chosen = ordered[:n_cells_per_group]
        selected_ids.extend(chosen)
        group_ids.append(chosen)

    index_lookup = pd.Series(np.arange(len(obs), dtype=np.int64), index=obs.index)
    sorted_positions = np.sort(index_lookup.loc[selected_ids].to_numpy(dtype=np.int64))
    position_to_local = {int(position): index for index, position in enumerate(sorted_positions)}
    counts = np.zeros((len(sorted_positions), max_tokens), dtype=np.float32)
    with h5py.File(raw_path, "r") as handle:
        matrix = handle["X"]
        if matrix.attrs.get("encoding-type") not in ("csc_matrix", b"csc_matrix"):
            raise ValueError("expected CSC raw count matrix")
        indptr = matrix["indptr"]
        for output_column, gene_index in enumerate(gene_indices):
            lo, hi = int(indptr[gene_index]), int(indptr[gene_index + 1])
            rows = np.asarray(matrix["indices"][lo:hi], dtype=np.int64)
            positions = np.searchsorted(sorted_positions, rows)
            valid = positions < len(sorted_positions)
            valid[valid] &= sorted_positions[positions[valid]] == rows[valid]
            if np.any(valid):
                counts[positions[valid], output_column] = np.asarray(
                    matrix["data"][lo:hi], dtype=np.float32
                )[valid]
    libraries = obs.iloc[sorted_positions]["ncounts"].to_numpy(dtype=np.float32)
    if np.any(libraries <= 0):
        raise ValueError("control cell has nonpositive library size")
    log_values = np.log1p(counts * (10000.0 / libraries[:, None]))
    binned = np.stack([_bin_row(row) for row in log_values])
    bank_values = np.stack(
        [binned[[position_to_local[int(index_lookup.loc[cell])] for cell in group]] for group in group_ids]
    )
    bank_ids = np.broadcast_to(token_ids, bank_values.shape).copy()
    bank = ControlTokenBank(bank_ids, bank_values, keys, input_genes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        gene_ids=bank.gene_ids,
        values=bank.values,
        splits=np.asarray([key[0] for key in keys]),
        gemgroups=np.asarray([key[1] for key in keys]),
        input_genes=np.asarray(input_genes),
    )
    return bank
