"""Small in-memory representation of precomputed pseudobulk samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import perturbation_genes


@dataclass
class PseudobulkDataset:
    expression: np.ndarray
    reference: np.ndarray
    delta: np.ndarray
    perturbation_features: np.ndarray
    samples: pd.DataFrame
    genes: pd.DataFrame
    perturbation_vocabulary: list[str]

    def split_mask(self, name: str, include_control: bool = True) -> np.ndarray:
        mask = self.samples["split"].to_numpy() == name
        if not include_control:
            mask &= self.samples["perturbation"].to_numpy() != "control"
        return mask


def _build_references(expression: np.ndarray, samples: pd.DataFrame) -> np.ndarray:
    control = samples.loc[samples["perturbation"] == "control"].copy()
    duplicated = control.duplicated(["split", "gemgroup"], keep=False)
    if duplicated.any():
        raise ValueError("expected exactly one control pseudobulk per split and gemgroup")
    lookup = {
        (row.split, int(row.gemgroup)): expression[index]
        for index, row in control.iterrows()
    }
    missing = sorted(
        {
            (row.split, int(row.gemgroup))
            for row in samples.itertuples()
            if (row.split, int(row.gemgroup)) not in lookup
        }
    )
    if missing:
        raise ValueError(f"missing control references for: {missing}")
    return np.stack(
        [lookup[(row.split, int(row.gemgroup))] for row in samples.itertuples()]
    ).astype(np.float32)


def _build_perturbation_features(
    samples: pd.DataFrame, vocabulary: list[str]
) -> np.ndarray:
    gene_to_index = {gene: index for index, gene in enumerate(vocabulary)}
    features = np.zeros((len(samples), len(vocabulary)), dtype=np.float32)
    for row_index, label in enumerate(samples["perturbation"].astype(str)):
        for gene in perturbation_genes(label):
            features[row_index, gene_to_index[gene]] = 1.0
    return features


def load_pseudobulk_dataset(directory: Path) -> PseudobulkDataset:
    expression = np.asarray(np.load(directory / "expression.npy"), dtype=np.float32)
    samples = pd.read_csv(directory / "samples.csv")
    genes = pd.read_csv(directory / "genes.csv")
    if expression.shape != (len(samples), len(genes)):
        raise ValueError("expression shape does not match sample/gene metadata")
    if not samples["sample_id"].is_unique:
        raise ValueError("sample_id must be unique")
    if not genes["gene"].is_unique:
        raise ValueError("gene names must be unique")

    vocabulary = sorted(
        {
            gene
            for label in samples["perturbation"].astype(str).unique()
            for gene in perturbation_genes(label)
        }
    )
    reference = _build_references(expression, samples)
    features = _build_perturbation_features(samples, vocabulary)
    return PseudobulkDataset(
        expression=expression,
        reference=reference,
        delta=expression - reference,
        perturbation_features=features,
        samples=samples,
        genes=genes,
        perturbation_vocabulary=vocabulary,
    )

