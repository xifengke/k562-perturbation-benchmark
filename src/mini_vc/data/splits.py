"""Leakage-aware split construction for Norman 2019.

Splits are constructed from metadata before HVG fitting. A split assignment is
stored for every QC-passing cell; held-out conditions are never divided across
train and test in the unseen-condition schemes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitResult:
    """Cell-level assignments plus human-readable held-out entities."""

    assignments: np.ndarray
    details: dict


def perturbation_genes(label: str) -> tuple[str, ...]:
    """Convert a harmonized Norman label into an order-independent gene tuple."""
    if label == "control":
        return ()
    return tuple(sorted(label.split("_")))


def _stable_uniform(cell_id: str, seed: int, namespace: str) -> float:
    payload = f"{namespace}|{seed}|{cell_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return value / float(2**64)


def _partition_control_cells(
    obs: pd.DataFrame,
    assignments: np.ndarray,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    namespace: str,
) -> None:
    control_positions = np.flatnonzero(obs["perturbation"].astype(str).to_numpy() == "control")
    train_cutoff = 1.0 - validation_fraction - test_fraction
    validation_cutoff = 1.0 - test_fraction
    for position in control_positions:
        value = _stable_uniform(str(obs.index[position]), seed, namespace)
        if value < train_cutoff:
            assignments[position] = "train"
        elif value < validation_cutoff:
            assignments[position] = "validation"
        else:
            assignments[position] = "test"


def _split_items(
    items: list[str], seed: int, validation_fraction: float, test_fraction: float
) -> tuple[list[str], list[str], list[str]]:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")
    if len(items) < 3:
        raise ValueError("at least three items are required for a three-way split")

    shuffled = np.asarray(sorted(items), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_validation = max(1, int(round(len(shuffled) * validation_fraction)))
    n_train = len(shuffled) - n_validation - n_test
    if n_train < 1:
        raise ValueError("split fractions leave no training items")
    train = sorted(shuffled[:n_train].tolist())
    validation = sorted(shuffled[n_train : n_train + n_validation].tolist())
    test = sorted(shuffled[n_train + n_validation :].tolist())
    return train, validation, test


def make_seen_split(obs: pd.DataFrame, seed: int) -> SplitResult:
    """Hold out complete gemgroups while retaining every perturbation."""
    batches = np.asarray(sorted(obs["gemgroup"].unique()), dtype=int)
    if len(batches) < 3:
        raise ValueError("seen split requires at least three gemgroups")
    rng = np.random.default_rng(seed)
    rng.shuffle(batches)
    test_batch = int(batches[-1])
    validation_batch = int(batches[-2])
    train_batches = sorted(int(value) for value in batches[:-2])

    batch_values = obs["gemgroup"].to_numpy()
    assignments = np.full(len(obs), "train", dtype="U10")
    assignments[batch_values == validation_batch] = "validation"
    assignments[batch_values == test_batch] = "test"
    details = {
        "meaning": "same perturbations, held-out gemgroups",
        "train_gemgroups": train_batches,
        "validation_gemgroups": [validation_batch],
        "test_gemgroups": [test_batch],
    }
    return SplitResult(assignments, details)


def make_unseen_combo_split(
    obs: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    """Hold out double-perturbation labels while keeping all singles in train."""
    labels = obs[["perturbation", "nperts"]].drop_duplicates()
    doubles = labels.loc[labels["nperts"] == 2, "perturbation"].astype(str).tolist()
    train_double, validation_double, test_double = _split_items(
        doubles, seed, validation_fraction, test_fraction
    )
    validation_set = set(validation_double)
    test_set = set(test_double)

    assignments = np.full(len(obs), "train", dtype="U10")
    perturbations = obs["perturbation"].astype(str).to_numpy()
    assignments[np.isin(perturbations, list(validation_set))] = "validation"
    assignments[np.isin(perturbations, list(test_set))] = "test"
    _partition_control_cells(
        obs,
        assignments,
        seed,
        validation_fraction,
        test_fraction,
        "unseen_combo_control",
    )

    single_genes = set(
        labels.loc[labels["nperts"] == 1, "perturbation"].astype(str).tolist()
    )
    missing_components = sorted(
        {
            gene
            for condition in validation_double + test_double
            for gene in perturbation_genes(condition)
            if gene not in single_genes
        }
    )
    if missing_components:
        raise ValueError(
            "held-out combinations have components without training singles: "
            + ", ".join(missing_components)
        )

    details = {
        "meaning": "held-out double perturbations; both component singles remain in train",
        "train_double_conditions": train_double,
        "validation_double_conditions": validation_double,
        "test_double_conditions": test_double,
        "all_single_conditions_in_train": True,
    }
    return SplitResult(assignments, details)


def make_unseen_combo_folds(
    obs: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    n_folds: int = 5,
) -> dict[str, SplitResult]:
    """Hold out each complete double perturbation in exactly one test fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    labels = obs[["perturbation", "nperts"]].drop_duplicates()
    doubles = sorted(labels.loc[labels["nperts"] == 2, "perturbation"].astype(str))
    if len(doubles) < n_folds:
        raise ValueError("fewer double perturbations than folds")
    n_validation = max(1, int(round(len(doubles) * validation_fraction)))
    if n_validation >= len(doubles) - max(len(chunk) for chunk in np.array_split(doubles, n_folds)):
        raise ValueError("validation fraction leaves no training doubles")

    shuffled = np.asarray(doubles, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    folds: dict[str, SplitResult] = {}
    perturbations = obs["perturbation"].astype(str).to_numpy()
    for fold_index, test_chunk in enumerate(np.array_split(shuffled, n_folds)):
        test_doubles = sorted(test_chunk.tolist())
        remaining = np.asarray(sorted(set(doubles) - set(test_doubles)), dtype=object)
        np.random.default_rng(seed + 1000 + fold_index).shuffle(remaining)
        validation_doubles = sorted(remaining[:n_validation].tolist())
        train_doubles = sorted(remaining[n_validation:].tolist())
        assignments = np.full(len(obs), "train", dtype="U10")
        assignments[np.isin(perturbations, validation_doubles)] = "validation"
        assignments[np.isin(perturbations, test_doubles)] = "test"
        _partition_control_cells(
            obs,
            assignments,
            seed,
            validation_fraction,
            1.0 / n_folds,
            f"unseen_combo_fold{fold_index}_control",
        )
        folds[f"unseen_combo_fold{fold_index}"] = SplitResult(
            assignments,
            {
                "meaning": "five-fold held-out double perturbations; singles remain in train",
                "fold_index": fold_index,
                "n_folds": n_folds,
                "train_double_conditions": train_doubles,
                "validation_double_conditions": validation_doubles,
                "test_double_conditions": test_doubles,
                "all_single_conditions_in_train": True,
            },
        )
    return folds


def make_unseen_gene_split(
    obs: pd.DataFrame,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    """Hold out genes and every condition containing those genes."""
    labels = obs[["perturbation", "nperts"]].drop_duplicates()
    genes = labels.loc[labels["nperts"] == 1, "perturbation"].astype(str).tolist()
    train_genes, validation_genes, test_genes = _split_items(
        genes, seed + 1, validation_fraction, test_fraction
    )
    validation_set = set(validation_genes)
    test_set = set(test_genes)

    condition_split: dict[str, str] = {"control": "control_cell_partition"}
    for condition in labels["perturbation"].astype(str):
        if condition == "control":
            continue
        components = set(perturbation_genes(condition))
        if components & test_set:
            condition_split[condition] = "test"
        elif components & validation_set:
            condition_split[condition] = "validation"
        else:
            condition_split[condition] = "train"

    assignments = np.asarray(
        [condition_split[str(label)] for label in obs["perturbation"].astype(str)],
        dtype="U22",
    )
    _partition_control_cells(
        obs,
        assignments,
        seed,
        validation_fraction,
        test_fraction,
        "unseen_gene_control",
    )
    if not set(np.unique(assignments)).issubset(SPLIT_NAMES):
        raise AssertionError("unseen-gene assignments contain an invalid split")

    train_conditions = sorted(
        condition for condition, split in condition_split.items() if split == "train"
    )
    validation_conditions = sorted(
        condition for condition, split in condition_split.items() if split == "validation"
    )
    test_conditions = sorted(
        condition for condition, split in condition_split.items() if split == "test"
    )
    details = {
        "meaning": "held-out genes and every condition containing them",
        "train_genes": train_genes,
        "validation_genes": validation_genes,
        "test_genes": test_genes,
        "train_conditions": train_conditions,
        "validation_conditions": validation_conditions,
        "test_conditions": test_conditions,
        "note": "MLP and Mini-VC mask held-out identities; Ridge has no fitted effect for their columns",
    }
    return SplitResult(assignments, details)


def make_splits(
    obs: pd.DataFrame,
    schemes: list[str],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, SplitResult]:
    """Build requested split schemes and validate complete assignments."""
    builders = {
        "seen": lambda: make_seen_split(obs, seed),
        "unseen_combo": lambda: make_unseen_combo_split(
            obs, seed, validation_fraction, test_fraction
        ),
        "unseen_gene": lambda: make_unseen_gene_split(
            obs, seed, validation_fraction, test_fraction
        ),
    }
    fold_names = {name for name in schemes if re.fullmatch(r"unseen_combo_fold[0-4]", name)}
    unknown = sorted(set(schemes) - set(builders) - fold_names)
    if unknown:
        raise ValueError("unknown split schemes: " + ", ".join(unknown))

    folds = (
        make_unseen_combo_folds(obs, seed, validation_fraction)
        if fold_names
        else {}
    )
    results = {
        name: folds[name] if name in fold_names else builders[name]()
        for name in schemes
    }
    for name, result in results.items():
        values = set(np.unique(result.assignments))
        if values != set(SPLIT_NAMES):
            raise AssertionError(f"{name} does not contain all split names: {values}")
        if len(result.assignments) != len(obs):
            raise AssertionError(f"{name} assignment length mismatch")
    return results
