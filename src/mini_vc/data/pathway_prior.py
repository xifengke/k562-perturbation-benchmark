"""Static Reactome pathway features for one- and two-gene perturbations.

These annotations describe published gene products and pathways. They are not
measurements of protein abundance or activity in the Norman K562 cells.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_reactome_features(
    identities: np.ndarray,
    vocabulary: list[str],
    archive_path: Path,
    *,
    min_genes: int = 2,
    max_genes: int = 30,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Add pathway union and shared-pathway indicators to gene identity.

    Pathways that have the same intersection with the perturbation vocabulary
    are collapsed because they would generate identical model columns. The
    selection uses only the fixed gene vocabulary and public annotations, never
    any expression target, validation metric, or held-out response.
    """
    identities = np.asarray(identities, dtype=np.float32)
    if identities.ndim != 2 or identities.shape[1] != len(vocabulary):
        raise ValueError("identity matrix does not match perturbation vocabulary")
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("perturbation vocabulary contains duplicate genes")
    if not 1 <= min_genes <= max_genes:
        raise ValueError("invalid pathway gene-count limits")
    if np.any((identities != 0) & (identities != 1)) or np.any(identities.sum(axis=1) > 2):
        raise ValueError("expected binary single- or double-perturbation identities")

    archive_bytes = archive_path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".gmt")]
        if len(members) != 1:
            raise ValueError("expected exactly one GMT member in Reactome archive")
        lines = archive.read(members[0]).decode("utf-8").splitlines()

    gene_set = set(vocabulary)
    grouped: dict[tuple[str, ...], list[tuple[str, str]]] = defaultdict(list)
    genes_in_any_pathway: set[str] = set()
    human_pathways = 0
    for line in lines:
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        name, pathway_id, *members = fields
        if not pathway_id.startswith("R-HSA-"):
            continue
        human_pathways += 1
        overlap = tuple(sorted(set(members) & gene_set))
        genes_in_any_pathway.update(overlap)
        if min_genes <= len(overlap) <= max_genes:
            grouped[overlap].append((pathway_id, name))

    patterns = sorted(grouped, key=lambda genes: (len(genes), genes))
    membership = np.asarray(
        [[gene in pattern for pattern in patterns] for gene in vocabulary],
        dtype=np.float32,
    )
    if membership.shape != (len(vocabulary), len(patterns)):
        raise AssertionError("pathway matrix has an unexpected shape")

    shuffled_from = None
    if shuffle_seed is not None:
        permutation = np.random.default_rng(shuffle_seed).permutation(len(vocabulary))
        membership = membership[permutation]
        shuffled_from = {
            gene: vocabulary[int(permutation[index])]
            for index, gene in enumerate(vocabulary)
        }

    pathway_ids = [sorted(grouped[pattern])[0][0] for pattern in patterns]
    counts = identities @ membership
    union = (counts >= 1).astype(np.float32)
    shared = (counts >= 2).astype(np.float32)
    features = np.concatenate([identities, union, shared], axis=1)
    feature_names = (
        [f"gene:{gene}" for gene in vocabulary]
        + [f"pathway_union:{pathway_id}" for pathway_id in pathway_ids]
        + [f"pathway_shared:{pathway_id}" for pathway_id in pathway_ids]
    )
    annotated = sorted(
        gene for index, gene in enumerate(vocabulary) if membership[index].any()
    )
    audit = {
        "source": "Reactome human pathway gene sets",
        "source_url": "https://reactome.org/download/current/ReactomePathways.gmt.zip",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "human_pathways_in_archive": human_pathways,
        "gene_count_filter": [min_genes, max_genes],
        "distinct_pathway_patterns": len(patterns),
        "n_features": features.shape[1],
        "perturbation_genes": len(vocabulary),
        "genes_in_any_reactome_pathway": len(genes_in_any_pathway),
        "genes_in_selected_patterns": len(annotated),
        "unannotated_genes": sorted(gene_set - set(annotated)),
        "shuffle_seed": shuffle_seed,
        "shuffled_from": shuffled_from,
        "pathway_patterns": [
            {
                "representative_id": pathway_id,
                "members_among_perturbations": list(pattern),
                "equivalent_pathways": [
                    {"id": pid, "name": name} for pid, name in sorted(grouped[pattern])
                ],
            }
            for pathway_id, pattern in zip(pathway_ids, patterns)
        ],
    }
    return features, feature_names, audit
