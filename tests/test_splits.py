"""Small deterministic tests for leakage-aware split construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.data.splits import (  # noqa: E402
    make_seen_split,
    make_unseen_combo_folds,
    make_unseen_combo_split,
    make_unseen_gene_split,
    perturbation_genes,
)


def example_obs() -> pd.DataFrame:
    rows = []
    index = []
    conditions = {
        "control": 0,
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
        "A_B": 2,
        "A_C": 2,
        "B_D": 2,
        "C_D": 2,
    }
    for gemgroup in (1, 2, 3):
        for condition, nperts in conditions.items():
            for replicate in range(5):
                index.append(f"{condition}-{gemgroup}-{replicate}")
                rows.append((condition, nperts, gemgroup))
    return pd.DataFrame(
        rows, index=index, columns=["perturbation", "nperts", "gemgroup"]
    )


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = example_obs()

    def test_perturbation_representation_is_order_independent(self) -> None:
        self.assertEqual(perturbation_genes("B_A"), ("A", "B"))
        self.assertEqual(perturbation_genes("control"), ())

    def test_seen_split_holds_out_complete_batches(self) -> None:
        result = make_seen_split(self.obs, seed=42)
        table = self.obs.assign(split=result.assignments)
        split_counts = table.groupby("gemgroup")["split"].nunique()
        self.assertTrue((split_counts == 1).all())
        for condition in self.obs["perturbation"].unique():
            self.assertEqual(
                set(table.loc[table.perturbation == condition, "split"]),
                {"train", "validation", "test"},
            )

    def test_unseen_combo_conditions_do_not_cross_splits(self) -> None:
        result = make_unseen_combo_split(self.obs, 42, 0.25, 0.25)
        table = self.obs.assign(split=result.assignments)
        for condition in self.obs.loc[self.obs.nperts == 2, "perturbation"].unique():
            self.assertEqual(
                table.loc[table.perturbation == condition, "split"].nunique(), 1
            )
        self.assertTrue(
            (table.loc[table.nperts == 1, "split"] == "train").all()
        )
        self.assertEqual(
            set(table.loc[table.perturbation == "control", "split"]),
            {"train", "validation", "test"},
        )

    def test_five_folds_test_every_double_once(self) -> None:
        doubles = {
            f"G{i}_H{i}": 2 for i in range(15)
        }
        singles = {f"G{i}": 1 for i in range(15)} | {
            f"H{i}": 1 for i in range(15)
        }
        rows = [
            (condition, nperts, gemgroup)
            for gemgroup in (1, 2)
            for condition, nperts in ({"control": 0} | singles | doubles).items()
            for _ in range(20)
        ]
        obs = pd.DataFrame(rows, columns=["perturbation", "nperts", "gemgroup"])
        obs.index = [f"cell-{i}" for i in range(len(obs))]
        folds = make_unseen_combo_folds(obs, seed=42, validation_fraction=0.10)
        tested = []
        for result in folds.values():
            table = obs.assign(split=result.assignments)
            test = set(table.loc[(table.nperts == 2) & (table.split == "test"), "perturbation"])
            tested.extend(test)
            self.assertTrue((table.loc[table.nperts == 1, "split"] == "train").all())
            self.assertTrue((table.loc[table.nperts == 2].groupby("perturbation")["split"].nunique() == 1).all())
        self.assertEqual(set(tested), set(doubles))
        self.assertEqual(len(tested), len(doubles))

    def test_unseen_gene_is_absent_from_training_conditions(self) -> None:
        result = make_unseen_gene_split(self.obs, 42, 0.25, 0.25)
        table = self.obs.assign(split=result.assignments)
        held_out = set(result.details["validation_genes"]) | set(
            result.details["test_genes"]
        )
        train_conditions = table.loc[
            (table.split == "train") & (table.perturbation != "control"),
            "perturbation",
        ].unique()
        train_genes = {
            gene for condition in train_conditions for gene in perturbation_genes(condition)
        }
        self.assertTrue(train_genes.isdisjoint(held_out))


if __name__ == "__main__":
    unittest.main()
