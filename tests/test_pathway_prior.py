"""Checks for static Reactome features used in the perturbation benchmark."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_vc.data.pathway_prior import build_reactome_features  # noqa: E402


class ReactomeFeatureTests(unittest.TestCase):
    def test_union_shared_and_duplicate_pathways(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "pathways.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "ReactomePathways.gmt",
                    "\n".join(
                        [
                            "AB pathway\tR-HSA-1\tA\tB",
                            "same observed genes\tR-HSA-2\tA\tB\tZ",
                            "BC pathway\tR-HSA-3\tB\tC",
                            "single gene\tR-HSA-4\tC",
                            "mouse pathway\tR-MMU-5\tA\tC",
                        ]
                    ),
                )
            identity = np.asarray(
                [
                    [0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            features, names, audit = build_reactome_features(
                identity, ["A", "B", "C", "D"], archive_path
            )
            self.assertEqual(features.shape, (5, 8))
            self.assertEqual(len(names), 8)
            self.assertEqual(audit["distinct_pathway_patterns"], 2)
            self.assertEqual(audit["unannotated_genes"], ["D"])
            np.testing.assert_array_equal(features[0], np.zeros(8))
            np.testing.assert_array_equal(features[1, 4:], [1, 0, 0, 0])
            np.testing.assert_array_equal(features[2, 4:], [1, 1, 1, 0])
            np.testing.assert_array_equal(features[3, 4:], [1, 1, 0, 1])
            np.testing.assert_array_equal(features[4, 4:], np.zeros(4))

    def test_shuffle_preserves_feature_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "pathways.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "ReactomePathways.gmt", "AB\tR-HSA-1\tA\tB\nBC\tR-HSA-2\tB\tC\n",
                )
            identity = np.eye(4, dtype=np.float32)
            real, _, _ = build_reactome_features(
                identity, ["A", "B", "C", "D"], archive_path
            )
            shuffled, _, audit = build_reactome_features(
                identity, ["A", "B", "C", "D"], archive_path, shuffle_seed=7
            )
            self.assertEqual(real.shape, shuffled.shape)
            np.testing.assert_array_equal(real[:, :4], shuffled[:, :4])
            np.testing.assert_array_equal(
                np.sort(real[:, 4:6], axis=0), np.sort(shuffled[:, 4:6], axis=0)
            )
            self.assertEqual(audit["shuffle_seed"], 7)


if __name__ == "__main__":
    unittest.main()
