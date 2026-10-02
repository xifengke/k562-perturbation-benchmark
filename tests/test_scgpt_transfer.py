"""Small checks for scGPT gene mapping and raw-control value preparation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.data.scgpt_controls import _bin_row  # noqa: E402
from mini_vc.models.scgpt_transfer import map_perturbation_genes  # noqa: E402


class ScGPTTransferTests(unittest.TestCase):
    def test_alias_and_oov_genes_are_explicit(self) -> None:
        tokens, oov, audit = map_perturbation_genes(
            ["ELMSAN1", "UNKNOWN_A", "UNKNOWN_B"],
            {"<pad>": 0, "MIDEAS": 5},
        )
        self.assertEqual(tokens, [5, 0, 0])
        self.assertEqual(oov, [-1, 0, 1])
        self.assertEqual(audit["aliases"], {"ELMSAN1": "MIDEAS"})
        self.assertEqual(audit["oov_genes"], ["UNKNOWN_A", "UNKNOWN_B"])

    def test_binning_keeps_zeros_and_is_repeatable(self) -> None:
        values = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        first = _bin_row(values)
        second = _bin_row(values)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:2], [0.0, 0.0])
        self.assertTrue(np.all((first[2:] >= 1) & (first[2:] <= 50)))


if __name__ == "__main__":
    unittest.main()
