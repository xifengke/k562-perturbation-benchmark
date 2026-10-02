"""Tests for the PyTorch MLP baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.models import MLPBaseline  # noqa: E402


class MLPBaselineTests(unittest.TestCase):
    def _model(self) -> MLPBaseline:
        torch.manual_seed(1)
        return MLPBaseline(
            n_genes=4,
            n_perturbations=3,
            hidden_dims=[8],
            dropout=0.0,
            active_perturbations=torch.tensor([True, True, False]),
        ).eval()

    def test_output_shape_and_backward(self) -> None:
        model = self._model().train()
        prediction = model(torch.randn(5, 4), torch.zeros(5, 3))
        self.assertEqual(tuple(prediction.shape), (5, 4))
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_unseen_perturbation_is_masked(self) -> None:
        model = self._model()
        reference = torch.randn(2, 4)
        no_perturbation = torch.zeros(2, 3)
        unseen_perturbation = no_perturbation.clone()
        unseen_perturbation[:, 2] = 1.0
        torch.testing.assert_close(
            model(reference, no_perturbation),
            model(reference, unseen_perturbation),
        )
        torch.testing.assert_close(
            model(reference, unseen_perturbation), torch.zeros(2, 4)
        )

    def test_delta_is_reference_invariant(self) -> None:
        model = self._model()
        perturbation = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(
            model(torch.zeros(1, 4), perturbation),
            model(torch.ones(1, 4), perturbation),
        )

    def test_double_perturbation_is_order_independent(self) -> None:
        model = self._model()
        reference = torch.randn(1, 4)
        first = torch.tensor([[1.0, 1.0, 0.0]])
        second = torch.tensor([[1.0, 1.0, 0.0]])
        torch.testing.assert_close(model(reference, first), model(reference, second))


if __name__ == "__main__":
    unittest.main()
