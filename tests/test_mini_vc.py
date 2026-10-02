"""Architecture and gradient tests for Mini Virtual Cell."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.models import GenePerturbationEncoder, MiniVirtualCell  # noqa: E402
from mini_vc.training import mini_vc_loss  # noqa: E402


class MiniVCTests(unittest.TestCase):
    def _model(self) -> MiniVirtualCell:
        torch.manual_seed(4)
        return MiniVirtualCell(
            n_genes=12,
            n_perturbations=4,
            encoder_hidden_dims=[8],
            latent_dim=5,
            perturbation_embedding_dim=3,
            transition_hidden_dims=[7],
            dropout=0.0,
            active_perturbations=torch.tensor([True, True, True, False]),
        )

    def test_forward_shapes_and_loss_backward(self) -> None:
        model = self._model()
        reference = torch.rand(6, 12)
        target = torch.rand(6, 12)
        perturbation = torch.zeros(6, 4)
        perturbation[:, 0] = 1.0
        output = model(reference, perturbation, target)
        self.assertEqual(tuple(output.predicted_expression.shape), (6, 12))
        self.assertEqual(tuple(output.predicted_latent.shape), (6, 5))
        loss, components = mini_vc_loss(
            output, reference, target, 1.0, 0.2, 0.1
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(components),
            {
                "response",
                "reference_reconstruction",
                "target_reconstruction",
                "reconstruction",
                "latent_alignment",
            },
        )
        self.assertIsNotNone(model.autoencoder.encoder.network[0].weight.grad)
        self.assertIsNotNone(model.transition.network[-1].weight.grad)

    def test_double_embedding_is_sum(self) -> None:
        encoder = GenePerturbationEncoder(
            3, 2, torch.tensor([True, True, True])
        )
        single_a, _ = encoder(torch.tensor([[1.0, 0.0, 0.0]]))
        single_b, _ = encoder(torch.tensor([[0.0, 1.0, 0.0]]))
        double, _ = encoder(torch.tensor([[1.0, 1.0, 0.0]]))
        torch.testing.assert_close(double, single_a + single_b)

    def test_zero_or_unknown_action_keeps_latent_state(self) -> None:
        model = self._model().eval()
        reference = torch.rand(2, 12)
        zero = torch.zeros(2, 4)
        unknown = zero.clone()
        unknown[:, 3] = 1.0
        with torch.no_grad():
            zero_output = model(reference, zero)
            unknown_output = model(reference, unknown)
        torch.testing.assert_close(
            zero_output.predicted_latent, zero_output.reference_latent
        )
        torch.testing.assert_close(
            unknown_output.predicted_latent, unknown_output.reference_latent
        )
        torch.testing.assert_close(
            zero_output.predicted_expression,
            zero_output.reconstructed_reference,
        )

    def test_inference_does_not_require_target(self) -> None:
        output = self._model().eval()(torch.rand(3, 12), torch.zeros(3, 4))
        self.assertIsNone(output.target_latent)
        self.assertIsNone(output.reconstructed_target)


if __name__ == "__main__":
    unittest.main()

