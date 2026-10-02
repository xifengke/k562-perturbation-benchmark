"""PyTorch MLP baseline for perturbation-response prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """Predict expression delta from perturbation identity.

    The perturbation input is multi-hot, so a double perturbation is invariant to
    gene order. ``active_perturbations`` masks identities that never occur in
    training; this prevents unseen-gene columns from retaining random weights.

    The matched control state is used as an explicit output skip connection by the
    caller: ``predicted_expression = reference + predicted_delta``. Keeping noisy
    batch-level control profiles out of the network limits its input to the
    perturbation identity.
    """

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        hidden_dims: Sequence[int],
        dropout: float,
        active_perturbations: torch.Tensor,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if active_perturbations.shape != (n_perturbations,):
            raise ValueError("active perturbation mask has the wrong shape")

        self.n_genes = int(n_genes)
        self.n_perturbations = int(n_perturbations)
        self.register_buffer(
            "active_perturbations", active_perturbations.float().clone()
        )

        dimensions = [n_perturbations, *map(int, hidden_dims)]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], n_genes))
        self.network = nn.Sequential(*layers)

    def forward(
        self, reference: torch.Tensor, perturbation: torch.Tensor
    ) -> torch.Tensor:
        if reference.shape[-1] != self.n_genes:
            raise ValueError("reference input has the wrong number of genes")
        known_perturbation = perturbation * self.active_perturbations
        has_known_action = (known_perturbation.sum(dim=-1, keepdim=True) > 0).float()
        return self.network(known_perturbation) * has_known_action
