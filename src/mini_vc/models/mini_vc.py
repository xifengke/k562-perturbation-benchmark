"""Encoder-Transition-Decoder Mini Virtual Cell model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .autoencoder import ExpressionAutoencoder


@dataclass
class MiniVCOutput:
    predicted_expression: torch.Tensor
    reconstructed_reference: torch.Tensor
    reference_latent: torch.Tensor
    predicted_latent: torch.Tensor
    reconstructed_target: torch.Tensor | None = None
    target_latent: torch.Tensor | None = None


class GenePerturbationEncoder(nn.Module):
    """Map an order-invariant multi-hot gene action to a summed embedding."""

    def __init__(
        self,
        n_perturbations: int,
        embedding_dim: int,
        active_perturbations: torch.Tensor,
    ) -> None:
        super().__init__()
        if active_perturbations.shape != (n_perturbations,):
            raise ValueError("active perturbation mask has the wrong shape")
        self.embedding = nn.Parameter(torch.empty(n_perturbations, embedding_dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)
        self.register_buffer(
            "active_perturbations", active_perturbations.float().clone()
        )

    def forward(
        self, perturbation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        known = perturbation * self.active_perturbations
        action_embedding = known @ self.embedding
        has_known_action = (known.sum(dim=-1, keepdim=True) > 0).float()
        return action_embedding, has_known_action


class LatentTransition(nn.Module):
    """Predict a residual update to the latent cellular state."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [latent_dim + action_dim, *map(int, hidden_dims)]
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
        final = nn.Linear(dimensions[-1], latent_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action_embedding: torch.Tensor,
        has_known_action: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.network(torch.cat([state, action_embedding], dim=-1))
        return state + delta * has_known_action


class MiniVirtualCell(nn.Module):
    """A compact transcriptome-only virtual-cell baseline."""

    def __init__(
        self,
        n_genes: int,
        n_perturbations: int,
        encoder_hidden_dims: Sequence[int],
        latent_dim: int,
        perturbation_embedding_dim: int,
        transition_hidden_dims: Sequence[int],
        dropout: float,
        active_perturbations: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.autoencoder = ExpressionAutoencoder(
            n_genes=n_genes,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.perturbation_encoder = GenePerturbationEncoder(
            n_perturbations=n_perturbations,
            embedding_dim=perturbation_embedding_dim,
            active_perturbations=active_perturbations,
        )
        self.transition = LatentTransition(
            latent_dim=latent_dim,
            action_dim=perturbation_embedding_dim,
            hidden_dims=transition_hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        reference_expression: torch.Tensor,
        perturbation: torch.Tensor,
        target_expression: torch.Tensor | None = None,
    ) -> MiniVCOutput:
        if reference_expression.shape[-1] != self.n_genes:
            raise ValueError("reference input has the wrong number of genes")
        reference_latent = self.autoencoder.encode(reference_expression)
        reconstructed_reference = self.autoencoder.decode(reference_latent)
        action_embedding, has_known_action = self.perturbation_encoder(perturbation)
        predicted_latent = self.transition(
            reference_latent, action_embedding, has_known_action
        )
        predicted_expression = self.autoencoder.decode(predicted_latent)

        target_latent = None
        reconstructed_target = None
        if target_expression is not None:
            target_latent = self.autoencoder.encode(target_expression)
            reconstructed_target = self.autoencoder.decode(target_latent)
        return MiniVCOutput(
            predicted_expression=predicted_expression,
            reconstructed_reference=reconstructed_reference,
            reference_latent=reference_latent,
            predicted_latent=predicted_latent,
            reconstructed_target=reconstructed_target,
            target_latent=target_latent,
        )


def build_mini_vc(
    config: dict,
    n_genes: int,
    n_perturbations: int,
    active_perturbations: torch.Tensor,
) -> MiniVirtualCell:
    """Build the model from the model section of the experiment config."""
    return MiniVirtualCell(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        encoder_hidden_dims=config["encoder_hidden_dims"],
        latent_dim=int(config["latent_dim"]),
        perturbation_embedding_dim=int(config["perturbation_embedding_dim"]),
        transition_hidden_dims=config["transition_hidden_dims"],
        dropout=float(config["dropout"]),
        active_perturbations=active_perturbations,
    )
