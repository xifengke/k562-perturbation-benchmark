"""Expression encoder/decoder building blocks for Mini-VC."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _hidden_stack(
    dimensions: Sequence[int], dropout: float
) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.extend(
            [
                nn.Linear(int(input_dim), int(output_dim)),
                nn.LayerNorm(int(output_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
    return layers


class ExpressionEncoder(nn.Module):
    """Compress a log-normalized expression vector into a latent state."""

    def __init__(
        self,
        n_genes: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [n_genes, *map(int, hidden_dims)]
        layers = _hidden_stack(dimensions, dropout)
        final_dim = dimensions[-1]
        layers.append(nn.Linear(final_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.network(expression)


class ExpressionDecoder(nn.Module):
    """Decode a latent cell state back to gene-expression space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int],
        n_genes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [latent_dim, *map(int, hidden_dims)]
        layers = _hidden_stack(dimensions, dropout)
        final_dim = dimensions[-1]
        layers.append(nn.Linear(final_dim, n_genes))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class ExpressionAutoencoder(nn.Module):
    """Convenience wrapper exposing encode/decode explicitly."""

    def __init__(
        self,
        n_genes: int,
        encoder_hidden_dims: Sequence[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = ExpressionEncoder(
            n_genes, encoder_hidden_dims, latent_dim, dropout
        )
        self.decoder = ExpressionDecoder(
            latent_dim, list(reversed(encoder_hidden_dims)), n_genes, dropout
        )

    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        return self.encoder(expression)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(expression))

