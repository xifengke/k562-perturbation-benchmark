"""Loss functions for the Mini-VC latent-state model."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mini_vc.models.mini_vc import MiniVCOutput


def mini_vc_loss(
    output: MiniVCOutput,
    reference_expression: torch.Tensor,
    target_expression: torch.Tensor,
    response_weight: float,
    reconstruction_weight: float,
    latent_alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine response, reconstruction, and latent alignment losses."""
    if output.reconstructed_target is None or output.target_latent is None:
        raise ValueError("target_expression must be passed to MiniVirtualCell.forward")
    response = F.mse_loss(output.predicted_expression, target_expression)
    reference_reconstruction = F.mse_loss(
        output.reconstructed_reference, reference_expression
    )
    target_reconstruction = F.mse_loss(
        output.reconstructed_target, target_expression
    )
    reconstruction = 0.5 * (reference_reconstruction + target_reconstruction)
    latent_alignment = F.mse_loss(
        output.predicted_latent, output.target_latent.detach()
    )
    total = (
        response_weight * response
        + reconstruction_weight * reconstruction
        + latent_alignment_weight * latent_alignment
    )
    return total, {
        "response": response,
        "reference_reconstruction": reference_reconstruction,
        "target_reconstruction": target_reconstruction,
        "reconstruction": reconstruction,
        "latent_alignment": latent_alignment,
    }
