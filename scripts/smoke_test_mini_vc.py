"""Run a short forward/backward check on training pseudobulk samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.data.dataset import load_pseudobulk_dataset  # noqa: E402
from mini_vc.models import build_mini_vc  # noqa: E402
from mini_vc.training import mini_vc_loss  # noqa: E402
from mini_vc.training.torch_runner import (  # noqa: E402
    choose_device,
    set_reproducible_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mini_vc.json"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    set_reproducible_seed(int(config["seed"]))
    device = choose_device(config["training"]["device"])
    scheme = config["schemes"][0]
    dataset = load_pseudobulk_dataset(
        REPO_ROOT / config["data_dir"] / scheme
    )
    train_mask = dataset.split_mask("train", include_control=False)
    active = dataset.perturbation_features[train_mask].sum(axis=0) > 0
    model = build_mini_vc(
        config["model"],
        n_genes=dataset.expression.shape[1],
        n_perturbations=dataset.perturbation_features.shape[1],
        active_perturbations=torch.from_numpy(active),
    ).to(device)

    indices = np.flatnonzero(train_mask)[: int(config["training"]["batch_size"])]
    reference = torch.from_numpy(dataset.reference[indices]).to(device)
    perturbation = torch.from_numpy(dataset.perturbation_features[indices]).to(device)
    target = torch.from_numpy(dataset.expression[indices]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    initial_embedding = model.perturbation_encoder.embedding.detach().clone()

    def objective() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = model(reference, perturbation, target)
        return mini_vc_loss(
            output,
            reference,
            target,
            response_weight=float(config["training"]["response_weight"]),
            reconstruction_weight=float(
                config["training"]["reconstruction_weight"]
            ),
            latent_alignment_weight=float(
                config["training"]["latent_alignment_weight"]
            ),
        )

    model.eval()
    with torch.no_grad():
        initial_loss, _ = objective()
    model.train()
    last_components = None
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, last_components = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss in Mini-VC smoke test")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss, final_components = objective()
        output = model(reference, perturbation)

    if final_loss >= initial_loss:
        raise RuntimeError(
            f"smoke loss did not decrease: {initial_loss.item()} -> {final_loss.item()}"
        )
    embedding_change = (
        model.perturbation_encoder.embedding.detach() - initial_embedding
    ).abs().max().item()
    if embedding_change == 0.0:
        raise RuntimeError("perturbation embeddings did not update")
    assert last_components is not None
    print(f"device={device}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"batch={len(indices)} genes={dataset.expression.shape[1]}")
    print(f"latent_shape={tuple(output.predicted_latent.shape)}")
    print(f"prediction_shape={tuple(output.predicted_expression.shape)}")
    print(f"initial_total_loss={initial_loss.item():.6f}")
    print(f"final_total_loss={final_loss.item():.6f}")
    print(f"max_embedding_update={embedding_change:.6f}")
    print(
        "final_components="
        + ", ".join(
            f"{name}:{value.item():.6f}"
            for name, value in final_components.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
