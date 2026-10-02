"""Verify official scGPT checkpoint loading and one GPU training step."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.data.dataset import load_pseudobulk_dataset  # noqa: E402
from mini_vc.data.scgpt_controls import build_control_token_bank  # noqa: E402
from mini_vc.models.scgpt_transfer import (  # noqa: E402
    ScGPTPerturbationRegressor,
    load_scgpt_encoder,
    map_perturbation_genes,
)


def main() -> None:
    torch.set_num_threads(4)
    scheme = "unseen_combo_fold0"
    data_dir = ROOT / "data" / "processed" / "norman_scgpt_folds"
    dataset = load_pseudobulk_dataset(data_dir / scheme)
    checkpoint_dir = ROOT / "checkpoints" / "scgpt_whole_human"
    backbone, vocab, audit = load_scgpt_encoder(checkpoint_dir)
    bank = build_control_token_bank(
        ROOT / "data" / "raw" / "NormanWeissman2019_filtered.h5ad",
        data_dir / "cell_splits.csv.gz",
        dataset,
        scheme,
        vocab,
        data_dir / scheme / "scgpt_control_tokens.npz",
    )
    token_ids, oov, coverage = map_perturbation_genes(
        dataset.perturbation_vocabulary, vocab
    )
    model = ScGPTPerturbationRegressor(
        backbone, token_ids, oov, len(dataset.genes), train_backbone=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    group = int(dataset.samples.loc[dataset.samples.split == "train", "gemgroup"].iloc[0])
    mask = (dataset.samples.split == "train") & (dataset.samples.gemgroup == group)
    mask &= dataset.samples.perturbation != "control"
    rows = mask.to_numpy().nonzero()[0][:8]
    control_gene_ids, control_values = bank.get("train", group)
    reference = torch.tensor(dataset.reference[rows], device=device)
    perturbation = torch.tensor(dataset.perturbation_features[rows], device=device)
    target = torch.tensor(dataset.expression[rows], device=device)
    predicted = model(
        reference,
        perturbation,
        torch.tensor(control_gene_ids, device=device),
        torch.tensor(control_values, device=device),
    )
    loss = (predicted - target).square().mean()
    loss.backward()
    print({"checkpoint": audit, "coverage": coverage})
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "prediction_shape": tuple(predicted.shape),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if device.type == "cuda"
            else None,
        }
    )


if __name__ == "__main__":
    main()
