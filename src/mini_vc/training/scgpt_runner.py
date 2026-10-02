"""Five-fold frozen and partial-fine-tuned scGPT perturbation experiments."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from mini_vc.data.dataset import load_pseudobulk_dataset
from mini_vc.data.scgpt_controls import build_control_token_bank
from mini_vc.evaluation import save_evaluation
from mini_vc.models.scgpt_transfer import (
    ScGPTPerturbationRegressor,
    load_scgpt_encoder,
    map_perturbation_genes,
)
from mini_vc.training.torch_runner import choose_device, set_reproducible_seed


def _rows_by_group(dataset, split: str) -> dict[int, np.ndarray]:
    samples = dataset.samples
    eligible = (samples["split"] == split) & (samples["perturbation"] != "control")
    return {
        int(group): rows.index.to_numpy(dtype=np.int64)
        for group, rows in samples.loc[eligible].groupby("gemgroup", sort=True)
    }


def _trainable_state(model: ScGPTPerturbationRegressor) -> dict[str, torch.Tensor]:
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable
    }


def _make_control_tensors(bank, device: torch.device) -> dict[tuple[str, int], tuple]:
    return {
        key: (
            torch.from_numpy(bank.gene_ids[index]).to(device),
            torch.from_numpy(bank.values[index]).to(device),
        )
        for index, key in enumerate(bank.keys)
    }


def _get_cached_contexts(model, controls: dict, device: torch.device) -> dict:
    model.eval()
    with torch.inference_mode():
        return {
            key: model.encode_control(gene_ids, values).detach()
            for key, (gene_ids, values) in controls.items()
        }


def _run_split(
    model,
    dataset,
    rows_by_group: dict[int, np.ndarray],
    split: str,
    controls: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    cached_contexts: dict | None,
    rng: np.random.Generator | None = None,
) -> tuple[float, dict[int, np.ndarray]]:
    training = optimizer is not None
    model.train(training)
    group_ids = list(rows_by_group)
    if rng is not None:
        rng.shuffle(group_ids)
    total_loss = 0.0
    total_rows = 0
    predictions = {}
    for group in group_ids:
        rows = rows_by_group[group].copy()
        if rng is not None:
            rng.shuffle(rows)
        reference = torch.from_numpy(dataset.reference[rows]).to(device)
        perturbation = torch.from_numpy(dataset.perturbation_features[rows]).to(device)
        target = torch.from_numpy(dataset.expression[rows]).to(device)
        gene_ids, values = controls[(split, group)]
        cached = None if cached_contexts is None else cached_contexts[(split, group)]
        with torch.set_grad_enabled(training):
            predicted = model(reference, perturbation, gene_ids, values, cached)
            loss = F.mse_loss(predicted, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=5.0,
                )
                optimizer.step()
        total_loss += float(loss.detach()) * len(rows)
        total_rows += len(rows)
        if not training:
            predictions[group] = (rows, predicted.detach().cpu().numpy())
    if total_rows == 0:
        raise ValueError(f"no non-control samples in {split}")
    return total_loss / total_rows, predictions


def train_scgpt_scheme(
    config: dict,
    repo_root: Path,
    scheme: str,
    mode: str,
) -> dict:
    if mode not in {"frozen", "finetuned"}:
        raise ValueError("mode must be frozen or finetuned")
    fold_index = (
        int(scheme.rsplit("fold", 1)[-1])
        if scheme.startswith("unseen_combo_fold")
        else 0
    )
    seed = int(config["seed"]) + fold_index
    set_reproducible_seed(seed)
    torch.set_num_threads(4)
    device = choose_device(config["training"].get("device", "auto"))
    data_dir = repo_root / config["data_dir"]
    dataset = load_pseudobulk_dataset(data_dir / scheme)
    checkpoint_dir = repo_root / config["checkpoint_dir"]
    backbone, vocab, load_audit = load_scgpt_encoder(checkpoint_dir)
    token_ids, oov_indices, coverage = map_perturbation_genes(
        dataset.perturbation_vocabulary, vocab
    )
    bank = build_control_token_bank(
        repo_root / config["raw_path"],
        data_dir / "cell_splits.csv.gz",
        dataset,
        scheme,
        vocab,
        data_dir / scheme / "scgpt_control_tokens.npz",
        n_cells_per_group=int(config["input"]["n_control_cells_per_group"]),
        max_tokens=int(config["input"]["max_tokens"]),
    )
    model = ScGPTPerturbationRegressor(
        backbone,
        token_ids,
        oov_indices,
        len(dataset.genes),
        train_backbone=mode == "finetuned",
        n_unfrozen_layers=int(config["training"]["n_unfrozen_layers"]),
    ).to(device)
    controls = _make_control_tensors(bank, device)
    cached = _get_cached_contexts(model, controls, device) if mode == "frozen" else None

    training = config["training"]
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("backbone.")
    ]
    parameter_groups = [{"params": head_params, "lr": float(training["head_lr"])}]
    if mode == "finetuned":
        backbone_params = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("backbone.")
        ]
        parameter_groups.append(
            {"params": backbone_params, "lr": float(training["backbone_lr"])}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=float(training["weight_decay"])
    )
    train_groups = _rows_by_group(dataset, "train")
    validation_groups = _rows_by_group(dataset, "validation")
    test_groups = _rows_by_group(dataset, "test")
    rng = np.random.default_rng(seed)
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    patience_count = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_mse, _ = _run_split(
            model, dataset, train_groups, "train", controls, device, optimizer, cached, rng
        )
        validation_mse, _ = _run_split(
            model, dataset, validation_groups, "validation", controls, device, None, cached
        )
        history.append(
            {"epoch": epoch, "train_response_mse": train_mse, "validation_response_mse": validation_mse}
        )
        if epoch == 1 or epoch % int(training["log_every"]) == 0:
            print(
                f"{scheme} {mode} epoch={epoch:03d} "
                f"train={train_mse:.6f} validation={validation_mse:.6f}",
                flush=True,
            )
        if validation_mse < best_validation - float(training["min_delta"]):
            best_validation = validation_mse
            best_epoch = epoch
            best_state = _trainable_state(model)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError("no scGPT checkpoint selected")
    model.load_state_dict(best_state, strict=False)
    test_mse, test_predictions = _run_split(
        model, dataset, test_groups, "test", controls, device, None, cached
    )
    predicted_delta = np.zeros_like(dataset.delta)
    for rows, values in test_predictions.values():
        predicted_delta[rows] = values - dataset.reference[rows]

    output_dir = repo_root / config["output_dir"]
    model_name = f"scgpt_{mode}"
    row = save_evaluation(
        output_dir,
        model_name,
        scheme,
        dataset,
        predicted_delta,
        config["de_top_k"],
        {
            "best_epoch": best_epoch,
            "best_validation_response_mse": best_validation,
            "test_sample_mse": test_mse,
            "training_seconds": time.perf_counter() - started,
            "device": str(device),
            "seed": seed,
            "checkpoint_sha256": load_audit["checkpoint_sha256"],
            "n_oov_perturbation_genes": len(coverage["oov_genes"]),
        },
    )
    model_dir = output_dir / scheme / model_name
    torch.save(
        {
            "state_dict": best_state,
            "mode": mode,
            "scheme": scheme,
            "seed": seed,
            "checkpoint_sha256": load_audit["checkpoint_sha256"],
            "gene_vocabulary": dataset.perturbation_vocabulary,
            "output_genes": dataset.genes["gene"].astype(str).tolist(),
            "config": config,
        },
        model_dir / "adapter_checkpoint.pt",
    )
    with (model_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (model_dir / "input_audit.json").write_text(
        json.dumps(
            {
                "checkpoint": load_audit,
                "perturbation_coverage": coverage,
                "input_genes": bank.input_genes,
                "n_control_cells_per_group": bank.gene_ids.shape[1],
                "n_input_genes": bank.gene_ids.shape[2],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return row
