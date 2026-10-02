"""Train, checkpoint, reload, and evaluate the Mini-VC model."""

from __future__ import annotations

import csv
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from mini_vc.data.dataset import load_pseudobulk_dataset
from mini_vc.evaluation import save_evaluation
from mini_vc.models import MiniVirtualCell, build_mini_vc
from mini_vc.training.losses import mini_vc_loss
from mini_vc.training.torch_runner import choose_device, set_reproducible_seed


def _tensor_dataset(dataset, mask: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(dataset.reference[mask]),
        torch.from_numpy(dataset.perturbation_features[mask]),
        torch.from_numpy(dataset.expression[mask]),
    )


def _run_epoch(
    model: MiniVirtualCell,
    loader: DataLoader,
    device: torch.device,
    training_config: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals: dict[str, float] = {}
    n_samples = 0
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for reference, perturbation, target in loader:
            reference = reference.to(device)
            perturbation = perturbation.to(device)
            target = target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(reference, perturbation, target)
            loss, components = mini_vc_loss(
                output,
                reference,
                target,
                response_weight=float(training_config["response_weight"]),
                reconstruction_weight=float(
                    training_config["reconstruction_weight"]
                ),
                latent_alignment_weight=float(
                    training_config["latent_alignment_weight"]
                ),
            )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training_config["gradient_clip_norm"]),
                )
                optimizer.step()
            batch_size = len(target)
            batch_values = {"total": loss, **components}
            for name, value in batch_values.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * batch_size
            n_samples += batch_size
    if n_samples == 0:
        raise ValueError("training/evaluation loader contains no samples")
    return {name: value / n_samples for name, value in totals.items()}


def _predict_all(
    model: MiniVirtualCell,
    dataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    all_mask = np.ones(len(dataset.expression), dtype=bool)
    loader = DataLoader(
        _tensor_dataset(dataset, all_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    model.eval()
    with torch.inference_mode():
        for reference, perturbation, _ in loader:
            output = model(reference.to(device), perturbation.to(device))
            predictions.append(output.predicted_expression.cpu().numpy())
    return np.concatenate(predictions).astype(np.float32, copy=False)


def load_mini_vc_checkpoint(
    checkpoint_path: Path, requested_device: str = "cpu"
) -> tuple[MiniVirtualCell, dict]:
    """Recreate an inference-ready Mini-VC from a saved checkpoint."""
    device = choose_device(requested_device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    state_dict = checkpoint["state_dict"]
    active = state_dict["perturbation_encoder.active_perturbations"]
    model = build_mini_vc(
        checkpoint["model_config"],
        n_genes=int(checkpoint["n_genes"]),
        n_perturbations=int(checkpoint["n_perturbations"]),
        active_perturbations=active,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def train_mini_vc_scheme(
    config: dict,
    dataset,
    scheme: str,
    output_dir: Path,
) -> dict:
    seed = int(config["seed"])
    training_config = config["training"]
    set_reproducible_seed(seed)
    device = choose_device(training_config.get("device", "auto"))

    train_mask = dataset.split_mask("train", include_control=False)
    validation_mask = dataset.split_mask("validation", include_control=False)
    active = dataset.perturbation_features[train_mask].sum(axis=0) > 0
    model = build_mini_vc(
        config["model"],
        n_genes=dataset.expression.shape[1],
        n_perturbations=dataset.perturbation_features.shape[1],
        active_perturbations=torch.from_numpy(active),
    ).to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _tensor_dataset(dataset, train_mask),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _tensor_dataset(dataset, validation_mask),
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["lr_scheduler_factor"]),
        patience=int(training_config["lr_scheduler_patience"]),
        min_lr=float(training_config["minimum_learning_rate"]),
    )

    patience = int(training_config["patience"])
    min_delta = float(training_config["min_delta"])
    best_response = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, training_config, optimizer
        )
        validation_metrics = _run_epoch(
            model, validation_loader, device, training_config, optimizer=None
        )
        validation_response = validation_metrics["response"]
        scheduler.step(validation_response)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history.append(row)

        if epoch == 1 or epoch % int(training_config["log_every"]) == 0:
            print(
                f"  {scheme} mini_vc epoch={epoch:03d} "
                f"train_response={train_metrics['response']:.6f} "
                f"validation_response={validation_response:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if validation_response < best_response - min_delta:
            best_response = validation_response
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"  {scheme} mini_vc early stop at epoch {epoch}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Mini-VC training did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed_seconds = time.perf_counter() - started

    model_dir = output_dir / scheme / "mini_vc"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": "mini_vc",
        "scheme": scheme,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "model_config": config["model"],
        "n_genes": dataset.expression.shape[1],
        "n_perturbations": dataset.perturbation_features.shape[1],
        "perturbation_vocabulary": dataset.perturbation_vocabulary,
        "genes": dataset.genes["gene"].astype(str).tolist(),
        "best_epoch": best_epoch,
        "best_validation_response_mse": best_response,
        "seed": seed,
    }
    torch.save(checkpoint, model_dir / "checkpoint.pt")
    with (model_dir / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    predicted_expression = _predict_all(
        model, dataset, int(training_config["batch_size"]), device
    )
    predicted_delta = predicted_expression - dataset.reference
    return save_evaluation(
        output_dir,
        "mini_vc",
        scheme,
        dataset,
        predicted_delta,
        config["de_top_k"],
        {
            "best_epoch": best_epoch,
            "best_validation_response_mse": best_response,
            "epochs_trained": len(history),
            "training_seconds": elapsed_seconds,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "n_train_active_perturbations": int(active.sum()),
            "n_total_perturbations": int(len(active)),
        },
    )


def run_mini_vc(config: dict, repo_root: Path) -> pd.DataFrame:
    data_dir = (repo_root / config["data_dir"]).resolve()
    output_dir = (repo_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scheme in config["schemes"]:
        print(f"Loading pseudobulk scheme: {scheme}", flush=True)
        dataset = load_pseudobulk_dataset(data_dir / scheme)
        rows.append(train_mini_vc_scheme(config, dataset, scheme, output_dir))

    comparison = pd.DataFrame(rows).sort_values(["scheme", "model"])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    baseline_path = repo_root / "results" / "baselines" / "model_comparison.csv"
    if baseline_path.exists():
        baselines = pd.read_csv(baseline_path)
        combined = pd.concat([baselines, comparison], ignore_index=True, sort=False)
        combined.sort_values(["scheme", "model"]).to_csv(
            output_dir / "combined_model_comparison.csv", index=False
        )
    return comparison

