"""PyTorch training and checkpoint loading for the MLP baseline."""

from __future__ import annotations

import csv
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mini_vc.evaluation import save_evaluation
from mini_vc.models.mlp import MLPBaseline


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_mlp_checkpoint(
    checkpoint_path: Path, requested_device: str = "cpu"
) -> tuple[MLPBaseline, dict]:
    """Reconstruct an inference-ready MLP from a saved checkpoint."""
    device = choose_device(requested_device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model_config = checkpoint["model_config"]
    active = checkpoint["state_dict"]["active_perturbations"]
    model = MLPBaseline(
        n_genes=int(model_config["n_genes"]),
        n_perturbations=int(model_config["n_perturbations"]),
        hidden_dims=model_config["hidden_dims"],
        dropout=float(model_config["dropout"]),
        active_perturbations=active,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _tensor_dataset(dataset, mask: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(dataset.reference[mask]),
        torch.from_numpy(dataset.perturbation_features[mask]),
        torch.from_numpy(dataset.delta[mask]),
    )


def _mean_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_samples = 0
    with torch.inference_mode():
        for reference, perturbation, target in loader:
            reference = reference.to(device)
            perturbation = perturbation.to(device)
            target = target.to(device)
            loss = criterion(model(reference, perturbation), target)
            total += loss.item() * len(target)
            n_samples += len(target)
    if not n_samples:
        raise ValueError("evaluation loader contains no samples")
    return total / n_samples


def _predict_all(
    model: nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        _tensor_dataset(dataset, np.ones(len(dataset.expression), dtype=bool)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs = []
    model.eval()
    with torch.inference_mode():
        for reference, perturbation, _ in loader:
            outputs.append(
                model(reference.to(device), perturbation.to(device)).cpu().numpy()
            )
    return np.concatenate(outputs).astype(np.float32, copy=False)


def train_mlp_baseline(
    config: dict,
    dataset,
    scheme: str,
    output_dir: Path,
    top_ks: list[int],
    seed: int,
) -> dict:
    """Train once on train, early-stop on validation, and evaluate test."""
    set_reproducible_seed(seed)
    device = choose_device(config.get("device", "auto"))
    train_mask = dataset.split_mask("train", include_control=False)
    validation_mask = dataset.split_mask("validation", include_control=False)

    active = dataset.perturbation_features[train_mask].sum(axis=0) > 0

    model = MLPBaseline(
        n_genes=dataset.expression.shape[1],
        n_perturbations=dataset.perturbation_features.shape[1],
        hidden_dims=config["hidden_dims"],
        dropout=float(config["dropout"]),
        active_perturbations=torch.from_numpy(active),
    ).to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _tensor_dataset(dataset, train_mask),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _tensor_dataset(dataset, validation_mask),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    model_dir = output_dir / scheme / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    patience = int(config["patience"])
    min_delta = float(config["min_delta"])
    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        train_total = 0.0
        train_samples = 0
        for reference, perturbation, target in train_loader:
            reference = reference.to(device)
            perturbation = perturbation.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(reference, perturbation)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            optimizer.step()
            train_total += loss.detach().item() * len(target)
            train_samples += len(target)

        train_loss = train_total / train_samples
        validation_loss = _mean_loss(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch == 1 or epoch % int(config["log_every"]) == 0:
            print(
                f"  {scheme} mlp epoch={epoch:03d} "
                f"train={train_loss:.6f} validation={validation_loss:.6f}",
                flush=True,
            )

        if validation_loss < best_validation - min_delta:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  {scheme} mlp early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed_seconds = time.perf_counter() - started

    checkpoint = {
        "model_name": "mlp",
        "scheme": scheme,
        "state_dict": {key: value.cpu() for key, value in best_state.items()},
        "model_config": {
            "architecture": "perturbation_response_mlp_v2",
            "n_genes": dataset.expression.shape[1],
            "n_perturbations": dataset.perturbation_features.shape[1],
            "hidden_dims": config["hidden_dims"],
            "dropout": config["dropout"],
        },
        "perturbation_vocabulary": dataset.perturbation_vocabulary,
        "genes": dataset.genes["gene"].astype(str).tolist(),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "seed": seed,
    }
    torch.save(checkpoint, model_dir / "checkpoint.pt")
    with (model_dir / "training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    predicted_delta = _predict_all(
        model, dataset, int(config["batch_size"]), device
    )
    return save_evaluation(
        output_dir,
        "mlp",
        scheme,
        dataset,
        predicted_delta,
        top_ks,
        {
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "epochs_trained": len(history),
            "training_seconds": elapsed_seconds,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "n_train_active_perturbations": int(active.sum()),
            "n_total_perturbations": int(len(active)),
        },
    )
