"""Reload a saved scGPT adapter and reproduce its stored test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.data.dataset import load_pseudobulk_dataset  # noqa: E402
from mini_vc.data.scgpt_controls import build_control_token_bank  # noqa: E402
from mini_vc.models.scgpt_transfer import load_scgpt_adapter  # noqa: E402
from mini_vc.training.scgpt_runner import (  # noqa: E402
    _make_control_tensors,
    _rows_by_group,
    _run_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scgpt_folds.json")
    parser.add_argument("--scheme", default="unseen_combo_fold0")
    parser.add_argument("--mode", choices=["frozen", "finetuned"], default="finetuned")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scheme = args.scheme
    data_dir = ROOT / config["data_dir"]
    model_dir = ROOT / config["output_dir"] / scheme / f"scgpt_{args.mode}"
    dataset = load_pseudobulk_dataset(data_dir / scheme)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_scgpt_adapter(
        model_dir / "adapter_checkpoint.pt",
        ROOT / config["checkpoint_dir"],
        dataset,
        device,
    )
    vocab = json.loads((ROOT / config["checkpoint_dir"] / "vocab.json").read_text(encoding="utf-8"))
    bank = build_control_token_bank(
        ROOT / config["raw_path"],
        data_dir / "cell_splits.csv.gz",
        dataset,
        scheme,
        vocab,
        data_dir / scheme / "scgpt_control_tokens.npz",
        n_cells_per_group=int(config["input"]["n_control_cells_per_group"]),
        max_tokens=int(config["input"]["max_tokens"]),
    )
    controls = _make_control_tensors(bank, device)
    groups = _rows_by_group(dataset, "test")
    _, by_group = _run_split(
        model, dataset, groups, "test", controls, device, None, None
    )
    recomputed = np.zeros_like(dataset.expression)
    for rows, predictions in by_group.values():
        recomputed[rows] = predictions
    test_mask = dataset.split_mask("test", include_control=False)
    with np.load(model_dir / "predictions_test.npz", allow_pickle=False) as saved:
        if not np.array_equal(
            dataset.samples.loc[test_mask, "sample_id"].to_numpy(dtype=str),
            saved["sample_id"].astype(str),
        ):
            raise ValueError("saved test samples are misaligned")
        difference = np.max(
            np.abs(recomputed[test_mask] - saved["predicted_expression"])
        )
    if not np.isfinite(difference) or difference > 1e-4:
        raise ValueError(f"checkpoint prediction mismatch: max error {difference}")
    print(f"Checkpoint predictions verified: {scheme}/{args.mode}; max error {difference:.3g}")


if __name__ == "__main__":
    main()
