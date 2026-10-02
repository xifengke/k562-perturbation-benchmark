"""Train frozen or fine-tuned scGPT transfer models on the five fixed folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.training.scgpt_runner import train_scgpt_scheme  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "scgpt_folds.json")
    parser.add_argument("--fold", choices=["seen", "unseen_gene"] + [f"unseen_combo_fold{i}" for i in range(5)])
    parser.add_argument("--mode", choices=["frozen", "finetuned"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schemes = [args.fold] if args.fold else config["schemes"]
    modes = [args.mode] if args.mode else config["modes"]
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scheme in schemes:
        for mode in modes:
            rows.append(train_scgpt_scheme(config, ROOT, scheme, mode))
            comparison = pd.DataFrame(rows)
            comparison.to_csv(output_dir / "model_comparison_partial.csv", index=False)
    if not args.fold and not args.mode:
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print(comparison[["scheme", "model", "mse_all", "pearson_delta", "top_100_recovery"]].to_string(index=False))


if __name__ == "__main__":
    main()
