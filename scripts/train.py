"""Train models from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.training import run_baselines, run_mini_vc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mini-VC models.")
    parser.add_argument("--config", type=Path, required=True, help="JSON config path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    experiment = config.get("experiment")
    if experiment == "baselines":
        comparison = run_baselines(config, REPO_ROOT)
    elif experiment == "mini_vc":
        comparison = run_mini_vc(config, REPO_ROOT)
    else:
        raise ValueError(
            "config must set experiment to 'baselines' or 'mini_vc'"
        )
    display_columns = [
        "scheme",
        "model",
        "mse_all",
        "pearson_delta",
        "top_100_recovery",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"Results: {(REPO_ROOT / config['output_dir']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
