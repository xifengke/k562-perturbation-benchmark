"""CLI entry point for leakage-aware preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.data import run_preprocessing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Norman 2019 for Mini-VC.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "preprocess.json",
        help="JSON configuration path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = run_preprocessing(config_path, REPO_ROOT)
    print(f"QC retained: {manifest['qc']['retained']:,} cells")
    for name, summary in manifest["schemes"].items():
        print(
            f"{name}: {summary['n_pseudobulk_samples']:,} pseudobulks x "
            f"{summary['n_hvg']:,} genes"
        )
    print(f"Output: {(REPO_ROOT / config['data']['output_dir']).resolve()}")
    print(f"Report: {(REPO_ROOT / config['data']['report_path']).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
