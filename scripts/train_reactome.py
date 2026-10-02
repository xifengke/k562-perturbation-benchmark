"""Train Reactome and shuffled-pathway controls on the established five folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_vc.training.baseline_runner import run_baselines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/reactome_pathway.json"
    )
    parser.add_argument("--arm", help="run only reactome or shuffle_<seed>")
    args = parser.parse_args()
    plan = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan["baseline_config"]).read_text(encoding="utf-8"))
    arms = [("reactome", None)] + [
        (f"shuffle_{seed}", int(seed)) for seed in plan["shuffle_seeds"]
    ]
    selected = [arm for arm in arms if args.arm is None or arm[0] == args.arm]
    if not selected:
        raise ValueError(f"unknown arm: {args.arm}")

    for name, shuffle_seed in selected:
        config = dict(base)
        config["models"] = ["ridge", "mlp"]
        config["output_dir"] = str(Path(plan["output_dir"]) / name)
        config["pathway_features"] = {
            "archive_path": plan["archive_path"],
            "min_genes": plan["min_genes"],
            "max_genes": plan["max_genes"],
            "shuffle_seed": shuffle_seed,
        }
        print(f"\n=== Reactome experiment: {name} ===", flush=True)
        comparison = run_baselines(config, ROOT)
        print(
            comparison[["scheme", "model", "pearson_delta", "mse_all"]].to_string(
                index=False
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
