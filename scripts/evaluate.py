"""Recompute condition-macro metrics from a saved test-prediction artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mini_vc.evaluation import evaluate_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--top-k", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    artifact = np.load(args.predictions)
    samples = pd.DataFrame(
        {
            "sample_id": artifact["sample_id"].astype(str),
            "perturbation": artifact["perturbation"].astype(str),
            "split": "test",
        }
    )
    macro, per_condition = evaluate_predictions(
        samples,
        artifact["true_expression"],
        artifact["predicted_expression"],
        artifact["reference"],
        split="test",
        top_ks=args.top_k,
    )

    if args.metrics is not None:
        saved = json.loads(args.metrics.read_text(encoding="utf-8"))
        for name, recomputed in macro.items():
            if name not in saved:
                raise ValueError(f"saved metrics are missing {name}")
            original = saved[name]
            if isinstance(recomputed, float):
                if not np.isclose(recomputed, original, rtol=1e-7, atol=1e-9):
                    raise ValueError(
                        f"metric mismatch for {name}: {recomputed} != {original}"
                    )
            elif recomputed != original:
                raise ValueError(
                    f"metric mismatch for {name}: {recomputed} != {original}"
                )
        print("Saved metrics verified.")

    print(json.dumps(macro, indent=2, ensure_ascii=False, allow_nan=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        per_condition.to_csv(args.output, index=False)
        print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

