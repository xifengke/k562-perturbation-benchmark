"""Compare scGPT against the existing seen and unseen-gene results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/mini_vc/combined_model_comparison.csv"
SCGPT = ROOT / "results/scgpt_secondary"
SCHEMES = ["seen", "unseen_gene"]
MODELS = ["ridge", "mlp", "mini_vc", "scgpt_frozen", "scgpt_finetuned"]


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in values
            )
            + " |"
        )
    return "\n".join(rows)


def main() -> None:
    original = pd.read_csv(ORIGINAL)
    rows = []
    for scheme in SCHEMES:
        for model in MODELS:
            if model.startswith("scgpt_"):
                path = SCGPT / scheme / model / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                metrics = json.loads(path.read_text(encoding="utf-8"))
            else:
                selected = original.loc[
                    (original.scheme == scheme) & (original.model == model)
                ]
                if len(selected) != 1:
                    raise ValueError(f"missing old result for {scheme}/{model}")
                metrics = selected.iloc[0].to_dict()
            rows.append(
                {
                    "scheme": scheme,
                    "model": model,
                    "n_conditions": int(metrics["n_conditions"]),
                    "pearson_delta": float(metrics["pearson_delta"]),
                    "mse_all": float(metrics["mse_all"]),
                    "top_20_recovery": float(metrics["top_20_recovery"]),
                    "top_100_recovery": float(metrics["top_100_recovery"]),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(SCGPT / "secondary_comparison.csv", index=False)
    indexed = table.set_index(["scheme", "model"])
    frozen_unseen = indexed.loc[("unseen_gene", "scgpt_frozen")]
    tuned_unseen = indexed.loc[("unseen_gene", "scgpt_finetuned")]
    ridge_unseen = indexed.loc[("unseen_gene", "ridge")]
    report = """# scGPT secondary K562 evaluations

These two schemes reuse the original V0.1 QC, training-only HVG selection,
split assignments, output genes, and condition-macro evaluation. The baseline
rows come from the saved V0.1 results; scGPT rows are newly trained on the same
split. The original test sets were already inspected, so these comparisons are
exploratory rather than independent confirmation.

`seen` holds out a gemgroup while retaining the perturbation identities.
`unseen_gene` holds out every condition containing specified genes; unlike the
V0.1 identity-only models, scGPT has pretrained tokens for most held-out genes.

""" + markdown_table(table) + """

The three missing scGPT perturbation tokens (C19orf26, C3orf72, KIAA1804)
receive separate trainable fallback vectors; an unseen fallback vector cannot
provide biological zero-shot semantics. The input audit in each model directory
records this mapping and the official checkpoint hashes.

""" + f"""
On `seen`, the two scGPT arms are close to each other and to Ridge, while MLP
has the highest response Pearson. On `unseen_gene`, partial fine-tuning changes
response Pearson from {frozen_unseen.pearson_delta:.4f} to {tuned_unseen.pearson_delta:.4f}
and all-gene MSE from {frozen_unseen.mse_all:.6f} to {tuned_unseen.mse_all:.6f}.
Ridge has response Pearson {ridge_unseen.pearson_delta:.4f}. The scGPT fine-tuned
arm has the lowest all-gene MSE in this {int(tuned_unseen.n_conditions)}-condition
split, but this does not establish broad unseen-gene transfer. The split is small,
was already examined in V0.1, and the target is a single K562 CRISPRa dataset.
"""
    (ROOT / "reports/scgpt_secondary_results.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
