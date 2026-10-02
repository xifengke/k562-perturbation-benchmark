"""Fit, select, and evaluate the perturbation-response baselines."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mini_vc.data.dataset import load_pseudobulk_dataset
from mini_vc.data.pathway_prior import build_reactome_features
from mini_vc.evaluation import save_evaluation
from mini_vc.models import MeanResponseBaseline, RidgeBaseline
from mini_vc.training.torch_runner import train_mlp_baseline


def _validation_mse(
    predicted_delta: np.ndarray, true_delta: np.ndarray, mask: np.ndarray
) -> float:
    return float(np.square(predicted_delta[mask] - true_delta[mask]).mean())


def run_baselines(config: dict, repo_root: Path) -> pd.DataFrame:
    data_dir = (repo_root / config["data_dir"]).resolve()
    output_dir = (repo_root / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for scheme in config["schemes"]:
        print(f"Loading pseudobulk scheme: {scheme}", flush=True)
        dataset = load_pseudobulk_dataset(data_dir / scheme)
        if "pathway_features" in config:
            prior = config["pathway_features"]
            features, feature_names, audit = build_reactome_features(
                dataset.perturbation_features,
                dataset.perturbation_vocabulary,
                (repo_root / prior["archive_path"]).resolve(),
                min_genes=int(prior.get("min_genes", 2)),
                max_genes=int(prior.get("max_genes", 30)),
                shuffle_seed=prior.get("shuffle_seed"),
            )
            audit["scheme"] = scheme
            dataset = replace(
                dataset,
                perturbation_features=features,
                perturbation_vocabulary=feature_names,
            )
            audit_path = output_dir / scheme / "pathway_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        train_noncontrol = dataset.split_mask("train", include_control=False)
        train = dataset.split_mask("train", include_control=True)
        validation_noncontrol = dataset.split_mask("validation", include_control=False)
        train_validation = train | dataset.split_mask("validation", include_control=True)

        if "no_change" in config["models"]:
            predicted_delta = np.zeros_like(dataset.delta)
            rows.append(
                save_evaluation(
                    output_dir,
                    "no_change",
                    scheme,
                    dataset,
                    predicted_delta,
                    config["de_top_k"],
                )
            )

        if "mean_response" in config["models"]:
            model = MeanResponseBaseline().fit(dataset.delta[train_noncontrol])
            predicted_delta = model.predict_delta(len(dataset.expression))
            rows.append(
                save_evaluation(
                    output_dir,
                    "mean_response",
                    scheme,
                    dataset,
                    predicted_delta,
                    config["de_top_k"],
                )
            )

        if "ridge" in config["models"]:
            validation_scores: dict[str, float] = {}
            for alpha in config["ridge_alphas"]:
                candidate = RidgeBaseline(float(alpha)).fit(
                    dataset.perturbation_features[train], dataset.delta[train]
                )
                candidate_delta = candidate.predict_delta(dataset.perturbation_features)
                validation_scores[str(alpha)] = _validation_mse(
                    candidate_delta, dataset.delta, validation_noncontrol
                )
            best_alpha = min(
                (float(alpha) for alpha in config["ridge_alphas"]),
                key=lambda value: validation_scores[str(value)],
            )
            model = RidgeBaseline(best_alpha).fit(
                dataset.perturbation_features[train_validation],
                dataset.delta[train_validation],
            )
            predicted_delta = model.predict_delta(dataset.perturbation_features)
            model_dir = output_dir / scheme / "ridge"
            model_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                model_dir / "model.npz",
                coefficients=model.coef_,
                vocabulary=np.asarray(dataset.perturbation_vocabulary, dtype=str),
                alpha=np.asarray(best_alpha),
            )
            rows.append(
                save_evaluation(
                    output_dir,
                    "ridge",
                    scheme,
                    dataset,
                    predicted_delta,
                    config["de_top_k"],
                    {
                        "best_alpha": best_alpha,
                        "validation_mse_by_alpha": validation_scores,
                    },
                )
            )

        if "mlp" in config["models"]:
            rows.append(
                train_mlp_baseline(
                    config["mlp"],
                    dataset,
                    scheme,
                    output_dir,
                    config["de_top_k"],
                    int(config["seed"]),
                )
            )

    comparison = pd.DataFrame(rows).sort_values(["scheme", "model"])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    return comparison
