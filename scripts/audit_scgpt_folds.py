"""Audit complete-condition isolation in the new five-fold split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/norman_scgpt_folds"


def main() -> None:
    all_test = []
    report = {}
    for fold_index in range(5):
        name = f"unseen_combo_fold{fold_index}"
        directory = DATA / name
        samples = pd.read_csv(directory / "samples.csv")
        details = json.loads((directory / "split_details.json").read_text(encoding="utf-8"))
        doubles = samples.loc[samples.nperts == 2]
        if (doubles.groupby("perturbation").split.nunique() != 1).any():
            raise ValueError(f"double condition crosses splits in {name}")
        if not (samples.loc[samples.nperts == 1, "split"] == "train").all():
            raise ValueError(f"single condition is not in train in {name}")
        test_labels = set(doubles.loc[doubles.split == "test", "perturbation"])
        if test_labels != set(details["test_double_conditions"]):
            raise ValueError(f"test labels differ from frozen split details in {name}")
        single_labels = set(samples.loc[samples.nperts == 1, "perturbation"])
        for label in test_labels:
            if not set(label.split("_")).issubset(single_labels):
                raise ValueError(f"missing training single for test combination {label}")
        all_test.extend(sorted(test_labels))
        report[name] = {
            "test_conditions": len(test_labels),
            "validation_conditions": len(details["validation_double_conditions"]),
            "training_conditions": len(details["train_double_conditions"]),
            "pseudobulks_by_split": samples.split.value_counts().to_dict(),
            "genes": len(pd.read_csv(directory / "genes.csv")),
        }
    if len(all_test) != 131 or len(set(all_test)) != 131:
        raise ValueError("test folds do not cover all 131 doubles exactly once")
    report["all_test_conditions"] = len(all_test)
    output = ROOT / "reports/scgpt_fold_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
