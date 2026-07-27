#!/usr/bin/env python3
"""Validate the complete Milestone 4 engineering analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from edge_underwater.cnn_evaluation import PROBABILITY_COLUMNS
from edge_underwater.cnn_training import load_cnn_checkpoint
from edge_underwater.risk_analysis import (
    STRATEGY_ORDER,
    binary_metrics,
    validate_label_audit,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone4"
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIT_FILE = PROJECT_FOLDER / "data/annotations/label_audit.csv"


def main() -> None:
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    runs = json.loads((REPORT_FOLDER / "run_configs.json").read_text())
    validation_predictions = pd.read_csv(
        REPORT_FOLDER / "validation_predictions.csv"
    )
    test_predictions = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")
    analysis = pd.read_csv(REPORT_FOLDER / "window_risk_analysis.csv")
    reference = json.loads((REPORT_FOLDER / "embedding_reference.json").read_text())
    manifest = pd.read_csv(MANIFEST_FILE)

    if metrics["strategy_order"] != list(STRATEGY_ORDER):
        raise RuntimeError("Imbalance strategy order changed.")
    if [run["strategy"] for run in runs] != list(STRATEGY_ORDER):
        raise RuntimeError("Saved run configurations are out of order.")
    selected = max(
        metrics["runs"],
        key=lambda row: (
            row["best_validation_macro_f1"],
            -list(STRATEGY_ORDER).index(row["strategy"]),
        ),
    )
    if metrics["selected_strategy"] != selected["strategy"]:
        raise RuntimeError("Selected strategy does not maximize validation macro F1.")
    if metrics["test_evaluated_strategies"] != [metrics["selected_strategy"]]:
        raise RuntimeError("More than the selected strategy reached test evaluation.")
    if len(validation_predictions) != 154 * 3:
        raise RuntimeError("Expected validation predictions for all three strategies.")
    if len(test_predictions) != 135 or set(test_predictions["split"]) != {"test"}:
        raise RuntimeError("Expected 135 selected-strategy test predictions.")
    if not np.allclose(test_predictions[PROBABILITY_COLUMNS].sum(axis=1), 1.0):
        raise RuntimeError("Test probabilities do not sum to one.")
    if np.asarray(
        metrics["selected_test"]["window"]["confusion_matrix"]
    ).shape != (4, 4):
        raise RuntimeError("Selected test confusion matrix is not 4x4.")

    selected_checkpoint = next(
        PROJECT_FOLDER / run["checkpoint"]
        for run in metrics["runs"]
        if run["strategy"] == metrics["selected_strategy"]
    )
    model, checkpoint = load_cnn_checkpoint(selected_checkpoint)
    if model.extract_embedding(
        torch.zeros((1, 1, 64, 155), dtype=torch.float32)
    ).shape != (1, 64):
        raise RuntimeError("Selected CNN does not expose a 64-value embedding.")
    if checkpoint["selection_metric"] != "validation_macro_f1":
        raise RuntimeError("Checkpoint selection metric changed.")

    if len(analysis) != 1_118 or analysis["window_id"].nunique() != 1_118:
        raise RuntimeError("Risk analysis must cover all 1,118 windows.")
    if set(analysis["window_id"]) != set(manifest["window_id"]):
        raise RuntimeError("Risk analysis and manifest traceability differ.")
    if len(reference["mean"]) != 64 or len(reference["centroids"]) != 4:
        raise RuntimeError("Embedding reference has the wrong shape.")
    if reference["fit_split"] != "train":
        raise RuntimeError("Embedding reference was not fitted on training rows.")
    if not np.isfinite(analysis["embedding_distance"]).all():
        raise RuntimeError("Embedding distances must be finite.")

    rare = metrics["rare_event"]["operating_point"]
    if rare["validation"]["false_positive_rate"] > 0.05:
        raise RuntimeError("Rare-event operating point exceeds 5% validation FPR.")
    test_labels = (test_predictions["class"] == "Tug").to_numpy()
    recomputed = binary_metrics(
        test_labels,
        test_predictions["probability_Tug"].to_numpy(),
        rare["threshold"],
    )
    if recomputed != rare["test"]:
        raise RuntimeError("Fixed Tug threshold was not applied unchanged to test.")
    if metrics["rejection"]["unknown_detection_validated"]:
        raise RuntimeError("Unknown detection cannot be validated without unknown data.")
    for method in ("maximum_probability", "embedding_distance"):
        validation_coverage = metrics["rejection"][method]["validation"]["coverage"]
        if validation_coverage < 0.89:
            raise RuntimeError(f"{method} retained too little validation coverage.")

    audit = validate_label_audit(
        AUDIT_FILE,
        expected_vessel_groups=set(manifest["vessel_group"]),
    )
    if audit["row_count"] != 48 or audit["vessel_group_count"] != 43:
        raise RuntimeError("Label audit coverage is incomplete.")

    print("Validated three imbalance strategies and validation-only selection")
    print("Validated selected-only test evaluation on 135 windows")
    print("Validated Tug PR/FPR and both rejection diagnostics")
    print("Validated 1,118 traceable risk rows and training-only embedding reference")
    print(
        f"Validated 48-row/43-group label audit; "
        f"{audit['completed_count']} manual reviews complete"
    )
if __name__ == "__main__":
    main()
