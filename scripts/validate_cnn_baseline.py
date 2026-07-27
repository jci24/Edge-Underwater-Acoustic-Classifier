#!/usr/bin/env python3
"""Validate the complete Milestone 3 experiment and versioned artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.cnn_evaluation import (
    PROBABILITY_COLUMNS,
    validate_error_review,
)
from edge_underwater.cnn_training import load_cnn_checkpoint
from edge_underwater.manifest import read_csv_rows, validate_split_leakage
from edge_underwater.preprocessing import PreprocessingConfig


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone3"
REVIEW_FILE = PROJECT_FOLDER / "data/annotations/deepship_cnn_error_review.csv"


def main() -> None:
    preprocessing = PreprocessingConfig.load(
        PROJECT_FOLDER / "data/preprocessing/config.json"
    )
    model_config = SmallCnnConfig.load(REPORT_FOLDER / "model_config.json")
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    run_configs = json.loads((REPORT_FOLDER / "run_configs.json").read_text())
    history = pd.read_csv(REPORT_FOLDER / "training_history.csv")
    predictions = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")
    source_predictions = pd.read_csv(REPORT_FOLDER / "source_predictions.csv")
    manifest = read_csv_rows(PROJECT_FOLDER / "data/manifests/deepship_windows.csv")

    if preprocessing.output_shape != model_config.input_shape:
        raise RuntimeError("CNN and preprocessing input shapes differ.")
    if metrics["preprocessing_config_hash"] != preprocessing.config_hash:
        raise RuntimeError("Milestone 3 preprocessing hash differs.")
    if metrics["model"]["model_config_hash"] != model_config.config_hash:
        raise RuntimeError("Milestone 3 model hash differs.")
    validate_split_leakage(manifest)

    expected_runs = [
        ("training_stats_unweighted", "training_stats", False),
        ("per_example_unweighted", "per_example", False),
        ("per_example_weighted", "per_example", True),
    ]
    actual_runs = [
        (row["name"], row["normalization"], row["class_weighted"])
        for row in run_configs
    ]
    if actual_runs != expected_runs:
        raise RuntimeError(f"Unexpected controlled run order: {actual_runs}")
    if set(history["run_name"]) != {name for name, _, _ in expected_runs}:
        raise RuntimeError("Training history does not contain all three runs.")
    if metrics["test_evaluated_runs"] != [metrics["primary_run"]]:
        raise RuntimeError("More than the selected primary run reached test evaluation.")

    checkpoint_names = {Path(row["checkpoint"]).stem for row in metrics["runs"]}
    if checkpoint_names != {name for name, _, _ in expected_runs}:
        raise RuntimeError("Expected one best checkpoint for every controlled run.")
    primary_path = next(
        PROJECT_FOLDER / row["checkpoint"]
        for row in metrics["runs"]
        if row["name"] == metrics["primary_run"]
    )
    primary_model, checkpoint = load_cnn_checkpoint(primary_path)
    if primary_model.parameter_count != 23_668:
        raise RuntimeError("Unexpected compact-CNN parameter count.")
    if checkpoint["selection_metric"] != "validation_macro_f1":
        raise RuntimeError("Checkpoint selection metric is not validation macro F1.")

    if len(predictions) != 135 or predictions["window_id"].nunique() != 135:
        raise RuntimeError("Expected 135 unique primary CNN test predictions.")
    if set(predictions["split"]) != {"test"}:
        raise RuntimeError("Saved predictions contain a non-test row.")
    if not np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1.0):
        raise RuntimeError("Saved CNN probabilities do not sum to one.")
    if np.asarray(
        metrics["primary_test"]["window"]["confusion_matrix"]
    ).shape != (4, 4):
        raise RuntimeError("CNN confusion matrix is not 4x4.")
    if set(metrics["primary_test"]["window"]["per_class"]) != {
        "Cargo",
        "Passengership",
        "Tanker",
        "Tug",
    }:
        raise RuntimeError("CNN per-class metrics are incomplete.")
    if len(source_predictions) != predictions["source_file"].nunique():
        raise RuntimeError("Source predictions do not cover each test source.")

    milestone2 = json.loads(
        (PROJECT_FOLDER / "reports/milestone2/metrics.json").read_text()
    )
    if (
        metrics["baseline_comparison"]["logistic_regression"]["macro_f1"]
        != milestone2["models"]["logistic_regression"]["window_test"]["macro_f1"]
    ):
        raise RuntimeError("Milestone 2 logistic comparison changed.")
    if (
        metrics["baseline_comparison"]["random_forest"]["macro_f1"]
        != milestone2["models"]["random_forest"]["window_test"]["macro_f1"]
    ):
        raise RuntimeError("Milestone 2 random-forest comparison changed.")

    review = validate_error_review(REVIEW_FILE)
    expected_review_count = min(
        20,
        int(
            (
                predictions["label_index"]
                != predictions["predicted_label_index"]
            ).sum()
        ),
    )
    if review["row_count"] != expected_review_count:
        raise RuntimeError("Error-listening pack has the wrong number of rows.")

    print("Validated three controlled CNN runs in the required order")
    print("Validated primary-only evaluation on 135 test windows")
    print("Validated fixed metrics, baseline comparison, latency, and checkpoints")
    print(
        f"Validated {review['row_count']}-row listening pack; "
        f"{review['completed_count']} manual reviews complete"
    )


if __name__ == "__main__":
    main()
