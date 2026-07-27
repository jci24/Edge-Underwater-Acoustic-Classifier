#!/usr/bin/env python3
"""Validate Milestone 2 feature, model, prediction, and report artifacts."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from edge_underwater.baseline import assert_probability_contract, validate_vessel_splits
from edge_underwater.features import HandcraftedFeatureConfig
from edge_underwater.manifest import read_csv_rows
from edge_underwater.preprocessing import PreprocessingConfig


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
FEATURE_FOLDER = PROJECT_FOLDER / "data/features"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone2"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models"


def main() -> None:
    preprocessing = PreprocessingConfig.load(
        PROJECT_FOLDER / "data/preprocessing/config.json"
    )
    config = HandcraftedFeatureConfig.load(
        FEATURE_FOLDER / "handcrafted_config.json"
    )
    manifest = pd.DataFrame(
        read_csv_rows(PROJECT_FOLDER / "data/manifests/deepship_windows.csv")
    )
    features = pd.read_csv(FEATURE_FOLDER / "deepship_handcrafted_features.csv")

    if len(features) != 1_118 or features["window_id"].nunique() != 1_118:
        raise RuntimeError("Expected exactly 1,118 unique feature rows.")
    if set(features["window_id"]) != set(manifest["window_id"]):
        raise RuntimeError("Feature and manifest window IDs differ.")
    if tuple(features.columns[-73:]) != config.feature_names:
        raise RuntimeError("Feature table order does not match its configuration.")
    if not np.isfinite(features.loc[:, config.feature_names].to_numpy()).all():
        raise RuntimeError("Feature table contains NaN or infinite values.")
    if set(features["config_hash"]) != {preprocessing.config_hash}:
        raise RuntimeError("Unexpected preprocessing configuration hash.")
    if set(features["feature_config_hash"]) != {config.config_hash}:
        raise RuntimeError("Unexpected feature configuration hash.")
    validate_vessel_splits(features)

    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    test = features.loc[features["split"] == "test"]
    for model_name in ("logistic_regression", "random_forest"):
        model = joblib.load(MODEL_FOLDER / f"{model_name}.joblib")
        probabilities = model.predict_proba(test.loc[:, config.feature_names])
        repeated = model.predict_proba(test.loc[:, config.feature_names])
        assert_probability_contract(model, probabilities)
        if not np.array_equal(probabilities, repeated):
            raise RuntimeError(f"{model_name} predictions are not deterministic.")
        matrix = metrics["models"][model_name]["window_test"]["confusion_matrix"]
        if np.asarray(matrix).shape != (4, 4):
            raise RuntimeError(f"{model_name} confusion matrix is not 4x4.")

    predictions = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")
    probability_columns = [
        "probability_Cargo",
        "probability_Passengership",
        "probability_Tanker",
        "probability_Tug",
    ]
    if len(predictions) != 270:
        raise RuntimeError("Expected 135 test predictions from each of two models.")
    if not np.allclose(predictions[probability_columns].sum(axis=1), 1.0):
        raise RuntimeError("Saved probabilities do not sum to one.")

    print("Validated 1,118 finite, traceable 73-feature rows")
    print("Validated deterministic four-class probabilities for both models")
    print("Validated fixed 4x4 confusion matrices and saved test predictions")


if __name__ == "__main__":
    main()

