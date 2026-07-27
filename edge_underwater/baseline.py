"""Leakage-controlled classical baselines for handcrafted features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import HandcraftedFeatureConfig
from .manifest import LABELS, validate_split_leakage


ORDERED_LABELS = tuple(LABELS.values())
ORDERED_CLASSES = tuple(LABELS.keys())
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0)


def split_features(
    table: pd.DataFrame,
    split: str,
    config: HandcraftedFeatureConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = table.loc[table["split"] == split]
    if rows.empty:
        raise ValueError(f"No rows found for split {split}.")
    features = rows.loc[:, config.feature_names]
    labels = rows["label_index"].to_numpy(dtype=np.int64)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(f"{split} features contain NaN or infinite values.")
    return features, labels


def training_sample_weights(training_rows: pd.DataFrame) -> np.ndarray:
    """Give equal total mass to classes and to vessels within each class."""

    if set(training_rows["split"]) != {"train"}:
        raise ValueError("Sample weights may only be derived from training rows.")
    class_count = training_rows["label_index"].nunique()
    if class_count != len(ORDERED_LABELS):
        raise ValueError("Training data must contain all four classes.")

    weights = pd.Series(index=training_rows.index, dtype=float)
    for _, class_rows in training_rows.groupby("label_index", sort=True):
        vessel_count = class_rows["vessel_group"].nunique()
        for _, vessel_rows in class_rows.groupby("vessel_group", sort=True):
            value = 1.0 / (class_count * vessel_count * len(vessel_rows))
            weights.loc[vessel_rows.index] = value
    weights *= len(training_rows) / weights.sum()
    return weights.loc[training_rows.index].to_numpy()


def logistic_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=5_000,
                    random_state=42,
                ),
            ),
        ]
    )


def random_forest_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0.0)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def fit_logistic_baseline(
    table: pd.DataFrame,
    config: HandcraftedFeatureConfig,
) -> tuple[Pipeline, list[dict[str, float]], float]:
    training_rows = table.loc[table["split"] == "train"]
    validation_rows = table.loc[table["split"] == "validation"]
    train_x, train_y = split_features(table, "train", config)
    validation_x, validation_y = split_features(table, "validation", config)
    weights = training_sample_weights(training_rows)

    candidates: list[dict[str, float]] = []
    best_model: Pipeline | None = None
    best_score = -1.0
    best_c = LOGISTIC_C_VALUES[0]
    for c_value in LOGISTIC_C_VALUES:
        model = logistic_pipeline(c_value)
        model.fit(train_x, train_y, classifier__sample_weight=weights)
        score = float(f1_score(validation_y, model.predict(validation_x), average="macro"))
        candidates.append({"C": c_value, "validation_macro_f1": score})
        if score > best_score:
            best_model = model
            best_score = score
            best_c = c_value

    if best_model is None:
        raise RuntimeError("No logistic model was fitted.")
    if len(validation_rows) != len(validation_x):
        raise RuntimeError("Validation rows changed during model selection.")
    return best_model, candidates, best_c


def fit_random_forest_baseline(
    table: pd.DataFrame,
    config: HandcraftedFeatureConfig,
) -> Pipeline:
    training_rows = table.loc[table["split"] == "train"]
    train_x, train_y = split_features(table, "train", config)
    weights = training_sample_weights(training_rows)
    model = random_forest_pipeline()
    model.fit(train_x, train_y, classifier__sample_weight=weights)
    return model


def assert_probability_contract(model: Pipeline, probabilities: np.ndarray) -> None:
    classes = tuple(int(value) for value in model.classes_)
    if classes != ORDERED_LABELS:
        raise ValueError(f"Model class order is {classes}, expected {ORDERED_LABELS}.")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(ORDERED_LABELS):
        raise ValueError("Expected four ordered probability columns.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities must be finite.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Predicted probabilities must sum to one.")


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=ORDERED_LABELS,
        zero_division=0,
    )
    per_class = {}
    for index, class_name in enumerate(ORDERED_CLASSES):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=ORDERED_LABELS,
    )
    if matrix.shape != (4, 4):
        raise RuntimeError("Confusion matrix must be 4x4.")
    return {
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro")),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def window_predictions(
    model: Pipeline,
    rows: pd.DataFrame,
    config: HandcraftedFeatureConfig,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = rows.loc[:, config.feature_names]
    probabilities = model.predict_proba(features)
    assert_probability_contract(model, probabilities)
    predicted = np.asarray(model.classes_)[probabilities.argmax(axis=1)]

    output = rows.loc[
        :,
        [
            "window_id",
            "source_file",
            "class",
            "label_index",
            "vessel_group",
            "split",
            "start_seconds",
            "end_seconds",
        ],
    ].copy()
    output.insert(0, "model", model_name)
    output["predicted_label_index"] = predicted
    output["predicted_class"] = [ORDERED_CLASSES[int(value)] for value in predicted]
    for index, class_name in enumerate(ORDERED_CLASSES):
        output[f"probability_{class_name}"] = probabilities[:, index]
    metrics = classification_metrics(
        rows["label_index"].to_numpy(dtype=np.int64),
        predicted,
    )
    return output, metrics


def source_predictions(
    window_output: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    probability_columns = [f"probability_{name}" for name in ORDERED_CLASSES]
    groups = []
    for source_file, rows in window_output.groupby("source_file", sort=True):
        labels = rows["label_index"].unique()
        classes = rows["class"].unique()
        vessels = rows["vessel_group"].unique()
        if len(labels) != 1 or len(classes) != 1 or len(vessels) != 1:
            raise ValueError(f"Source traceability conflict for {source_file}.")
        probabilities = rows[probability_columns].mean().to_numpy()
        predicted = int(np.argmax(probabilities))
        group: dict[str, Any] = {
            "model": rows["model"].iloc[0],
            "source_file": source_file,
            "class": classes[0],
            "label_index": int(labels[0]),
            "vessel_group": vessels[0],
            "window_count": len(rows),
            "predicted_label_index": predicted,
            "predicted_class": ORDERED_CLASSES[predicted],
        }
        group.update(dict(zip(probability_columns, probabilities, strict=True)))
        groups.append(group)
    output = pd.DataFrame(groups)
    metrics = classification_metrics(
        output["label_index"].to_numpy(dtype=np.int64),
        output["predicted_label_index"].to_numpy(dtype=np.int64),
    )
    return output, metrics


def support_by_split(table: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for (split, class_name), rows in table.groupby(["split", "class"], sort=True):
        output.append(
            {
                "split": split,
                "class": class_name,
                "windows": len(rows),
                "source_files": rows["source_file"].nunique(),
                "vessel_groups": rows["vessel_group"].nunique(),
            }
        )
    return output


def logistic_coefficients(
    model: Pipeline,
    config: HandcraftedFeatureConfig,
) -> pd.DataFrame:
    selected = model.named_steps["variance"].get_support()
    names = np.asarray(config.feature_names)[selected]
    classifier = model.named_steps["classifier"]
    rows = []
    for class_index, class_label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        positive_order = np.argsort(-coefficients)
        negative_order = np.argsort(coefficients)
        positive_rank = {int(item): rank + 1 for rank, item in enumerate(positive_order)}
        negative_rank = {int(item): rank + 1 for rank, item in enumerate(negative_order)}
        for feature_index, feature_name in enumerate(names):
            rows.append(
                {
                    "class": ORDERED_CLASSES[int(class_label)],
                    "label_index": int(class_label),
                    "feature": feature_name,
                    "standardized_coefficient": coefficients[feature_index],
                    "positive_rank": positive_rank[feature_index],
                    "negative_rank": negative_rank[feature_index],
                }
            )
    return pd.DataFrame(rows)


def random_forest_permutation_importance(
    model: Pipeline,
    table: pd.DataFrame,
    config: HandcraftedFeatureConfig,
) -> pd.DataFrame:
    validation_x, validation_y = split_features(table, "validation", config)
    result = permutation_importance(
        model,
        validation_x,
        validation_y,
        scoring="f1_macro",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    output = pd.DataFrame(
        {
            "feature": config.feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return output.sort_values(
        ["importance_mean", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )


def validate_vessel_splits(table: pd.DataFrame) -> None:
    rows = table[["vessel_group", "split"]].astype(str).to_dict("records")
    validate_split_leakage(rows)
