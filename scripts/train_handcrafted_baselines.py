#!/usr/bin/env python3
"""Train, evaluate, inspect, and benchmark both Milestone 2 baselines."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from edge_underwater.baseline import (
    ORDERED_CLASSES,
    fit_logistic_baseline,
    fit_random_forest_baseline,
    logistic_coefficients,
    random_forest_permutation_importance,
    source_predictions,
    support_by_split,
    validate_vessel_splits,
    window_predictions,
)
from edge_underwater.feature_table import CachedIntervalDecoder
from edge_underwater.features import (
    HandcraftedFeatureConfig,
    HandcraftedFeatureExtractor,
)
from edge_underwater.manifest import read_csv_rows
from edge_underwater.preprocessing import PreprocessingConfig


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_FOLDER / "data/features/deepship_handcrafted_features.csv"
FEATURE_CONFIG_FILE = PROJECT_FOLDER / "data/features/handcrafted_config.json"
EXTRACTION_METADATA_FILE = PROJECT_FOLDER / "data/features/extraction_metadata.json"
PREPROCESSING_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone2"


def percentile_summary(seconds: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(seconds) * 1_000
    return {
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "total_seconds": float(np.sum(seconds)),
        "call_count": len(seconds),
    }


def timed_calls(call: Callable[[int], None], count: int) -> list[float]:
    times = []
    for index in range(count):
        started = time.perf_counter()
        call(index)
        times.append(time.perf_counter() - started)
    return times


def benchmark_feature_extraction(
    manifest_rows: list[dict[str, str]],
    extractor: HandcraftedFeatureExtractor,
) -> dict[str, float]:
    decoder = CachedIntervalDecoder(AUDIO_FOLDER)
    for row in manifest_rows[:10]:
        waveform, sample_rate = decoder.decode(row)
        extractor.extract(waveform, sample_rate)

    def extract(index: int) -> None:
        waveform, sample_rate = decoder.decode(manifest_rows[index])
        extractor.extract(waveform, sample_rate)

    return percentile_summary(timed_calls(extract, len(manifest_rows)))


def benchmark_classifier(
    model: Pipeline,
    features: pd.DataFrame,
) -> dict[str, float]:
    for index in range(10):
        model.predict_proba(features.iloc[[index % len(features)]])

    def predict(index: int) -> None:
        row = index % len(features)
        model.predict_proba(features.iloc[[row]])

    return percentile_summary(timed_calls(predict, 1_000))


def benchmark_end_to_end(
    model: Pipeline,
    manifest_rows: list[dict[str, str]],
    extractor: HandcraftedFeatureExtractor,
) -> dict[str, float]:
    decoder = CachedIntervalDecoder(AUDIO_FOLDER)

    def process(row: dict[str, str]) -> None:
        waveform, sample_rate = decoder.decode(row)
        values = extractor.extract(waveform, sample_rate)
        features = pd.DataFrame([values])
        model.predict_proba(features)

    for row in manifest_rows[:10]:
        process(row)

    return percentile_summary(
        timed_calls(lambda index: process(manifest_rows[index]), len(manifest_rows))
    )


def save_confusion_matrices(metrics: dict[str, Any]) -> None:
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, model_name in zip(axes, ("logistic_regression", "random_forest"), strict=True):
        matrix = np.asarray(metrics["models"][model_name]["window_test"]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name.replace("_", " ").title())
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(4), ORDERED_CLASSES, rotation=30, ha="right")
        axis.set_yticks(range(4), ORDERED_CLASSES)
        for true_index in range(4):
            for predicted_index in range(4):
                value = int(matrix[true_index, predicted_index])
                axis.text(predicted_index, true_index, value, ha="center", va="center")
                rows.append(
                    {
                        "model": model_name,
                        "true_class": ORDERED_CLASSES[true_index],
                        "predicted_class": ORDERED_CLASSES[predicted_index],
                        "count": value,
                    }
                )
        figure.colorbar(image, ax=axis, fraction=0.046)
    pd.DataFrame(rows).to_csv(REPORT_FOLDER / "confusion_matrices.csv", index=False)
    figure.savefig(REPORT_FOLDER / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def report_markdown(
    metrics: dict[str, Any],
    coefficients: pd.DataFrame,
    importance: pd.DataFrame,
) -> str:
    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for row in frame.itertuples(index=False, name=None):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    logistic = metrics["models"]["logistic_regression"]
    forest = metrics["models"]["random_forest"]
    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic regression",
                "role": "primary",
                "window macro F1": logistic["window_test"]["macro_f1"],
                "window accuracy": logistic["window_test"]["accuracy"],
                "source macro F1": logistic["source_test_secondary"]["macro_f1"],
            },
            {
                "model": "Random forest",
                "role": "secondary",
                "window macro F1": forest["window_test"]["macro_f1"],
                "window accuracy": forest["window_test"]["accuracy"],
                "source macro F1": forest["source_test_secondary"]["macro_f1"],
            },
        ]
    )
    per_class_rows = []
    for model_name, model_metrics in (
        ("Logistic regression", logistic),
        ("Random forest", forest),
    ):
        for class_name, values in model_metrics["window_test"]["per_class"].items():
            per_class_rows.append(
                {
                    "model": model_name,
                    "class": class_name,
                    **values,
                }
            )
    supports = pd.DataFrame(metrics["support_by_split"])
    comparison_table = markdown_table(comparison)
    per_class_table = markdown_table(pd.DataFrame(per_class_rows))
    support_table = markdown_table(supports)
    positive = coefficients.loc[coefficients["positive_rank"] <= 3].assign(
        direction="positive",
        rank=lambda frame: frame["positive_rank"],
    )
    negative = coefficients.loc[coefficients["negative_rank"] <= 3].assign(
        direction="negative",
        rank=lambda frame: frame["negative_rank"],
    )
    top_coefficients = pd.concat([positive, negative]).sort_values(
        ["class", "direction", "rank"]
    )[["class", "direction", "rank", "feature", "standardized_coefficient"]]
    top_coefficients = markdown_table(top_coefficients)
    top_importance = markdown_table(importance.head(10))
    timing = metrics["timing"]

    return f"""# Milestone 2 — Hand-Engineered Feature Baseline

## Outcome

The primary regularized multinomial logistic regression selected `C={logistic["selected_C"]}`
using validation macro F1. Its untouched test-window macro F1 was
**{logistic["window_test"]["macro_f1"]:.3f}** with accuracy
**{logistic["window_test"]["accuracy"]:.3f}**. The fixed random-forest comparison
reached test-window macro F1 **{forest["window_test"]["macro_f1"]:.3f}** and
accuracy **{forest["window_test"]["accuracy"]:.3f}**.

Source-level aggregation is secondary: the public test split contains only one
Cargo, one Passengership, and one Tug vessel group. Windows from the same source
and vessel are correlated and must not be interpreted as independent vessel
observations.

{comparison_table}

## Test metrics by class

{per_class_table}

## Support

{support_table}

## Validation selection

| C | Validation macro F1 |
|---:|---:|
{chr(10).join(f'| {row["C"]:g} | {row["validation_macro_f1"]:.3f} |' for row in logistic["validation_candidates"])}

Selectors, scaling, sample weights, and both classifiers were fitted only on
the committed training split. The logistic model was not refitted on
training-plus-validation after selection.

## Model inspection

The logistic coefficients below are the three strongest positive and negative
standardized coefficients per class. They show association within this fitted
model, not causation.

{top_coefficients}

The random-forest values are validation-set permutation importance using macro
F1, 20 repeats, and seed 42. They measure fitted-model reliance; correlated
features can dilute each other's permutation importance.

{top_importance}

## Timing and size

All latency values are warm-cache, batch-size-one measurements on the current
development Mac with one numerical-library thread. They are not target
edge-device performance.

- Full feature-table extraction:
  {timing["feature_table_extraction"]["total_extraction_seconds"]:.3f} seconds
- Test source decoding plus feature extraction:
  median {timing["decode_and_feature"]["median_ms"]:.3f} ms,
  p95 {timing["decode_and_feature"]["p95_ms"]:.3f} ms
- Logistic `predict_proba`:
  median {timing["logistic_regression_inference"]["median_ms"]:.3f} ms,
  p95 {timing["logistic_regression_inference"]["p95_ms"]:.3f} ms,
  model size {logistic["model_size_bytes"]} bytes
- Random-forest `predict_proba`:
  median {timing["random_forest_inference"]["median_ms"]:.3f} ms,
  p95 {timing["random_forest_inference"]["p95_ms"]:.3f} ms,
  model size {forest["model_size_bytes"]} bytes
- Logistic end to end:
  median {timing["logistic_regression_end_to_end"]["median_ms"]:.3f} ms,
  p95 {timing["logistic_regression_end_to_end"]["p95_ms"]:.3f} ms
- Random-forest end to end:
  median {timing["random_forest_end_to_end"]["median_ms"]:.3f} ms,
  p95 {timing["random_forest_end_to_end"]["p95_ms"]:.3f} ms

Environment: Python {metrics["environment"]["python"]},
scikit-learn {metrics["environment"]["scikit_learn"]},
PyTorch {metrics["environment"]["torch"]}.

## Reporting limits

This baseline validates the reproducible mechanics of the public subset. It
does not generalize performance to the full 47-hour DeepShip dataset, and no
minimum F1 threshold was imposed. Source-level results are reported only as a
secondary view.
"""


def main() -> None:
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig.load(PREPROCESSING_FILE)
    feature_config = HandcraftedFeatureConfig.load(FEATURE_CONFIG_FILE)
    if feature_config.preprocessing_config_hash != preprocessing.config_hash:
        raise ValueError("Feature and preprocessing configurations differ.")

    table = pd.read_csv(FEATURE_FILE)
    if len(table) != 1_118 or table["window_id"].nunique() != 1_118:
        raise ValueError("Expected 1,118 unique feature rows.")
    if set(table["config_hash"]) != {preprocessing.config_hash}:
        raise ValueError("Feature rows use an unexpected preprocessing hash.")
    if set(table["feature_config_hash"]) != {feature_config.config_hash}:
        raise ValueError("Feature rows use an unexpected feature hash.")
    validate_vessel_splits(table)

    with threadpool_limits(limits=1):
        logistic, candidates, selected_c = fit_logistic_baseline(table, feature_config)
        forest = fit_random_forest_baseline(table, feature_config)

        test_rows = table.loc[table["split"] == "test"].reset_index(drop=True)
        logistic_windows, logistic_window_metrics = window_predictions(
            logistic, test_rows, feature_config, "logistic_regression"
        )
        forest_windows, forest_window_metrics = window_predictions(
            forest, test_rows, feature_config, "random_forest"
        )
        logistic_sources, logistic_source_metrics = source_predictions(logistic_windows)
        forest_sources, forest_source_metrics = source_predictions(forest_windows)

        coefficients = logistic_coefficients(logistic, feature_config)
        importance = random_forest_permutation_importance(
            forest, table, feature_config
        )

        joblib.dump(logistic, MODEL_FOLDER / "logistic_regression.joblib")
        joblib.dump(forest, MODEL_FOLDER / "random_forest.joblib")
        logistic_size = (MODEL_FOLDER / "logistic_regression.joblib").stat().st_size
        forest_size = (MODEL_FOLDER / "random_forest.joblib").stat().st_size

        test_manifest = [
            row for row in read_csv_rows(MANIFEST_FILE) if row["split"] == "test"
        ]
        extractor = HandcraftedFeatureExtractor(feature_config, preprocessing)
        feature_timing = benchmark_feature_extraction(test_manifest, extractor)
        test_features = test_rows.loc[:, feature_config.feature_names]
        logistic_inference = benchmark_classifier(logistic, test_features)
        forest_inference = benchmark_classifier(forest, test_features)
        logistic_end_to_end = benchmark_end_to_end(
            logistic, test_manifest, extractor
        )
        forest_end_to_end = benchmark_end_to_end(forest, test_manifest, extractor)

    all_windows = pd.concat([logistic_windows, forest_windows], ignore_index=True)
    all_sources = pd.concat([logistic_sources, forest_sources], ignore_index=True)
    all_windows.to_csv(REPORT_FOLDER / "window_predictions.csv", index=False)
    all_sources.to_csv(REPORT_FOLDER / "source_predictions.csv", index=False)
    coefficients.to_csv(REPORT_FOLDER / "logistic_coefficients.csv", index=False)
    importance.to_csv(
        REPORT_FOLDER / "random_forest_permutation_importance.csv",
        index=False,
    )

    extraction_metadata = json.loads(EXTRACTION_METADATA_FILE.read_text())
    metrics = {
        "primary_metric": "test window-level macro F1",
        "models": {
            "logistic_regression": {
                "role": "primary",
                "selected_C": selected_c,
                "validation_candidates": candidates,
                "window_test": logistic_window_metrics,
                "source_test_secondary": logistic_source_metrics,
                "model_size_bytes": logistic_size,
            },
            "random_forest": {
                "role": "secondary",
                "window_test": forest_window_metrics,
                "source_test_secondary": forest_source_metrics,
                "model_size_bytes": forest_size,
            },
        },
        "support_by_split": support_by_split(table),
        "timing": {
            "measurement_note": (
                "Warm-cache Mac development-machine measurements with one thread "
                "and batch size one; not target-edge-device performance."
            ),
            "feature_table_extraction": extraction_metadata,
            "decode_and_feature": feature_timing,
            "logistic_regression_inference": logistic_inference,
            "random_forest_inference": forest_inference,
            "logistic_regression_end_to_end": logistic_end_to_end,
            "random_forest_end_to_end": forest_end_to_end,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "thread_count": 1,
            "inference_batch_size": 1,
        },
        "reporting_limits": [
            "Windows within recordings and vessels are correlated.",
            "The public test split has only one Cargo, Passengership, and Tug vessel group.",
            "Results do not generalize to the full 47-hour DeepShip dataset.",
        ],
        "preprocessing_config_hash": preprocessing.config_hash,
        "feature_config_hash": feature_config.config_hash,
    }
    (REPORT_FOLDER / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    save_confusion_matrices(metrics)
    (REPORT_FOLDER / "README.md").write_text(
        report_markdown(metrics, coefficients, importance),
        encoding="utf-8",
    )
    print(
        "Logistic test macro F1:",
        f"{logistic_window_metrics['macro_f1']:.3f}",
    )
    print(
        "Random forest test macro F1:",
        f"{forest_window_metrics['macro_f1']:.3f}",
    )
    print(f"Wrote versioned results to {REPORT_FOLDER}")


if __name__ == "__main__":
    main()
