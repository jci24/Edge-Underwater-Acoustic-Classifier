#!/usr/bin/env python3
"""Run Milestone 4 imbalance, rare-event, and rejection analyses."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch

from edge_underwater.baseline import ORDERED_CLASSES, support_by_split
from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.cnn_data import PreloadedLogMelStore
from edge_underwater.cnn_evaluation import (
    PROBABILITY_COLUMNS,
    aggregate_source_predictions,
    evaluate_cnn,
)
from edge_underwater.cnn_training import (
    CnnRunConfig,
    CnnTrainingConfig,
    TrainingResult,
    train_cnn,
)
from edge_underwater.preprocessing import PreprocessingConfig, TrainingStatistics
from edge_underwater.risk_analysis import (
    STRATEGY_ORDER,
    EmbeddingReference,
    ImbalanceExperimentResult,
    ImbalanceRunConfig,
    binary_metrics,
    class_vessel_sampling_weights,
    coverage_threshold,
    fit_embedding_reference,
    precision_recall_table,
    select_imbalance_strategy,
    select_threshold_at_fpr,
    selective_metrics,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
PREPROCESSING_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
MILESTONE2_METRICS = PROJECT_FOLDER / "reports/milestone2/metrics.json"
MILESTONE3_METRICS = PROJECT_FOLDER / "reports/milestone3/metrics.json"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models/imbalance"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone4"
EMBEDDING_COLUMNS = [f"embedding_{index}" for index in range(64)]


def show_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"Preloaded {done}/{total} windows")


def make_configs(
    strategy: str,
    preprocessing: PreprocessingConfig,
    model: SmallCnnConfig,
    training: CnnTrainingConfig,
) -> tuple[ImbalanceRunConfig, CnnRunConfig]:
    imbalance = ImbalanceRunConfig(
        name=strategy,
        strategy=strategy,
        normalization="per_example",
        preprocessing_config_hash=preprocessing.config_hash,
        model_config_hash=model.config_hash,
        training_config_hash=training.config_hash,
    )
    cnn_run = CnnRunConfig(
        name=f"imbalance_{strategy}",
        normalization="per_example",
        class_weighted=strategy == "class_weighted",
        preprocessing_config_hash=preprocessing.config_hash,
        model_config_hash=model.config_hash,
        training_config_hash=training.config_hash,
    )
    return imbalance, cnn_run


def run_strategy(
    strategy: str,
    store: PreloadedLogMelStore,
    preprocessing: PreprocessingConfig,
    statistics: TrainingStatistics,
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
) -> tuple[ImbalanceExperimentResult, pd.DataFrame]:
    imbalance_config, cnn_run = make_configs(
        strategy,
        preprocessing,
        model_config,
        training_config,
    )
    training_dataset = store.subset(
        "train", "per_example", preprocessing, statistics
    )
    validation_dataset = store.subset(
        "validation", "per_example", preprocessing, statistics
    )
    sampling_weights = None
    if strategy == "class_vessel_balanced_sampling":
        training_rows = [store.rows[index] for index in training_dataset.indexes]
        sampling_weights = class_vessel_sampling_weights(training_rows)
    print(f"Training imbalance strategy: {strategy}")
    training_result = train_cnn(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model_config=model_config,
        training_config=training_config,
        run_config=cnn_run,
        checkpoint_path=MODEL_FOLDER / f"{strategy}.pt",
        sampling_weights=sampling_weights,
    )
    validation_predictions, validation_metrics = evaluate_cnn(
        training_result.model,
        validation_dataset,
        batch_size=training_config.batch_size,
    )
    validation_predictions.insert(0, "strategy", strategy)
    result = ImbalanceExperimentResult(
        imbalance_config=imbalance_config,
        training_result=training_result,
        validation_metrics=validation_metrics,
    )
    print(
        f"Finished {strategy}: best epoch {training_result.best_epoch}, "
        f"validation macro F1 {training_result.best_validation_macro_f1:.3f}"
    )
    return result, validation_predictions


def save_training_artifacts(
    results: list[ImbalanceExperimentResult],
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
) -> None:
    model_config.save(REPORT_FOLDER / "model_config.json")
    training_payload = asdict(training_config)
    training_payload["config_hash"] = training_config.config_hash
    (REPORT_FOLDER / "training_config.json").write_text(
        json.dumps(training_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    configs = []
    history_rows = []
    for result in results:
        config = asdict(result.imbalance_config)
        config["config_hash"] = result.imbalance_config.config_hash
        config["best_epoch"] = result.training_result.best_epoch
        config["epochs_completed"] = len(result.training_result.history)
        config["best_validation_macro_f1"] = (
            result.training_result.best_validation_macro_f1
        )
        config["validation_per_class_recall"] = {
            class_name: values["recall"]
            for class_name, values in result.validation_metrics["per_class"].items()
        }
        config["checkpoint"] = str(
            result.training_result.checkpoint_path.relative_to(PROJECT_FOLDER)
        )
        configs.append(config)
        for epoch in result.training_result.history:
            history_rows.append(
                {
                    "strategy": result.imbalance_config.strategy,
                    **epoch,
                }
            )
    (REPORT_FOLDER / "run_configs.json").write_text(
        json.dumps(configs, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(history_rows).to_csv(
        REPORT_FOLDER / "training_history.csv",
        index=False,
    )

    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row_index, result in enumerate(results):
        history = pd.DataFrame(result.training_result.history)
        for column_index, metric in enumerate(("loss", "accuracy", "macro_f1")):
            axis = axes[row_index, column_index]
            axis.plot(history["epoch"], history[f"training_{metric}"], label="Training")
            axis.plot(
                history["epoch"],
                history[f"validation_{metric}"],
                label="Validation",
            )
            axis.axvline(
                result.training_result.best_epoch,
                color="black",
                linestyle="--",
                linewidth=1,
            )
            axis.set_title(f"{result.imbalance_config.strategy}\n{metric}")
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            if column_index == 0:
                axis.legend()
    figure.savefig(REPORT_FOLDER / "training_curves.png", dpi=160)
    plt.close(figure)


def add_analysis_columns(
    predictions: pd.DataFrame,
    reference: EmbeddingReference,
    store: PreloadedLogMelStore,
) -> pd.DataFrame:
    output = predictions.copy()
    probabilities = output[PROBABILITY_COLUMNS].to_numpy()
    labels = output["label_index"].to_numpy(dtype=int)
    output["maximum_probability"] = probabilities.max(axis=1)
    output["true_class_probability"] = probabilities[np.arange(len(output)), labels]
    output["correct"] = output["label_index"] == output["predicted_label_index"]
    output["embedding_distance"] = reference.distances(
        output[EMBEDDING_COLUMNS].to_numpy(),
        output["predicted_label_index"].to_numpy(),
    )
    trace = pd.DataFrame(store.rows)[
        ["window_id", "rms", "low_level", "vessel_name", "session_identifier"]
    ]
    output = output.merge(trace, on="window_id", how="left", validate="one_to_one")
    last_end = output.groupby("source_file")["end_seconds"].transform("max")
    output["source_boundary"] = (
        np.isclose(output["start_seconds"], 0.0)
        | np.isclose(output["end_seconds"], last_end)
    )
    output["file"] = output["source_file"]
    output["current_label"] = output["class"]
    return output


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = [
        float(np.nextafter(scores.max(), np.inf)),
        *sorted({float(value) for value in scores}, reverse=True),
    ]
    return pd.DataFrame(
        [binary_metrics(labels, scores, threshold) for threshold in thresholds]
    )


def rare_event_analysis(
    analysis: pd.DataFrame,
) -> tuple[dict[str, Any], float]:
    tables = []
    metrics: dict[str, Any] = {
        "rare_class": "Tug",
        "score": "probability_Tug",
        "interpretation": (
            "One-vs-rest diagnostic of the selected four-class model, "
            "not a separately validated detector."
        ),
    }
    for split in ("validation", "test"):
        rows = analysis.loc[analysis["split"] == split]
        labels = (rows["class"] == "Tug").to_numpy()
        scores = rows["probability_Tug"].to_numpy()
        table, average_precision = precision_recall_table(labels, scores)
        table.insert(0, "split", split)
        tables.append(table)
        metrics[split] = {
            "average_precision": average_precision,
            "positive_support": int(labels.sum()),
            "negative_support": int((~labels).sum()),
        }
    validation = analysis.loc[analysis["split"] == "validation"]
    validation_labels = (validation["class"] == "Tug").to_numpy()
    validation_scores = validation["probability_Tug"].to_numpy()
    threshold, validation_operating = select_threshold_at_fpr(
        validation_labels,
        validation_scores,
        maximum_fpr=0.05,
    )
    test = analysis.loc[analysis["split"] == "test"]
    test_operating = binary_metrics(
        (test["class"] == "Tug").to_numpy(),
        test["probability_Tug"].to_numpy(),
        threshold,
    )
    source_rows = (
        test.assign(is_tug=test["class"] == "Tug")
        .groupby("source_file", as_index=False)
        .agg(is_tug=("is_tug", "first"), probability_Tug=("probability_Tug", "mean"))
    )
    source_operating = binary_metrics(
        source_rows["is_tug"].to_numpy(),
        source_rows["probability_Tug"].to_numpy(),
        threshold,
    )
    metrics["operating_point"] = {
        "maximum_validation_fpr": 0.05,
        "threshold": threshold,
        "validation": validation_operating,
        "test": test_operating,
        "test_source_secondary": source_operating,
    }
    pd.concat(tables, ignore_index=True).to_csv(
        REPORT_FOLDER / "rare_event_precision_recall.csv",
        index=False,
    )
    threshold_sweep(validation_labels, validation_scores).to_csv(
        REPORT_FOLDER / "rare_event_validation_thresholds.csv",
        index=False,
    )
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for table in tables:
        split = table["split"].iloc[0]
        axis.step(table["recall"], table["precision"], where="post", label=split.title())
    axis.scatter(
        [validation_operating["recall"]],
        [validation_operating["precision"]],
        color="black",
        marker="x",
        s=80,
        label="5% validation-FPR point",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Tug One-vs-Rest Precision–Recall")
    axis.set_xlim(0, 1.02)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(REPORT_FOLDER / "rare_event_precision_recall.png", dpi=160)
    plt.close(figure)
    return metrics, threshold


def rejection_curve(
    rows: pd.DataFrame,
    score_column: str,
    higher_is_accepted: bool,
) -> pd.DataFrame:
    scores = rows[score_column].to_numpy()
    thresholds = sorted({float(value) for value in scores})
    output = []
    for threshold in thresholds:
        accepted = scores >= threshold if higher_is_accepted else scores <= threshold
        metrics = selective_metrics(
            rows["label_index"].to_numpy(),
            rows["predicted_label_index"].to_numpy(),
            accepted,
        )
        output.append(
            {
                "score": score_column,
                "threshold": threshold,
                "coverage": metrics["coverage"],
                "selective_accuracy": metrics["classification_on_accepted"]["accuracy"],
                "selective_macro_f1": metrics["classification_on_accepted"]["macro_f1"],
            }
        )
    return pd.DataFrame(output)


def rejection_analysis(
    analysis: pd.DataFrame,
) -> tuple[dict[str, Any], float]:
    validation = analysis.loc[analysis["split"] == "validation"]
    test = analysis.loc[analysis["split"] == "test"]
    methods = {
        "maximum_probability": {
            "column": "maximum_probability",
            "higher_is_accepted": True,
        },
        "embedding_distance": {
            "column": "embedding_distance",
            "higher_is_accepted": False,
        },
    }
    results: dict[str, Any] = {
        "target_validation_coverage": 0.90,
        "unknown_detection_validated": False,
        "limitation": (
            "DeepShip provides no unknown/background evaluation examples; "
            "these are confidence and in-distribution outlier diagnostics."
        ),
    }
    curves = []
    for method, settings in methods.items():
        column = settings["column"]
        higher = settings["higher_is_accepted"]
        threshold = coverage_threshold(
            validation[column].to_numpy(),
            target_coverage=0.90,
            higher_is_accepted=higher,
        )
        validation_accepted = (
            validation[column].to_numpy() >= threshold
            if higher
            else validation[column].to_numpy() <= threshold
        )
        test_accepted = (
            test[column].to_numpy() >= threshold
            if higher
            else test[column].to_numpy() <= threshold
        )
        results[method] = {
            "threshold": threshold,
            "validation": selective_metrics(
                validation["label_index"].to_numpy(),
                validation["predicted_label_index"].to_numpy(),
                validation_accepted,
            ),
            "test": selective_metrics(
                test["label_index"].to_numpy(),
                test["predicted_label_index"].to_numpy(),
                test_accepted,
            ),
        }
        curve = rejection_curve(validation, column, higher)
        curve.insert(0, "method", method)
        curves.append(curve)
    curve_table = pd.concat(curves, ignore_index=True)
    curve_table.to_csv(REPORT_FOLDER / "rejection_risk_coverage.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, rows in curve_table.groupby("method"):
        axes[0].plot(rows["coverage"], rows["selective_accuracy"], label=method)
        axes[1].plot(rows["coverage"], rows["selective_macro_f1"], label=method)
    axes[0].set_title("Validation Coverage vs Selective Accuracy")
    axes[1].set_title("Validation Coverage vs Selective Macro F1")
    for axis in axes:
        axis.set_xlabel("Coverage")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(REPORT_FOLDER / "rejection_risk_coverage.png", dpi=160)
    plt.close(figure)
    return results, float(results["embedding_distance"]["threshold"])


def save_domain_summaries(analysis: pd.DataFrame) -> dict[str, Any]:
    class_summary = (
        analysis.groupby(["split", "class"], as_index=False)
        .agg(
            windows=("window_id", "size"),
            accuracy=("correct", "mean"),
            mean_maximum_probability=("maximum_probability", "mean"),
            mean_embedding_distance=("embedding_distance", "mean"),
        )
    )
    source_summary = (
        analysis.groupby(["split", "class", "source_file"], as_index=False)
        .agg(
            windows=("window_id", "size"),
            accuracy=("correct", "mean"),
            mean_maximum_probability=("maximum_probability", "mean"),
            mean_embedding_distance=("embedding_distance", "mean"),
        )
    )
    class_summary.to_csv(REPORT_FOLDER / "domain_shift_class_summary.csv", index=False)
    source_summary.to_csv(REPORT_FOLDER / "domain_shift_source_summary.csv", index=False)
    return {
        "interpretation": (
            "Confidence and training-centroid distance are domain-shift indicators, "
            "not proof of recording-site exploitation."
        ),
        "class_summary_rows": len(class_summary),
        "source_summary_rows": len(source_summary),
    }


def save_confusion_matrix(matrix: list[list[int]]) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    axis.set_title("Selected Imbalance Strategy — Test")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(4), ORDERED_CLASSES, rotation=30)
    axis.set_yticks(range(4), ORDERED_CLASSES)
    rows = []
    for true_index in range(4):
        for predicted_index in range(4):
            count = int(values[true_index, predicted_index])
            axis.text(predicted_index, true_index, count, ha="center", va="center")
            rows.append(
                {
                    "true_class": ORDERED_CLASSES[true_index],
                    "predicted_class": ORDERED_CLASSES[predicted_index],
                    "count": count,
                }
            )
    figure.colorbar(image, ax=axis)
    figure.savefig(REPORT_FOLDER / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    pd.DataFrame(rows).to_csv(REPORT_FOLDER / "confusion_matrix.csv", index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(metrics: dict[str, Any]) -> None:
    runs = []
    for run in metrics["runs"]:
        runs.append(
            {
                "strategy": run["strategy"],
                "validation macro F1": run["best_validation_macro_f1"],
                "Cargo recall": run["validation_per_class_recall"]["Cargo"],
                "Passengership recall": run["validation_per_class_recall"][
                    "Passengership"
                ],
                "Tanker recall": run["validation_per_class_recall"]["Tanker"],
                "Tug recall": run["validation_per_class_recall"]["Tug"],
            }
        )
    test = metrics["selected_test"]["window"]
    rare = metrics["rare_event"]["operating_point"]
    rejection = metrics["rejection"]
    support = pd.DataFrame(metrics["support_by_split"])
    audit = metrics["label_audit"]
    if audit["status"] == "complete":
        audit_detail = (
            "All 48 manual reviews are complete. Reviewed disposition counts: "
            f"`{audit['disposition_counts']}`. See `label_audit_summary.md` for "
            "the human-reviewed breakdown."
        )
        exit_status = (
            "The engineering analysis and human label audit are complete. "
            "Interpretation remains limited to this public subset."
        )
    else:
        audit_detail = (
            "Until all 48 rows are reviewed, counts of model error, dataset "
            "ambiguity, domain shift, mixed cases, and unresolved cases remain "
            "**pending**."
        )
        exit_status = (
            "The engineering analysis is complete. The full milestone exit "
            "remains pending human completion of all 48 label-audit rows and "
            "regeneration of the disposition summary."
        )
    report = f"""# Milestone 4 — Imbalance, Rare Events, and Label Risk

## Outcome

The validation-selected imbalance strategy was
`{metrics["selected_strategy"]}`. Selection used validation macro F1 and only
the selected strategy was evaluated on test. The test set had already been
used in Milestone 3, so this is transparent repeated test-set use rather than
a pristine confirmatory evaluation.

{markdown_table(pd.DataFrame(runs))}

Selected-strategy test macro F1 was **{test["macro_f1"]:.3f}** with accuracy
**{test["accuracy"]:.3f}**. Accuracy is secondary; per-class recall and macro F1
remain primary.

## Coverage and imbalance risk

{markdown_table(support)}

Tug has 118 windows but only three independent vessel groups—one in each
split. Window oversampling cannot create new acoustic coverage.

## Tug one-vs-rest diagnostic

The selected four-class CNN's Tug probability is analyzed as a separate score;
this is not a dedicated or independently validated rare-event detector.

- Validation average precision:
  {metrics["rare_event"]["validation"]["average_precision"]:.3f}
- Test average precision: {metrics["rare_event"]["test"]["average_precision"]:.3f}
- Validation-selected threshold at FPR ≤5%: {rare["threshold"]:.6f}
- Fixed-threshold test recall: {rare["test"]["recall"]:.3f}
- Fixed-threshold test precision: {rare["test"]["precision"]:.3f}
- Fixed-threshold test FPR: {rare["test"]["false_positive_rate"]:.3f}
- Test counts: TP {rare["test"]["true_positive"]}, FP {rare["test"]["false_positive"]},
  FN {rare["test"]["false_negative"]}, TN {rare["test"]["true_negative"]}

Source-level rare-event results are secondary because the test set contains one
Tug source group.

## Rejection diagnostics

Both thresholds retain approximately 90% of validation windows:

| method | threshold | test coverage | selective test accuracy | selective test macro F1 |
|---|---:|---:|---:|---:|
| maximum probability | {rejection["maximum_probability"]["threshold"]:.6f} | {rejection["maximum_probability"]["test"]["coverage"]:.3f} | {rejection["maximum_probability"]["test"]["classification_on_accepted"]["accuracy"]:.3f} | {rejection["maximum_probability"]["test"]["classification_on_accepted"]["macro_f1"]:.3f} |
| embedding distance | {rejection["embedding_distance"]["threshold"]:.6f} | {rejection["embedding_distance"]["test"]["coverage"]:.3f} | {rejection["embedding_distance"]["test"]["classification_on_accepted"]["accuracy"]:.3f} | {rejection["embedding_distance"]["test"]["classification_on_accepted"]["macro_f1"]:.3f} |

There are no unknown/background examples. These results describe confidence
and in-distribution outliers only and do not validate unknown detection.

## Label and domain risk

`data/annotations/label_audit.csv` contains 48 pending manual reviews: one for
every vessel group plus five targeted extremes. RMS, probabilities, embedding
distance, and source boundaries are selection proxies—not evidence that a
label is wrong or that target-vessel energy is absent.

{audit_detail} Confidence and embedding-distance summaries are domain-shift
indicators, not proof that the model exploited recording-site cues.

## Exit status

{exit_status}
"""
    (REPORT_FOLDER / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig.load(PREPROCESSING_FILE)
    statistics = TrainingStatistics.load(STATISTICS_FILE, preprocessing)
    model_config = SmallCnnConfig(
        input_mel_bands=preprocessing.mel_bands,
        input_frames=preprocessing.output_frames,
    )
    training_config = CnnTrainingConfig()
    preload_started = time.perf_counter()
    store = PreloadedLogMelStore.from_files(
        MANIFEST_FILE,
        AUDIO_FOLDER,
        preprocessing,
        progress=show_progress,
    )
    preload_seconds = time.perf_counter() - preload_started

    results: list[ImbalanceExperimentResult] = []
    validation_predictions = []
    for strategy in STRATEGY_ORDER:
        result, predictions = run_strategy(
            strategy,
            store,
            preprocessing,
            statistics,
            model_config,
            training_config,
        )
        results.append(result)
        validation_predictions.append(predictions)
    selected = select_imbalance_strategy(results)
    selected_model = selected.training_result.model
    print(f"Selected strategy: {selected.imbalance_config.strategy}")

    split_predictions = []
    split_metrics = {}
    for split in ("train", "validation", "test"):
        dataset = store.subset(split, "per_example", preprocessing, statistics)
        predictions, metrics = evaluate_cnn(
            selected_model,
            dataset,
            batch_size=training_config.batch_size,
            include_embeddings=True,
        )
        predictions.insert(0, "strategy", selected.imbalance_config.strategy)
        split_predictions.append(predictions)
        split_metrics[split] = metrics
    all_predictions = pd.concat(split_predictions, ignore_index=True)

    training_rows = all_predictions.loc[all_predictions["split"] == "train"]
    reference = fit_embedding_reference(
        training_rows[EMBEDDING_COLUMNS].to_numpy(),
        training_rows["label_index"].to_numpy(),
        split="train",
    )
    analysis = add_analysis_columns(all_predictions, reference, store)
    rare_metrics, _ = rare_event_analysis(analysis)
    rejection_metrics, embedding_outlier_threshold = rejection_analysis(analysis)
    domain_metrics = save_domain_summaries(analysis)

    test_predictions = analysis.loc[analysis["split"] == "test"].copy()
    source_predictions, source_metrics = aggregate_source_predictions(test_predictions)
    test_predictions.drop(columns=EMBEDDING_COLUMNS).to_csv(
        REPORT_FOLDER / "window_predictions.csv",
        index=False,
    )
    source_predictions.to_csv(REPORT_FOLDER / "source_predictions.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_csv(
        REPORT_FOLDER / "validation_predictions.csv",
        index=False,
    )
    analysis.drop(columns=EMBEDDING_COLUMNS).to_csv(
        REPORT_FOLDER / "window_risk_analysis.csv",
        index=False,
    )
    save_confusion_matrix(split_metrics["test"]["confusion_matrix"])
    save_training_artifacts(results, model_config, training_config)

    reference_payload = {
        "mean": list(reference.mean),
        "standard_deviation": list(reference.standard_deviation),
        "centroids": [list(row) for row in reference.centroids],
        "fit_split": "train",
        "embedding_dimensions": 64,
    }
    (REPORT_FOLDER / "embedding_reference.json").write_text(
        json.dumps(reference_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    run_summaries = []
    for result in results:
        run_summaries.append(
            {
                "strategy": result.imbalance_config.strategy,
                "name": result.imbalance_config.name,
                "config_hash": result.imbalance_config.config_hash,
                "best_epoch": result.training_result.best_epoch,
                "epochs_completed": len(result.training_result.history),
                "best_validation_macro_f1": result.best_validation_macro_f1,
                "validation_per_class_recall": {
                    class_name: values["recall"]
                    for class_name, values in result.validation_metrics[
                        "per_class"
                    ].items()
                },
                "checkpoint": str(
                    result.training_result.checkpoint_path.relative_to(PROJECT_FOLDER)
                ),
            }
        )
    milestone2 = json.loads(MILESTONE2_METRICS.read_text())
    milestone3 = json.loads(MILESTONE3_METRICS.read_text())
    metrics = {
        "primary_metric": "validation window-level macro F1",
        "strategy_order": list(STRATEGY_ORDER),
        "selected_strategy": selected.imbalance_config.strategy,
        "test_evaluated_strategies": [selected.imbalance_config.strategy],
        "runs": run_summaries,
        "selected_test": {
            "window": split_metrics["test"],
            "source_secondary": source_metrics,
        },
        "rare_event": rare_metrics,
        "rejection": rejection_metrics,
        "domain_shift": domain_metrics,
        "label_audit": {
            "status": "pending_manual_review",
            "expected_rows": 48,
            "expected_vessel_groups": 43,
            "embedding_outlier_threshold": embedding_outlier_threshold,
            "disposition_counts": {},
        },
        "support_by_split": support_by_split(pd.DataFrame(store.rows)),
        "preload": {
            "window_count": len(store.rows),
            "memory_bytes": store.memory_bytes,
            "elapsed_seconds": preload_seconds,
            "persistent_tensor_cache": False,
        },
        "prior_results": {
            "milestone2_logistic_test_macro_f1": milestone2["models"][
                "logistic_regression"
            ]["window_test"]["macro_f1"],
            "milestone3_selected_test_macro_f1": milestone3["primary_test"]["window"][
                "macro_f1"
            ],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "training_device": "cpu",
        },
        "preprocessing_config_hash": preprocessing.config_hash,
        "test_reuse_disclosure": (
            "The test split was previously evaluated in Milestone 3 and is not "
            "a pristine confirmatory set."
        ),
    }
    (REPORT_FOLDER / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics)
    print(
        f"Selected test macro F1: {split_metrics['test']['macro_f1']:.3f}"
    )
    print(f"Wrote Milestone 4 analysis to {REPORT_FOLDER}")


if __name__ == "__main__":
    main()
