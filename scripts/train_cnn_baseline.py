#!/usr/bin/env python3
"""Run the three controlled Milestone 3 CNN experiments."""

from __future__ import annotations

import io
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch

from edge_underwater.baseline import support_by_split
from edge_underwater.cnn import SmallCnn, SmallCnnConfig
from edge_underwater.cnn_data import PreloadedLogMelStore
from edge_underwater.cnn_evaluation import (
    aggregate_source_predictions,
    evaluate_cnn,
)
from edge_underwater.cnn_training import (
    CnnRunConfig,
    CnnTrainingConfig,
    TrainingResult,
    select_primary_cnn,
    select_unweighted_normalization,
    train_cnn,
)
from edge_underwater.dataset import DeepShipWindowDataset
from edge_underwater.preprocessing import PreprocessingConfig, TrainingStatistics


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
PREPROCESSING_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
MILESTONE2_METRICS = PROJECT_FOLDER / "reports/milestone2/metrics.json"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models/cnn"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone3"


def show_preload_progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"Preloaded {done}/{total} windows")


def make_run_config(
    name: str,
    normalization: str,
    class_weighted: bool,
    preprocessing: PreprocessingConfig,
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
) -> CnnRunConfig:
    return CnnRunConfig(
        name=name,
        normalization=normalization,
        class_weighted=class_weighted,
        preprocessing_config_hash=preprocessing.config_hash,
        model_config_hash=model_config.config_hash,
        training_config_hash=training_config.config_hash,
    )


def run_experiment(
    store: PreloadedLogMelStore,
    preprocessing: PreprocessingConfig,
    statistics: TrainingStatistics,
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
    run_config: CnnRunConfig,
) -> TrainingResult:
    print(
        f"Training {run_config.name}: normalization={run_config.normalization}, "
        f"class_weighted={run_config.class_weighted}"
    )
    training_dataset = store.subset(
        "train", run_config.normalization, preprocessing, statistics
    )
    validation_dataset = store.subset(
        "validation", run_config.normalization, preprocessing, statistics
    )
    result = train_cnn(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model_config=model_config,
        training_config=training_config,
        run_config=run_config,
        checkpoint_path=MODEL_FOLDER / f"{run_config.name}.pt",
    )
    print(
        f"Finished {run_config.name}: best epoch {result.best_epoch}, "
        f"validation macro F1 {result.best_validation_macro_f1:.3f}"
    )
    return result


def save_configs(
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
    results: list[TrainingResult],
) -> None:
    model_config.save(REPORT_FOLDER / "model_config.json")
    training_payload = asdict(training_config)
    training_payload["config_hash"] = training_config.config_hash
    (REPORT_FOLDER / "training_config.json").write_text(
        json.dumps(training_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    run_payload = []
    for result in results:
        values = asdict(result.run_config)
        values["config_hash"] = result.run_config.config_hash
        values["best_epoch"] = result.best_epoch
        values["best_validation_macro_f1"] = result.best_validation_macro_f1
        values["stopped_early"] = result.stopped_early
        run_payload.append(values)
    (REPORT_FOLDER / "run_configs.json").write_text(
        json.dumps(run_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def save_histories(results: list[TrainingResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for epoch in result.history:
            rows.append(
                {
                    "run_name": result.run_config.name,
                    "normalization": result.run_config.normalization,
                    "class_weighted": result.run_config.class_weighted,
                    **epoch,
                }
            )
    history = pd.DataFrame(rows)
    history.to_csv(REPORT_FOLDER / "training_history.csv", index=False)
    return history


def plot_training_curves(results: list[TrainingResult]) -> None:
    figure, axes = plt.subplots(
        len(results),
        3,
        figsize=(14, 10),
        constrained_layout=True,
    )
    metrics = (
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
    )
    for row_index, result in enumerate(results):
        history = pd.DataFrame(result.history)
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            axis.plot(
                history["epoch"],
                history[f"training_{metric}"],
                label="Training",
            )
            axis.plot(
                history["epoch"],
                history[f"validation_{metric}"],
                label="Validation",
            )
            axis.axvline(
                result.best_epoch,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Best epoch" if column_index == 0 else None,
            )
            axis.set_title(f"{result.run_config.name}\n{title}")
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            if column_index == 0:
                axis.legend()
    figure.savefig(REPORT_FOLDER / "training_curves.png", dpi=160)
    plt.close(figure)


def percentile_summary(seconds: list[float]) -> dict[str, float | int]:
    milliseconds = np.asarray(seconds) * 1_000
    return {
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "total_seconds": float(np.sum(seconds)),
        "call_count": len(seconds),
    }


def time_calls(call: Callable[[int], None], count: int) -> list[float]:
    times = []
    for index in range(count):
        started = time.perf_counter()
        call(index)
        times.append(time.perf_counter() - started)
    return times


def benchmark_primary(
    model: SmallCnn,
    test_dataset,
    normalization: str,
    preprocessing: PreprocessingConfig,
    statistics: TrainingStatistics,
) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model.eval()
    inference_tensors = [
        test_dataset[index]["features"].unsqueeze(0)
        for index in range(len(test_dataset))
    ]
    with torch.inference_mode():
        for index in range(10):
            model(inference_tensors[index % len(inference_tensors)])

        def infer(index: int) -> None:
            model(inference_tensors[index % len(inference_tensors)])

        inference = percentile_summary(time_calls(infer, 1_000))

    preprocessing_dataset = DeepShipWindowDataset(
        manifest_path=MANIFEST_FILE,
        audio_root=AUDIO_FOLDER,
        split="test",
        normalization=normalization,
        statistics=statistics,
        config=preprocessing,
    )
    for index in range(10):
        preprocessing_dataset[index]
    preprocessing_timing = percentile_summary(
        time_calls(lambda index: preprocessing_dataset[index], len(preprocessing_dataset))
    )

    end_to_end_dataset = DeepShipWindowDataset(
        manifest_path=MANIFEST_FILE,
        audio_root=AUDIO_FOLDER,
        split="test",
        normalization=normalization,
        statistics=statistics,
        config=preprocessing,
    )
    with torch.inference_mode():
        for index in range(10):
            item = end_to_end_dataset[index]
            model(item["features"].unsqueeze(0))

        def end_to_end(index: int) -> None:
            item = end_to_end_dataset[index]
            model(item["features"].unsqueeze(0))

        full_timing = percentile_summary(
            time_calls(end_to_end, len(end_to_end_dataset))
        )
    return {
        "measurement_note": (
            "Warm-cache development-Mac CPU measurements, one PyTorch thread, "
            "batch size one; not target-edge-device performance."
        ),
        "preprocessing": preprocessing_timing,
        "inference": inference,
        "end_to_end": full_timing,
        "thread_count": 1,
        "previous_training_thread_count": previous_threads,
    }


def state_dict_size_bytes(model: SmallCnn) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def save_confusion_matrix(matrix: list[list[int]]) -> None:
    values = np.asarray(matrix)
    rows = []
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    axis.set_title("Small CNN Test Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(4), ("Cargo", "Passengership", "Tanker", "Tug"), rotation=30)
    axis.set_yticks(range(4), ("Cargo", "Passengership", "Tanker", "Tug"))
    for true_index in range(4):
        for predicted_index in range(4):
            count = int(values[true_index, predicted_index])
            axis.text(predicted_index, true_index, count, ha="center", va="center")
            rows.append(
                {
                    "true_class": ("Cargo", "Passengership", "Tanker", "Tug")[
                        true_index
                    ],
                    "predicted_class": (
                        "Cargo",
                        "Passengership",
                        "Tanker",
                        "Tug",
                    )[predicted_index],
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
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(metrics: dict[str, Any]) -> None:
    cnn = metrics["primary_test"]["window"]
    baseline = metrics["baseline_comparison"]
    comparison = pd.DataFrame(
        [
            {
                "model": "Small CNN",
                "role": "Milestone 3 primary",
                "test macro F1": cnn["macro_f1"],
                "test accuracy": cnn["accuracy"],
            },
            {
                "model": "Logistic regression",
                "role": "Milestone 2 primary",
                "test macro F1": baseline["logistic_regression"]["macro_f1"],
                "test accuracy": baseline["logistic_regression"]["accuracy"],
            },
            {
                "model": "Random forest",
                "role": "Milestone 2 secondary",
                "test macro F1": baseline["random_forest"]["macro_f1"],
                "test accuracy": baseline["random_forest"]["accuracy"],
            },
        ]
    )
    runs = pd.DataFrame(metrics["runs"])[
        [
            "name",
            "normalization",
            "class_weighted",
            "best_epoch",
            "epochs_completed",
            "best_validation_macro_f1",
        ]
    ]
    class_rows = []
    for class_name, values in cnn["per_class"].items():
        class_rows.append({"class": class_name, **values})
    support = pd.DataFrame(metrics["support_by_split"])
    timing = metrics["timing"]
    review_count = metrics["error_review"]["selected_error_count"]
    report = f"""# Milestone 3 — Small CNN Baseline

## Outcome

The validation-selected primary run was `{metrics["primary_run"]}`. It used
`{metrics["primary_normalization"]}` normalization and
{"class-weighted" if metrics["primary_class_weighted"] else "unweighted"} loss.
Only this selected CNN was evaluated on the test split.

{markdown_table(comparison)}

Window-level macro F1 remains the primary metric. Source-level aggregation is
secondary because the test split contains only one Cargo, one Passengership,
and one Tug vessel group.

## Controlled training runs

{markdown_table(runs)}

Both unweighted normalization runs completed before class weighting was
calculated and tested. Selection used validation macro F1 only, with
`training_stats` preferred for a normalization tie and unweighted loss
preferred for a weighting tie. No train-plus-validation refit was performed.

## Primary CNN test metrics

{markdown_table(pd.DataFrame(class_rows))}

The source-level test macro F1 was
{metrics["primary_test"]["source_secondary"]["macro_f1"]:.3f}; this is a
secondary descriptive result.

## Support

{markdown_table(support)}

## Size and latency

- Parameters: {metrics["model"]["parameter_count"]:,}
- Deployable state-dict size: {metrics["model"]["state_dict_size_bytes"]:,} bytes
- Full checkpoint size: {metrics["model"]["checkpoint_size_bytes"]:,} bytes
- Preprocessing: median {timing["preprocessing"]["median_ms"]:.3f} ms,
  p95 {timing["preprocessing"]["p95_ms"]:.3f} ms
- CNN inference: median {timing["inference"]["median_ms"]:.3f} ms,
  p95 {timing["inference"]["p95_ms"]:.3f} ms
- End to end: median {timing["end_to_end"]["median_ms"]:.3f} ms,
  p95 {timing["end_to_end"]["p95_ms"]:.3f} ms

These are warm-cache, batch-one measurements on the current development Mac
with one PyTorch thread, not target-edge-device performance.

## Error listening review

A deterministic pack of {review_count} incorrect test windows is stored in
`data/annotations/deepship_cnn_error_review.csv`. Its listening fields are
intentionally blank. Human listening remains pending and must be completed
before the listening checklist can be marked done.

## Reporting limits

Windows within recordings and vessels are correlated. The 135 test windows are
not 135 independent vessel observations, and these public-subset results do not
generalize to the complete 47-hour DeepShip dataset. No minimum CNN F1 was
required.
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
        progress=show_preload_progress,
    )
    preload_seconds = time.perf_counter() - preload_started
    split_counts = pd.Series([row["split"] for row in store.rows]).value_counts()
    if split_counts.to_dict() != {"train": 829, "validation": 154, "test": 135}:
        raise RuntimeError(f"Unexpected preloaded split counts: {split_counts.to_dict()}")

    training_stats_run = make_run_config(
        "training_stats_unweighted",
        "training_stats",
        False,
        preprocessing,
        model_config,
        training_config,
    )
    per_example_run = make_run_config(
        "per_example_unweighted",
        "per_example",
        False,
        preprocessing,
        model_config,
        training_config,
    )
    first = run_experiment(
        store,
        preprocessing,
        statistics,
        model_config,
        training_config,
        training_stats_run,
    )
    second = run_experiment(
        store,
        preprocessing,
        statistics,
        model_config,
        training_config,
        per_example_run,
    )
    best_unweighted = select_unweighted_normalization(first, second)

    weighted_run = make_run_config(
        f"{best_unweighted.run_config.normalization}_weighted",
        best_unweighted.run_config.normalization,
        True,
        preprocessing,
        model_config,
        training_config,
    )
    third = run_experiment(
        store,
        preprocessing,
        statistics,
        model_config,
        training_config,
        weighted_run,
    )
    results = [first, second, third]
    primary = select_primary_cnn(best_unweighted, third)

    # Test data is exposed to a model only after all validation selection is complete.
    test_dataset = store.subset(
        "test",
        primary.run_config.normalization,
        preprocessing,
        statistics,
    )
    window_predictions, window_metrics = evaluate_cnn(
        primary.model,
        test_dataset,
        batch_size=training_config.batch_size,
    )
    source_output, source_metrics = aggregate_source_predictions(window_predictions)
    window_predictions.to_csv(REPORT_FOLDER / "window_predictions.csv", index=False)
    source_output.to_csv(REPORT_FOLDER / "source_predictions.csv", index=False)

    save_configs(model_config, training_config, results)
    save_histories(results)
    plot_training_curves(results)
    save_confusion_matrix(window_metrics["confusion_matrix"])

    baseline_metrics = json.loads(MILESTONE2_METRICS.read_text())
    timing = benchmark_primary(
        primary.model,
        test_dataset,
        primary.run_config.normalization,
        preprocessing,
        statistics,
    )
    run_summaries = []
    for result in results:
        run_summaries.append(
            {
                "name": result.run_config.name,
                "normalization": result.run_config.normalization,
                "class_weighted": result.run_config.class_weighted,
                "best_epoch": result.best_epoch,
                "epochs_completed": len(result.history),
                "best_validation_macro_f1": result.best_validation_macro_f1,
                "stopped_early": result.stopped_early,
                "checkpoint": str(result.checkpoint_path.relative_to(PROJECT_FOLDER)),
                "run_config_hash": result.run_config.config_hash,
            }
        )
    metrics = {
        "primary_metric": "test window-level macro F1",
        "primary_run": primary.run_config.name,
        "primary_normalization": primary.run_config.normalization,
        "primary_class_weighted": primary.run_config.class_weighted,
        "test_evaluated_runs": [primary.run_config.name],
        "runs": run_summaries,
        "primary_test": {
            "window": window_metrics,
            "source_secondary": source_metrics,
        },
        "baseline_comparison": {
            "logistic_regression": {
                "macro_f1": baseline_metrics["models"]["logistic_regression"][
                    "window_test"
                ]["macro_f1"],
                "accuracy": baseline_metrics["models"]["logistic_regression"][
                    "window_test"
                ]["accuracy"],
            },
            "random_forest": {
                "macro_f1": baseline_metrics["models"]["random_forest"][
                    "window_test"
                ]["macro_f1"],
                "accuracy": baseline_metrics["models"]["random_forest"][
                    "window_test"
                ]["accuracy"],
            },
        },
        "support_by_split": support_by_split(pd.DataFrame(store.rows)),
        "model": {
            "parameter_count": primary.model.parameter_count,
            "state_dict_size_bytes": state_dict_size_bytes(primary.model),
            "checkpoint_size_bytes": primary.checkpoint_path.stat().st_size,
            "model_config_hash": model_config.config_hash,
            "training_config_hash": training_config.config_hash,
            "primary_run_config_hash": primary.run_config.config_hash,
        },
        "preload": {
            "window_count": len(store.rows),
            "memory_bytes": store.memory_bytes,
            "elapsed_seconds": preload_seconds,
            "persistent_tensor_cache": False,
        },
        "timing": timing,
        "error_review": {
            "status": "pending_manual_listening",
            "selected_error_count": min(
                20,
                int(
                    (
                        window_predictions["label_index"]
                        != window_predictions["predicted_label_index"]
                    ).sum()
                ),
            ),
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
        "reporting_limits": [
            "Windows within recordings and vessels are correlated.",
            "The public test split has one Cargo, Passengership, and Tug vessel group.",
            "Results do not generalize to the full 47-hour DeepShip dataset.",
        ],
    }
    (REPORT_FOLDER / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics)
    print(f"Selected primary run: {primary.run_config.name}")
    print(f"Test window macro F1: {window_metrics['macro_f1']:.3f}")
    print(f"Wrote Milestone 3 results to {REPORT_FOLDER}")


if __name__ == "__main__":
    main()
