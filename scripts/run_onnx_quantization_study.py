#!/usr/bin/env python3
"""Run Milestone 6 FP32 parity and ONNX quantization comparisons."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import torch

from edge_underwater.baseline import ORDERED_CLASSES
from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.cnn_data import PreloadedLogMelStore
from edge_underwater.cnn_training import load_cnn_checkpoint
from edge_underwater.edge_benchmark import (
    create_onnx_session,
    export_onnx_model,
    measure_calls,
    sha256_file,
    summarize_durations,
    validate_onnx_parity,
)
from edge_underwater.onnx_quantization import (
    VARIANT_ORDER,
    OnnxQuantizationConfig,
    TrainingCalibrationReader,
    comparison_with_fp32,
    create_quantized_models,
    evaluate_onnx_session,
    select_deployment_variant,
    strip_intermediate_value_info,
    variant_qualifies,
)
from edge_underwater.preprocessing import PreprocessingConfig, TrainingStatistics


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
PREPROCESSING_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
MILESTONE4_METRICS = PROJECT_FOLDER / "reports/milestone4/metrics.json"
MILESTONE4_RUNS = PROJECT_FOLDER / "reports/milestone4/run_configs.json"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models/onnx"
FP32_MODEL = MODEL_FOLDER / "small_cnn_fp32.onnx"
QUANTIZATION_SOURCE = MODEL_FOLDER / "small_cnn_quantization_source.onnx"
DYNAMIC_MODEL = MODEL_FOLDER / "small_cnn_dynamic_int8.onnx"
STATIC_MODEL = MODEL_FOLDER / "small_cnn_static_int8.onnx"
MODEL_PATHS = {
    "fp32": FP32_MODEL,
    "dynamic_int8": DYNAMIC_MODEL,
    "static_int8": STATIC_MODEL,
}
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone6"


def selected_model_details():
    metrics = json.loads(MILESTONE4_METRICS.read_text(encoding="utf-8"))
    runs = json.loads(MILESTONE4_RUNS.read_text(encoding="utf-8"))
    selected_run = next(
        run for run in runs if run["strategy"] == metrics["selected_strategy"]
    )
    checkpoint_path = PROJECT_FOLDER / selected_run["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing selected checkpoint: {checkpoint_path}. "
            "Run python3 scripts/train_imbalance_risk.py first."
        )
    if metrics["selected_strategy"] != "class_vessel_balanced_sampling":
        raise RuntimeError("Milestone 4 selected strategy changed unexpectedly.")
    model, checkpoint = load_cnn_checkpoint(checkpoint_path)
    return checkpoint_path, metrics, selected_run, model, checkpoint


def make_config(
    checkpoint_path: Path,
    milestone4: dict[str, Any],
    selected_run: dict[str, Any],
    checkpoint: dict[str, Any],
) -> OnnxQuantizationConfig:
    return OnnxQuantizationConfig(
        source_checkpoint=str(checkpoint_path.relative_to(PROJECT_FOLDER)),
        source_checkpoint_sha256=sha256_file(checkpoint_path),
        preprocessing_config_hash=milestone4["preprocessing_config_hash"],
        model_config_hash=checkpoint["model_config_hash"],
        training_config_hash=checkpoint["training_config_hash"],
        selected_run_config_hash=selected_run["config_hash"],
    )


def progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        print(f"Preloaded {done}/{total} log-mel windows", flush=True)


def dataset_features(dataset) -> list[torch.Tensor]:
    return [dataset[index]["features"] for index in range(len(dataset))]


def calibration_rows(dataset) -> list[dict[str, Any]]:
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        rows.append(
            {
                "window_id": item["window_id"],
                "split": item["split"],
                "class": item["class"],
                "vessel_group": item["vessel_group"],
            }
        )
    return rows


def validate_graph(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if model.graph.input[0].name != "features":
        raise RuntimeError(f"Unexpected input name in {path}.")
    if model.graph.output[0].name != "logits":
        raise RuntimeError(f"Unexpected output name in {path}.")
    input_shape = [
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    if input_shape != [1, 1, 64, 155] or output_shape != [1, 4]:
        raise RuntimeError(f"Unexpected fixed shapes in {path}.")
    if not model.graph.initializer:
        raise RuntimeError(f"{path} contains no embedded weights.")
    operator_counts: dict[str, int] = {}
    for node in model.graph.node:
        operator_counts[node.op_type] = operator_counts.get(node.op_type, 0) + 1
    return {
        "checker_valid": True,
        "input_name": model.graph.input[0].name,
        "output_name": model.graph.output[0].name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "operator_counts": operator_counts,
        "initializer_count": len(model.graph.initializer),
    }


def benchmark_sessions(
    sessions: dict[str, Any],
    validation_features: list[torch.Tensor],
    config: OnnxQuantizationConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    numpy_features = [
        tensor.unsqueeze(0).numpy().astype(np.float32, copy=False)
        for tensor in validation_features
    ]
    raw_rows = []
    summaries = {}
    for variant in VARIANT_ORDER:
        session = sessions[variant]

        def infer(index: int) -> None:
            session.run(
                ["logits"],
                {"features": numpy_features[index % len(numpy_features)]},
            )

        durations = measure_calls(
            infer,
            config.warmup_calls,
            config.measured_calls,
        )
        summaries[variant] = summarize_durations(durations)
        for iteration, duration_ns in enumerate(durations):
            raw_rows.append(
                {
                    "variant": variant,
                    "thread_count": config.intra_op_threads,
                    "iteration": iteration,
                    "benchmark_split": "validation",
                    "benchmark_window_index": iteration % len(validation_features),
                    "duration_ns": duration_ns,
                }
            )
    return pd.DataFrame(raw_rows), summaries


def save_confusion_artifacts(
    metrics_by_split: dict[str, dict[str, dict[str, Any]]],
) -> None:
    rows = []
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
        constrained_layout=True,
    )
    for split_index, split in enumerate(("validation", "test")):
        for variant_index, variant in enumerate(VARIANT_ORDER):
            matrix = np.asarray(
                metrics_by_split[split][variant]["confusion_matrix"],
                dtype=np.int64,
            )
            axis = axes[split_index, variant_index]
            axis.imshow(matrix, cmap="Blues")
            axis.set_title(f"{split} — {variant}")
            axis.set_xticks(range(4), ORDERED_CLASSES, rotation=35, ha="right")
            axis.set_yticks(range(4), ORDERED_CLASSES)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            for true_index in range(4):
                for predicted_index in range(4):
                    count = int(matrix[true_index, predicted_index])
                    axis.text(
                        predicted_index,
                        true_index,
                        str(count),
                        ha="center",
                        va="center",
                    )
                    rows.append(
                        {
                            "split": split,
                            "variant": variant,
                            "true_class": ORDERED_CLASSES[true_index],
                            "predicted_class": ORDERED_CLASSES[predicted_index],
                            "count": count,
                        }
                    )
    pd.DataFrame(rows).to_csv(
        REPORT_FOLDER / "confusion_matrices.csv",
        index=False,
    )
    figure.savefig(REPORT_FOLDER / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def save_tradeoff_plots(comparison: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for row in comparison:
        variant = row["variant"]
        size_kb = row["size_bytes"] / 1_000
        p99 = row["timing"]["p99_ms"]
        macro_f1 = row["validation_metrics"]["macro_f1"]
        axes[0].scatter(size_kb, p99, s=100)
        axes[0].annotate(variant, (size_kb, p99), xytext=(5, 5), textcoords="offset points")
        axes[1].scatter(size_kb, macro_f1, s=100)
        axes[1].annotate(
            variant,
            (size_kb, macro_f1),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axes[0].set_title("Model size and p99 latency")
    axes[0].set_xlabel("Model size (kB)")
    axes[0].set_ylabel("Single-thread p99 (ms)")
    axes[1].set_title("Model size and validation macro F1")
    axes[1].set_xlabel("Model size (kB)")
    axes[1].set_ylabel("Validation macro F1")
    figure.savefig(REPORT_FOLDER / "size_latency_accuracy_tradeoff.png", dpi=160)
    plt.close(figure)


def report_table_rows(comparison: list[dict[str, Any]]) -> str:
    rows = []
    for row in comparison:
        qualification = row["qualification"]
        rows.append(
            f"| {row['variant']} | {row['size_bytes']:,} | "
            f"{row['timing']['median_ms']:.3f} | "
            f"{row['timing']['p99_ms']:.3f} | "
            f"{row['validation_metrics']['macro_f1']:.3f} | "
            f"{row['test_metrics']['macro_f1']:.3f} | "
            f"{row['validation_vs_fp32']['prediction_agreement']:.3f} | "
            f"{'yes' if qualification['qualifies'] else 'no'} |"
        )
    return "\n".join(rows)


def quantization_findings(comparison: list[dict[str, Any]]) -> str:
    findings = []
    for row in comparison[1:]:
        checks = row["qualification"]
        failed_checks = [
            name
            for name, key in (
                ("validation macro F1", "macro_f1_pass"),
                ("per-class validation recall", "per_class_recall_pass"),
                ("p99 latency", "latency_pass"),
                ("file size", "size_pass"),
            )
            if not checks[key]
        ]
        if failed_checks:
            outcome = "failed " + ", ".join(failed_checks)
        else:
            outcome = "passed every qualification rule"
        findings.append(
            f"- `{row['variant']}` {outcome}. Its validation macro F1 loss was "
            f"`{checks['macro_f1_loss']:.4f}`, maximum class-recall loss was "
            f"`{checks['maximum_recall_loss']:.4f}`, p99 latency change was "
            f"`{checks['p99_latency_increase_ratio'] * 100:.1f}%`, and size "
            f"reduction was `{row['size_reduction_ratio'] * 100:.1f}%`."
        )
    return "\n".join(findings)


def write_report(metrics: dict[str, Any]) -> None:
    comparison = metrics["comparison"]
    report = f"""# Milestone 6 — ONNX Parity and Quantization Study

## Outcome

The fixed-shape FP32 ONNX model reproduced all 135 PyTorch test predictions.
Maximum absolute FP32 logit error was
`{metrics['fp32_pytorch_parity']['maximum_absolute_error']:.8f}`, within the
configured `1e-4` absolute and relative tolerances.

The validation-only deployment recommendation is
**`{metrics['recommended_variant']}`**.

| variant | bytes | median ms | p99 ms | validation macro F1 | reused-test macro F1 | validation agreement with FP32 | qualifies |
|---|---:|---:|---:|---:|---:|---:|---|
{report_table_rows(comparison)}

Latency uses ONNX Runtime CPU, sequential execution, one intra-op thread,
batch size one, 50 warm-up calls, and 1,000 measured calls.

## Quantization scope

- Dynamic INT8 targets only the exported `Gemm`/`MatMul` head. The three
  convolution operators remain FP32.
- Static INT8 uses signed per-channel QDQ quantization for `Conv`, `Gemm`, and
  `MatMul`, calibrated with all 829 per-example-normalized training windows.
- Calibration contains no validation or test rows.

Quantization is treated as an experiment. Smaller files are not declared
better unless validation quality and measured latency also satisfy the
predefined limits.

## Trade-off interpretation

{quantization_findings(comparison)}

## Selection policy

A quantized variant must lose no more than `0.01` validation macro F1 or
`0.05` recall in any class, must be no more than 5% slower at p99, and must be
smaller than FP32. Selection uses validation metrics only; reused-test results
do not change the recommendation.

## Limits

- The test set has already been evaluated in earlier milestones and is not a
  pristine confirmatory set.
- Results describe ONNX Runtime CPU on the current Apple M2 development
  laptop, not the eventual underwater edge device.
- No retraining, quantization-aware training, pruning, float16 conversion, or
  dynamic input shape is included.
"""
    (REPORT_FOLDER / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint_path, milestone4, selected_run, model, checkpoint = (
        selected_model_details()
    )
    preprocessing = PreprocessingConfig.load(PREPROCESSING_FILE)
    statistics = TrainingStatistics.load(STATISTICS_FILE, preprocessing)
    config = make_config(
        checkpoint_path,
        milestone4,
        selected_run,
        checkpoint,
    )
    config.save(REPORT_FOLDER / "quantization_config.json")

    store = PreloadedLogMelStore.from_files(
        MANIFEST_FILE,
        AUDIO_FOLDER,
        preprocessing,
        progress=progress,
    )
    datasets = {
        split: store.subset(
            split,
            "per_example",
            preprocessing,
            statistics,
        )
        for split in ("train", "validation", "test")
    }
    training_features = dataset_features(datasets["train"])
    reader = TrainingCalibrationReader(
        calibration_rows(datasets["train"]),
        training_features,
        expected_count=config.calibration_window_count,
    )

    example = torch.zeros(config.input_shape, dtype=torch.float32)
    export_onnx_model(
        model,
        FP32_MODEL,
        example,
        config.onnx_opset,
    )
    strip_intermediate_value_info(FP32_MODEL, QUANTIZATION_SOURCE)
    create_quantized_models(
        QUANTIZATION_SOURCE,
        DYNAMIC_MODEL,
        STATIC_MODEL,
        reader,
        config,
    )
    graph_summaries = {
        variant: validate_graph(path)
        for variant, path in MODEL_PATHS.items()
    }
    if graph_summaries["dynamic_int8"]["operator_counts"].get("Conv") != 3:
        raise RuntimeError("Dynamic quantization changed convolution operators.")
    if graph_summaries["dynamic_int8"]["operator_counts"].get(
        "MatMulInteger", 0
    ) < 1:
        raise RuntimeError("Dynamic quantization did not quantize the linear head.")
    if graph_summaries["static_int8"]["operator_counts"].get(
        "QuantizeLinear", 0
    ) < 1 or graph_summaries["static_int8"]["operator_counts"].get(
        "DequantizeLinear", 0
    ) < 1:
        raise RuntimeError("Static quantization did not produce a QDQ graph.")

    sessions = {
        variant: create_onnx_session(path, threads=config.intra_op_threads)
        for variant, path in MODEL_PATHS.items()
    }
    test_features = dataset_features(datasets["test"])
    fp32_pytorch_parity = validate_onnx_parity(
        model,
        sessions["fp32"],
        test_features,
        absolute_tolerance=config.absolute_tolerance,
        relative_tolerance=config.relative_tolerance,
    )

    predictions_by_split = {}
    metrics_by_split = {}
    logits_by_split = {}
    prediction_tables = []
    for split in ("validation", "test"):
        predictions_by_split[split] = {}
        metrics_by_split[split] = {}
        logits_by_split[split] = {}
        for variant in VARIANT_ORDER:
            predictions, variant_metrics, logits = evaluate_onnx_session(
                sessions[variant],
                datasets[split],
                variant,
            )
            predictions_by_split[split][variant] = predictions
            metrics_by_split[split][variant] = variant_metrics
            logits_by_split[split][variant] = logits
            prediction_tables.append(predictions)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        REPORT_FOLDER / "window_predictions.csv",
        index=False,
    )

    comparisons_by_split = {}
    for split in ("validation", "test"):
        comparisons_by_split[split] = {}
        fp32_predictions = predictions_by_split[split]["fp32"]
        fp32_logits = logits_by_split[split]["fp32"]
        for variant in VARIANT_ORDER:
            comparisons_by_split[split][variant] = comparison_with_fp32(
                fp32_predictions,
                fp32_logits,
                predictions_by_split[split][variant],
                logits_by_split[split][variant],
            )

    validation_features = dataset_features(datasets["validation"])
    raw_timings, timing_summaries = benchmark_sessions(
        sessions,
        validation_features,
        config,
    )
    raw_timings.to_csv(REPORT_FOLDER / "inference_timings.csv", index=False)

    fp32_size = FP32_MODEL.stat().st_size
    fp32_validation = metrics_by_split["validation"]["fp32"]
    fp32_p99 = timing_summaries["fp32"]["p99_ms"]
    validation_selection_rows = []
    comparison = []
    for variant in VARIANT_ORDER:
        size_bytes = MODEL_PATHS[variant].stat().st_size
        if variant == "fp32":
            qualification = {
                "qualifies": True,
                "baseline": True,
                "macro_f1_pass": True,
                "per_class_recall_pass": True,
                "latency_pass": True,
                "size_pass": False,
            }
        else:
            _, qualification = variant_qualifies(
                fp32_validation,
                metrics_by_split["validation"][variant],
                fp32_p99,
                timing_summaries[variant]["p99_ms"],
                fp32_size,
                size_bytes,
                config,
            )
        row = {
            "variant": variant,
            "size_bytes": size_bytes,
            "size_reduction_bytes": fp32_size - size_bytes,
            "size_reduction_ratio": (fp32_size - size_bytes) / fp32_size,
            "timing": timing_summaries[variant],
            "validation_metrics": metrics_by_split["validation"][variant],
            "test_metrics": metrics_by_split["test"][variant],
            "validation_vs_fp32": comparisons_by_split["validation"][variant],
            "test_vs_fp32": comparisons_by_split["test"][variant],
            "qualification": qualification,
        }
        comparison.append(row)
        validation_selection_rows.append(
            {
                "variant": variant,
                "qualification": qualification,
                "timing": timing_summaries[variant],
                "size_bytes": size_bytes,
            }
        )
    recommendation = select_deployment_variant(validation_selection_rows)

    save_confusion_artifacts(metrics_by_split)
    save_tradeoff_plots(comparison)
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "torch",
                "torchaudio",
                "torchcodec",
                "onnx",
                "onnxruntime",
                "onnxscript",
            )
        },
        "provider": "CPUExecutionProvider",
        "intra_op_threads": config.intra_op_threads,
        "execution_mode": "sequential",
    }
    metrics = {
        "quantization_config_hash": config.config_hash,
        "selected_strategy": milestone4["selected_strategy"],
        "fp32_pytorch_parity": fp32_pytorch_parity,
        "calibration": reader.coverage,
        "graphs": graph_summaries,
        "model_files": {
            variant: {
                "path": str(path.relative_to(PROJECT_FOLDER)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for variant, path in MODEL_PATHS.items()
        },
        "comparison": comparison,
        "recommended_variant": recommendation,
        "selection_split": "validation",
        "recommended_test_metrics": metrics_by_split["test"][recommendation],
        "test_reuse_disclosure": (
            "The test split was evaluated in earlier milestones and is not a "
            "pristine confirmatory set."
        ),
        "environment": environment,
    }
    (REPORT_FOLDER / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(metrics)
    print(f"FP32 parity: {fp32_pytorch_parity['window_count']}/135")
    print(f"Recommended deployment variant: {recommendation}")
    print(f"Wrote Milestone 6 study to {REPORT_FOLDER}")


if __name__ == "__main__":
    main()
