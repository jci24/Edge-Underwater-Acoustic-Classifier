#!/usr/bin/env python3
"""Validate the committed Milestone 6 study and ignored ONNX models."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pandas as pd

from edge_underwater.edge_benchmark import sha256_file
from edge_underwater.onnx_quantization import (
    VARIANT_ORDER,
    OnnxQuantizationConfig,
    select_deployment_variant,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone6"
CONFIG_FILE = REPORT_FOLDER / "quantization_config.json"
METRICS_FILE = REPORT_FOLDER / "metrics.json"
PREDICTIONS_FILE = REPORT_FOLDER / "window_predictions.csv"
TIMINGS_FILE = REPORT_FOLDER / "inference_timings.csv"


def check_ignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=PROJECT_FOLDER,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Generated ONNX model must remain ignored: {path}")


def graph_operator_counts(model: onnx.ModelProto) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def main() -> None:
    config = OnnxQuantizationConfig.load(CONFIG_FILE)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS_FILE)
    timings = pd.read_csv(TIMINGS_FILE)

    if metrics["quantization_config_hash"] != config.config_hash:
        raise RuntimeError("Metrics use a different Milestone 6 configuration.")
    if metrics["selected_strategy"] != "class_vessel_balanced_sampling":
        raise RuntimeError("Study did not use the Milestone 4 selected CNN.")

    calibration = metrics["calibration"]
    if calibration != {
        "window_count": 829,
        "unique_window_count": 829,
        "classes": ["Cargo", "Passengership", "Tanker", "Tug"],
        "vessel_group_count": 29,
        "split": "train",
    }:
        raise RuntimeError("Static calibration coverage or split differs.")

    model_files = metrics["model_files"]
    for variant in VARIANT_ORDER:
        details = model_files[variant]
        path = PROJECT_FOLDER / details["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {variant} model: {path}. Run the study script first."
            )
        check_ignored(path)
        if path.stat().st_size != details["size_bytes"]:
            raise RuntimeError(f"{variant} model size differs from the report.")
        if sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"{variant} model checksum differs from the report.")

        model = onnx.load(path)
        onnx.checker.check_model(model)
        if model.graph.input[0].name != "features":
            raise RuntimeError(f"{variant} input must be named features.")
        if model.graph.output[0].name != "logits":
            raise RuntimeError(f"{variant} output must be named logits.")
        input_shape = [
            dimension.dim_value
            for dimension in model.graph.input[0].type.tensor_type.shape.dim
        ]
        output_shape = [
            dimension.dim_value
            for dimension in model.graph.output[0].type.tensor_type.shape.dim
        ]
        if input_shape != [1, 1, 64, 155] or output_shape != [1, 4]:
            raise RuntimeError(f"{variant} fixed shapes differ.")
        if not model.graph.initializer:
            raise RuntimeError(f"{variant} has no embedded weights.")

        operators = graph_operator_counts(model)
        if variant == "dynamic_int8":
            if operators.get("Conv") != 3:
                raise RuntimeError("Dynamic INT8 must leave all convolutions FP32.")
            if operators.get("MatMulInteger", 0) < 1:
                raise RuntimeError("Dynamic INT8 did not quantize the linear head.")
        if variant == "static_int8":
            if operators.get("Conv") != 3:
                raise RuntimeError("Static INT8 must contain three QDQ convolutions.")
            if (
                operators.get("QuantizeLinear", 0) < 1
                or operators.get("DequantizeLinear", 0) < 1
            ):
                raise RuntimeError("Static INT8 graph is not QDQ.")

    expected_prediction_counts = {
        ("test", variant): 135 for variant in VARIANT_ORDER
    } | {
        ("validation", variant): 154 for variant in VARIANT_ORDER
    }
    prediction_counts = predictions.groupby(["split", "variant"]).size().to_dict()
    if prediction_counts != expected_prediction_counts:
        raise RuntimeError("Validation/test prediction coverage is incomplete.")
    logit_columns = [
        column for column in predictions if column.startswith("logit_")
    ]
    if not np.isfinite(predictions[logit_columns]).all().all():
        raise RuntimeError("Predictions contain non-finite logits.")

    parity = metrics["fp32_pytorch_parity"]
    if (
        parity["window_count"] != 135
        or parity["matching_prediction_count"] != 135
        or not parity["all_predictions_match"]
        or parity["maximum_absolute_error"] > config.absolute_tolerance
    ):
        raise RuntimeError("FP32 ONNX/PyTorch parity contract failed.")

    timing_counts = timings.groupby("variant").size().to_dict()
    if timing_counts != {variant: 1_000 for variant in VARIANT_ORDER}:
        raise RuntimeError("Each ONNX variant must have 1,000 timing samples.")
    if set(timings["thread_count"]) != {1} or (timings["duration_ns"] <= 0).any():
        raise RuntimeError("Timing samples must be positive and single threaded.")
    if set(timings["benchmark_split"]) != {"validation"}:
        raise RuntimeError("Selection timing must use validation inputs only.")

    comparison = metrics["comparison"]
    selection_rows = [
        {
            "variant": row["variant"],
            "qualification": row["qualification"],
            "timing": row["timing"],
            "size_bytes": row["size_bytes"],
        }
        for row in comparison
    ]
    recommendation = select_deployment_variant(selection_rows)
    if metrics["selection_split"] != "validation":
        raise RuntimeError("Deployment recommendation did not use validation.")
    if recommendation != metrics["recommended_variant"]:
        raise RuntimeError("Deployment recommendation cannot be reproduced.")
    if set(metrics["environment"]["versions"]) != {
        "numpy",
        "torch",
        "torchaudio",
        "torchcodec",
        "onnx",
        "onnxruntime",
        "onnxscript",
    }:
        raise RuntimeError("Runtime version evidence is incomplete.")
    if metrics["environment"]["provider"] != "CPUExecutionProvider":
        raise RuntimeError("Study did not use ONNX Runtime CPU.")

    print("Validated 829 training-only calibration windows")
    print("Validated fixed FP32, dynamic INT8, and static QDQ INT8 graphs")
    print("Validated FP32 parity across all 135 reused-test windows")
    print("Validated 1,000 single-thread timing samples per ONNX model")
    print(f"Validated validation-only recommendation: {recommendation}")


if __name__ == "__main__":
    main()
