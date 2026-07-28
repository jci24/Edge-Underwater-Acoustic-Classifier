"""ONNX parity, calibration, quantization, and deployment selection helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pandas as pd
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from torch import Tensor

from .baseline import ORDERED_CLASSES, classification_metrics


VARIANT_ORDER = ("fp32", "dynamic_int8", "static_int8")


@dataclass(frozen=True)
class OnnxQuantizationConfig:
    source_checkpoint: str
    source_checkpoint_sha256: str
    preprocessing_config_hash: str
    model_config_hash: str
    training_config_hash: str
    selected_run_config_hash: str
    input_shape: tuple[int, int, int, int] = (1, 1, 64, 155)
    output_shape: tuple[int, int] = (1, 4)
    onnx_opset: int = 18
    absolute_tolerance: float = 1e-4
    relative_tolerance: float = 1e-4
    normalization: str = "per_example"
    calibration_split: str = "train"
    calibration_window_count: int = 829
    calibration_method: str = "MinMax"
    static_quant_format: str = "QDQ"
    static_activation_type: str = "QInt8"
    static_weight_type: str = "QInt8"
    static_per_channel: bool = True
    dynamic_weight_type: str = "QInt8"
    dynamic_operators: tuple[str, ...] = ("Gemm", "MatMul")
    static_operators: tuple[str, ...] = ("Conv", "Gemm", "MatMul")
    warmup_calls: int = 50
    measured_calls: int = 1_000
    intra_op_threads: int = 1
    maximum_macro_f1_loss: float = 0.01
    maximum_per_class_recall_loss: float = 0.05
    maximum_p99_latency_increase_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.input_shape != (1, 1, 64, 155) or self.output_shape != (1, 4):
            raise ValueError("Milestone 6 requires fixed batch-one CNN shapes.")
        if self.normalization != "per_example" or self.calibration_split != "train":
            raise ValueError("Calibration must use per-example-normalized training rows.")
        if self.calibration_window_count != 829:
            raise ValueError("Milestone 6 requires all 829 training windows.")
        if self.warmup_calls < 1 or self.measured_calls < 500:
            raise ValueError("Benchmark requires warm-up and at least 500 calls.")
        if self.intra_op_threads != 1:
            raise ValueError("Primary ONNX benchmark must use one thread.")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, output_path: Path) -> None:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_shape"] = list(self.output_shape)
        payload["dynamic_operators"] = list(self.dynamic_operators)
        payload["static_operators"] = list(self.static_operators)
        payload["config_hash"] = self.config_hash
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "OnnxQuantizationConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        values = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__
        }
        for key in ("input_shape", "output_shape", "dynamic_operators", "static_operators"):
            values[key] = tuple(values[key])
        config = cls(**values)
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("Saved ONNX quantization configuration hash differs.")
        return config


class TrainingCalibrationReader(CalibrationDataReader):
    """Yield deterministic training-only model inputs to ONNX Runtime."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        features: list[Tensor],
        expected_count: int = 829,
    ) -> None:
        if len(rows) != len(features) or len(rows) != expected_count:
            raise ValueError(f"Calibration requires exactly {expected_count} aligned rows.")
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Calibration data may contain only training rows.")
        window_ids = [str(row["window_id"]) for row in rows]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("Calibration window IDs must be unique.")
        for tensor in features:
            if tuple(tensor.shape) != (1, 64, 155):
                raise ValueError("Calibration feature must have shape [1,64,155].")
            if not torch.isfinite(tensor).all():
                raise ValueError("Calibration features must be finite.")
        self.rows = rows
        self.features = features
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.features):
            return None
        tensor = self.features[self.position].unsqueeze(0)
        self.position += 1
        return {
            "features": tensor.numpy().astype(np.float32, copy=False),
        }

    def rewind(self) -> None:
        self.position = 0

    @property
    def coverage(self) -> dict[str, Any]:
        return {
            "window_count": len(self.rows),
            "unique_window_count": len({row["window_id"] for row in self.rows}),
            "classes": sorted({row["class"] for row in self.rows}),
            "vessel_group_count": len({row["vessel_group"] for row in self.rows}),
            "split": "train",
        }


def strip_intermediate_value_info(input_path: Path, output_path: Path) -> None:
    """Remove stale intermediates before ORT rewrites Gemm into MatMul."""

    model = onnx.load(input_path)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)


def create_quantized_models(
    quantization_source: Path,
    dynamic_output: Path,
    static_output: Path,
    calibration_reader: TrainingCalibrationReader,
    config: OnnxQuantizationConfig,
) -> None:
    quantize_dynamic(
        quantization_source,
        dynamic_output,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(config.dynamic_operators),
        use_external_data_format=False,
    )
    calibration_reader.rewind()
    quantize_static(
        quantization_source,
        static_output,
        calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=config.static_per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=list(config.static_operators),
        use_external_data_format=False,
    )


def evaluate_onnx_session(
    session,
    dataset,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"Unknown ONNX variant: {variant}")
    rows = []
    logits_rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = item["features"].unsqueeze(0).numpy().astype(np.float32, copy=False)
        logits = session.run(["logits"], {"features": batch})[0][0]
        if logits.shape != (4,) or not np.isfinite(logits).all():
            raise ValueError(f"{variant} produced invalid logits.")
        logits_rows.append(logits)
        predicted_index = int(np.argmax(logits))
        rows.append(
            {
                "variant": variant,
                "window_id": item["window_id"],
                "source_file": item["source_file"],
                "vessel_group": item["vessel_group"],
                "split": item["split"],
                "class": item["class"],
                "label_index": int(item["label"]),
                "predicted_label_index": predicted_index,
                "predicted_class": ORDERED_CLASSES[predicted_index],
                **{
                    f"logit_{class_name}": float(logits[class_index])
                    for class_index, class_name in enumerate(ORDERED_CLASSES)
                },
            }
        )
    output = pd.DataFrame(rows)
    metrics = classification_metrics(
        output["label_index"].to_numpy(dtype=np.int64),
        output["predicted_label_index"].to_numpy(dtype=np.int64),
    )
    return output, metrics, np.stack(logits_rows)


def comparison_with_fp32(
    fp32_predictions: pd.DataFrame,
    fp32_logits: np.ndarray,
    variant_predictions: pd.DataFrame,
    variant_logits: np.ndarray,
) -> dict[str, Any]:
    if list(fp32_predictions["window_id"]) != list(variant_predictions["window_id"]):
        raise ValueError("Variant and FP32 predictions are not aligned.")
    absolute_error = np.abs(variant_logits - fp32_logits)
    agreement = (
        fp32_predictions["predicted_label_index"].to_numpy()
        == variant_predictions["predicted_label_index"].to_numpy()
    )
    return {
        "window_count": len(agreement),
        "matching_prediction_count": int(agreement.sum()),
        "prediction_agreement": float(agreement.mean()),
        "mean_absolute_logit_error": float(absolute_error.mean()),
        "maximum_absolute_logit_error": float(absolute_error.max()),
    }


def variant_qualifies(
    fp32_metrics: dict[str, Any],
    variant_metrics: dict[str, Any],
    fp32_p99_ms: float,
    variant_p99_ms: float,
    fp32_size_bytes: int,
    variant_size_bytes: int,
    config: OnnxQuantizationConfig,
) -> tuple[bool, dict[str, Any]]:
    macro_f1_loss = fp32_metrics["macro_f1"] - variant_metrics["macro_f1"]
    recall_losses = {
        class_name: (
            fp32_metrics["per_class"][class_name]["recall"]
            - variant_metrics["per_class"][class_name]["recall"]
        )
        for class_name in ORDERED_CLASSES
    }
    latency_increase_ratio = (variant_p99_ms - fp32_p99_ms) / fp32_p99_ms
    checks = {
        "macro_f1_loss": macro_f1_loss,
        "maximum_recall_loss": max(recall_losses.values()),
        "recall_loss_by_class": recall_losses,
        "p99_latency_increase_ratio": latency_increase_ratio,
        "size_reduction_bytes": fp32_size_bytes - variant_size_bytes,
        "macro_f1_pass": macro_f1_loss <= config.maximum_macro_f1_loss,
        "per_class_recall_pass": max(recall_losses.values())
        <= config.maximum_per_class_recall_loss,
        "latency_pass": latency_increase_ratio
        <= config.maximum_p99_latency_increase_ratio,
        "size_pass": variant_size_bytes < fp32_size_bytes,
    }
    qualifies = all(
        checks[key]
        for key in (
            "macro_f1_pass",
            "per_class_recall_pass",
            "latency_pass",
            "size_pass",
        )
    )
    checks["qualifies"] = qualifies
    return qualifies, checks


def select_deployment_variant(
    validation_rows: list[dict[str, Any]],
) -> str:
    expected = list(VARIANT_ORDER)
    if [row["variant"] for row in validation_rows] != expected:
        raise ValueError(f"Variant order must be {expected}.")
    candidates = [
        row
        for row in validation_rows[1:]
        if row["qualification"]["qualifies"]
    ]
    if not candidates:
        return "fp32"
    return min(
        candidates,
        key=lambda row: (
            row["timing"]["p99_ms"],
            row["size_bytes"],
            VARIANT_ORDER.index(row["variant"]),
        ),
    )["variant"]
