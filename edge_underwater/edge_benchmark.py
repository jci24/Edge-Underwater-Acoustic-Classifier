"""Reusable interfaces for deterministic edge-performance measurements."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .cnn import SmallCnn, SmallCnnConfig


PRIMARY_THREAD_POLICY = "single_thread"
SECONDARY_THREAD_POLICY = "runtime_default"


@dataclass(frozen=True)
class EdgeBenchmarkConfig:
    """Values that define a reproducible Milestone 5 benchmark."""

    source_checkpoint: str
    source_checkpoint_sha256: str
    preprocessing_config_hash: str
    model_config_hash: str
    training_config_hash: str
    selected_run_config_hash: str
    normalization: str = "per_example"
    input_shape: tuple[int, int, int, int] = (1, 1, 64, 155)
    batch_size: int = 1
    warmup_calls: int = 50
    measured_calls: int = 1_000
    cold_start_runs: int = 20
    primary_threads: int = 1
    onnx_opset: int = 18
    onnx_size_target_bytes: int = 5_000_000
    inference_p99_target_ms: float = 50.0
    full_pipeline_p99_target_ms: float = 500.0

    def __post_init__(self) -> None:
        if self.normalization != "per_example":
            raise ValueError("Milestone 5 must use the selected per-example normalization.")
        if self.input_shape != (1, 1, 64, 155) or self.batch_size != 1:
            raise ValueError("Milestone 5 requires static batch-one [1,1,64,155] input.")
        if self.warmup_calls < 1 or self.measured_calls < 500:
            raise ValueError("Benchmark requires warm-up and at least 500 measured calls.")
        if self.cold_start_runs < 1 or self.primary_threads != 1:
            raise ValueError("Cold-start count must be positive and primary threads must be one.")
        if self.onnx_opset <= 0:
            raise ValueError("ONNX opset must be positive.")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, output_path: Path) -> None:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["config_hash"] = self.config_hash
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "EdgeBenchmarkConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        values = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__
        }
        values["input_shape"] = tuple(values["input_shape"])
        config = cls(**values)
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("Saved edge benchmark configuration hash does not match.")
        return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_durations(duration_ns: list[int]) -> dict[str, float | int]:
    if not duration_ns:
        raise ValueError("At least one duration is required.")
    values_ms = np.asarray(duration_ns, dtype=np.float64) / 1_000_000
    return {
        "call_count": len(duration_ns),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "total_seconds": float(values_ms.sum() / 1_000),
    }


def measure_calls(
    call: Callable[[int], Any],
    warmup_calls: int,
    measured_calls: int,
) -> list[int]:
    if warmup_calls < 0 or measured_calls < 1:
        raise ValueError("Warm-up cannot be negative and measured calls must be positive.")
    for index in range(warmup_calls):
        call(index)
    durations = []
    for index in range(measured_calls):
        started = time.perf_counter_ns()
        call(index)
        durations.append(time.perf_counter_ns() - started)
    return durations


def count_conv_linear_macs(
    model: SmallCnn,
    example: Tensor,
) -> dict[str, int]:
    """Count Conv2d and Linear MACs; omit BN, activation, and pooling work."""

    total_macs = 0
    layer_macs: dict[str, int] = {}
    module_names = {module: name for name, module in model.named_modules()}

    def count_layer(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            batch, output_channels, height, width = output.shape
            kernel_height, kernel_width = module.kernel_size
            operations_per_output = (
                module.in_channels // module.groups
            ) * kernel_height * kernel_width
            macs = (
                batch
                * output_channels
                * height
                * width
                * operations_per_output
            )
        elif isinstance(module, nn.Linear):
            output_values = output.numel()
            macs = output_values * module.in_features
        else:
            return
        layer_macs[module_names[module]] = int(macs)
        total_macs += int(macs)

    hooks = [
        module.register_forward_hook(count_layer)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        with torch.inference_mode():
            model(example)
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "multiply_accumulates": total_macs,
        "approximate_flops": total_macs * 2,
        **{f"layer:{name}": value for name, value in layer_macs.items()},
    }


def save_deployment_state_dict(model: SmallCnn, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_deployment_model(
    model_path: Path,
    model_config: SmallCnnConfig,
) -> SmallCnn:
    model = SmallCnn(model_config)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def export_onnx_model(
    model: SmallCnn,
    output_path: Path,
    example: Tensor,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        model.eval(),
        (example,),
        f=None,
        input_names=["features"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    if program is None:
        raise RuntimeError("PyTorch ONNX exporter did not return a program.")
    program.save(output_path, external_data=False)


def create_onnx_session(
    model_path: Path,
    threads: int | None,
):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        if threads <= 0:
            raise ValueError("Explicit ONNX thread count must be positive.")
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def validate_onnx_parity(
    model: SmallCnn,
    session,
    features: list[Tensor],
    absolute_tolerance: float = 1e-4,
    relative_tolerance: float = 1e-4,
) -> dict[str, Any]:
    maximum_absolute_error = 0.0
    matching_predictions = 0
    with torch.inference_mode():
        for tensor in features:
            batch = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
            pytorch_logits = model(batch).numpy()
            onnx_logits = session.run(
                ["logits"],
                {"features": batch.numpy().astype(np.float32, copy=False)},
            )[0]
            if not np.isfinite(onnx_logits).all():
                raise ValueError("ONNX produced NaN or infinite logits.")
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(np.max(np.abs(pytorch_logits - onnx_logits))),
            )
            np.testing.assert_allclose(
                onnx_logits,
                pytorch_logits,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
            matching_predictions += int(
                np.argmax(onnx_logits) == np.argmax(pytorch_logits)
            )
    return {
        "window_count": len(features),
        "maximum_absolute_error": maximum_absolute_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "matching_prediction_count": matching_predictions,
        "all_predictions_match": matching_predictions == len(features),
    }
