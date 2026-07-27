#!/usr/bin/env python3
"""Validate Milestone 5 model export and benchmark artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pandas as pd

from edge_underwater.edge_benchmark import (
    PRIMARY_THREAD_POLICY,
    SECONDARY_THREAD_POLICY,
    EdgeBenchmarkConfig,
    sha256_file,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone5"
CONFIG_FILE = REPORT_FOLDER / "benchmark_config.json"
METRICS_FILE = REPORT_FOLDER / "metrics.json"
TIMINGS_FILE = REPORT_FOLDER / "steady_state_timings.csv"
COLD_FILE = REPORT_FOLDER / "cold_start_timings.csv"


def check_ignored(path: Path) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", str(path)],
        cwd=PROJECT_FOLDER,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Generated model must remain ignored: {path}")


def main() -> None:
    config = EdgeBenchmarkConfig.load(CONFIG_FILE)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    timings = pd.read_csv(TIMINGS_FILE)
    cold = pd.read_csv(COLD_FILE)

    if metrics["benchmark_config_hash"] != config.config_hash:
        raise RuntimeError("Metrics use a different benchmark configuration.")
    if metrics["selected_strategy"] != "class_vessel_balanced_sampling":
        raise RuntimeError("Benchmark did not use the Milestone 4 selected strategy.")
    if metrics["normalization"] != "per_example" or metrics["batch_size"] != 1:
        raise RuntimeError("Benchmark input policy changed.")

    model = metrics["model"]
    source_checkpoint = PROJECT_FOLDER / model["source_checkpoint"]
    pytorch_path = PROJECT_FOLDER / model["pytorch_path"]
    onnx_path = PROJECT_FOLDER / model["onnx_path"]
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing selected Milestone 4 checkpoint: {source_checkpoint}"
        )
    if sha256_file(source_checkpoint) != model["source_checkpoint_sha256"]:
        raise RuntimeError("Selected Milestone 4 checkpoint checksum differs.")
    for path in (pytorch_path, onnx_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing generated deployment model: {path}")
        check_ignored(path)
    if pytorch_path.stat().st_size != model["pytorch_size_bytes"]:
        raise RuntimeError("PyTorch model size differs from the report.")
    if onnx_path.stat().st_size != model["onnx_size_bytes"]:
        raise RuntimeError("ONNX model size differs from the report.")
    if sha256_file(pytorch_path) != model["pytorch_sha256"]:
        raise RuntimeError("PyTorch deployment model checksum differs.")
    if sha256_file(onnx_path) != model["onnx_sha256"]:
        raise RuntimeError("ONNX model checksum differs.")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    input_shape = [
        dimension.dim_value
        for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dimension.dim_value
        for dimension in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    if input_shape != [1, 1, 64, 155] or output_shape != [1, 4]:
        raise RuntimeError("ONNX input/output shape changed.")
    if not onnx_model.graph.initializer:
        raise RuntimeError("ONNX model has no embedded weights.")

    expected_groups = {
        ("dsp_preprocessing", "pytorch_dsp", PRIMARY_THREAD_POLICY),
        ("model_inference", "pytorch", PRIMARY_THREAD_POLICY),
        ("model_inference", "onnxruntime", PRIMARY_THREAD_POLICY),
        ("model_inference", "pytorch", SECONDARY_THREAD_POLICY),
        ("model_inference", "onnxruntime", SECONDARY_THREAD_POLICY),
        ("compute_pipeline", "pytorch", PRIMARY_THREAD_POLICY),
        ("compute_pipeline", "onnxruntime", PRIMARY_THREAD_POLICY),
        ("full_product_pipeline", "pytorch", PRIMARY_THREAD_POLICY),
        ("full_product_pipeline", "onnxruntime", PRIMARY_THREAD_POLICY),
    }
    observed_groups = set(
        timings[["operation", "runtime", "thread_policy"]]
        .itertuples(index=False, name=None)
    )
    if observed_groups != expected_groups:
        raise RuntimeError("Steady-state benchmark groups are incomplete.")
    counts = timings.groupby(
        ["operation", "runtime", "thread_policy"]
    ).size()
    if len(timings) != 9_000 or not (counts == config.measured_calls).all():
        raise RuntimeError("Every steady-state group must contain 1,000 samples.")
    if (timings["duration_ns"] <= 0).any():
        raise RuntimeError("Timing samples must be positive.")
    preprocessing = timings.loc[timings["operation"] == "dsp_preprocessing"]
    full_pipeline = timings.loc[
        timings["operation"] == "full_product_pipeline"
    ]
    if preprocessing["includes_decoding"].any():
        raise RuntimeError("DSP preprocessing must exclude decoding.")
    if not full_pipeline["includes_decoding"].all():
        raise RuntimeError("Full product pipeline must include decoding.")

    if len(cold) != 40 or cold["process_id"].nunique() != 40:
        raise RuntimeError("Cold starts must use 40 fresh processes.")
    if cold.groupby("runtime").size().to_dict() != {
        "onnxruntime": 20,
        "pytorch": 20,
    }:
        raise RuntimeError("Cold-start runtime coverage is incomplete.")
    memory_columns = [
        "baseline_rss_bytes",
        "rss_after_load_bytes",
        "rss_after_inference_bytes",
        "rss_after_full_pipeline_bytes",
        "peak_rss_bytes",
    ]
    if not np.isfinite(cold[memory_columns]).all().all():
        raise RuntimeError("Memory values must be finite.")
    if (cold[memory_columns] <= 0).any().any():
        raise RuntimeError("Memory values must be positive bytes.")

    parity = metrics["parity"]
    if parity["window_count"] != 135 or not parity["all_predictions_match"]:
        raise RuntimeError("ONNX parity does not cover all 135 test windows.")
    if parity["maximum_absolute_error"] > parity["absolute_tolerance"]:
        raise RuntimeError("ONNX/PyTorch logit error exceeds tolerance.")
    if metrics["model"]["parameter_count"] != 23_668:
        raise RuntimeError("Model parameter count changed.")
    if metrics["computation"]["multiply_accumulates"] != 23_989_504:
        raise RuntimeError("MAC count changed.")
    if not all(target["passed"] for target in metrics["targets"].values()):
        raise RuntimeError("At least one Milestone 5 engineering target failed.")

    environment = metrics["environment"]
    required_versions = {
        "numpy",
        "torch",
        "torchaudio",
        "torchcodec",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "psutil",
    }
    if set(environment["versions"]) != required_versions:
        raise RuntimeError("Runtime version evidence is incomplete.")
    if environment["onnxruntime"]["providers"] != ["CPUExecutionProvider"]:
        raise RuntimeError("ONNX benchmark did not use only the CPU provider.")
    if environment["onnxruntime"]["primary_intra_op_threads"] != 1:
        raise RuntimeError("Primary ONNX benchmark is not single threaded.")
    if environment["pytorch"]["primary_intra_op_threads"] != 1:
        raise RuntimeError("Primary PyTorch benchmark is not single threaded.")

    print("Validated static ONNX model and embedded weights")
    print("Validated parity across all 135 test windows")
    print("Validated nine 1,000-call steady-state groups")
    print("Validated 40 isolated cold starts and RSS evidence")
    print("Validated all three Milestone 5 engineering targets")


if __name__ == "__main__":
    main()
