#!/usr/bin/env python3
"""Export and benchmark the Milestone 4 selected CNN."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import psutil
import torch

from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.cnn_training import load_cnn_checkpoint
from edge_underwater.dataset import DeepShipWindowDataset, decode_audio_interval
from edge_underwater.edge_benchmark import (
    PRIMARY_THREAD_POLICY,
    SECONDARY_THREAD_POLICY,
    EdgeBenchmarkConfig,
    count_conv_linear_macs,
    create_onnx_session,
    export_onnx_model,
    measure_calls,
    save_deployment_state_dict,
    sha256_file,
    summarize_durations,
    validate_onnx_parity,
)
from edge_underwater.manifest import read_csv_rows
from edge_underwater.preprocessing import PreprocessingConfig, WaveformToLogMel


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
PREPROCESSING_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
MILESTONE4_METRICS = PROJECT_FOLDER / "reports/milestone4/metrics.json"
MILESTONE4_RUNS = PROJECT_FOLDER / "reports/milestone4/run_configs.json"
MILESTONE4_MODEL_CONFIG = PROJECT_FOLDER / "reports/milestone4/model_config.json"
MODEL_FOLDER = PROJECT_FOLDER / "artifacts/models/edge"
PYTORCH_MODEL_FILE = MODEL_FOLDER / "selected_cnn_state_dict.pt"
ONNX_MODEL_FILE = MODEL_FOLDER / "selected_cnn.onnx"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone5"
CONFIG_FILE = REPORT_FOLDER / "benchmark_config.json"
RAW_TIMINGS_FILE = REPORT_FOLDER / "steady_state_timings.csv"
COLD_START_FILE = REPORT_FOLDER / "cold_start_timings.csv"
METRICS_FILE = REPORT_FOLDER / "metrics.json"


def selected_model_details() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    metrics = json.loads(MILESTONE4_METRICS.read_text(encoding="utf-8"))
    runs = json.loads(MILESTONE4_RUNS.read_text(encoding="utf-8"))
    selected_strategy = metrics["selected_strategy"]
    selected_run = next(
        run for run in runs if run["strategy"] == selected_strategy
    )
    checkpoint = PROJECT_FOLDER / selected_run["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Selected Milestone 4 checkpoint is missing: {checkpoint}. "
            "Run python3 scripts/train_imbalance_risk.py first."
        )
    if selected_strategy != "class_vessel_balanced_sampling":
        raise RuntimeError("Milestone 4 selected model changed unexpectedly.")
    return checkpoint, metrics, selected_run


def make_config(
    checkpoint_path: Path,
    milestone4_metrics: dict[str, Any],
    selected_run: dict[str, Any],
    checkpoint: dict[str, Any],
) -> EdgeBenchmarkConfig:
    return EdgeBenchmarkConfig(
        source_checkpoint=str(checkpoint_path.relative_to(PROJECT_FOLDER)),
        source_checkpoint_sha256=sha256_file(checkpoint_path),
        preprocessing_config_hash=milestone4_metrics["preprocessing_config_hash"],
        model_config_hash=checkpoint["model_config_hash"],
        training_config_hash=checkpoint["training_config_hash"],
        selected_run_config_hash=selected_run["config_hash"],
    )


def load_test_waveforms(
    rows: list[dict[str, str]],
) -> list[tuple[torch.Tensor, int]]:
    waveforms = []
    for index, row in enumerate(rows, start=1):
        waveform, sample_rate = decode_audio_interval(
            AUDIO_FOLDER / row["source_file"],
            float(row["start_seconds"]),
            float(row["end_seconds"]),
        )
        waveforms.append((waveform, sample_rate))
        if index % 25 == 0 or index == len(rows):
            print(f"Decoded benchmark inputs {index}/{len(rows)}")
    return waveforms


def make_features(
    transform: WaveformToLogMel,
    waveforms: list[tuple[torch.Tensor, int]],
) -> list[torch.Tensor]:
    features = []
    with torch.inference_mode():
        for waveform, sample_rate in waveforms:
            tensor, _, _ = transform(
                waveform,
                sample_rate,
                normalization="per_example",
            )
            features.append(tensor)
    return features


def append_timings(
    rows: list[dict[str, Any]],
    operation: str,
    runtime: str,
    thread_policy: str,
    configured_threads: int,
    includes_decoding: bool,
    durations: list[int],
) -> None:
    for iteration, duration_ns in enumerate(durations):
        rows.append(
            {
                "operation": operation,
                "runtime": runtime,
                "thread_policy": thread_policy,
                "configured_threads": configured_threads,
                "includes_decoding": includes_decoding,
                "iteration": iteration,
                "duration_ns": duration_ns,
            }
        )


def benchmark_steady_state(
    config: EdgeBenchmarkConfig,
    model,
    primary_onnx,
    default_onnx,
    transform: WaveformToLogMel,
    waveforms: list[tuple[torch.Tensor, int]],
    features: list[torch.Tensor],
    default_torch_threads: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numpy_features = [
        tensor.unsqueeze(0).numpy().astype(np.float32, copy=False)
        for tensor in features
    ]

    torch.set_num_threads(1)
    with torch.inference_mode():
        durations = measure_calls(
            lambda index: transform(
                *waveforms[index % len(waveforms)],
                normalization="per_example",
            ),
            config.warmup_calls,
            config.measured_calls,
        )
    append_timings(
        rows,
        "dsp_preprocessing",
        "pytorch_dsp",
        PRIMARY_THREAD_POLICY,
        1,
        False,
        durations,
    )

    def torch_inference(index: int) -> None:
        model(features[index % len(features)].unsqueeze(0))

    with torch.inference_mode():
        durations = measure_calls(
            torch_inference,
            config.warmup_calls,
            config.measured_calls,
        )
    append_timings(
        rows,
        "model_inference",
        "pytorch",
        PRIMARY_THREAD_POLICY,
        1,
        False,
        durations,
    )

    def onnx_inference(index: int) -> None:
        primary_onnx.run(
            ["logits"],
            {"features": numpy_features[index % len(numpy_features)]},
        )

    durations = measure_calls(
        onnx_inference,
        config.warmup_calls,
        config.measured_calls,
    )
    append_timings(
        rows,
        "model_inference",
        "onnxruntime",
        PRIMARY_THREAD_POLICY,
        1,
        False,
        durations,
    )

    torch.set_num_threads(default_torch_threads)
    with torch.inference_mode():
        durations = measure_calls(
            torch_inference,
            config.warmup_calls,
            config.measured_calls,
        )
    append_timings(
        rows,
        "model_inference",
        "pytorch",
        SECONDARY_THREAD_POLICY,
        default_torch_threads,
        False,
        durations,
    )

    def default_onnx_inference(index: int) -> None:
        default_onnx.run(
            ["logits"],
            {"features": numpy_features[index % len(numpy_features)]},
        )

    durations = measure_calls(
        default_onnx_inference,
        config.warmup_calls,
        config.measured_calls,
    )
    append_timings(
        rows,
        "model_inference",
        "onnxruntime",
        SECONDARY_THREAD_POLICY,
        0,
        False,
        durations,
    )

    torch.set_num_threads(1)

    def torch_compute_pipeline(index: int) -> None:
        waveform, sample_rate = waveforms[index % len(waveforms)]
        tensor, _, _ = transform(
            waveform,
            sample_rate,
            normalization="per_example",
        )
        model(tensor.unsqueeze(0))

    with torch.inference_mode():
        durations = measure_calls(
            torch_compute_pipeline,
            config.warmup_calls,
            config.measured_calls,
        )
    append_timings(
        rows,
        "compute_pipeline",
        "pytorch",
        PRIMARY_THREAD_POLICY,
        1,
        False,
        durations,
    )

    def onnx_compute_pipeline(index: int) -> None:
        waveform, sample_rate = waveforms[index % len(waveforms)]
        with torch.inference_mode():
            tensor, _, _ = transform(
                waveform,
                sample_rate,
                normalization="per_example",
            )
        primary_onnx.run(
            ["logits"],
            {"features": tensor.unsqueeze(0).numpy()},
        )

    durations = measure_calls(
        onnx_compute_pipeline,
        config.warmup_calls,
        config.measured_calls,
    )
    append_timings(
        rows,
        "compute_pipeline",
        "onnxruntime",
        PRIMARY_THREAD_POLICY,
        1,
        False,
        durations,
    )

    for runtime_name in ("pytorch", "onnxruntime"):
        dataset = DeepShipWindowDataset(
            manifest_path=MANIFEST_FILE,
            audio_root=AUDIO_FOLDER,
            split="test",
            normalization="per_example",
            config=transform.config,
        )

        def full_pipeline(index: int) -> None:
            item = dataset[index % len(dataset)]
            batch = item["features"].unsqueeze(0)
            if runtime_name == "pytorch":
                with torch.inference_mode():
                    model(batch)
            else:
                primary_onnx.run(
                    ["logits"],
                    {"features": batch.numpy().astype(np.float32, copy=False)},
                )

        durations = measure_calls(
            full_pipeline,
            config.warmup_calls,
            config.measured_calls,
        )
        append_timings(
            rows,
            "full_product_pipeline",
            runtime_name,
            PRIMARY_THREAD_POLICY,
            1,
            True,
            durations,
        )

    torch.set_num_threads(default_torch_threads)
    return pd.DataFrame(rows)


def run_cold_starts(
    config: EdgeBenchmarkConfig,
) -> pd.DataFrame:
    rows = []
    worker = PROJECT_FOLDER / "scripts/edge_cold_start_worker.py"
    for runtime_name in ("pytorch", "onnxruntime"):
        for run_index in range(config.cold_start_runs):
            parent_started = time.perf_counter_ns()
            command = [
                sys.executable,
                str(worker),
                "--runtime",
                runtime_name,
                "--model",
                str(PYTORCH_MODEL_FILE),
                "--model-config",
                str(MILESTONE4_MODEL_CONFIG),
                "--onnx-model",
                str(ONNX_MODEL_FILE),
                "--manifest",
                str(MANIFEST_FILE),
                "--audio-root",
                str(AUDIO_FOLDER),
                "--preprocessing-config",
                str(PREPROCESSING_FILE),
                "--threads",
                "1",
                "--parent-start-ns",
                str(parent_started),
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_FOLDER,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            process_total_ms = (
                time.perf_counter_ns() - parent_started
            ) / 1_000_000
            output_lines = [
                line for line in completed.stdout.splitlines() if line.strip()
            ]
            result = json.loads(output_lines[-1])
            result["run_index"] = run_index
            result["process_total_ms"] = process_total_ms
            rows.append(result)
            pd.DataFrame(rows).to_csv(COLD_START_FILE, index=False)
            print(
                f"Cold starts {runtime_name}: "
                f"{run_index + 1}/{config.cold_start_runs}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summary_rows(timings: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    group_columns = [
        "operation",
        "runtime",
        "thread_policy",
        "configured_threads",
        "includes_decoding",
    ]
    for group, values in timings.groupby(group_columns, sort=False):
        summary = dict(zip(group_columns, group, strict=True))
        summary.update(summarize_durations(values["duration_ns"].tolist()))
        summaries.append(summary)
    return summaries


def cold_start_summaries(cold: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "startup_import_ms",
        "load_ms",
        "first_inference_ms",
        "first_full_pipeline_ms",
        "process_total_ms",
        "baseline_rss_bytes",
        "load_rss_increase_bytes",
        "full_pipeline_rss_increase_bytes",
        "peak_rss_bytes",
    ]
    summaries = []
    for runtime_name, values in cold.groupby("runtime", sort=False):
        result: dict[str, Any] = {
            "runtime": runtime_name,
            "run_count": len(values),
        }
        for column in columns:
            series = values[column].to_numpy(dtype=np.float64)
            result[column] = {
                "median": float(np.median(series)),
                "p95": float(np.percentile(series, 95)),
                "p99": float(np.percentile(series, 99)),
                "maximum": float(np.max(series)),
            }
        summaries.append(result)
    return summaries


def system_environment(
    default_torch_threads: int,
    primary_onnx,
    default_onnx,
) -> dict[str, Any]:
    hardware = {}
    if sys.platform == "darwin":
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            text=True,
            capture_output=True,
            check=True,
        )
        details = json.loads(completed.stdout)["SPHardwareDataType"][0]
        hardware = {
            "machine_name": details.get("machine_name", ""),
            "chip": details.get("chip_type", ""),
            "memory": details.get("physical_memory", ""),
        }
    versions = {}
    for package in (
        "numpy",
        "torch",
        "torchaudio",
        "torchcodec",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "psutil",
    ):
        versions[package] = importlib.metadata.version(package)
    return {
        "hardware": hardware,
        "architecture": platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_bytes": psutil.virtual_memory().total,
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "versions": versions,
        "pytorch": {
            "primary_intra_op_threads": 1,
            "default_intra_op_threads": default_torch_threads,
        },
        "onnxruntime": {
            "providers": primary_onnx.get_providers(),
            "primary_intra_op_threads": primary_onnx.get_session_options().intra_op_num_threads,
            "default_intra_op_threads": default_onnx.get_session_options().intra_op_num_threads,
            "execution_mode": "sequential",
            "graph_optimization": "ORT_ENABLE_ALL",
        },
    }


def save_plots(
    timing_summaries: list[dict[str, Any]],
    cold_summaries: list[dict[str, Any]],
) -> None:
    primary = [
        row
        for row in timing_summaries
        if row["thread_policy"] == PRIMARY_THREAD_POLICY
        and row["runtime"] in ("pytorch", "onnxruntime")
    ]
    labels = [
        f"{row['runtime']}\n{row['operation'].replace('_', ' ')}"
        for row in primary
    ]
    positions = np.arange(len(primary))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for offset, percentile in enumerate(("median_ms", "p95_ms", "p99_ms")):
        axis.bar(
            positions + (offset - 1) * width,
            [row[percentile] for row in primary],
            width,
            label=percentile.replace("_ms", ""),
        )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Latency (ms, log scale)")
    axis.set_yscale("log")
    axis.set_title("Single-thread batch-one edge latency")
    axis.legend()
    figure.savefig(REPORT_FOLDER / "latency_percentiles.png", dpi=160)
    plt.close(figure)

    memory_labels = [row["runtime"] for row in cold_summaries]
    baseline = [
        row["baseline_rss_bytes"]["median"] / (1024**2)
        for row in cold_summaries
    ]
    load_increase = [
        row["load_rss_increase_bytes"]["median"] / (1024**2)
        for row in cold_summaries
    ]
    peak = [
        row["peak_rss_bytes"]["maximum"] / (1024**2)
        for row in cold_summaries
    ]
    positions = np.arange(len(memory_labels))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(positions - width, baseline, width, label="baseline RSS median")
    axis.bar(positions, load_increase, width, label="load increase median")
    axis.bar(positions + width, peak, width, label="maximum peak RSS")
    axis.set_xticks(positions, memory_labels)
    axis.set_ylabel("Memory (MiB)")
    axis.set_title("Isolated-process memory")
    axis.legend()
    figure.savefig(REPORT_FOLDER / "memory_summary.png", dpi=160)
    plt.close(figure)


def find_timing(
    summaries: list[dict[str, Any]],
    operation: str,
    runtime: str,
    thread_policy: str = PRIMARY_THREAD_POLICY,
) -> dict[str, Any]:
    return next(
        row
        for row in summaries
        if row["operation"] == operation
        and row["runtime"] == runtime
        and row["thread_policy"] == thread_policy
    )


def write_report(metrics: dict[str, Any]) -> None:
    timing = metrics["steady_state"]
    pytorch_inference = find_timing(timing, "model_inference", "pytorch")
    onnx_inference = find_timing(timing, "model_inference", "onnxruntime")
    pytorch_default = find_timing(
        timing,
        "model_inference",
        "pytorch",
        SECONDARY_THREAD_POLICY,
    )
    onnx_default = find_timing(
        timing,
        "model_inference",
        "onnxruntime",
        SECONDARY_THREAD_POLICY,
    )
    preprocessing = find_timing(timing, "dsp_preprocessing", "pytorch_dsp")
    pytorch_compute = find_timing(timing, "compute_pipeline", "pytorch")
    onnx_compute = find_timing(timing, "compute_pipeline", "onnxruntime")
    pytorch_full = find_timing(timing, "full_product_pipeline", "pytorch")
    onnx_full = find_timing(timing, "full_product_pipeline", "onnxruntime")
    targets = metrics["targets"]
    cold_rows = metrics["cold_start"]
    cold_lines = []
    for row in cold_rows:
        cold_lines.append(
            f"| {row['runtime']} | "
            f"{row['startup_import_ms']['median']:.1f} | "
            f"{row['load_ms']['median']:.1f} | "
            f"{row['first_inference_ms']['median']:.3f} | "
            f"{row['first_full_pipeline_ms']['median']:.3f} | "
            f"{row['baseline_rss_bytes']['median'] / (1024**2):.1f} | "
            f"{row['load_rss_increase_bytes']['median'] / (1024**2):.1f} | "
            f"{row['full_pipeline_rss_increase_bytes']['median'] / (1024**2):.1f} | "
            f"{row['peak_rss_bytes']['maximum'] / (1024**2):.1f} |"
        )
    target_lines = [
        f"| {name} | {values['observed']:.3f} {values['unit']} | "
        f"{values['limit']:.3f} {values['unit']} | "
        f"{'PASS' if values['passed'] else 'FAIL'} |"
        for name, values in targets.items()
    ]
    report = f"""# Milestone 5 — Edge-Performance Evaluation

## Outcome

The unchanged Milestone 4 `class_vessel_balanced_sampling` CNN was exported
to ONNX and benchmarked at batch size one. Single-threaded ONNX Runtime is the
primary deployment condition; PyTorch eager is the reference.

| measurement | median ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| DSP preprocessing, decoded waveform | {preprocessing['median_ms']:.3f} | {preprocessing['p95_ms']:.3f} | {preprocessing['p99_ms']:.3f} |
| PyTorch model inference | {pytorch_inference['median_ms']:.3f} | {pytorch_inference['p95_ms']:.3f} | {pytorch_inference['p99_ms']:.3f} |
| ONNX Runtime model inference | {onnx_inference['median_ms']:.3f} | {onnx_inference['p95_ms']:.3f} | {onnx_inference['p99_ms']:.3f} |
| PyTorch compute pipeline | {pytorch_compute['median_ms']:.3f} | {pytorch_compute['p95_ms']:.3f} | {pytorch_compute['p99_ms']:.3f} |
| ONNX compute pipeline | {onnx_compute['median_ms']:.3f} | {onnx_compute['p95_ms']:.3f} | {onnx_compute['p99_ms']:.3f} |
| PyTorch full product pipeline | {pytorch_full['median_ms']:.3f} | {pytorch_full['p95_ms']:.3f} | {pytorch_full['p99_ms']:.3f} |
| ONNX full product pipeline | {onnx_full['median_ms']:.3f} | {onnx_full['p95_ms']:.3f} | {onnx_full['p99_ms']:.3f} |

The DSP measurement excludes decoding and disk access. The full product
pipeline includes requested-interval decoding from a warm file cache,
preprocessing, and inference.

Secondary default-thread model-only results:

| runtime | configured threads | median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| PyTorch | {pytorch_default['configured_threads']} | {pytorch_default['median_ms']:.3f} | {pytorch_default['p95_ms']:.3f} | {pytorch_default['p99_ms']:.3f} |
| ONNX Runtime | runtime default (`0`) | {onnx_default['median_ms']:.3f} | {onnx_default['p95_ms']:.3f} | {onnx_default['p99_ms']:.3f} |

## Engineering targets

| target | observed | limit | result |
|---|---:|---:|---|
{chr(10).join(target_lines)}

## Model and export

- Parameters: {metrics['model']['parameter_count']:,}
- PyTorch deployment state dict: {metrics['model']['pytorch_size_bytes']:,} bytes
- ONNX model: {metrics['model']['onnx_size_bytes']:,} bytes
- Conv2d/Linear MACs: {metrics['computation']['multiply_accumulates']:,}
- Approximate FLOPs: {metrics['computation']['approximate_flops']:,}
- ONNX/PyTorch parity windows: {metrics['parity']['matching_prediction_count']}/{metrics['parity']['window_count']}
- Maximum absolute logit difference: {metrics['parity']['maximum_absolute_error']:.8f}

MACs exclude batch normalization, activation, and pooling work. Approximate
FLOPs use two operations per multiply-accumulate.

## Cold start and memory

Twenty fresh processes were measured per runtime.

| runtime | startup/import median ms | load median ms | first inference median ms | first full pipeline median ms | baseline RSS median MiB | load increase median MiB | full increase median MiB | maximum peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(cold_lines)}

RSS includes the Python process and required DSP/runtime libraries. The report
also preserves post-import baselines and load/full-pipeline increases so model
cost is not confused with the whole interpreter.

## Environment

- Machine: {metrics['environment']['hardware'].get('machine_name', 'unknown')}
- CPU: {metrics['environment']['hardware'].get('chip', 'unknown')}
- Architecture: {metrics['environment']['architecture']}
- CPU cores: {metrics['environment']['physical_cpu_count']} physical,
  {metrics['environment']['logical_cpu_count']} logical
- RAM: {metrics['environment']['hardware'].get('memory', 'unknown')}
- Operating system: {metrics['environment']['operating_system']}
- Python: {metrics['environment']['python']}
- PyTorch / torchaudio / TorchCodec:
  {metrics['environment']['versions']['torch']} /
  {metrics['environment']['versions']['torchaudio']} /
  {metrics['environment']['versions']['torchcodec']}
- ONNX / ONNX Runtime / ONNX Script:
  {metrics['environment']['versions']['onnx']} /
  {metrics['environment']['versions']['onnxruntime']} /
  {metrics['environment']['versions']['onnxscript']}
- Primary threads: one PyTorch intra-op thread or one ONNX Runtime intra-op
  thread; ONNX execution is sequential with the CPU execution provider.

## Measurement limits

- Results come from the current {metrics['environment']['hardware'].get('chip', 'CPU')} development laptop, not the eventual underwater edge device.
- Default-thread results are secondary and do not replace the reproducible single-thread acceptance condition.
- Timing uses a warm model and warm file cache after 50 warm-up calls; cold-start measurements are reported separately.
- Fast execution does not repair the selected CNN's weak reused-test classification score from Milestone 4.
"""
    (REPORT_FOLDER / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint_path, milestone4_metrics, selected_run = selected_model_details()
    model, checkpoint = load_cnn_checkpoint(checkpoint_path)
    preprocessing = PreprocessingConfig.load(PREPROCESSING_FILE)
    if checkpoint["model_config_hash"] != SmallCnnConfig.load(
        MILESTONE4_MODEL_CONFIG
    ).config_hash:
        raise RuntimeError("Milestone 4 model configuration and checkpoint differ.")
    config = make_config(
        checkpoint_path,
        milestone4_metrics,
        selected_run,
        checkpoint,
    )
    config.save(CONFIG_FILE)

    example = torch.zeros(config.input_shape, dtype=torch.float32)
    save_deployment_state_dict(model, PYTORCH_MODEL_FILE)
    export_onnx_model(model, ONNX_MODEL_FILE, example, config.onnx_opset)
    onnx_model = onnx.load(ONNX_MODEL_FILE)
    onnx.checker.check_model(onnx_model)
    if len(onnx_model.graph.initializer) == 0:
        raise RuntimeError("ONNX file does not contain embedded model weights.")

    default_torch_threads = torch.get_num_threads()
    primary_onnx = create_onnx_session(ONNX_MODEL_FILE, threads=1)
    default_onnx = create_onnx_session(ONNX_MODEL_FILE, threads=None)
    test_rows = [
        row
        for row in read_csv_rows(MANIFEST_FILE)
        if row["split"] == "test"
    ]
    if len(test_rows) != 135:
        raise RuntimeError("Milestone 5 expects the unchanged 135-window test split.")
    waveforms = load_test_waveforms(test_rows)
    transform = WaveformToLogMel(preprocessing).eval()
    features = make_features(transform, waveforms)
    parity = validate_onnx_parity(model, primary_onnx, features)

    computation = count_conv_linear_macs(model, example)
    timings = benchmark_steady_state(
        config,
        model,
        primary_onnx,
        default_onnx,
        transform,
        waveforms,
        features,
        default_torch_threads,
    )
    timings.to_csv(RAW_TIMINGS_FILE, index=False)
    timing_summaries = summary_rows(timings)

    cold_starts = run_cold_starts(config)
    cold_starts.to_csv(COLD_START_FILE, index=False)
    cold_summaries = cold_start_summaries(cold_starts)

    onnx_size = ONNX_MODEL_FILE.stat().st_size
    onnx_inference = find_timing(
        timing_summaries,
        "model_inference",
        "onnxruntime",
    )
    onnx_full = find_timing(
        timing_summaries,
        "full_product_pipeline",
        "onnxruntime",
    )
    targets = {
        "onnx_file_size": {
            "observed": onnx_size,
            "limit": config.onnx_size_target_bytes,
            "unit": "bytes",
            "passed": onnx_size < config.onnx_size_target_bytes,
        },
        "onnx_single_thread_inference_p99": {
            "observed": onnx_inference["p99_ms"],
            "limit": config.inference_p99_target_ms,
            "unit": "ms",
            "passed": onnx_inference["p99_ms"] < config.inference_p99_target_ms,
        },
        "onnx_full_pipeline_p99": {
            "observed": onnx_full["p99_ms"],
            "limit": config.full_pipeline_p99_target_ms,
            "unit": "ms",
            "passed": onnx_full["p99_ms"] < config.full_pipeline_p99_target_ms,
        },
    }
    environment = system_environment(
        default_torch_threads,
        primary_onnx,
        default_onnx,
    )
    metrics = {
        "benchmark_config_hash": config.config_hash,
        "selected_strategy": milestone4_metrics["selected_strategy"],
        "normalization": config.normalization,
        "batch_size": config.batch_size,
        "model": {
            "parameter_count": model.parameter_count,
            "source_checkpoint": config.source_checkpoint,
            "source_checkpoint_sha256": config.source_checkpoint_sha256,
            "pytorch_path": str(PYTORCH_MODEL_FILE.relative_to(PROJECT_FOLDER)),
            "pytorch_size_bytes": PYTORCH_MODEL_FILE.stat().st_size,
            "pytorch_sha256": sha256_file(PYTORCH_MODEL_FILE),
            "onnx_path": str(ONNX_MODEL_FILE.relative_to(PROJECT_FOLDER)),
            "onnx_size_bytes": onnx_size,
            "onnx_sha256": sha256_file(ONNX_MODEL_FILE),
            "onnx_opset": config.onnx_opset,
            "onnx_external_data": False,
            "input_name": "features",
            "output_name": "logits",
            "input_shape": list(config.input_shape),
            "output_shape": [1, 4],
        },
        "parity": parity,
        "computation": computation,
        "steady_state": timing_summaries,
        "cold_start": cold_summaries,
        "targets": targets,
        "environment": environment,
        "measurement_boundaries": {
            "dsp_preprocessing": "Decoded waveform to per-example normalized log-mel; no decoding or disk I/O.",
            "model_inference": "Prepared [1,1,64,155] tensor to four logits.",
            "compute_pipeline": "Decoded waveform to four logits; no decoding or disk I/O.",
            "full_product_pipeline": "Warm-cache source interval decoding, preprocessing, and four logits.",
            "cold_start": "Fresh process startup/import, runtime load, first inference, and first full pipeline.",
        },
        "classification_limit": (
            "Edge performance does not repair the selected model's weak reused-test "
            "classification score from Milestone 4."
        ),
    }
    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    save_plots(timing_summaries, cold_summaries)
    write_report(metrics)
    print(f"ONNX parity: {parity['window_count']} windows")
    print(f"ONNX size: {onnx_size:,} bytes")
    print(
        "ONNX single-thread inference p99: "
        f"{onnx_inference['p99_ms']:.3f} ms"
    )
    print(f"Wrote Milestone 5 analysis to {REPORT_FOLDER}")


if __name__ == "__main__":
    main()
