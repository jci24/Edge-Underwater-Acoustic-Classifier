#!/usr/bin/env python3
"""Run one isolated cold-start and memory measurement."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.dataset import DeepShipWindowDataset
from edge_underwater.edge_benchmark import (
    create_onnx_session,
    load_deployment_model,
)
from edge_underwater.preprocessing import PreprocessingConfig


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith(("linux", "freebsd")):
        return value * 1_024
    return value


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("pytorch", "onnxruntime"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--parent-start-ns", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    startup_import_ms = (
        time.perf_counter_ns() - arguments.parent_start_ns
    ) / 1_000_000
    if arguments.threads != 1:
        raise ValueError("Cold-start acceptance measurements require one thread.")
    torch.set_num_threads(arguments.threads)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss

    input_tensor = torch.zeros(1, 1, 64, 155, dtype=torch.float32)
    load_started = time.perf_counter_ns()
    if arguments.runtime == "pytorch":
        model_config = SmallCnnConfig.load(arguments.model_config)
        runtime = load_deployment_model(arguments.model, model_config)
    else:
        runtime = create_onnx_session(arguments.onnx_model, threads=arguments.threads)
    load_ms = elapsed_ms(load_started)
    rss_after_load = process.memory_info().rss

    inference_started = time.perf_counter_ns()
    if arguments.runtime == "pytorch":
        with torch.inference_mode():
            runtime(input_tensor)
    else:
        runtime.run(
            ["logits"],
            {"features": input_tensor.numpy().astype(np.float32, copy=False)},
        )
    first_inference_ms = elapsed_ms(inference_started)
    rss_after_inference = process.memory_info().rss

    preprocessing = PreprocessingConfig.load(arguments.preprocessing_config)
    dataset = DeepShipWindowDataset(
        manifest_path=arguments.manifest,
        audio_root=arguments.audio_root,
        split="test",
        normalization="per_example",
        config=preprocessing,
    )
    full_started = time.perf_counter_ns()
    item = dataset[0]
    features = item["features"].unsqueeze(0)
    if arguments.runtime == "pytorch":
        with torch.inference_mode():
            runtime(features)
    else:
        runtime.run(
            ["logits"],
            {"features": features.numpy().astype(np.float32, copy=False)},
        )
    first_full_pipeline_ms = elapsed_ms(full_started)
    rss_after_full_pipeline = process.memory_info().rss

    result = {
        "runtime": arguments.runtime,
        "process_id": process.pid,
        "startup_import_ms": startup_import_ms,
        "load_ms": load_ms,
        "first_inference_ms": first_inference_ms,
        "first_full_pipeline_ms": first_full_pipeline_ms,
        "baseline_rss_bytes": baseline_rss,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_inference_bytes": rss_after_inference,
        "rss_after_full_pipeline_bytes": rss_after_full_pipeline,
        "load_rss_increase_bytes": max(0, rss_after_load - baseline_rss),
        "full_pipeline_rss_increase_bytes": max(
            0,
            rss_after_full_pipeline - baseline_rss,
        ),
        "peak_rss_bytes": peak_rss_bytes(),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
