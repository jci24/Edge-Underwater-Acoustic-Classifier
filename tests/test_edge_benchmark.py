from pathlib import Path

import pytest
import torch

from edge_underwater.cnn import SmallCnn
from edge_underwater.edge_benchmark import (
    EdgeBenchmarkConfig,
    count_conv_linear_macs,
    measure_calls,
    summarize_durations,
)


def benchmark_config() -> EdgeBenchmarkConfig:
    return EdgeBenchmarkConfig(
        source_checkpoint="artifacts/models/imbalance/selected.pt",
        source_checkpoint_sha256="a" * 64,
        preprocessing_config_hash="preprocessing",
        model_config_hash="model",
        training_config_hash="training",
        selected_run_config_hash="run",
    )


def test_benchmark_config_is_deterministic_and_requires_minimum_calls(tmp_path: Path):
    first = benchmark_config()
    second = benchmark_config()
    assert first.config_hash == second.config_hash

    output = tmp_path / "config.json"
    first.save(output)
    assert EdgeBenchmarkConfig.load(output) == first

    with pytest.raises(ValueError, match="at least 500"):
        EdgeBenchmarkConfig(
            source_checkpoint="checkpoint.pt",
            source_checkpoint_sha256="a" * 64,
            preprocessing_config_hash="preprocessing",
            model_config_hash="model",
            training_config_hash="training",
            selected_run_config_hash="run",
            measured_calls=499,
        )


def test_duration_summary_includes_p99_and_warmups_are_not_measured():
    calls = []

    def call(index: int) -> None:
        calls.append(index)

    durations = measure_calls(call, warmup_calls=3, measured_calls=5)
    assert len(calls) == 8
    assert len(durations) == 5
    summary = summarize_durations([1_000_000, 2_000_000, 3_000_000])
    assert summary["call_count"] == 3
    assert summary["median_ms"] == 2.0
    assert summary["p99_ms"] == pytest.approx(2.98)


def test_small_cnn_macs_follow_documented_conv_linear_convention():
    model = SmallCnn().eval()
    result = count_conv_linear_macs(
        model,
        torch.zeros(1, 1, 64, 155),
    )

    assert model.parameter_count == 23_668
    assert result["multiply_accumulates"] == 23_989_504
    assert result["approximate_flops"] == 47_979_008
    assert set(key for key in result if key.startswith("layer:")) == {
        "layer:blocks.0.0",
        "layer:blocks.1.0",
        "layer:blocks.2.0",
        "layer:head.2",
    }


def test_duration_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one"):
        summarize_durations([])
