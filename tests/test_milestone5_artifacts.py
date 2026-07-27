import json
from pathlib import Path

import pandas as pd

from edge_underwater.edge_benchmark import (
    PRIMARY_THREAD_POLICY,
    SECONDARY_THREAD_POLICY,
    EdgeBenchmarkConfig,
)


REPORT_FOLDER = "reports/milestone5"


def test_committed_edge_results_follow_the_benchmark_contract():
    config = EdgeBenchmarkConfig.load(
        Path(REPORT_FOLDER) / "benchmark_config.json"
    )
    metrics = json.loads(
        (Path(REPORT_FOLDER) / "metrics.json").read_text(encoding="utf-8")
    )
    timings = pd.read_csv(f"{REPORT_FOLDER}/steady_state_timings.csv")

    assert metrics["benchmark_config_hash"] == config.config_hash
    assert metrics["selected_strategy"] == "class_vessel_balanced_sampling"
    assert metrics["normalization"] == "per_example"
    assert len(timings) == 9_000
    assert set(timings["thread_policy"]) == {
        PRIMARY_THREAD_POLICY,
        SECONDARY_THREAD_POLICY,
    }
    assert (
        timings.groupby(["operation", "runtime", "thread_policy"]).size()
        == 1_000
    ).all()


def test_committed_export_parity_memory_and_targets_are_complete():
    metrics = json.loads(
        (Path(REPORT_FOLDER) / "metrics.json").read_text(encoding="utf-8")
    )
    cold = pd.read_csv(f"{REPORT_FOLDER}/cold_start_timings.csv")

    assert metrics["parity"]["window_count"] == 135
    assert metrics["parity"]["all_predictions_match"]
    assert metrics["model"]["input_shape"] == [1, 1, 64, 155]
    assert metrics["model"]["output_shape"] == [1, 4]
    assert metrics["model"]["onnx_external_data"] is False
    assert metrics["computation"]["multiply_accumulates"] == 23_989_504
    assert all(value["passed"] for value in metrics["targets"].values())
    assert len(cold) == 40
    assert cold["process_id"].nunique() == 40
    assert (cold["peak_rss_bytes"] > 0).all()
