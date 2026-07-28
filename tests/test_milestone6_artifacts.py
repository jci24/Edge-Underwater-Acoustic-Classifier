import json
from pathlib import Path

import pandas as pd

from edge_underwater.onnx_quantization import (
    VARIANT_ORDER,
    OnnxQuantizationConfig,
    select_deployment_variant,
)


REPORT_FOLDER = Path("reports/milestone6")


def load_results():
    config = OnnxQuantizationConfig.load(
        REPORT_FOLDER / "quantization_config.json"
    )
    metrics = json.loads(
        (REPORT_FOLDER / "metrics.json").read_text(encoding="utf-8")
    )
    return config, metrics


def test_committed_study_has_complete_parity_calibration_and_predictions():
    config, metrics = load_results()
    predictions = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")

    assert metrics["quantization_config_hash"] == config.config_hash
    assert metrics["fp32_pytorch_parity"]["window_count"] == 135
    assert metrics["fp32_pytorch_parity"]["all_predictions_match"]
    assert (
        metrics["fp32_pytorch_parity"]["maximum_absolute_error"]
        <= config.absolute_tolerance
    )
    assert metrics["calibration"]["window_count"] == 829
    assert metrics["calibration"]["unique_window_count"] == 829
    assert metrics["calibration"]["split"] == "train"
    assert metrics["calibration"]["vessel_group_count"] == 29
    assert predictions.groupby(["split", "variant"]).size().to_dict() == {
        **{("validation", variant): 154 for variant in VARIANT_ORDER},
        **{("test", variant): 135 for variant in VARIANT_ORDER},
    }


def test_committed_study_has_complete_timings_and_validation_only_selection():
    config, metrics = load_results()
    timings = pd.read_csv(REPORT_FOLDER / "inference_timings.csv")
    comparison = metrics["comparison"]

    assert timings.groupby("variant").size().to_dict() == {
        variant: config.measured_calls for variant in VARIANT_ORDER
    }
    assert set(timings["thread_count"]) == {config.intra_op_threads}
    assert set(timings["benchmark_split"]) == {"validation"}
    assert [row["variant"] for row in comparison] == list(VARIANT_ORDER)
    assert all(
        {
            "accuracy",
            "macro_f1",
            "per_class",
            "confusion_matrix",
        }.issubset(row["validation_metrics"])
        for row in comparison
    )
    assert all(
        {
            "prediction_agreement",
            "mean_absolute_logit_error",
            "maximum_absolute_logit_error",
        }.issubset(row["validation_vs_fp32"])
        and row["size_bytes"] > 0
        and row["timing"]["call_count"] == config.measured_calls
        for row in comparison
    )
    selection_rows = [
        {
            "variant": row["variant"],
            "qualification": row["qualification"],
            "timing": row["timing"],
            "size_bytes": row["size_bytes"],
        }
        for row in comparison
    ]
    assert metrics["selection_split"] == "validation"
    assert (
        select_deployment_variant(selection_rows)
        == metrics["recommended_variant"]
    )
