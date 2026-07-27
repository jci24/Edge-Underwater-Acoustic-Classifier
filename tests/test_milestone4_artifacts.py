import json
from pathlib import Path

import numpy as np
import pandas as pd

from edge_underwater.cnn_evaluation import PROBABILITY_COLUMNS
from edge_underwater.risk_analysis import STRATEGY_ORDER, validate_label_audit


REPORT_FOLDER = Path("reports/milestone4")


def test_committed_imbalance_results_follow_selection_contract():
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    validation = pd.read_csv(REPORT_FOLDER / "validation_predictions.csv")
    test = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")

    assert metrics["strategy_order"] == list(STRATEGY_ORDER)
    assert metrics["test_evaluated_strategies"] == [metrics["selected_strategy"]]
    assert len(validation) == 154 * 3
    assert len(test) == 135
    assert test["window_id"].is_unique
    assert np.allclose(test[PROBABILITY_COLUMNS].sum(axis=1), 1.0)


def test_committed_rare_event_and_rejection_are_limited():
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    rare = metrics["rare_event"]
    rejection = metrics["rejection"]

    assert rare["rare_class"] == "Tug"
    assert rare["operating_point"]["validation"]["false_positive_rate"] <= 0.05
    assert rejection["target_validation_coverage"] == 0.90
    assert not rejection["unknown_detection_validated"]
    assert "not a separately validated detector" in rare["interpretation"]


def test_committed_label_audit_is_pending_and_covers_all_groups():
    manifest = pd.read_csv("data/manifests/deepship_windows.csv")
    result = validate_label_audit(
        Path("data/annotations/label_audit.csv"),
        expected_vessel_groups=set(manifest["vessel_group"]),
    )

    assert result["row_count"] == 48
    assert result["vessel_group_count"] == 43
    assert result["completed_count"] == 0
    assert not result["complete"]
