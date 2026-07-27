import json
from pathlib import Path

import numpy as np
import pandas as pd

from edge_underwater.cnn_evaluation import PROBABILITY_COLUMNS, validate_error_review


REPORT_FOLDER = Path("reports/milestone3")


def test_committed_cnn_results_have_primary_only_test_contract():
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    predictions = pd.read_csv(REPORT_FOLDER / "window_predictions.csv")

    assert metrics["test_evaluated_runs"] == [metrics["primary_run"]]
    assert len(metrics["runs"]) == 3
    assert [run["class_weighted"] for run in metrics["runs"]] == [
        False,
        False,
        True,
    ]
    assert len(predictions) == 135
    assert predictions["window_id"].is_unique
    assert set(predictions["split"]) == {"test"}
    assert np.allclose(predictions[PROBABILITY_COLUMNS].sum(axis=1), 1.0)
    assert np.asarray(
        metrics["primary_test"]["window"]["confusion_matrix"]
    ).shape == (4, 4)


def test_committed_cnn_comparison_preserves_milestone2_metrics():
    cnn = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    baseline = json.loads(Path("reports/milestone2/metrics.json").read_text())

    assert (
        cnn["baseline_comparison"]["logistic_regression"]["macro_f1"]
        == baseline["models"]["logistic_regression"]["window_test"]["macro_f1"]
    )
    assert (
        cnn["baseline_comparison"]["random_forest"]["macro_f1"]
        == baseline["models"]["random_forest"]["window_test"]["macro_f1"]
    )


def test_committed_error_review_is_a_pending_20_error_template():
    result = validate_error_review(
        Path("data/annotations/deepship_cnn_error_review.csv")
    )

    assert result == {"row_count": 20, "completed_count": 0, "complete": False}
