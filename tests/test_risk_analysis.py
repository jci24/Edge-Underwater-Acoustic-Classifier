from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

from edge_underwater.cnn_training import make_weighted_sampler
from edge_underwater.risk_analysis import (
    ImbalanceRunConfig,
    binary_metrics,
    class_vessel_sampling_weights,
    coverage_threshold,
    fit_embedding_reference,
    precision_recall_table,
    select_imbalance_strategy,
    select_label_audit_rows,
    select_threshold_at_fpr,
    selective_metrics,
    validate_label_audit,
    write_label_audit,
)


def sampling_rows():
    rows = []
    counts = {
        ("Cargo", "cargo-a"): 8,
        ("Cargo", "cargo-b"): 2,
        ("Passengership", "passenger-a"): 3,
        ("Passengership", "passenger-b"): 1,
        ("Tanker", "tanker-a"): 5,
        ("Tanker", "tanker-b"): 5,
        ("Tug", "tug-a"): 1,
        ("Tug", "tug-b"): 4,
    }
    for (class_name, vessel), count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "split": "train",
                    "class": class_name,
                    "vessel_group": vessel,
                    "window_id": f"{vessel}-{index}",
                }
            )
    return rows


def test_class_vessel_sampling_equalizes_class_and_vessel_mass():
    rows = sampling_rows()
    weights = class_vessel_sampling_weights(rows).numpy()
    table = pd.DataFrame(rows).assign(weight=weights)

    class_mass = table.groupby("class")["weight"].sum()
    vessel_mass = table.groupby(["class", "vessel_group"])["weight"].sum()

    np.testing.assert_allclose(class_mass, 0.25)
    for _, values in vessel_mass.groupby(level=0):
        np.testing.assert_allclose(values, values.iloc[0])


def test_weighted_sampler_draws_exact_seeded_epoch():
    weights = torch.arange(1, 830, dtype=torch.float64)
    first = list(make_weighted_sampler(weights, 829, seed=42))
    second = list(make_weighted_sampler(weights, 829, seed=42))

    assert len(first) == 829
    assert first == second


def test_sampling_rejects_non_training_rows():
    rows = sampling_rows()
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="training"):
        class_vessel_sampling_weights(rows)


@dataclass
class DummyResult:
    imbalance_config: ImbalanceRunConfig
    best_validation_macro_f1: float


def make_result(strategy, score):
    return DummyResult(
        ImbalanceRunConfig(
            name=strategy,
            strategy=strategy,
            normalization="per_example",
            preprocessing_config_hash="preprocessing",
            model_config_hash="model",
            training_config_hash="training",
        ),
        score,
    )


def test_imbalance_selection_uses_macro_f1_and_tie_order():
    results = [
        make_result("unweighted", 0.4),
        make_result("class_weighted", 0.4),
        make_result("class_vessel_balanced_sampling", 0.4),
    ]
    assert select_imbalance_strategy(results) is results[0]
    results[2].best_validation_macro_f1 = 0.5
    assert select_imbalance_strategy(results) is results[2]


def test_tug_threshold_maximizes_recall_under_five_percent_fpr():
    labels = np.array([1, 1] + [0] * 20)
    scores = np.array([0.8, 0.6, 0.7] + [0.1] * 19)

    threshold, validation = select_threshold_at_fpr(labels, scores, 0.05)
    test = binary_metrics(labels, scores, threshold)
    table, average_precision = precision_recall_table(labels, scores)

    assert threshold == pytest.approx(0.6)
    assert validation["recall"] == 1.0
    assert validation["false_positive_rate"] == 0.05
    assert test["true_positive"] == 2
    assert len(table) >= 2
    assert 0 <= average_precision <= 1


def test_tug_threshold_can_choose_zero_false_positive_operating_point():
    labels = np.array([1, 0, 0])
    scores = np.array([0.2, 0.9, 0.8])

    threshold, metrics = select_threshold_at_fpr(labels, scores, 0.0)

    assert threshold > scores.max()
    assert metrics["false_positive"] == 0
    assert metrics["recall"] == 0


def test_embedding_reference_is_training_only_and_finite():
    generator = np.random.default_rng(42)
    embeddings = generator.normal(size=(40, 64))
    labels = np.repeat(np.arange(4), 10)

    reference = fit_embedding_reference(embeddings, labels, split="train")
    distances = reference.distances(embeddings, labels)

    assert len(reference.mean) == 64
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()
    with pytest.raises(ValueError, match="training"):
        fit_embedding_reference(embeddings, labels, split="validation")


def test_rejection_thresholds_and_selective_metrics_handle_rejected_class():
    scores = np.arange(1, 11, dtype=float)
    probability_threshold = coverage_threshold(scores, 0.9, True)
    distance_threshold = coverage_threshold(scores, 0.9, False)
    true = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 0])
    accepted = np.array([True, True, False, False])

    metrics = selective_metrics(true, predicted, accepted)

    assert probability_threshold == 1
    assert distance_threshold == 9
    assert metrics["coverage"] == 0.5
    assert metrics["per_class_retention"]["Passengership"]["accepted"] == 0


def audit_analysis_rows():
    rows = []
    classes = ("Cargo", "Passengership", "Tanker", "Tug")
    for vessel_index in range(43):
        for window_index in range(2):
            label = vessel_index % 4
            rows.append(
                {
                    "window_id": f"window-{vessel_index:02d}-{window_index}",
                    "source_file": f"{classes[label]}/{vessel_index}.wav",
                    "class": classes[label],
                    "label_index": label,
                    "predicted_class": classes[(label + window_index) % 4],
                    "predicted_label_index": (label + window_index) % 4,
                    "vessel_group": f"vessel-{vessel_index:02d}",
                    "split": ("train", "validation", "test")[vessel_index % 3],
                    "start_seconds": float(window_index * 5),
                    "end_seconds": float(window_index * 5 + 5),
                    "maximum_probability": 0.8 - vessel_index / 100,
                    "true_class_probability": 0.7 - window_index / 2,
                    "embedding_distance": float(vessel_index + window_index),
                    "rms": 0.001 + vessel_index / 1000 + window_index / 10000,
                    "source_boundary": window_index == 0,
                }
            )
    return pd.DataFrame(rows)


def test_label_audit_selects_all_groups_and_five_unique_extremes(tmp_path):
    analysis = audit_analysis_rows()
    selected = select_label_audit_rows(analysis, embedding_outlier_threshold=35)
    output = tmp_path / "label_audit.csv"
    write_label_audit(selected, output)
    result = validate_label_audit(
        output,
        expected_vessel_groups=set(analysis["vessel_group"]),
    )

    assert len(selected) == 48
    assert selected["window_id"].is_unique
    assert selected["vessel_group"].nunique() == 43
    assert result["row_count"] == 48
    assert result["completed_count"] == 0
    with pytest.raises(FileExistsError):
        write_label_audit(selected, output)


def test_completed_label_audit_summarizes_dispositions(tmp_path):
    analysis = audit_analysis_rows()
    selected = select_label_audit_rows(analysis, embedding_outlier_threshold=35)
    output = tmp_path / "label_audit.csv"
    write_label_audit(selected, output)
    rows = pd.read_csv(output)
    rows["reviewer_confidence"] = "medium"
    rows["target_audibility"] = "partly_masked"
    rows["overlapping_sources"] = "uncertain"
    rows["category_ambiguity"] = "possible"
    rows["boundary_metadata_concern"] = "no"
    rows["environmental_site_cue_concern"] = "uncertain"
    rows["final_disposition"] = "uncertain"
    rows["reviewer_notes"] = "Vessel is audible but partially masked."
    rows.to_csv(output, index=False)

    result = validate_label_audit(
        output,
        expected_vessel_groups=set(analysis["vessel_group"]),
        require_complete=True,
    )

    assert result["complete"]
    assert result["disposition_counts"] == {"uncertain": 48}
