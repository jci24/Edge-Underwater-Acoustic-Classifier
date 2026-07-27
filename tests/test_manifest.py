from collections import defaultdict

import pytest

from edge_underwater.manifest import (
    LABELS,
    assign_vessel_splits,
    build_window_rows,
    validate_split_leakage,
)
from edge_underwater.preprocessing import PreprocessingConfig


def make_catalogue_rows(groups_per_class=3, duration_seconds=11):
    rows = []
    for class_name in LABELS:
        for group_index in range(groups_per_class):
            rows.append(
                {
                    "file": f"{class_name}/{group_index}.wav",
                    "class": class_name,
                    "duration_seconds": str(duration_seconds),
                    "sample_rate_hz": "32000",
                    "source_recording": str(group_index),
                    "vessel_name": f"{class_name} vessel {group_index}",
                    "session_identifier": f"{class_name}-{group_index}",
                }
            )
    return rows


def test_vessel_split_is_deterministic_and_disjoint():
    rows = make_catalogue_rows(groups_per_class=5)

    first = assign_vessel_splits(rows, seed=42)
    second = assign_vessel_splits(list(reversed(rows)), seed=42)

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_manifest_drops_incomplete_final_windows():
    config = PreprocessingConfig(window_seconds=5.0)
    rows = make_catalogue_rows(duration_seconds=11)

    windows = build_window_rows(rows, config)

    assert len(windows) == len(rows) * 2
    assert {row["start_seconds"] for row in windows} == {"0.000", "5.000"}
    assert {row["end_seconds"] for row in windows} == {"5.000", "10.000"}
    assert all(row["config_hash"] == config.config_hash for row in windows)
    validate_split_leakage(windows)


def test_each_vessel_appears_in_only_one_split():
    rows = make_catalogue_rows(groups_per_class=5)
    windows = build_window_rows(rows, PreprocessingConfig())
    splits_by_vessel = defaultdict(set)

    for row in windows:
        splits_by_vessel[row["vessel_group"]].add(row["split"])

    assert all(len(splits) == 1 for splits in splits_by_vessel.values())


def test_conflicting_vessel_classes_are_rejected():
    rows = make_catalogue_rows()
    rows[3]["vessel_name"] = rows[0]["vessel_name"]

    with pytest.raises(ValueError, match="conflicting classes"):
        assign_vessel_splits(rows)


def test_public_catalogue_produces_expected_window_count():
    from pathlib import Path

    from edge_underwater.manifest import read_csv_rows

    catalogue_path = Path("data/catalogues/deepship_recordings.csv")
    rows = read_csv_rows(catalogue_path)
    windows = build_window_rows(rows, PreprocessingConfig())

    assert len(windows) == 1_118
    assert sum(row["class"] == "Cargo" for row in windows) == 459
    assert sum(row["class"] == "Passengership" for row in windows) == 221
    assert sum(row["class"] == "Tanker" for row in windows) == 320
    assert sum(row["class"] == "Tug" for row in windows) == 118
