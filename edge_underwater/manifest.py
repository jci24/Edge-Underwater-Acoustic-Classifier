"""Deterministic source-group splits and fixed-window manifest generation."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from .preprocessing import PreprocessingConfig


LABELS = {
    "Cargo": 0,
    "Passengership": 1,
    "Tanker": 2,
    "Tug": 3,
}
SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}
MANIFEST_COLUMNS = [
    "window_id",
    "source_file",
    "class",
    "label_index",
    "vessel_name",
    "vessel_group",
    "session_identifier",
    "split",
    "start_seconds",
    "end_seconds",
    "source_sample_rate_hz",
    "config_hash",
]


def normalize_vessel_name(vessel_name: str) -> str:
    return " ".join(vessel_name.casefold().split())


def _split_counts(group_count: int) -> dict[str, int]:
    if group_count < len(SPLIT_RATIOS):
        raise ValueError("Each class needs at least three vessel groups.")

    split_names = list(SPLIT_RATIOS)
    targets = {name: group_count * SPLIT_RATIOS[name] for name in split_names}
    counts = {
        name: max(1, math.floor(targets[name]))
        for name in split_names
    }

    while sum(counts.values()) < group_count:
        name = max(split_names, key=lambda item: targets[item] - counts[item])
        counts[name] += 1

    while sum(counts.values()) > group_count:
        candidates = [name for name in split_names if counts[name] > 1]
        name = max(candidates, key=lambda item: counts[item] - targets[item])
        counts[name] -= 1

    return counts


def _stable_group_order(
    class_name: str,
    vessel_groups: list[str],
    seed: int,
) -> list[str]:
    def group_key(group_name: str) -> str:
        value = f"{seed}|{class_name}|{group_name}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    return sorted(vessel_groups, key=group_key)


def assign_vessel_splits(
    catalogue_rows: list[dict[str, str]],
    seed: int = 42,
) -> dict[str, str]:
    vessel_classes: dict[str, set[str]] = defaultdict(set)

    for row in catalogue_rows:
        vessel_group = normalize_vessel_name(row["vessel_name"])
        if not vessel_group:
            raise ValueError(f"Missing vessel name for {row['file']}.")
        vessel_classes[vessel_group].add(row["class"])

    conflicts = {
        vessel: classes
        for vessel, classes in vessel_classes.items()
        if len(classes) != 1
    }
    if conflicts:
        raise ValueError(f"Vessels map to conflicting classes: {conflicts}")

    groups_by_class: dict[str, set[str]] = defaultdict(set)
    for vessel_group, classes in vessel_classes.items():
        class_name = next(iter(classes))
        groups_by_class[class_name].add(vessel_group)

    assignments: dict[str, str] = {}
    for class_name in LABELS:
        ordered_groups = _stable_group_order(
            class_name,
            list(groups_by_class[class_name]),
            seed,
        )
        counts = _split_counts(len(ordered_groups))
        position = 0
        for split_name in SPLIT_RATIOS:
            next_position = position + counts[split_name]
            for vessel_group in ordered_groups[position:next_position]:
                assignments[vessel_group] = split_name
            position = next_position

    return assignments


def build_window_rows(
    catalogue_rows: list[dict[str, str]],
    config: PreprocessingConfig,
    seed: int = 42,
) -> list[dict[str, str | int | float]]:
    assignments = assign_vessel_splits(catalogue_rows, seed)
    window_rows: list[dict[str, str | int | float]] = []

    for recording in sorted(catalogue_rows, key=lambda row: row["file"]):
        class_name = recording["class"]
        if class_name not in LABELS:
            raise ValueError(f"Unknown class: {class_name}")

        duration_seconds = float(recording["duration_seconds"])
        full_window_count = math.floor(duration_seconds / config.window_seconds)
        vessel_group = normalize_vessel_name(recording["vessel_name"])
        source_recording = Path(recording["file"]).stem

        for window_index in range(full_window_count):
            start_seconds = window_index * config.window_seconds
            end_seconds = start_seconds + config.window_seconds
            start_milliseconds = round(start_seconds * 1_000)
            window_id = (
                f"{class_name.casefold()}-{source_recording}-"
                f"{start_milliseconds:09d}ms"
            )
            window_rows.append(
                {
                    "window_id": window_id,
                    "source_file": recording["file"],
                    "class": class_name,
                    "label_index": LABELS[class_name],
                    "vessel_name": recording["vessel_name"],
                    "vessel_group": vessel_group,
                    "session_identifier": recording["session_identifier"],
                    "split": assignments[vessel_group],
                    "start_seconds": f"{start_seconds:.3f}",
                    "end_seconds": f"{end_seconds:.3f}",
                    "source_sample_rate_hz": int(recording["sample_rate_hz"]),
                    "config_hash": config.config_hash,
                }
            )

    window_ids = [row["window_id"] for row in window_rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Window IDs must be unique.")

    return window_rows


def read_csv_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_window_manifest(
    window_rows: list[dict[str, str | int | float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=MANIFEST_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(window_rows)


def validate_split_leakage(window_rows: list[dict[str, str]]) -> None:
    splits_by_vessel: dict[str, set[str]] = defaultdict(set)
    for row in window_rows:
        splits_by_vessel[row["vessel_group"]].add(row["split"])

    leaked_vessels = {
        vessel: splits
        for vessel, splits in splits_by_vessel.items()
        if len(splits) > 1
    }
    if leaked_vessels:
        raise ValueError(f"Vessels appear in multiple splits: {leaked_vessels}")
