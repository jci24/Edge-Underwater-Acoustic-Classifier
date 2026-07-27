"""CNN prediction, metrics, and listening-review helpers."""

from __future__ import annotations

import csv
import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .baseline import (
    ORDERED_CLASSES,
    classification_metrics,
    source_predictions,
)
from .cnn import SmallCnn


PROBABILITY_COLUMNS = [f"probability_{name}" for name in ORDERED_CLASSES]
NOISE_PRESENT = {"yes", "no", "uncertain"}
NOISE_TYPES = {
    "broadband",
    "tonal",
    "transient",
    "flow_or_wave",
    "biological",
    "other_vessel",
    "electrical_or_recording",
    "unknown",
    "none",
}
VESSEL_AUDIBILITY = {"clear", "partly_masked", "weak", "not_obvious"}
AMBIGUITY = {
    "none",
    "possible_other_source",
    "class_unclear",
    "multiple_sources",
    "low_signal_to_noise",
    "recording_artifact",
    "unknown",
}
CONFIDENCE = {"high", "medium", "low"}


def evaluate_cnn(
    model: SmallCnn,
    dataset: Dataset,
    batch_size: int = 32,
    include_embeddings: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(torch.float32)
            embeddings = model.extract_embedding(features)
            logits = model.head(embeddings)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("CNN probabilities contain NaN or infinite values.")
            if not np.allclose(probabilities.sum(axis=1), 1.0):
                raise ValueError("CNN probabilities must sum to one.")
            predicted = probabilities.argmax(axis=1)
            for index in range(len(predicted)):
                row = {
                    "model": "small_cnn",
                    "window_id": batch["window_id"][index],
                    "source_file": batch["source_file"][index],
                    "class": batch["class"][index],
                    "label_index": int(batch["label"][index]),
                    "vessel_group": batch["vessel_group"][index],
                    "split": batch["split"][index],
                    "start_seconds": float(batch["start_seconds"][index]),
                    "end_seconds": float(batch["end_seconds"][index]),
                    "predicted_label_index": int(predicted[index]),
                    "predicted_class": ORDERED_CLASSES[int(predicted[index])],
                }
                for class_index, class_name in enumerate(ORDERED_CLASSES):
                    row[f"probability_{class_name}"] = float(
                        probabilities[index, class_index]
                    )
                if include_embeddings:
                    for embedding_index, value in enumerate(embeddings[index]):
                        row[f"embedding_{embedding_index}"] = float(value)
                rows.append(row)
    output = pd.DataFrame(rows)
    metrics = classification_metrics(
        output["label_index"].to_numpy(dtype=np.int64),
        output["predicted_label_index"].to_numpy(dtype=np.int64),
    )
    return output, metrics


def aggregate_source_predictions(
    window_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return source_predictions(window_predictions)


def select_error_review_rows(
    predictions: pd.DataFrame,
    requested_count: int = 20,
) -> pd.DataFrame:
    errors = predictions.loc[
        predictions["label_index"] != predictions["predicted_label_index"]
    ].copy()
    if errors.empty:
        return errors
    probabilities = errors[PROBABILITY_COLUMNS].to_numpy()
    sorted_probabilities = np.sort(probabilities, axis=1)
    errors["predicted_probability"] = probabilities.max(axis=1)
    errors["confidence_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    errors = errors.sort_values(
        ["predicted_probability", "confidence_margin", "window_id"],
        ascending=[False, False, True],
    )

    selected_indexes: list[int] = []

    def add_first_per(columns: list[str]) -> None:
        seen = set()
        for index, row in errors.iterrows():
            key = tuple(row[column] for column in columns)
            if key not in seen and index not in selected_indexes:
                selected_indexes.append(index)
                seen.add(key)
            if len(selected_indexes) >= requested_count:
                return

    add_first_per(["class"])
    add_first_per(["source_file"])
    add_first_per(["class", "predicted_class"])
    for index in errors.index:
        if index not in selected_indexes:
            selected_indexes.append(index)
        if len(selected_indexes) >= requested_count:
            break
    return errors.loc[selected_indexes[:requested_count]].reset_index(drop=True)


def write_error_review(
    selected: pd.DataFrame,
    output_path: Path,
    audio_root_relative: str = "data/raw/deepship",
) -> None:
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists; it may contain manual notes."
        )
    manual_columns = [
        "playback_command",
        "noise_present",
        "noise_type",
        "vessel_audibility",
        "ambiguity",
        "confidence",
        "notes",
    ]
    output_columns = [*selected.columns, *manual_columns]
    output_rows = []
    for _, row in selected.iterrows():
        audio_path = f"{audio_root_relative}/{row['source_file']}"
        playback = (
            f"ffplay -nodisp -autoexit -ss {float(row['start_seconds']):.3f} "
            f"-t 5 {shlex.quote(audio_path)}"
        )
        output = row.to_dict()
        output["playback_command"] = playback
        output["noise_present"] = ""
        output["noise_type"] = ""
        output["vessel_audibility"] = ""
        output["ambiguity"] = ""
        output["confidence"] = ""
        output["notes"] = ""
        output_rows.append(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=output_columns).to_csv(output_path, index=False)


def validate_error_review(
    input_path: Path,
    require_complete: bool = False,
) -> dict[str, int | bool]:
    with input_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    window_ids = [row["window_id"] for row in rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Error-review window IDs must be unique.")
    if any(row["class"] == row["predicted_class"] for row in rows):
        raise ValueError("The review pack may contain only classification errors.")

    completed = 0
    for row in rows:
        controlled_values = (
            row["noise_present"],
            row["noise_type"],
            row["vessel_audibility"],
            row["ambiguity"],
            row["confidence"],
        )
        if not any(controlled_values):
            if require_complete:
                raise ValueError(f"Review is incomplete for {row['window_id']}.")
            continue
        if not all(controlled_values):
            raise ValueError(f"Review is partially completed for {row['window_id']}.")
        if row["noise_present"] not in NOISE_PRESENT:
            raise ValueError(f"Invalid noise_present for {row['window_id']}.")
        noise_types = set(row["noise_type"].split(";"))
        if not noise_types or not noise_types <= NOISE_TYPES:
            raise ValueError(f"Invalid noise_type for {row['window_id']}.")
        if row["vessel_audibility"] not in VESSEL_AUDIBILITY:
            raise ValueError(f"Invalid vessel_audibility for {row['window_id']}.")
        if row["ambiguity"] not in AMBIGUITY:
            raise ValueError(f"Invalid ambiguity for {row['window_id']}.")
        if row["confidence"] not in CONFIDENCE:
            raise ValueError(f"Invalid confidence for {row['window_id']}.")
        completed += 1
    return {
        "row_count": len(rows),
        "completed_count": completed,
        "complete": completed == len(rows),
    }
