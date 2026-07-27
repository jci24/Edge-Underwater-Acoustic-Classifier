"""Imbalance, rare-event, rejection, and label-risk analysis helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
from torch import Tensor

from .baseline import ORDERED_CLASSES, classification_metrics


ImbalanceStrategy = Literal[
    "unweighted",
    "class_weighted",
    "class_vessel_balanced_sampling",
]
STRATEGY_ORDER: tuple[ImbalanceStrategy, ...] = (
    "unweighted",
    "class_weighted",
    "class_vessel_balanced_sampling",
)


@dataclass(frozen=True)
class ImbalanceRunConfig:
    name: str
    strategy: ImbalanceStrategy
    normalization: str
    preprocessing_config_hash: str
    model_config_hash: str
    training_config_hash: str

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGY_ORDER:
            raise ValueError(f"Unknown imbalance strategy: {self.strategy}")
        if self.normalization != "per_example":
            raise ValueError("Milestone 4 fixes per-example normalization.")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ImbalanceExperimentResult:
    imbalance_config: ImbalanceRunConfig
    training_result: Any
    validation_metrics: dict[str, Any]

    @property
    def best_validation_macro_f1(self) -> float:
        return float(self.training_result.best_validation_macro_f1)


def class_vessel_sampling_weights(rows: list[dict[str, Any]]) -> Tensor:
    """Equalize total class mass and vessel mass within each class."""

    if not rows or {str(row["split"]) for row in rows} != {"train"}:
        raise ValueError("Sampling weights may use training rows only.")
    classes = sorted({str(row["class"]) for row in rows})
    if classes != sorted(ORDERED_CLASSES):
        raise ValueError("Training rows must contain all four classes.")
    vessel_counts: Counter[tuple[str, str]] = Counter(
        (str(row["class"]), str(row["vessel_group"])) for row in rows
    )
    vessels_by_class: dict[str, set[str]] = {
        class_name: {
            str(row["vessel_group"])
            for row in rows
            if str(row["class"]) == class_name
        }
        for class_name in classes
    }
    values = []
    for row in rows:
        class_name = str(row["class"])
        vessel = str(row["vessel_group"])
        values.append(
            1.0
            / (
                len(classes)
                * len(vessels_by_class[class_name])
                * vessel_counts[(class_name, vessel)]
            )
        )
    return torch.tensor(values, dtype=torch.float64)


def select_imbalance_strategy(results: list[Any]) -> Any:
    if [result.imbalance_config.strategy for result in results] != list(
        STRATEGY_ORDER
    ):
        raise ValueError("Imbalance runs are not in the required order.")
    best = results[0]
    for candidate in results[1:]:
        if candidate.best_validation_macro_f1 > best.best_validation_macro_f1:
            best = candidate
    return best


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    positive_count = true_positive + false_negative
    negative_count = false_positive + true_negative
    recall = true_positive / positive_count if positive_count else None
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    false_positive_rate = false_positive / negative_count if negative_count else None
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_support": positive_count,
        "negative_support": negative_count,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
    }


def select_threshold_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_fpr: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or (~labels).sum() == 0:
        raise ValueError("Threshold selection needs positive and negative examples.")
    candidates = [
        float(np.nextafter(scores.max(), math.inf)),
        *sorted({float(value) for value in scores}, reverse=True),
    ]
    feasible = []
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, threshold)
        if metrics["false_positive_rate"] <= maximum_fpr:
            feasible.append(metrics)
    best = sorted(
        feasible,
        key=lambda item: (
            -(item["recall"] if item["recall"] is not None else -1),
            item["threshold"],
        ),
    )[0]
    return float(best["threshold"]), best


def precision_recall_table(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    threshold_values = [*thresholds.tolist(), float("nan")]
    table = pd.DataFrame(
        {
            "threshold": threshold_values,
            "precision": precision,
            "recall": recall,
        }
    )
    return table, float(average_precision_score(labels, scores))


@dataclass(frozen=True)
class EmbeddingReference:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    centroids: tuple[tuple[float, ...], ...]

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean)
        standard_deviation = np.asarray(self.standard_deviation)
        return (np.asarray(embeddings) - mean) / standard_deviation

    def distances(
        self,
        embeddings: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> np.ndarray:
        standardized = self.transform(embeddings)
        centroids = np.asarray(self.centroids)
        selected = centroids[np.asarray(predicted_labels, dtype=int)]
        return np.linalg.norm(standardized - selected, axis=1)


def fit_embedding_reference(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: str,
    epsilon: float = 1e-6,
) -> EmbeddingReference:
    if split != "train":
        raise ValueError("Embedding reference may use training rows only.")
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError("Expected 64-dimensional embeddings.")
    mean = embeddings.mean(axis=0)
    standard_deviation = embeddings.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, epsilon)
    standardized = (embeddings - mean) / standard_deviation
    centroids = []
    for label in range(4):
        selected = standardized[labels == label]
        if len(selected) == 0:
            raise ValueError(f"Training embeddings are missing class {label}.")
        centroids.append(selected.mean(axis=0))
    return EmbeddingReference(
        mean=tuple(mean.tolist()),
        standard_deviation=tuple(standard_deviation.tolist()),
        centroids=tuple(tuple(row.tolist()) for row in centroids),
    )


def coverage_threshold(
    scores: np.ndarray,
    target_coverage: float,
    higher_is_accepted: bool,
) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("Target coverage must be in (0, 1].")
    scores = np.asarray(scores, dtype=float)
    quantile = 1 - target_coverage if higher_is_accepted else target_coverage
    return float(np.quantile(scores, quantile, method="inverted_cdf"))


def selective_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    true_labels = np.asarray(true_labels, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    accepted = np.asarray(accepted, dtype=bool)
    accepted_count = int(accepted.sum())
    total_count = len(accepted)
    per_class_retention = {}
    for label, class_name in enumerate(ORDERED_CLASSES):
        class_rows = true_labels == label
        per_class_retention[class_name] = {
            "accepted": int(np.sum(accepted & class_rows)),
            "total": int(class_rows.sum()),
            "retention": (
                float(np.sum(accepted & class_rows) / class_rows.sum())
                if class_rows.sum()
                else None
            ),
        }
    if accepted_count:
        classification = classification_metrics(
            true_labels[accepted],
            predicted_labels[accepted],
        )
    else:
        classification = {
            "macro_f1": None,
            "accuracy": None,
            "per_class": {},
            "confusion_matrix": [[0] * 4 for _ in range(4)],
        }
    return {
        "coverage": accepted_count / total_count if total_count else None,
        "accepted_count": accepted_count,
        "rejected_count": total_count - accepted_count,
        "classification_on_accepted": classification,
        "per_class_retention": per_class_retention,
    }


AUDIT_REVIEW_VALUES = {
    "reviewer_confidence": {"high", "medium", "low"},
    "target_audibility": {"clear", "partly_masked", "weak", "not_obvious"},
    "overlapping_sources": {"yes", "no", "uncertain"},
    "category_ambiguity": {"none", "possible", "class_unclear", "unknown"},
    "boundary_metadata_concern": {"yes", "no", "uncertain"},
    "environmental_site_cue_concern": {"yes", "no", "uncertain"},
    "final_disposition": {
        "model_error",
        "dataset_ambiguity",
        "domain_shift",
        "mixed",
        "uncertain",
    },
}


def select_label_audit_rows(
    analysis_rows: pd.DataFrame,
    embedding_outlier_threshold: float,
) -> pd.DataFrame:
    rows = analysis_rows.copy()
    required = {
        "window_id",
        "vessel_group",
        "class",
        "predicted_class",
        "maximum_probability",
        "true_class_probability",
        "embedding_distance",
        "rms",
        "source_boundary",
    }
    if not required <= set(rows):
        raise ValueError(f"Audit analysis is missing columns: {required - set(rows)}")
    rows["model_disagreement"] = rows["class"] != rows["predicted_class"]
    rows["embedding_outlier"] = (
        rows["embedding_distance"] > embedding_outlier_threshold
    )
    low_rms_threshold = rows.groupby("class")["rms"].transform(
        lambda values: values.quantile(0.10)
    )
    rows["low_rms"] = rows["rms"] <= low_rms_threshold
    rows = rows.sort_values(
        [
            "model_disagreement",
            "embedding_outlier",
            "low_rms",
            "source_boundary",
            "true_class_probability",
            "window_id",
        ],
        ascending=[False, False, False, False, True, True],
    )

    extra_indexes: list[int] = []
    extra_groups: set[str] = set()
    reasons: dict[int, list[str]] = {}

    def reserve_extreme(reason: str, candidates: pd.DataFrame) -> None:
        for index in candidates.index:
            index = int(index)
            vessel_group = str(rows.loc[index, "vessel_group"])
            if index not in extra_indexes and vessel_group not in extra_groups:
                extra_indexes.append(index)
                extra_groups.add(vessel_group)
                reasons[index] = [reason]
                return
        raise ValueError(f"No unique audit candidate remained for {reason}.")

    reserve_extreme("lowest_rms", rows.sort_values(["rms", "window_id"]))
    reserve_extreme(
        "highest_embedding_distance",
        rows.sort_values(["embedding_distance", "window_id"], ascending=[False, True]),
    )
    disagreements = rows.loc[rows["model_disagreement"]]
    reserve_extreme(
        "highest_confidence_disagreement",
        disagreements.sort_values(
            ["maximum_probability", "window_id"],
            ascending=[False, True],
        ),
    )
    reserve_extreme(
        "lowest_true_class_probability",
        rows.sort_values(["true_class_probability", "window_id"]),
    )
    boundaries = rows.loc[rows["source_boundary"]]
    reserve_extreme(
        "source_boundary",
        boundaries.sort_values(["true_class_probability", "window_id"]),
    )
    selected: list[int] = []
    for _, group_rows in rows.groupby("vessel_group", sort=True):
        available = [int(index) for index in group_rows.index if index not in extra_indexes]
        if not available:
            raise ValueError("A vessel group has no audit row beyond reserved extremes.")
        index = available[0]
        selected.append(index)
        reasons[index] = ["vessel_group_coverage"]
    selected.extend(extra_indexes)
    output = rows.loc[selected].copy()
    output["selection_reasons"] = [
        ";".join(
            [
                *reasons[int(index)],
                *(["model_disagreement"] if row["model_disagreement"] else []),
                *(["embedding_outlier"] if row["embedding_outlier"] else []),
                *(["low_rms"] if row["low_rms"] else []),
                *(["source_boundary"] if row["source_boundary"] else []),
            ]
        )
        for index, row in output.iterrows()
    ]
    return output.reset_index(drop=True)


def write_label_audit(
    selected: pd.DataFrame,
    output_path: Path,
    audio_root_relative: str = "data/raw/deepship",
) -> None:
    if output_path.exists():
        raise FileExistsError(f"{output_path} may contain manual review notes.")
    review_columns = [*AUDIT_REVIEW_VALUES, "reviewer_notes"]
    rows = []
    for _, row in selected.iterrows():
        output = row.to_dict()
        audio_path = f"{audio_root_relative}/{row['source_file']}"
        output["playback_command"] = (
            f"ffplay -nodisp -autoexit -ss {float(row['start_seconds']):.3f} "
            f"-t 5 {shlex.quote(audio_path)}"
        )
        for column in review_columns:
            output[column] = ""
        rows.append(output)
    columns = [*selected.columns, "playback_command", *review_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def validate_label_audit(
    input_path: Path,
    expected_vessel_groups: set[str],
    require_complete: bool = False,
) -> dict[str, Any]:
    with input_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if len(rows) != 48:
        raise ValueError("Label audit must contain exactly 48 rows.")
    window_ids = [row["window_id"] for row in rows]
    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Label audit window IDs must be unique.")
    represented = {row["vessel_group"] for row in rows}
    if not expected_vessel_groups <= represented:
        raise ValueError("Label audit does not cover every vessel group.")
    completed = 0
    dispositions: Counter[str] = Counter()
    review_columns = list(AUDIT_REVIEW_VALUES)
    for row in rows:
        values = [row[column] for column in review_columns]
        if not any(values):
            if require_complete:
                raise ValueError(f"Audit is incomplete for {row['window_id']}.")
            continue
        if not all(values) or not row["reviewer_notes"].strip():
            raise ValueError(f"Audit is partially completed for {row['window_id']}.")
        for column, allowed in AUDIT_REVIEW_VALUES.items():
            if row[column] not in allowed:
                raise ValueError(f"Invalid {column} for {row['window_id']}.")
        completed += 1
        dispositions[row["final_disposition"]] += 1
    return {
        "row_count": len(rows),
        "vessel_group_count": len(represented),
        "completed_count": completed,
        "complete": completed == len(rows),
        "disposition_counts": dict(sorted(dispositions.items())),
    }
