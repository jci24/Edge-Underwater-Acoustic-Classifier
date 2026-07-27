"""In-memory log-mel storage for deterministic CNN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DeepShipWindowDataset
from .preprocessing import (
    PreprocessingConfig,
    TrainingStatistics,
    WaveformToLogMel,
)


@dataclass
class PreloadedLogMelStore:
    """Unnormalized tensors and trace metadata loaded once from the manifest."""

    features: Tensor
    labels: Tensor
    rows: list[dict[str, Any]]
    config_hash: str

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("Stored features must have [window, channel, mel, frame].")
        if len(self.features) != len(self.labels) or len(self.features) != len(self.rows):
            raise ValueError("Stored features, labels, and rows must align.")
        if not torch.isfinite(self.features).all():
            raise ValueError("Stored features contain NaN or infinite values.")
        window_ids = [row["window_id"] for row in self.rows]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("Stored window IDs must be unique.")

    @classmethod
    def from_files(
        cls,
        manifest_path: Path,
        audio_root: Path,
        config: PreprocessingConfig,
        progress: Callable[[int, int], None] | None = None,
    ) -> "PreloadedLogMelStore":
        dataset = DeepShipWindowDataset(
            manifest_path=manifest_path,
            audio_root=audio_root,
            normalization="none",
            config=config,
        )
        tensors = []
        labels = []
        rows = []
        for index in range(len(dataset)):
            item = dataset[index]
            tensors.append(item["features"])
            labels.append(item["label"])
            rows.append(
                {
                    "window_id": item["window_id"],
                    "source_file": item["source_file"],
                    "class": dataset.rows[index]["class"],
                    "label_index": item["label"],
                    "vessel_name": dataset.rows[index]["vessel_name"],
                    "vessel_group": dataset.rows[index]["vessel_group"],
                    "session_identifier": dataset.rows[index]["session_identifier"],
                    "split": item["split"],
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                    "config_hash": dataset.rows[index]["config_hash"],
                    "rms": item["rms"],
                    "low_level": item["low_level"],
                }
            )
            if progress is not None:
                progress(index + 1, len(dataset))
        return cls(
            features=torch.stack(tensors),
            labels=torch.tensor(labels, dtype=torch.long),
            rows=rows,
            config_hash=config.config_hash,
        )

    @property
    def memory_bytes(self) -> int:
        return self.features.nelement() * self.features.element_size()

    def subset(
        self,
        split: str,
        normalization: str,
        config: PreprocessingConfig,
        statistics: TrainingStatistics,
    ) -> "NormalizedLogMelDataset":
        indexes = [
            index for index, row in enumerate(self.rows) if row["split"] == split
        ]
        if not indexes:
            raise ValueError(f"No preloaded rows found for split {split}.")
        return NormalizedLogMelDataset(
            store=self,
            indexes=indexes,
            normalization=normalization,
            config=config,
            statistics=statistics,
        )


class NormalizedLogMelDataset(Dataset):
    """Apply one supported normalization mode to preloaded log-mel tensors."""

    def __init__(
        self,
        store: PreloadedLogMelStore,
        indexes: list[int],
        normalization: str,
        config: PreprocessingConfig,
        statistics: TrainingStatistics,
    ) -> None:
        if store.config_hash != config.config_hash:
            raise ValueError("Preloaded data and preprocessing configuration differ.")
        if normalization not in {"training_stats", "per_example"}:
            raise ValueError(f"Unsupported CNN normalization: {normalization}")
        statistics.validate(config)
        self.store = store
        self.indexes = indexes
        self.normalization = normalization
        self.transform = WaveformToLogMel(config).eval()
        self.statistics = statistics

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        store_index = self.indexes[index]
        log_mel = self.store.features[store_index]
        features = self.transform.normalize(
            log_mel,
            self.normalization,
            self.statistics,
        )
        if not torch.isfinite(features).all():
            raise ValueError("Normalized CNN input contains NaN or infinite values.")
        row = self.store.rows[store_index]
        return {
            "features": features,
            "label": self.store.labels[store_index],
            "window_id": row["window_id"],
            "source_file": row["source_file"],
            "vessel_group": row["vessel_group"],
            "class": row["class"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "split": row["split"],
        }
