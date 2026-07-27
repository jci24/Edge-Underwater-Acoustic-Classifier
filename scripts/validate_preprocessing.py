#!/usr/bin/env python3
"""Validate every manifest window and both normalization modes."""

from collections import Counter
from pathlib import Path

import torch

from edge_underwater.dataset import DeepShipWindowDataset
from edge_underwater.manifest import read_csv_rows, validate_split_leakage
from edge_underwater.preprocessing import PreprocessingConfig, TrainingStatistics


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
CONFIG_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"


def main():
    config = PreprocessingConfig.load(CONFIG_FILE)
    statistics = TrainingStatistics.load(STATISTICS_FILE, config)
    manifest_rows = read_csv_rows(MANIFEST_FILE)
    validate_split_leakage(manifest_rows)

    dataset = DeepShipWindowDataset(
        manifest_path=MANIFEST_FILE,
        audio_root=AUDIO_FOLDER,
        normalization="none",
        config=config,
    )

    first_features = dataset[0]["features"].clone()
    repeated_features = dataset[0]["features"]
    if not torch.equal(first_features, repeated_features):
        raise RuntimeError("Repeated evaluation did not produce an identical tensor.")

    low_level_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        log_mel = item["features"]
        if tuple(log_mel.shape) != config.output_shape:
            raise RuntimeError(f"Wrong shape for {item['window_id']}.")

        for mode in ("per_example", "training_stats"):
            normalized = dataset.transform.normalize(
                log_mel,
                mode,
                statistics=statistics,
            )
            if not torch.isfinite(normalized).all():
                raise RuntimeError(
                    f"{mode} produced invalid values for {item['window_id']}."
                )

        low_level_count += int(item["low_level"])
        if (index + 1) % 100 == 0:
            print(f"Validated {index + 1}/{len(dataset)} windows")

    split_counts = Counter(row["split"] for row in manifest_rows)
    class_counts = Counter(row["class"] for row in manifest_rows)
    print(f"Validated {len(dataset)} windows under both normalization modes")
    print(f"Splits: {dict(sorted(split_counts.items()))}")
    print(f"Classes: {dict(sorted(class_counts.items()))}")
    print(f"Low-level windows: {low_level_count}")
    print(f"Tensor shape: {config.output_shape}")


if __name__ == "__main__":
    main()
