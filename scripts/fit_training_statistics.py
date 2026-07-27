#!/usr/bin/env python3
"""Fit per-mel-band normalization statistics from training windows only."""

from pathlib import Path

from edge_underwater.dataset import DeepShipWindowDataset
from edge_underwater.preprocessing import (
    PreprocessingConfig,
    RunningMelStatistics,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
CONFIG_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"


def main():
    if not MANIFEST_FILE.is_file() or not CONFIG_FILE.is_file():
        raise SystemExit("Build the window manifest first.")

    config = PreprocessingConfig.load(CONFIG_FILE)
    dataset = DeepShipWindowDataset(
        manifest_path=MANIFEST_FILE,
        audio_root=AUDIO_FOLDER,
        split="train",
        normalization="none",
        config=config,
    )
    accumulator = RunningMelStatistics(config)

    for index in range(len(dataset)):
        item = dataset[index]
        accumulator.update(item["features"], split=item["split"])
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(dataset)} training windows")

    statistics = accumulator.finalize()
    statistics.save(STATISTICS_FILE)
    print(f"Saved statistics from {statistics.window_count} windows")
    print(f"Frame count per mel band: {statistics.frame_count}")
    print(f"Output: {STATISTICS_FILE}")


if __name__ == "__main__":
    main()
