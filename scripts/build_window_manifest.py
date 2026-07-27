#!/usr/bin/env python3
"""Build deterministic vessel splits and fixed-window metadata."""

from collections import Counter
from pathlib import Path

from edge_underwater.manifest import (
    build_window_rows,
    read_csv_rows,
    validate_split_leakage,
    write_window_manifest,
)
from edge_underwater.preprocessing import PreprocessingConfig


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
CATALOGUE_FILE = PROJECT_FOLDER / "data/catalogues/deepship_recordings.csv"
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
CONFIG_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"


def main():
    if not CATALOGUE_FILE.is_file():
        raise SystemExit("Catalogue not found. Run catalogue_deepship.py first.")

    config = PreprocessingConfig()
    catalogue_rows = read_csv_rows(CATALOGUE_FILE)
    window_rows = build_window_rows(catalogue_rows, config, seed=42)
    validate_split_leakage(window_rows)

    write_window_manifest(window_rows, MANIFEST_FILE)
    config.save(CONFIG_FILE)

    split_counts = Counter(row["split"] for row in window_rows)
    class_counts = Counter(row["class"] for row in window_rows)
    print(f"Saved {len(window_rows)} windows to {MANIFEST_FILE}")
    print(f"Splits: {dict(sorted(split_counts.items()))}")
    print(f"Classes: {dict(sorted(class_counts.items()))}")
    print(f"Tensor shape: {config.output_shape}")
    print(f"Config hash: {config.config_hash}")


if __name__ == "__main__":
    main()
