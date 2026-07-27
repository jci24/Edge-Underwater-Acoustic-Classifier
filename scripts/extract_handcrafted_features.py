#!/usr/bin/env python3
"""Create the deterministic Milestone 2 feature table."""

from pathlib import Path

from edge_underwater.feature_table import build_feature_table


PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def main() -> None:
    build_feature_table(
        manifest_path=PROJECT_FOLDER / "data/manifests/deepship_windows.csv",
        audio_root=PROJECT_FOLDER / "data/raw/deepship",
        preprocessing_path=PROJECT_FOLDER / "data/preprocessing/config.json",
        feature_config_path=PROJECT_FOLDER / "data/features/handcrafted_config.json",
        output_path=PROJECT_FOLDER / "data/features/deepship_handcrafted_features.csv",
        metadata_path=PROJECT_FOLDER / "data/features/extraction_metadata.json",
    )


if __name__ == "__main__":
    main()
