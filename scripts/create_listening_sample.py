#!/usr/bin/env python3
"""Create a balanced CSV template for manual listening notes."""

from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
CATALOGUE_FILE = PROJECT_FOLDER / "data/catalogues/deepship_recordings.csv"
OUTPUT_FILE = PROJECT_FOLDER / "data/annotations/deepship_listening_annotations.csv"
EXAMPLES_PER_CLASS = 3
RANDOM_SEED = 42


def main():
    if not CATALOGUE_FILE.is_file():
        raise SystemExit("Catalogue not found. Run catalogue_deepship.py first.")

    if OUTPUT_FILE.exists():
        raise SystemExit(
            f"{OUTPUT_FILE} already exists. Move it before creating a new sample."
        )

    recordings = pd.read_csv(CATALOGUE_FILE)
    required_columns = {"file", "class", "duration_seconds", "vessel_name"}
    missing_columns = required_columns - set(recordings.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise SystemExit(f"Catalogue is missing columns: {missing_names}")

    class_sizes = recordings.groupby("class").size()
    small_classes = class_sizes[class_sizes < EXAMPLES_PER_CLASS]
    if not small_classes.empty:
        raise SystemExit("A class has too few recordings for the requested sample.")

    listening_sample = recordings.groupby("class", group_keys=False).sample(
        n=EXAMPLES_PER_CLASS,
        random_state=RANDOM_SEED,
    )

    listening_sample["noise_present"] = ""
    listening_sample["noise_type"] = ""
    listening_sample["vessel_audibility"] = ""
    listening_sample["ambiguity"] = ""
    listening_sample["confidence"] = ""
    listening_sample["notes"] = ""

    output_columns = [
        "file",
        "class",
        "duration_seconds",
        "vessel_name",
        "noise_present",
        "noise_type",
        "vessel_audibility",
        "ambiguity",
        "confidence",
        "notes",
    ]

    listening_sample = listening_sample.sort_values(["class", "file"])
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    listening_sample[output_columns].to_csv(OUTPUT_FILE, index=False)

    print(listening_sample.groupby("class").size())
    print(f"Saved annotation template to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
