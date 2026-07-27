#!/usr/bin/env python3
"""Plot file, duration, and source-group totals for each DeepShip class."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
CATALOGUE_FILE = PROJECT_FOLDER / "data/catalogues/deepship_recordings.csv"
OUTPUT_FILE = PROJECT_FOLDER / "data/plots/deepship_class_distribution.png"


def main():
    if not CATALOGUE_FILE.is_file():
        raise SystemExit("Catalogue not found. Run catalogue_deepship.py first.")

    recordings = pd.read_csv(CATALOGUE_FILE)

    required_columns = {"class", "duration_seconds", "vessel_name"}
    missing_columns = required_columns - set(recordings.columns)
    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise SystemExit(f"Catalogue is missing columns: {missing_names}")

    if recordings.empty:
        raise SystemExit("The catalogue is empty.")

    # A source group means one unique vessel within a class.
    class_summary = recordings.groupby("class").agg(
        files=("class", "size"),
        duration_minutes=("duration_seconds", lambda values: values.sum() / 60),
        source_groups=("vessel_name", "nunique"),
    )

    print(class_summary.round(2))

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    class_summary["files"].plot(
        kind="bar",
        ax=axes[0],
        color="steelblue",
        title="Files by class",
    )
    class_summary["duration_minutes"].plot(
        kind="bar",
        ax=axes[1],
        color="darkorange",
        title="Duration by class",
    )
    class_summary["source_groups"].plot(
        kind="bar",
        ax=axes[2],
        color="seagreen",
        title="Unique vessels by class",
    )

    axes[0].set_ylabel("Number of files")
    axes[1].set_ylabel("Duration in minutes")
    axes[2].set_ylabel("Number of unique vessels")

    for axis in axes:
        axis.set_xlabel("Ship class")
        axis.tick_params(axis="x", rotation=30)

    figure.suptitle("DeepShip public dataset class distribution")
    figure.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
