#!/usr/bin/env python3
"""Create a non-overwriting listening pack from primary CNN test errors."""

from pathlib import Path

import pandas as pd

from edge_underwater.cnn_evaluation import (
    select_error_review_rows,
    write_error_review,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = PROJECT_FOLDER / "reports/milestone3/window_predictions.csv"
OUTPUT_FILE = PROJECT_FOLDER / "data/annotations/deepship_cnn_error_review.csv"


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_FILE)
    selected = select_error_review_rows(predictions, requested_count=20)
    write_error_review(selected, OUTPUT_FILE)
    error_count = int(
        (predictions["label_index"] != predictions["predicted_label_index"]).sum()
    )
    print(f"Selected {len(selected)} of {error_count} CNN test errors")
    print(f"Saved manual listening pack to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
