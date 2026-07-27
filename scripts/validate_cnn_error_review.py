#!/usr/bin/env python3
"""Validate the CNN error-listening template or completed annotations."""

import argparse
from pathlib import Path

from edge_underwater.cnn_evaluation import validate_error_review


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REVIEW_FILE = PROJECT_FOLDER / "data/annotations/deepship_cnn_error_review.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any listening row is still blank.",
    )
    arguments = parser.parse_args()
    result = validate_error_review(
        REVIEW_FILE,
        require_complete=arguments.require_complete,
    )
    print(f"Review rows: {result['row_count']}")
    print(f"Completed rows: {result['completed_count']}")
    print(f"Manual listening complete: {result['complete']}")


if __name__ == "__main__":
    main()
