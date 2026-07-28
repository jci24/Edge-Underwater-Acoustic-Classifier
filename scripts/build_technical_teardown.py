#!/usr/bin/env python3
"""Build the deterministic two-page Milestone 7 PDF."""

from pathlib import Path

from edge_underwater.technical_teardown import build_technical_teardown


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
OUTPUT_FILE = (
    PROJECT_FOLDER
    / "output/pdf/edge_underwater_classifier_technical_teardown.pdf"
)


if __name__ == "__main__":
    output = build_technical_teardown(PROJECT_FOLDER, OUTPUT_FILE)
    print(f"Wrote {output}")
