#!/usr/bin/env python3
"""Validate pending or completed Milestone 4 label-review annotations."""

import argparse
from pathlib import Path

import pandas as pd

from edge_underwater.risk_analysis import validate_label_audit


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIT_FILE = PROJECT_FOLDER / "data/annotations/label_audit.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless all 48 rows contain valid reviewer fields and notes.",
    )
    arguments = parser.parse_args()
    vessel_groups = set(pd.read_csv(MANIFEST_FILE)["vessel_group"])
    result = validate_label_audit(
        AUDIT_FILE,
        expected_vessel_groups=vessel_groups,
        require_complete=arguments.require_complete,
    )
    print(f"Audit rows: {result['row_count']}")
    print(f"Vessel groups represented: {result['vessel_group_count']}")
    print(f"Completed reviews: {result['completed_count']}")
    print(f"Manual audit complete: {result['complete']}")
    print(f"Disposition counts: {result['disposition_counts']}")


if __name__ == "__main__":
    main()
