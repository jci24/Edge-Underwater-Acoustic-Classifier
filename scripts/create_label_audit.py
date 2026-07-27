#!/usr/bin/env python3
"""Create the non-overwriting 48-window Milestone 4 label audit."""

import json
from pathlib import Path

import pandas as pd

from edge_underwater.risk_analysis import (
    select_label_audit_rows,
    write_label_audit,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone4"
OUTPUT_FILE = PROJECT_FOLDER / "data/annotations/label_audit.csv"


def main() -> None:
    metrics = json.loads((REPORT_FOLDER / "metrics.json").read_text())
    analysis = pd.read_csv(REPORT_FOLDER / "window_risk_analysis.csv")
    selected = select_label_audit_rows(
        analysis,
        embedding_outlier_threshold=metrics["label_audit"][
            "embedding_outlier_threshold"
        ],
    )
    write_label_audit(selected, OUTPUT_FILE)
    print(
        f"Saved {len(selected)} audit rows covering "
        f"{selected['vessel_group'].nunique()} vessel groups"
    )
    print(f"Manual label audit: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
