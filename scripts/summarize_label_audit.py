#!/usr/bin/env python3
"""Write the label-risk disposition summary and refresh the main report."""

import json
from pathlib import Path

import pandas as pd

from edge_underwater.risk_analysis import validate_label_audit
from train_imbalance_risk import write_report


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIT_FILE = PROJECT_FOLDER / "data/annotations/label_audit.csv"
REPORT_FOLDER = PROJECT_FOLDER / "reports/milestone4"


def main() -> None:
    vessel_groups = set(pd.read_csv(MANIFEST_FILE)["vessel_group"])
    result = validate_label_audit(
        AUDIT_FILE,
        expected_vessel_groups=vessel_groups,
    )
    status = "complete" if result["complete"] else "pending_manual_review"
    summary = {
        "status": status,
        **result,
        "interpretation": (
            "Disposition counts come from human review. Quantitative risk flags "
            "were used only to select audit candidates."
        ),
    }
    (REPORT_FOLDER / "label_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Label Audit Summary",
        "",
        f"- Status: `{status}`",
        f"- Rows: {result['row_count']}",
        f"- Vessel groups: {result['vessel_group_count']}",
        f"- Completed reviews: {result['completed_count']}",
        "",
        "## Dispositions",
        "",
    ]
    if result["disposition_counts"]:
        lines.extend(
            f"- `{name}`: {count}"
            for name, count in result["disposition_counts"].items()
        )
    else:
        lines.append("- Pending manual review; no dispositions are claimed.")
    lines.extend(
        [
            "",
            "Quantitative risk flags selected candidates but do not establish "
            "label correctness, target audibility, or domain shift.",
        ]
    )
    (REPORT_FOLDER / "label_audit_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    metrics_path = REPORT_FOLDER / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["label_audit"]["status"] = status
    metrics["label_audit"]["completed_count"] = result["completed_count"]
    metrics["label_audit"]["disposition_counts"] = result["disposition_counts"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_report(metrics)
    print(f"Label audit status: {status}")
    print(f"Completed reviews: {result['completed_count']}/48")


if __name__ == "__main__":
    main()
