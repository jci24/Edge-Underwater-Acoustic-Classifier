"""Evidence loading and deterministic rendering for the Milestone 7 teardown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib.colors import Color, HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ORDERED_CLASSES = ("Cargo", "Passengership", "Tanker", "Tug")
NAVY = HexColor("#102A43")
SLATE = HexColor("#486581")
TEAL = HexColor("#008C8C")
ORANGE = HexColor("#E07A35")
PALE_TEAL = HexColor("#E6F5F3")
PALE_ORANGE = HexColor("#FFF1E8")
PALE_BLUE = HexColor("#EAF1F8")
PALE_GRAY = HexColor("#F4F7FA")
MID_GRAY = HexColor("#CBD5E1")
DARK = HexColor("#172B4D")
MUTED = HexColor("#5D6B7A")


@dataclass(frozen=True)
class TeardownEvidence:
    logistic_accuracy: float
    logistic_macro_f1: float
    logistic_tug_recall: float
    logistic_model_bytes: int
    logistic_feature_median_ms: float
    logistic_inference_median_ms: float
    logistic_end_to_end_median_ms: float
    logistic_confusion: tuple[tuple[int, ...], ...]
    cnn_accuracy: float
    cnn_macro_f1: float
    cnn_tug_recall: float
    cnn_confusion: tuple[tuple[int, ...], ...]
    tug_windows: int
    tug_vessel_groups: int
    tug_threshold: float
    tug_threshold_test_recall: float
    cnn_parameters: int
    onnx_model_bytes: int
    dsp_median_ms: float
    onnx_inference_median_ms: float
    onnx_inference_p99_ms: float
    full_pipeline_median_ms: float
    full_pipeline_p99_ms: float
    onnx_full_rss_increase_bytes: float
    onnx_peak_rss_bytes: float
    deployment_variant: str
    dynamic_size_reduction_ratio: float
    dynamic_p99_change_ratio: float
    static_size_reduction_ratio: float
    static_macro_f1_loss: float
    static_maximum_recall_loss: float


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Could not read evidence file {path}: {error}") from error


def _require(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    traversed = []
    for key in keys:
        traversed.append(key)
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing evidence key: {'.'.join(traversed)}")
        value = value[key]
    return value


def _confusion_from_csv(
    path: Path,
    model_name: str | None = None,
) -> tuple[tuple[int, ...], ...]:
    if not path.exists():
        raise FileNotFoundError(f"Required confusion matrix is missing: {path}")
    table = pd.read_csv(path)
    required = {"true_class", "predicted_class", "count"}
    if not required.issubset(table.columns):
        raise ValueError(f"Confusion matrix columns are incomplete: {path}")
    if model_name is not None:
        if "model" not in table.columns:
            raise ValueError(f"Confusion matrix has no model column: {path}")
        table = table.loc[table["model"] == model_name]
    expected_pairs = {
        (true_class, predicted_class)
        for true_class in ORDERED_CLASSES
        for predicted_class in ORDERED_CLASSES
    }
    observed_pairs = set(
        table[["true_class", "predicted_class"]].itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs or len(table) != 16:
        raise ValueError(f"Confusion matrix class ordering/coverage differs: {path}")
    lookup = {
        (row.true_class, row.predicted_class): int(row.count)
        for row in table.itertuples()
    }
    matrix = tuple(
        tuple(lookup[(true_class, predicted_class)] for predicted_class in ORDERED_CLASSES)
        for true_class in ORDERED_CLASSES
    )
    if sum(sum(row) for row in matrix) != 135:
        raise ValueError(f"Confusion matrix must contain 135 test windows: {path}")
    return matrix


def _steady_state(
    metrics: dict[str, Any],
    operation: str,
    runtime: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in _require(metrics, "steady_state")
        if row["operation"] == operation
        and row["runtime"] == runtime
        and row["thread_policy"] == "single_thread"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one single-thread timing for {operation}/{runtime}."
        )
    return matches[0]


def _cold_start(metrics: dict[str, Any], runtime: str) -> dict[str, Any]:
    matches = [
        row for row in _require(metrics, "cold_start") if row["runtime"] == runtime
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one cold-start record for {runtime}.")
    return matches[0]


def load_teardown_evidence(project_root: Path) -> TeardownEvidence:
    milestone2 = _read_json(project_root / "reports/milestone2/metrics.json")
    milestone4 = _read_json(project_root / "reports/milestone4/metrics.json")
    milestone5 = _read_json(project_root / "reports/milestone5/metrics.json")
    milestone6 = _read_json(project_root / "reports/milestone6/metrics.json")

    if _require(milestone4, "selected_strategy") != (
        "class_vessel_balanced_sampling"
    ):
        raise ValueError("Milestone 4 selected model differs from the teardown.")
    if _require(milestone6, "recommended_variant") != "fp32":
        raise ValueError("Milestone 6 deployment recommendation is not FP32.")
    if _require(milestone5, "selected_strategy") != (
        "class_vessel_balanced_sampling"
    ):
        raise ValueError("Milestone 5 benchmark used a different CNN.")

    logistic = _require(milestone2, "models", "logistic_regression")
    logistic_test = _require(logistic, "window_test")
    cnn_test = _require(milestone4, "selected_test", "window")
    rare_event = _require(milestone4, "rare_event", "operating_point")
    model = _require(milestone5, "model")
    dsp = _steady_state(milestone5, "dsp_preprocessing", "pytorch_dsp")
    inference = _steady_state(milestone5, "model_inference", "onnxruntime")
    full_pipeline = _steady_state(
        milestone5,
        "full_product_pipeline",
        "onnxruntime",
    )
    memory = _cold_start(milestone5, "onnxruntime")
    variants = {
        row["variant"]: row for row in _require(milestone6, "comparison")
    }
    if set(variants) != {"fp32", "dynamic_int8", "static_int8"}:
        raise ValueError("Milestone 6 quantization variants differ.")

    tug_support = [
        row
        for row in _require(milestone4, "support_by_split")
        if row["class"] == "Tug"
    ]
    if {row["split"] for row in tug_support} != {"train", "validation", "test"}:
        raise ValueError("Tug split support is incomplete.")

    logistic_matrix = _confusion_from_csv(
        project_root / "reports/milestone2/confusion_matrices.csv",
        "logistic_regression",
    )
    cnn_matrix = _confusion_from_csv(
        project_root / "reports/milestone4/confusion_matrix.csv"
    )
    if [list(row) for row in logistic_matrix] != logistic_test["confusion_matrix"]:
        raise ValueError("Milestone 2 JSON and CSV confusion matrices differ.")
    if [list(row) for row in cnn_matrix] != cnn_test["confusion_matrix"]:
        raise ValueError("Milestone 4 JSON and CSV confusion matrices differ.")

    timing = _require(milestone2, "timing")
    dynamic = variants["dynamic_int8"]
    static = variants["static_int8"]
    return TeardownEvidence(
        logistic_accuracy=float(logistic_test["accuracy"]),
        logistic_macro_f1=float(logistic_test["macro_f1"]),
        logistic_tug_recall=float(logistic_test["per_class"]["Tug"]["recall"]),
        logistic_model_bytes=int(logistic["model_size_bytes"]),
        logistic_feature_median_ms=float(timing["decode_and_feature"]["median_ms"]),
        logistic_inference_median_ms=float(
            timing["logistic_regression_inference"]["median_ms"]
        ),
        logistic_end_to_end_median_ms=float(
            timing["logistic_regression_end_to_end"]["median_ms"]
        ),
        logistic_confusion=logistic_matrix,
        cnn_accuracy=float(cnn_test["accuracy"]),
        cnn_macro_f1=float(cnn_test["macro_f1"]),
        cnn_tug_recall=float(cnn_test["per_class"]["Tug"]["recall"]),
        cnn_confusion=cnn_matrix,
        tug_windows=sum(int(row["windows"]) for row in tug_support),
        tug_vessel_groups=sum(int(row["vessel_groups"]) for row in tug_support),
        tug_threshold=float(rare_event["threshold"]),
        tug_threshold_test_recall=float(rare_event["test"]["recall"]),
        cnn_parameters=int(model["parameter_count"]),
        onnx_model_bytes=int(model["onnx_size_bytes"]),
        dsp_median_ms=float(dsp["median_ms"]),
        onnx_inference_median_ms=float(inference["median_ms"]),
        onnx_inference_p99_ms=float(inference["p99_ms"]),
        full_pipeline_median_ms=float(full_pipeline["median_ms"]),
        full_pipeline_p99_ms=float(full_pipeline["p99_ms"]),
        onnx_full_rss_increase_bytes=float(
            memory["full_pipeline_rss_increase_bytes"]["median"]
        ),
        onnx_peak_rss_bytes=float(memory["peak_rss_bytes"]["maximum"]),
        deployment_variant=str(milestone6["recommended_variant"]),
        dynamic_size_reduction_ratio=float(dynamic["size_reduction_ratio"]),
        dynamic_p99_change_ratio=float(
            dynamic["qualification"]["p99_latency_increase_ratio"]
        ),
        static_size_reduction_ratio=float(static["size_reduction_ratio"]),
        static_macro_f1_loss=float(static["qualification"]["macro_f1_loss"]),
        static_maximum_recall_loss=float(
            static["qualification"]["maximum_recall_loss"]
        ),
    )


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    size: float = 7.5,
    leading: float = 9.3,
    color: Color = DARK,
    font: str = "Helvetica",
    max_lines: int | None = None,
) -> float:
    lines = _wrap(text, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _section_title(pdf: canvas.Canvas, text: str, x: float, y: float) -> None:
    pdf.setFillColor(TEAL)
    pdf.rect(x, y - 1, 3, 12, fill=1, stroke=0)
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x + 8, y, text)


def _rounded_box(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: Color,
    stroke: Color = MID_GRAY,
    radius: float = 5,
) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.6)
    pdf.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def _header(pdf: canvas.Canvas, page: int, title: str, subtitle: str) -> None:
    page_width, page_height = A4
    pdf.setFillColor(NAVY)
    pdf.rect(0, page_height - 11, page_width, 11, fill=1, stroke=0)
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(34, page_height - 42, title)
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(MUTED)
    pdf.drawString(34, page_height - 56, subtitle)
    pdf.setStrokeColor(MID_GRAY)
    pdf.line(34, page_height - 66, page_width - 34, page_height - 66)
    pdf.setFont("Helvetica-Bold", 7)
    pdf.setFillColor(TEAL)
    pdf.drawRightString(
        page_width - 34,
        page_height - 25,
        f"MILESTONE 7 / PAGE {page}",
    )


def _footer(pdf: canvas.Canvas, page: int) -> None:
    page_width, _ = A4
    pdf.setStrokeColor(MID_GRAY)
    pdf.line(34, 31, page_width - 34, 31)
    pdf.setFont("Helvetica", 6.8)
    pdf.setFillColor(MUTED)
    pdf.drawString(34, 19, "Public-subset engineering evidence; not deployment-ready.")
    pdf.drawRightString(
        page_width - 34,
        19,
        f"Edge Underwater Acoustic Classifier  |  {page} / 2",
    )


def _arrow(pdf: canvas.Canvas, x1: float, y: float, x2: float) -> None:
    pdf.setStrokeColor(SLATE)
    pdf.setFillColor(SLATE)
    pdf.setLineWidth(0.8)
    pdf.line(x1, y, x2 - 4, y)
    path = pdf.beginPath()
    path.moveTo(x2, y)
    path.lineTo(x2 - 5, y + 2.5)
    path.lineTo(x2 - 5, y - 2.5)
    path.close()
    pdf.drawPath(path, fill=1, stroke=0)


def _pipeline(
    pdf: canvas.Canvas,
    labels: list[tuple[str, str]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    gap = 10
    node_width = (width - gap * (len(labels) - 1)) / len(labels)
    for index, (top, bottom) in enumerate(labels):
        node_x = x + index * (node_width + gap)
        _rounded_box(pdf, node_x, y, node_width, height, PALE_BLUE, MID_GRAY, 4)
        pdf.setFillColor(NAVY)
        pdf.setFont("Helvetica-Bold", 6.6)
        pdf.drawCentredString(node_x + node_width / 2, y + height - 13, top)
        pdf.setFillColor(MUTED)
        pdf.setFont("Helvetica", 5.8)
        pdf.drawCentredString(node_x + node_width / 2, y + 9, bottom)
        if index < len(labels) - 1:
            _arrow(
                pdf,
                node_x + node_width + 1,
                y + height / 2,
                node_x + node_width + gap - 1,
            )


def _metric_table(pdf: canvas.Canvas, evidence: TeardownEvidence) -> None:
    x, y, width, height = 34, 566, A4[0] - 68, 82
    _rounded_box(pdf, x, y, width, height, white)
    columns = [x, x + 174, x + 292, x + 410, x + width]
    row_height = height / 3
    pdf.setFillColor(PALE_BLUE)
    pdf.rect(x, y + height - row_height, width, row_height, fill=1, stroke=0)
    headings = ("Model", "Accuracy", "Macro F1", "Tug recall")
    for index, heading in enumerate(headings):
        pdf.setFont("Helvetica-Bold", 7)
        pdf.setFillColor(NAVY)
        pdf.drawString(columns[index] + 7, y + height - 17, heading)
    rows = [
        (
            "Logistic regression / 73 handcrafted features",
            evidence.logistic_accuracy,
            evidence.logistic_macro_f1,
            evidence.logistic_tug_recall,
        ),
        (
            "Selected CNN / FP32 ONNX deployment candidate",
            evidence.cnn_accuracy,
            evidence.cnn_macro_f1,
            evidence.cnn_tug_recall,
        ),
    ]
    for row_index, row in enumerate(rows):
        baseline = y + height - row_height * (row_index + 2) + 9
        pdf.setFont("Helvetica", 6.8)
        pdf.setFillColor(DARK)
        pdf.drawString(columns[0] + 7, baseline, row[0])
        for value_index, value in enumerate(row[1:]):
            pdf.setFont("Helvetica-Bold", 8)
            pdf.drawString(columns[value_index + 1] + 7, baseline, f"{value:.3f}")
    pdf.setStrokeColor(MID_GRAY)
    for column in columns[1:-1]:
        pdf.line(column, y, column, y + height)
    for row in range(1, 3):
        pdf.line(x, y + row * row_height, x + width, y + row * row_height)


def _confusion_matrix(
    pdf: canvas.Canvas,
    matrix: tuple[tuple[int, ...], ...],
    title: str,
    x: float,
    y: float,
    width: float,
) -> None:
    pdf.setFont("Helvetica-Bold", 7.6)
    pdf.setFillColor(NAVY)
    pdf.drawString(x, y + 128, title)
    grid_x = x + 50
    grid_y = y + 10
    cell = 24
    maximum = max(max(row) for row in matrix)
    short = ("C", "P", "Ta", "Tu")
    pdf.setFont("Helvetica", 6)
    pdf.setFillColor(MUTED)
    pdf.drawString(grid_x, y + 116, "predicted")
    pdf.saveState()
    pdf.translate(x + 9, grid_y + 22)
    pdf.rotate(90)
    pdf.drawString(0, 0, "true")
    pdf.restoreState()
    for index, label in enumerate(short):
        pdf.drawCentredString(grid_x + cell * index + cell / 2, grid_y + 96, label)
        pdf.drawRightString(grid_x - 6, grid_y + cell * (3 - index) + 8, label)
    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            intensity = count / maximum if maximum else 0
            fill = Color(
                0.92 - 0.55 * intensity,
                0.96 - 0.42 * intensity,
                0.98 - 0.20 * intensity,
            )
            cell_x = grid_x + predicted_index * cell
            cell_y = grid_y + (3 - true_index) * cell
            pdf.setFillColor(fill)
            pdf.setStrokeColor(white)
            pdf.rect(cell_x, cell_y, cell, cell, fill=1, stroke=1)
            pdf.setFillColor(white if intensity > 0.62 else NAVY)
            pdf.setFont("Helvetica-Bold", 7)
            pdf.drawCentredString(cell_x + cell / 2, cell_y + 8, str(count))
    pdf.setFont("Helvetica", 5.7)
    pdf.setFillColor(MUTED)
    pdf.drawRightString(x + width, y + 1, "counts; n = 135 windows")


def _latency_table(pdf: canvas.Canvas, evidence: TeardownEvidence) -> None:
    x, y, width, height = 34, 85, A4[0] - 68, 156
    _rounded_box(pdf, x, y, width, height, white)
    columns = [x, x + 178, x + 265, x + 352, x + 439, x + width]
    row_height = 26
    pdf.setFillColor(PALE_BLUE)
    pdf.rect(x, y + height - row_height, width, row_height, fill=1, stroke=0)
    headings = ("Path / boundary", "median", "p99", "model", "memory")
    for index, heading in enumerate(headings):
        pdf.setFont("Helvetica-Bold", 6.7)
        pdf.setFillColor(NAVY)
        pdf.drawString(columns[index] + 6, y + height - 17, heading)
    rows = [
        (
            "Logistic feature extraction",
            f"{evidence.logistic_feature_median_ms:.2f} ms",
            "not measured",
            "-",
            "not measured",
        ),
        (
            "Logistic inference",
            f"{evidence.logistic_inference_median_ms:.2f} ms",
            "not measured",
            f"{evidence.logistic_model_bytes:,} B",
            "not measured",
        ),
        (
            "Logistic end to end",
            f"{evidence.logistic_end_to_end_median_ms:.2f} ms",
            "not measured",
            "-",
            "not measured",
        ),
        (
            "ONNX DSP / inference",
            f"{evidence.dsp_median_ms:.2f} / {evidence.onnx_inference_median_ms:.2f} ms",
            f"- / {evidence.onnx_inference_p99_ms:.2f} ms",
            f"{evidence.onnx_model_bytes:,} B",
            f"+{evidence.onnx_full_rss_increase_bytes / 1024**2:.1f} MiB*",
        ),
        (
            "ONNX full warm-cache pipeline",
            f"{evidence.full_pipeline_median_ms:.2f} ms",
            f"{evidence.full_pipeline_p99_ms:.2f} ms",
            f"{evidence.cnn_parameters:,} params",
            f"{evidence.onnx_peak_rss_bytes / 1024**2:.1f} MiB peak*",
        ),
    ]
    for row_index, row in enumerate(rows):
        baseline = y + height - row_height * (row_index + 2) + 9
        for value_index, value in enumerate(row):
            pdf.setFont("Helvetica-Bold" if value_index == 0 else "Helvetica", 6.4)
            pdf.setFillColor(DARK)
            pdf.drawString(columns[value_index] + 6, baseline, value)
    pdf.setStrokeColor(MID_GRAY)
    for column in columns[1:-1]:
        pdf.line(column, y, column, y + height)
    for row in range(1, 6):
        pdf.line(x, y + row * row_height, x + width, y + row * row_height)
    pdf.setFont("Helvetica", 5.8)
    pdf.setFillColor(MUTED)
    pdf.drawString(
        x + 5,
        y - 10,
        "* Python process + ONNX Runtime + DSP libraries; not model-only memory.",
    )


def _draw_page_one(pdf: canvas.Canvas, evidence: TeardownEvidence) -> None:
    _header(
        pdf,
        1,
        "System and evidence",
        "Five-second vessel classification from public DeepShip recordings",
    )
    _rounded_box(pdf, 34, 741, A4[0] - 68, 34, PALE_TEAL, TEAL)
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.setFillColor(NAVY)
    pdf.drawString(45, 760, "OPERATING ASSUMPTION")
    _paragraph(
        pdf,
        "A near-sensor processor receives hydrophone audio, converts each non-overlapping 5 s interval to one four-class prediction, and may retain scores locally. Hydrophone, ADC and field hardware are not yet measured.",
        145,
        760,
        402,
        size=6.7,
        leading=8.2,
        max_lines=3,
    )
    _section_title(pdf, "Implemented signal and model path", 34, 724)
    _pipeline(
        pdf,
        [
            ("AUDIO", "hydrophone file"),
            ("MONO", "16 kHz / DC"),
            ("WINDOW", "5 s / 80k"),
            ("LOG-MEL", "1 x 64 x 155"),
            ("SMALL CNN", "3 conv blocks"),
            ("OUTPUT", "4 logits"),
        ],
        34,
        674,
        A4[0] - 68,
        38,
    )
    _section_title(pdf, "Feature baseline versus deployment CNN", 34, 654)
    _metric_table(pdf, evidence)
    _rounded_box(pdf, 34, 516, A4[0] - 68, 36, PALE_ORANGE, ORANGE)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.setFillColor(NAVY)
    pdf.drawString(45, 537, "EVIDENCE CHECK")
    _paragraph(
        pdf,
        f"The strongest test result remains logistic regression at macro F1 {evidence.logistic_macro_f1:.3f}; the final CNN reached {evidence.cnn_macro_f1:.3f}. Fast edge execution does not establish classification readiness.",
        141,
        537,
        405,
        size=7,
        leading=8.5,
        font="Helvetica-Bold",
        max_lines=2,
    )
    _section_title(pdf, "Test confusion: where the models fail", 34, 497)
    column_width = (A4[0] - 82) / 2
    _confusion_matrix(
        pdf,
        evidence.logistic_confusion,
        "LOGISTIC REGRESSION",
        34,
        343,
        column_width,
    )
    _confusion_matrix(
        pdf,
        evidence.cnn_confusion,
        "SELECTED CNN / FP32 ONNX",
        48 + column_width,
        343,
        column_width,
    )
    _section_title(pdf, "Development-machine latency, memory and size", 34, 257)
    _latency_table(pdf, evidence)
    _footer(pdf, 1)


def _card(
    pdf: canvas.Canvas,
    title: str,
    body: str,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: Color,
) -> None:
    _rounded_box(pdf, x, y, width, height, fill)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.setFillColor(NAVY)
    pdf.drawString(x + 10, y + height - 17, title)
    _paragraph(
        pdf,
        body,
        x + 10,
        y + height - 31,
        width - 20,
        size=6.8,
        leading=8.5,
    )


def _risk_row(
    pdf: canvas.Canvas,
    y: float,
    label: str,
    signal: str,
    consequence: str,
) -> None:
    x, width, height = 34, A4[0] - 68, 27
    pdf.setFillColor(PALE_GRAY if int(y) % 2 == 0 else white)
    pdf.rect(x, y, width, height, fill=1, stroke=0)
    pdf.setStrokeColor(MID_GRAY)
    pdf.line(x, y, x + width, y)
    pdf.setFont("Helvetica-Bold", 6.5)
    pdf.setFillColor(NAVY)
    pdf.drawString(x + 7, y + 10, label)
    pdf.setFont("Helvetica", 6.2)
    pdf.setFillColor(DARK)
    pdf.drawString(x + 120, y + 10, signal)
    pdf.drawString(x + 328, y + 10, consequence)


def _draw_page_two(pdf: canvas.Canvas, evidence: TeardownEvidence) -> None:
    _header(
        pdf,
        2,
        "Edge-first design and risks",
        "Proposed deployment path; only preprocessing and inference are implemented",
    )
    _section_title(pdf, "Proposed near-sensor architecture", 34, 758)
    _pipeline(
        pdf,
        [
            ("HYDROPHONE", "analog signal"),
            ("ADC", "digitize"),
            ("RING BUFFER", "5 s frames"),
            ("PREPROCESS", "log-mel"),
            ("ONNX", "FP32 inference"),
            ("SMOOTH", "future policy"),
            ("EVENT", "store / uplink"),
        ],
        34,
        704,
        A4[0] - 68,
        40,
    )
    pdf.setStrokeColor(TEAL)
    pdf.setLineWidth(0.8)
    pdf.setDash(2, 2)
    pdf.line(473, 698, 473, 683)
    pdf.line(473, 683, 550, 683)
    pdf.setDash()
    pdf.setFont("Helvetica", 5.8)
    pdf.setFillColor(MUTED)
    pdf.drawRightString(550, 674, "optional connectivity; event metadata or selected audio")

    column_width = (A4[0] - 82) / 2
    _card(
        pdf,
        "WHY INFER NEAR THE SENSOR",
        "Avoid continuous audio uplink, bound response time, tolerate intermittent connectivity and store only events or review clips. This benefit depends on calibrated false-alarm behavior and target-device power.",
        34,
        586,
        column_width,
        76,
        PALE_TEAL,
    )
    _card(
        pdf,
        "COMPUTE / POWER / STORAGE / LINK",
        f"FP32 remains recommended: {evidence.onnx_model_bytes / 1024:.1f} KiB and {evidence.full_pipeline_p99_ms:.1f} ms warm-cache p99 on Apple M2. Dynamic INT8 saved {evidence.dynamic_size_reduction_ratio * 100:.1f}% but was {evidence.dynamic_p99_change_ratio * 100:.1f}% slower at p99. Static INT8 saved {evidence.static_size_reduction_ratio * 100:.1f}% but lost {evidence.static_macro_f1_loss:.3f} macro F1 and {evidence.static_maximum_recall_loss:.3f} maximum recall.",
        48 + column_width,
        586,
        column_width,
        76,
        PALE_BLUE,
    )
    _section_title(pdf, "Decision logic that is not deployment-calibrated", 34, 565)
    _card(
        pdf,
        "THRESHOLD + TEMPORAL AGGREGATION",
        "Retain the four per-window scores. Calibrate class thresholds on representative field data, then require evidence across multiple windows before an event. No production threshold, K-of-N rule or smoothing horizon has been validated.",
        34,
        472,
        column_width,
        78,
        PALE_ORANGE,
    )
    _card(
        pdf,
        "RARE / UNKNOWN EVENTS",
        f"Tug has {evidence.tug_windows} windows but only {evidence.tug_vessel_groups} vessel groups. A validation-selected Tug threshold ({evidence.tug_threshold:.3f}) produced test recall {evidence.tug_threshold_test_recall:.3f}. No unknown/background set exists, so confidence and embedding distance do not validate unknown rejection.",
        48 + column_width,
        472,
        column_width,
        78,
        PALE_ORANGE,
    )

    _section_title(pdf, "Risk register: do not treat all errors as model errors", 34, 450)
    pdf.setFillColor(NAVY)
    pdf.rect(34, 411, A4[0] - 68, 24, fill=1, stroke=0)
    for x, text in ((41, "RISK"), (142, "CURRENT SIGNAL"), (362, "DEPLOYMENT CONSEQUENCE")):
        pdf.setFillColor(white)
        pdf.setFont("Helvetica-Bold", 6.3)
        pdf.drawString(x, 420, text)
    _risk_row(pdf, 384, "Model error", "CNN test macro F1 0.031", "Misclassification is the observed default.")
    _risk_row(pdf, 357, "Label ambiguity", "48 manual audits pending", "Cannot yet separate bad labels from errors.")
    _risk_row(pdf, 330, "Domain shift", "Few independent vessels / sites", "Field spectra may differ from public subset.")
    _risk_row(pdf, 303, "False alarms", "No long-duration background run", "Precision at scale is unknown.")
    _risk_row(pdf, 276, "Site shortcuts", "Confidence/distance are proxies", "Recording context may drive predictions.")

    _section_title(pdf, "Required before real deployment", 34, 252)
    checklist = [
        "Representative hydrophones, ADCs, sites and operating conditions",
        "Unknown/background and overlapping-source collection",
        "Complete 48 label audits and 20 error-listening reviews",
        "Independent vessel-disjoint holdout with no prior test reuse",
        "Long-duration false-alarm and missed-event evaluation",
        "Threshold, smoothing and event-policy calibration",
        "Target-device power, thermal, latency and RSS benchmarks",
        "Field, enclosure, storage and connectivity validation",
    ]
    left_x, right_x = 34, 305
    for index, item in enumerate(checklist):
        col_x = left_x if index < 4 else right_x
        row = index if index < 4 else index - 4
        item_y = 224 - row * 30
        pdf.setFillColor(white)
        pdf.setStrokeColor(TEAL)
        pdf.setLineWidth(1)
        pdf.rect(col_x, item_y - 2, 8, 8, fill=0, stroke=1)
        _paragraph(
            pdf,
            item,
            col_x + 15,
            item_y,
            238,
            size=6.6,
            leading=8,
            max_lines=2,
        )
    _rounded_box(pdf, 34, 60, A4[0] - 68, 37, PALE_ORANGE, ORANGE)
    pdf.setFont("Helvetica-Bold", 7.4)
    pdf.setFillColor(NAVY)
    pdf.drawString(45, 81, "BOTTOM LINE")
    _paragraph(
        pdf,
        "The pipeline is computationally feasible on a laptop CPU. The evidence is not yet sufficient for a safety-, monitoring- or operations-critical underwater deployment.",
        118,
        81,
        428,
        size=7,
        leading=8.5,
        font="Helvetica-Bold",
        max_lines=2,
    )
    _footer(pdf, 2)


def build_technical_teardown(project_root: Path, output_path: Path) -> Path:
    evidence = load_teardown_evidence(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle("Edge Underwater Acoustic Classifier - Technical Teardown")
    pdf.setAuthor("Edge Underwater Acoustic Classifier")
    pdf.setSubject("Milestone 7 system evidence and edge-first deployment risks")
    _draw_page_one(pdf, evidence)
    pdf.showPage()
    _draw_page_two(pdf, evidence)
    pdf.showPage()
    pdf.save()
    return output_path
