from hashlib import sha256
from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4

from edge_underwater.technical_teardown import (
    build_technical_teardown,
    load_teardown_evidence,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def test_evidence_loader_uses_expected_models_metrics_and_confusions():
    evidence = load_teardown_evidence(PROJECT_FOLDER)

    assert evidence.logistic_accuracy == pytest.approx(0.3925925926)
    assert evidence.logistic_macro_f1 == pytest.approx(0.2832017989)
    assert evidence.logistic_tug_recall == pytest.approx(0.025)
    assert evidence.logistic_model_bytes == 7_945
    assert sum(map(sum, evidence.logistic_confusion)) == 135

    assert evidence.cnn_accuracy == pytest.approx(0.0666666667)
    assert evidence.cnn_macro_f1 == pytest.approx(0.0314685315)
    assert evidence.cnn_tug_recall == 0
    assert evidence.deployment_variant == "fp32"
    assert sum(map(sum, evidence.cnn_confusion)) == 135

    assert evidence.tug_windows == 118
    assert evidence.tug_vessel_groups == 3
    assert evidence.tug_threshold_test_recall == pytest.approx(0.025)
    assert evidence.onnx_model_bytes == 116_187
    assert evidence.cnn_parameters == 23_668


def test_evidence_loader_fails_when_required_files_are_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Required evidence file"):
        load_teardown_evidence(tmp_path)


def test_pdf_is_deterministic_exactly_two_a4_pages_with_required_text(
    tmp_path: Path,
):
    first = build_technical_teardown(PROJECT_FOLDER, tmp_path / "first.pdf")
    second = build_technical_teardown(PROJECT_FOLDER, tmp_path / "second.pdf")

    assert sha256(first.read_bytes()).hexdigest() == sha256(
        second.read_bytes()
    ).hexdigest()
    reader = PdfReader(first)
    assert len(reader.pages) == 2
    texts = [page.extract_text() or "" for page in reader.pages]
    assert "System and evidence" in texts[0]
    assert "0.283" in texts[0]
    assert "0.031" in texts[0]
    assert "Edge-first design and risks" in texts[1]
    assert "Required before real deployment" in texts[1]
    for page in reader.pages:
        assert float(page.mediabox.width) == pytest.approx(A4[0], abs=0.1)
        assert float(page.mediabox.height) == pytest.approx(A4[1], abs=0.1)
