#!/usr/bin/env python3
"""Validate Milestone 7 PDF structure and required reader-facing content."""

from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
PDF_FILE = (
    PROJECT_FOLDER
    / "output/pdf/edge_underwater_classifier_technical_teardown.pdf"
)
REQUIRED_TEXT = (
    "System and evidence",
    "Edge-first design and risks",
    "0.283",
    "0.031",
    "Public-subset engineering evidence; not deployment-ready.",
    "Required before real deployment",
    "1 / 2",
    "2 / 2",
)


def main() -> None:
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"Missing teardown PDF: {PDF_FILE}. Run the build script first."
        )
    reader = PdfReader(PDF_FILE)
    if len(reader.pages) != 2:
        raise RuntimeError("Technical teardown must contain exactly two pages.")
    expected_width, expected_height = A4
    page_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - expected_width) > 0.1 or abs(height - expected_height) > 0.1:
            raise RuntimeError(f"Page {page_number} is not A4.")
        text = page.extract_text() or ""
        if len(text.strip()) < 500:
            raise RuntimeError(f"Page {page_number} has too little extractable text.")
        page_text.append(text)
    combined = "\n".join(page_text)
    missing = [value for value in REQUIRED_TEXT if value not in combined]
    if missing:
        raise RuntimeError(f"Required PDF text is missing: {missing}")
    print("Validated exactly two non-empty A4 pages")
    print("Validated headings, core metrics, limitations, footer and page numbers")


if __name__ == "__main__":
    main()
