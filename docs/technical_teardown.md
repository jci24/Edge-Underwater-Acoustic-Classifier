# Two-page technical teardown

Milestone 7 turns the committed experimental evidence into a compact
engineering-review artifact. The source remains data-driven: the PDF generator
reads Milestone 2, 4, 5, and 6 metrics and confusion matrices instead of
copying numerical results into a separate document.

## Build and validate

Install the documentation dependencies:

```bash
python3 -m pip install -e '.[docs,test]'
```

Generate and structurally validate the PDF:

```bash
python3 scripts/build_technical_teardown.py
python3 scripts/validate_technical_teardown.py
```

The committed artifact is:

```text
output/pdf/edge_underwater_classifier_technical_teardown.pdf
```

For visual QA, install Poppler and render both pages:

```bash
mkdir -p tmp/pdfs
pdftoppm -png -r 180 \
  output/pdf/edge_underwater_classifier_technical_teardown.pdf \
  tmp/pdfs/technical_teardown
```

Inspect both PNG files at full resolution. Check for clipping, overlaps,
unreadable tables, broken glyphs, poor contrast, uneven alignment, and an
unexpected third or blank page. Files under `tmp/pdfs/` are ignored.

## Evidence boundary

The PDF distinguishes implemented and measured software from proposed edge
architecture. Hydrophone/ADC behavior, ring buffering, temporal smoothing,
event policies, power consumption, and target underwater hardware remain
unimplemented or unmeasured.
