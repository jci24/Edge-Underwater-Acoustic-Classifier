# Edge Underwater Acoustic Classifier

This repository contains reproducible tooling for acquiring and cataloguing the
publicly available portion of the
[DeepShip dataset](https://github.com/irfankamboh/DeepShip).

## DeepShip public recordings

DeepShip describes 47 hours and 4 minutes of underwater audio from 265 ships in
four commercial ship classes. The GitHub repository contains only a public
subset; its README directs users to request the remaining recordings from the
dataset author.

## Setup

Create a local environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Install FFmpeg separately if `ffprobe` and `ffplay` are not available. On
macOS:

```bash
brew install ffmpeg
```

## Dataset workflow

Download the public files:

```bash
python3 scripts/download_deepship.py
```

Build the catalogue:

```bash
python3 scripts/catalogue_deepship.py
```

Catalogue generation requires `ffprobe` (provided by FFmpeg) because the source
recordings use 32-bit floating-point WAV encoding, which Python's standard
library WAV reader does not support.

The downloader stores the untracked source files in `data/raw/deepship/`. The
catalogue command creates:

- `data/catalogues/deepship_recordings.csv`: one row per available WAV file,
  including its class, duration, sample rate, source recording ID, vessel name,
  and a derived session identifier when date and time metadata are available.

Both commands are safe to rerun. Existing downloads are skipped only when their
sizes match the upstream file list. The downloader follows the upstream `main`
branch and records the exact revision in
`data/raw/deepship/source_commit.txt`.

Plot the number of files, total duration, and unique vessels in each class:

```bash
python3 scripts/plot_class_distribution.py
```

The generated plot is saved to
`data/plots/deepship_class_distribution.png` and is excluded from Git because
it can be recreated from the catalogue.

In the plot, a source group means one unique vessel name within a class. It is
not an independently verified recording session.

## Listening annotations

Create a reproducible listening sample with three recordings from each class:

```bash
python3 scripts/create_listening_sample.py
```

This creates
`data/annotations/deepship_listening_annotations.csv`. The script refuses to
overwrite an existing file so that manual notes are not lost. Listen to an
entry with:

```bash
ffplay -nodisp -autoexit data/raw/deepship/Cargo/103.wav
```

Use consistent labels such as:

- `noise_present`: `yes`, `no`, or `uncertain`
- `vessel_audibility`: `clear`, `partly_masked`, `weak`, or `not_obvious`
- `confidence`: `high`, `medium`, or `low`

Write observations rather than inferred mechanical causes. For example,
“steady low-frequency tone with intermittent broadband bursts” is appropriate;
an unsupported diagnosis of a damaged component is not.

## Model preprocessing

Install the project and its test dependencies:

```bash
python3 -m pip install -e ".[test]"
```

Build the deterministic five-second window manifest and fit normalization
statistics using training windows only:

```bash
python3 scripts/build_window_manifest.py
python3 scripts/fit_training_statistics.py
```

Validate every window under both supported normalization modes and create the
visual comparison:

```bash
python3 scripts/validate_preprocessing.py
MPLBACKEND=Agg python3 scripts/inspect_preprocessing.py
```

The model input is a finite `float32` tensor with shape `[1, 64, 155]`:
one channel, 64 mel bands, and 155 time frames. The committed manifest traces
every tensor to its source file and half-open time interval. See
[the preprocessing specification](docs/preprocessing.md) for the complete
configuration, split policy, and normalization behavior.

## Hand-engineered baseline

Extract the fixed 73-feature representation, train the primary logistic
regression and secondary random forest, then validate every artifact:

```bash
python3 scripts/extract_handcrafted_features.py
MPLBACKEND=Agg python3 scripts/train_handcrafted_baselines.py
python3 scripts/validate_handcrafted_baseline.py
```

The committed results include window- and source-level metrics, per-class
support, fixed confusion matrices, model inspection tables, and separate
feature-extraction, classifier-inference, end-to-end latency, and model-size
measurements. See
[the baseline specification and limits](docs/handcrafted_baseline.md).

## Small CNN baseline

Train the three controlled compact-CNN runs, create the non-overwriting manual
error-review pack, and validate the experiment:

```bash
MPLBACKEND=Agg python3 scripts/train_cnn_baseline.py
python3 scripts/create_cnn_error_review.py
python3 scripts/validate_cnn_baseline.py
python3 scripts/validate_cnn_error_review.py
```

The validation-selected CNN alone is evaluated on the unchanged test split and
compared with the committed feature baselines. Model checkpoints stay ignored;
configurations, histories, learning curves, predictions, latency, and metrics
are versioned. The generated 20-error listening pack still requires human
annotation. See [the CNN specification](docs/cnn_baseline.md).

## Dataset access and interpretation

The upstream metafiles do not provide column headers. This project uses only
the fields whose formats and values clearly support the vessel-name, date, and
time interpretations. Do not treat the other source fields as defined physical
quantities without a primary source.

The full dataset is not downloadable from the linked GitHub repository. Follow
the access instruction in the
[upstream README](https://github.com/irfankamboh/DeepShip/blob/main/README.txt)
to request it from the author.

No licence file was present in the upstream repository when this workflow was
created. Confirm the current dataset terms with the author before
redistributing the recordings or using them beyond research permitted by the
applicable terms.
