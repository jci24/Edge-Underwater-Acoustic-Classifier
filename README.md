# Edge Underwater Acoustic Classifier

This repository contains reproducible tooling for acquiring and cataloguing the
publicly available portion of the
[DeepShip dataset](https://github.com/irfankamboh/DeepShip).

## DeepShip public recordings

DeepShip describes 47 hours and 4 minutes of underwater audio from 265 ships in
four commercial ship classes. The GitHub repository contains only a public
subset; its README directs users to request the remaining recordings from the
dataset author.

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
sizes match the upstream file list.

Plot the number of files, total duration, and unique vessels in each class:

```bash
python3 -m pip install pandas matplotlib
python3 scripts/plot_class_distribution.py
```

The generated plot is saved to
`data/plots/deepship_class_distribution.png` and is excluded from Git because
it can be recreated from the catalogue.

## Dataset access and interpretation

The upstream metafiles do not provide column headers. This project therefore
preserves their fields conservatively as `source_field_2` through
`source_field_7`; the vessel name, date, and time aliases are included because
their formats and values make those interpretations clear. Do not treat the
remaining fields as defined physical quantities without a primary source.

The full dataset is not downloadable from the linked GitHub repository. Follow
the access instruction in the
[upstream README](https://github.com/irfankamboh/DeepShip/blob/main/README.txt)
to request it from the author.

No licence file is present in the upstream repository at the revision recorded
in the generated catalogue. Confirm the dataset terms with the author before
redistributing the recordings or using them beyond research permitted by the
applicable terms.
