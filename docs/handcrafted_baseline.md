# Hand-Engineered Feature Baseline

Milestone 2 represents each five-second Milestone 1 window with 73 deterministic
features. It uses the committed vessel-disjoint train, validation, and test
assignments without resplitting.

## Feature contract

The feature configuration and exact column order are stored in
`data/features/handcrafted_config.json`. The vector contains:

- RMS, crest factor, and zero-crossing rate;
- mean and standard deviation of spectral centroid, bandwidth, 85% roll-off,
  and flatness;
- eight non-overlapping energy-band ratios from 10 Hz to 8 kHz;
- frequency and relative power for five dominant peaks, plus the number of
  peaks detected with at least 3 dB prominence and 25 Hz separation;
- mean and standard deviation for 20 MFCCs; and
- mean, standard deviation, and maximum spectral flux.

The extractor first applies the same channel averaging, 16 kHz resampling,
DC removal, and exact-length policy as Milestone 1. Silence, empty bands, and
missing peaks use epsilon protection or documented zero padding. NaN and
infinite inputs or outputs are rejected.

`data/features/deepship_handcrafted_features.csv` contains one row per
`window_id`. It preserves source, class, vessel, session, split, time interval,
and both preprocessing configuration hashes. Timing is intentionally excluded
from the deterministic feature table.

## Leakage controls

Both classifiers use scikit-learn pipelines. `VarianceThreshold`, scaling, and
model fitting receive training rows only. Training sample weights give each
class equal total weight and each vessel within a class equal total weight.

The primary model selects logistic-regression `C` from 0.01, 0.1, 1, and 10
using validation window-level macro F1, with the smaller value winning a tie.
It is not refitted on combined training and validation data. The random forest
uses one fixed configuration and remains a secondary comparison.

The test split is evaluated only after logistic model selection. Source-level
probabilities are averages of window probabilities and are a secondary metric.

## Reproduce

After downloading and cataloguing the public audio and completing Milestone 1:

```bash
python3 scripts/extract_handcrafted_features.py
MPLBACKEND=Agg python3 scripts/train_handcrafted_baselines.py
python3 scripts/validate_handcrafted_baseline.py
python3 -m pytest
```

The trained `.joblib` pipelines are reproducible but ignored under
`artifacts/models/`. The feature table, configurations, metrics, predictions,
coefficient table, permutation-importance table, confusion matrices, and
results report are versioned.

## Interpretation limits

Window-level macro F1 is the primary metric. Windows from a recording or vessel
are correlated, and the 135 public test windows are not 135 independent vessel
observations. In particular, the public test split contains only one Cargo,
one Passengership, and one Tug vessel group. These results do not establish
performance on the full 47-hour DeepShip dataset or on a target edge device.

