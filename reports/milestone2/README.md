# Milestone 2 — Hand-Engineered Feature Baseline

## Outcome

The primary regularized multinomial logistic regression selected `C=1.0`
using validation macro F1. Its untouched test-window macro F1 was
**0.283** with accuracy
**0.393**. The fixed random-forest comparison
reached test-window macro F1 **0.167** and
accuracy **0.193**.

Source-level aggregation is secondary: the public test split contains only one
Cargo, one Passengership, and one Tug vessel group. Windows from the same source
and vessel are correlated and must not be interpreted as independent vessel
observations.

| model | role | window macro F1 | window accuracy | source macro F1 |
|---|---|---|---|---|
| Logistic regression | primary | 0.283202 | 0.392593 | 0.2875 |
| Random forest | secondary | 0.166602 | 0.192593 | 0.236111 |

## Test metrics by class

| model | class | precision | recall | f1 | support |
|---|---|---|---|---|---|
| Logistic regression | Cargo | 0.375 | 0.0810811 | 0.133333 | 37 |
| Logistic regression | Passengership | 0.547945 | 0.888889 | 0.677966 | 45 |
| Logistic regression | Tanker | 0.169811 | 0.692308 | 0.272727 | 13 |
| Logistic regression | Tug | 1 | 0.025 | 0.0487805 | 40 |
| Random forest | Cargo | 0 | 0 | 0 | 37 |
| Random forest | Passengership | 0.9375 | 0.333333 | 0.491803 | 45 |
| Random forest | Tanker | 0.0973451 | 0.846154 | 0.174603 | 13 |
| Random forest | Tug | 0 | 0 | 0 | 40 |

## Support

| split | class | windows | source_files | vessel_groups |
|---|---|---|---|---|
| test | Cargo | 37 | 1 | 1 |
| test | Passengership | 45 | 3 | 1 |
| test | Tanker | 13 | 3 | 3 |
| test | Tug | 40 | 1 | 1 |
| train | Cargo | 345 | 9 | 7 |
| train | Passengership | 167 | 15 | 8 |
| train | Tanker | 279 | 22 | 13 |
| train | Tug | 38 | 1 | 1 |
| validation | Cargo | 77 | 2 | 2 |
| validation | Passengership | 9 | 2 | 2 |
| validation | Tanker | 28 | 3 | 3 |
| validation | Tug | 40 | 1 | 1 |

## Validation selection

| C | Validation macro F1 |
|---:|---:|
| 0.01 | 0.360 |
| 0.1 | 0.360 |
| 1 | 0.364 |
| 10 | 0.355 |

Selectors, scaling, sample weights, and both classifiers were fitted only on
the committed training split. The logistic model was not refitted on
training-plus-validation after selection.

## Model inspection

The logistic coefficients below are the three strongest positive and negative
standardized coefficients per class. They show association within this fitted
model, not causation.

| class | direction | rank | feature | standardized_coefficient |
|---|---|---|---|---|
| Cargo | negative | 1 | mfcc_3_mean | -1.24215 |
| Cargo | negative | 2 | spectral_rolloff_85_hz_std | -1.21469 |
| Cargo | negative | 3 | mfcc_3_std | -1.06732 |
| Cargo | positive | 1 | mfcc_1_std | 1.04304 |
| Cargo | positive | 2 | dominant_peak_1_relative_power | 1.01568 |
| Cargo | positive | 3 | mfcc_19_mean | 0.776692 |
| Passengership | negative | 1 | band_energy_ratio_100_250_hz | -1.53237 |
| Passengership | negative | 2 | dominant_peak_5_relative_power | -1.18339 |
| Passengership | negative | 3 | mfcc_4_mean | -0.976291 |
| Passengership | positive | 1 | band_energy_ratio_50_100_hz | 1.2466 |
| Passengership | positive | 2 | mfcc_20_mean | 1.23331 |
| Passengership | positive | 3 | spectral_bandwidth_hz_mean | 0.750731 |
| Tanker | negative | 1 | mfcc_1_std | -1.51901 |
| Tanker | negative | 2 | band_energy_ratio_50_100_hz | -1.46399 |
| Tanker | negative | 3 | spectral_flatness_mean | -0.919808 |
| Tanker | positive | 1 | mfcc_3_mean | 1.38785 |
| Tanker | positive | 2 | band_energy_ratio_100_250_hz | 1.28783 |
| Tanker | positive | 3 | mfcc_15_mean | 1.03177 |
| Tug | negative | 1 | mfcc_10_mean | -0.477328 |
| Tug | negative | 2 | mfcc_6_mean | -0.473099 |
| Tug | negative | 3 | mfcc_20_mean | -0.421805 |
| Tug | positive | 1 | spectral_flatness_std | 0.497232 |
| Tug | positive | 2 | spectral_centroid_hz_std | 0.414914 |
| Tug | positive | 3 | mfcc_18_mean | 0.386771 |

The random-forest values are validation-set permutation importance using macro
F1, 20 repeats, and seed 42. They measure fitted-model reliance; correlated
features can dilute each other's permutation importance.

| feature | importance_mean | importance_std |
|---|---|---|
| spectral_centroid_hz_std | 0.0248771 | 0.00630001 |
| spectral_flatness_std | 0.0244313 | 0.00726265 |
| spectral_flux_mean | 0.0149486 | 0.00926242 |
| mfcc_17_mean | 0.0119861 | 0.0164738 |
| mfcc_10_mean | 0.00768293 | 0.00558581 |
| spectral_rolloff_85_hz_mean | 0.0045393 | 0.00633689 |
| spectral_bandwidth_hz_std | 0.00398388 | 0.00430293 |
| dominant_peak_5_relative_power | 0.0031128 | 0.00631955 |
| mfcc_9_mean | 0.00292429 | 0.00940355 |
| mfcc_7_mean | 0.00254968 | 0.00696952 |

## Timing and size

All latency values are warm-cache, batch-size-one measurements on the current
development Mac with one numerical-library thread. They are not target
edge-device performance.

- Full feature-table extraction:
  9.035 seconds
- Test source decoding plus feature extraction:
  median 7.184 ms,
  p95 12.712 ms
- Logistic `predict_proba`:
  median 0.816 ms,
  p95 1.148 ms,
  model size 7945 bytes
- Random-forest `predict_proba`:
  median 14.156 ms,
  p95 20.025 ms,
  model size 5580562 bytes
- Logistic end to end:
  median 8.130 ms,
  p95 11.046 ms
- Random-forest end to end:
  median 20.354 ms,
  p95 23.141 ms

Environment: Python 3.14.6,
scikit-learn 1.9.0,
PyTorch 2.10.0.

## Reporting limits

This baseline validates the reproducible mechanics of the public subset. It
does not generalize performance to the full 47-hour DeepShip dataset, and no
minimum F1 threshold was imposed. Source-level results are reported only as a
secondary view.
