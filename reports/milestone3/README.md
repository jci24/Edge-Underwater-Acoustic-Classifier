# Milestone 3 — Small CNN Baseline

## Outcome

The validation-selected primary run was `per_example_weighted`. It used
`per_example` normalization and
class-weighted loss.
Only this selected CNN was evaluated on the test split.

| model | role | test macro F1 | test accuracy |
|---|---|---|---|
| Small CNN | Milestone 3 primary | 0.126188 | 0.133333 |
| Logistic regression | Milestone 2 primary | 0.283202 | 0.392593 |
| Random forest | Milestone 2 secondary | 0.166602 | 0.192593 |

Window-level macro F1 remains the primary metric. Source-level aggregation is
secondary because the test split contains only one Cargo, one Passengership,
and one Tug vessel group.

## Controlled training runs

| name | normalization | class_weighted | best_epoch | epochs_completed | best_validation_macro_f1 |
|---|---|---|---|---|---|
| training_stats_unweighted | training_stats | False | 28 | 36 | 0.279213 |
| per_example_unweighted | per_example | False | 8 | 16 | 0.300629 |
| per_example_weighted | per_example | True | 2 | 10 | 0.325027 |

Both unweighted normalization runs completed before class weighting was
calculated and tested. Selection used validation macro F1 only, with
`training_stats` preferred for a normalization tie and unweighted loss
preferred for a weighting tie. No train-plus-validation refit was performed.

## Primary CNN test metrics

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Cargo | 0 | 0 | 0 | 37 |
| Passengership | 0 | 0 | 0 | 45 |
| Tanker | 0.0762712 | 0.692308 | 0.137405 | 13 |
| Tug | 1 | 0.225 | 0.367347 | 40 |

The source-level test macro F1 was
0.056; this is a
secondary descriptive result.

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

## Size and latency

- Parameters: 23,668
- Deployable state-dict size: 102,375 bytes
- Full checkpoint size: 304,899 bytes
- Preprocessing: median 5.831 ms,
  p95 11.973 ms
- CNN inference: median 1.133 ms,
  p95 1.237 ms
- End to end: median 5.649 ms,
  p95 8.235 ms

These are warm-cache, batch-one measurements on the current development Mac
with one PyTorch thread, not target-edge-device performance.

## Error listening review

A deterministic pack of 20 incorrect test windows is stored in
`data/annotations/deepship_cnn_error_review.csv`. Its listening fields are
intentionally blank. Human listening remains pending and must be completed
before the listening checklist can be marked done.

## Reporting limits

Windows within recordings and vessels are correlated. The 135 test windows are
not 135 independent vessel observations, and these public-subset results do not
generalize to the complete 47-hour DeepShip dataset. No minimum CNN F1 was
required.
