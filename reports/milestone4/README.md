# Milestone 4 — Imbalance, Rare Events, and Label Risk

## Outcome

The validation-selected imbalance strategy was
`class_vessel_balanced_sampling`. Selection used validation macro F1 and only
the selected strategy was evaluated on test. The test set had already been
used in Milestone 3, so this is transparent repeated test-set use rather than
a pristine confirmatory evaluation.

| strategy | validation macro F1 | Cargo recall | Passengership recall | Tanker recall | Tug recall |
|---|---|---|---|---|---|
| unweighted | 0.300629 | 0.415584 | 0.666667 | 0 | 0 |
| class_weighted | 0.325027 | 0.194805 | 0.222222 | 0 | 0.825 |
| class_vessel_balanced_sampling | 0.396017 | 0.636364 | 0.333333 | 0.214286 | 0.2 |

Selected-strategy test macro F1 was **0.031** with accuracy
**0.067**. Accuracy is secondary; per-class recall and macro F1
remain primary.

## Coverage and imbalance risk

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

Tug has 118 windows but only three independent vessel groups—one in each
split. Window oversampling cannot create new acoustic coverage.

## Tug one-vs-rest diagnostic

The selected four-class CNN's Tug probability is analyzed as a separate score;
this is not a dedicated or independently validated rare-event detector.

- Validation average precision:
  0.817
- Test average precision: 0.999
- Validation-selected threshold at FPR ≤5%: 0.194970
- Fixed-threshold test recall: 0.025
- Fixed-threshold test precision: 1.000
- Fixed-threshold test FPR: 0.000
- Test counts: TP 1, FP 0,
  FN 39, TN 95

Source-level rare-event results are secondary because the test set contains one
Tug source group.

## Rejection diagnostics

Both thresholds retain approximately 90% of validation windows:

| method | threshold | test coverage | selective test accuracy | selective test macro F1 |
|---|---:|---:|---:|---:|
| maximum probability | 0.322080 | 1.000 | 0.067 | 0.031 |
| embedding distance | 11.561041 | 1.000 | 0.067 | 0.031 |

There are no unknown/background examples. These results describe confidence
and in-distribution outliers only and do not validate unknown detection.

## Label and domain risk

`data/annotations/label_audit.csv` contains 48 pending manual reviews: one for
every vessel group plus five targeted extremes. RMS, probabilities, embedding
distance, and source boundaries are selection proxies—not evidence that a
label is wrong or that target-vessel energy is absent.

Until all 48 rows are reviewed, counts of model error, dataset ambiguity, domain shift, mixed cases, and unresolved cases remain **pending**. Confidence and embedding-distance summaries are domain-shift
indicators, not proof that the model exploited recording-site cues.

## Exit status

The engineering analysis is complete. The full milestone exit remains pending human completion of all 48 label-audit rows and regeneration of the disposition summary.
