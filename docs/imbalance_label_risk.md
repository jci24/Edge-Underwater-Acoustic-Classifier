# Imbalance, Rare Events, and Label Risk

Milestone 4 separates window imbalance, rare-event operating behavior, rejection
diagnostics, label ambiguity, and domain-shift indicators. It does not treat all
classification mistakes as equivalent.

## Imbalance experiment

Three fresh CNN runs use per-example normalization and the unchanged Milestone 3
architecture, optimizer, seed, early stopping, and vessel-disjoint splits:

1. unweighted cross-entropy;
2. class-weighted cross-entropy; and
3. class-and-vessel-balanced sampling.

The sampler draws 829 examples with replacement per epoch. Its window weights
give equal total mass to each class and equal mass to each vessel within a
class, preventing long recordings from dominating merely through window count.
Sampling still cannot create new acoustic coverage: Tug has only three vessel
groups, one per split.

Validation macro F1 selects one strategy. Ties prefer unweighted,
class-weighted, then balanced sampling. Only the selected strategy is evaluated
on test, and the report discloses that Milestone 3 already used this test set.

## Tug and rejection diagnostics

Tug probability from the selected four-class CNN is treated as a one-vs-rest
score. The threshold maximizes validation recall while keeping empirical
window-level FPR at or below 5%, then remains fixed on test. Precision-recall
curves, average precision, counts, and secondary source aggregation are
reported.

Maximum class probability and standardized embedding distance are calibrated
to retain 90% of validation windows. The embedding reference—per-dimension
mean/std and four class centroids—is fitted only from training embeddings.

DeepShip supplies no unknown/background evaluation examples. Rejection results
therefore describe confidence and in-distribution outliers, not validated
unknown detection.

## Label audit

`data/annotations/label_audit.csv` contains 48 non-overwriting review rows: one
for every vessel group plus five targeted extremes. Quantitative flags select
review candidates but do not prove that target-vessel energy is absent, labels
are incorrect, or recording-site cues caused a prediction.

Complete the controlled reviewer fields and observational notes, then run:

```bash
python3 scripts/validate_label_audit.py --require-complete
python3 scripts/summarize_label_audit.py
```

The summary distinguishes reviewed model errors, dataset ambiguity,
domain-shift indicators, mixed cases, and unresolved cases. Full milestone exit
remains pending until all 48 rows are manually reviewed.

## Reproduce

```bash
MPLBACKEND=Agg python3 scripts/train_imbalance_risk.py
python3 scripts/create_label_audit.py
python3 scripts/validate_imbalance_risk.py
python3 scripts/validate_label_audit.py
python3 scripts/summarize_label_audit.py
python3 -m pytest
```

Checkpoints remain ignored. Configurations, histories, predictions, threshold
tables, PR and risk-coverage plots, domain summaries, audit state, and the main
report are versioned under `reports/milestone4/`.
