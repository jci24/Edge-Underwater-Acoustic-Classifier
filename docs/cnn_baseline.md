# Small CNN Baseline

Milestone 3 compares a compact convolutional classifier with the committed
hand-engineered baselines using exactly the same five-second windows, class
mapping, vessel-disjoint splits, and primary test metric.

## Architecture

The model accepts `[batch, 1, 64, 155]` log-mel tensors. Three blocks use a
3×3 convolution, batch normalization, ReLU, 2×2 max pooling, and dropout, with
16, 32, and 64 output channels. Adaptive average pooling reduces the final
feature map to one value per channel. Dropout and one `64 → 4` linear layer
produce the class logits.

The model has 23,668 parameters and no large fully connected layer. Milestone 3
uses no training-time augmentation.

## Controlled selection

All runs use seed 42, batch size 32, AdamW, learning rate `1e-3`, weight decay
`1e-4`, at most 50 epochs, and patience 8. Early stopping maximizes validation
macro F1; an improvement must exceed `1e-4`, and ties retain the earlier epoch.

The fixed run order is:

1. unweighted loss with training-stat normalization;
2. unweighted loss with per-example normalization; and
3. class-weighted loss with the better validation normalization.

Normalization ties prefer training statistics. Weighting ties prefer the
unweighted model. Class weights use only training counts and equal
`N / (4 × class_count)`. The selected model is not refitted on combined
training and validation data. Only the validation-selected model is evaluated
on test.

All 1,118 unnormalized log-mel tensors are decoded once into approximately
44 MiB of memory. Normalization is applied on dataset access. No tensor cache
is written to disk.

## Reproduce and validate

```bash
MPLBACKEND=Agg python3 scripts/train_cnn_baseline.py
python3 scripts/create_cnn_error_review.py
python3 scripts/validate_cnn_baseline.py
python3 scripts/validate_cnn_error_review.py
python3 -m pytest
```

The error-review generator refuses to overwrite an existing CSV because it may
contain manual notes. After listening to every selected error, validate the
controlled vocabulary with:

```bash
python3 scripts/validate_cnn_error_review.py --require-complete
```

Best `.pt` checkpoints are reproducible and remain ignored under
`artifacts/models/cnn/`. Configurations, histories, curves, predictions,
metrics, confusion matrices, and the result report are versioned.

## Interpretation limits

Window-level macro F1 is primary. Source-level aggregation is secondary because
the public test split has only one Cargo, one Passengership, and one Tug vessel
group. Windows are correlated within recordings and vessels. Development-Mac
latency is not target-device performance, and this public subset cannot
establish performance on the full DeepShip dataset.
