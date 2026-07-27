# Edge-performance evaluation

Milestone 5 benchmarks the unchanged validation-selected Milestone 4 CNN. It
does not retrain, quantize, compile, or change the learned weights.

## Setup

Install the optional edge and test dependencies:

```bash
python3 -m pip install -e '.[edge,test]'
```

The ignored selected checkpoint and raw DeepShip recordings must be available.
If the selected checkpoint is missing, regenerate Milestone 4 first:

```bash
MPLBACKEND=Agg python3 scripts/train_imbalance_risk.py
```

## Run and validate

```bash
MPLBACKEND=Agg python3 scripts/benchmark_edge_performance.py
python3 scripts/validate_edge_performance.py
python3 -m pytest
```

The benchmark creates ignored deployment files in `artifacts/models/edge/`.
Configuration, raw timing samples, cold-start samples, plots, metrics, and the
reader-facing report are versioned under `reports/milestone5/`.

## Measurement boundaries

- DSP preprocessing starts with an already-decoded waveform and ends with the
  per-example-normalized `[1, 64, 155]` tensor. It excludes decoding and disk
  access.
- Model inference starts with `[1, 1, 64, 155]` and ends with four logits.
- The compute pipeline includes DSP preprocessing and model inference but no
  decoding.
- The full product pipeline includes warm-cache interval decoding,
  preprocessing, and model inference.
- Cold start uses a fresh process for every run and records startup/import,
  runtime loading, first inference, first full-pipeline prediction, current
  RSS, and OS high-water RSS.

Single-threaded ONNX Runtime is the primary deployment condition.
Single-threaded PyTorch eager is the reference. Runtime-default thread results
are secondary and apply only to model inference.

## Interpretation limits

Latency and RSS depend on the CPU, operating system, runtime versions, system
load, and thermal state. These measurements describe the development laptop,
not the eventual underwater edge device. Fast execution also does not repair
the selected CNN's weak reused-test classification score from Milestone 4.
