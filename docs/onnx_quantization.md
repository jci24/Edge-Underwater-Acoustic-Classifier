# ONNX parity and quantization study

Milestone 6 evaluates three fixed-shape exports of the unchanged
validation-selected Milestone 4 CNN:

- FP32 ONNX
- dynamic INT8 applied only to the small linear head
- calibrated static signed INT8 QDQ applied to convolution and linear operators

No model is retrained, pruned, converted to float16, or changed to accept
dynamic input shapes.

## Setup and execution

Install the edge dependencies and ensure that the ignored Milestone 4
checkpoint and raw DeepShip recordings are present:

```bash
python3 -m pip install -e '.[edge,test]'
MPLBACKEND=Agg python3 scripts/run_onnx_quantization_study.py
python3 scripts/validate_onnx_quantization.py
python3 -m pytest
```

The generated ONNX files remain ignored under `artifacts/models/onnx/`.
Configuration, predictions, metrics, confusion matrices, 1,000-call timing
samples, plots, and the report are versioned under `reports/milestone6/`.

## Data boundaries

The FP32 parity check covers every one of the 135 reused-test windows. Static
INT8 calibration uses all 829 training windows in deterministic manifest order
with per-example normalization. Validation and test rows are rejected by the
calibration reader.

All three variants are evaluated on the same 154 validation and 135 reused-test
windows. Only validation quality, latency, and file size determine the
deployment recommendation. Test metrics are reporting-only and cannot change
that decision.

## Benchmark procedure

ONNX Runtime uses the CPU provider, sequential execution, one intra-op thread,
batch size one, 50 warm-up calls, and 1,000 measured calls per model. Timing
cycles through validation inputs, starts with a ready normalized tensor, and
ends after four logits are returned; it therefore excludes audio decoding and
DSP preprocessing.

Dynamic quantization is not presumed to help this convolution-heavy model. It
targets only the exported `Gemm`/`MatMul` head, while static QDQ quantization
uses MinMax ranges, signed INT8 activations and weights, and per-channel weight
quantization for `Conv`, `Gemm`, and `MatMul`.

## Recommendation rules and limits

A quantized model qualifies only when validation macro F1 drops by at most
0.01, no validation class recall drops by more than 0.05, p99 latency is at
most 5% slower than FP32, and its file is smaller. If more than one qualifies,
the lower p99 wins, followed by smaller file size. Otherwise FP32 remains the
recommendation.

The test set has been used by earlier milestones and is not a pristine
confirmatory set. Results describe ONNX Runtime on the current Apple M2
development laptop, not the eventual underwater edge device.
