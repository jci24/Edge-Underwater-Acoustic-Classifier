# Milestone 6 — ONNX Parity and Quantization Study

## Outcome

The fixed-shape FP32 ONNX model reproduced all 135 PyTorch test predictions.
Maximum absolute FP32 logit error was
`0.00000095`, within the
configured `1e-4` absolute and relative tolerances.

The validation-only deployment recommendation is
**`fp32`**.

| variant | bytes | median ms | p99 ms | validation macro F1 | reused-test macro F1 | validation agreement with FP32 | qualifies |
|---|---:|---:|---:|---:|---:|---:|---|
| fp32 | 116,187 | 0.696 | 0.836 | 0.396 | 0.031 | 1.000 | yes |
| dynamic_int8 | 114,138 | 0.708 | 0.890 | 0.396 | 0.031 | 1.000 | no |
| static_int8 | 51,877 | 0.457 | 0.625 | 0.371 | 0.031 | 0.968 | no |

Latency uses ONNX Runtime CPU, sequential execution, one intra-op thread,
batch size one, 50 warm-up calls, and 1,000 measured calls.

## Quantization scope

- Dynamic INT8 targets only the exported `Gemm`/`MatMul` head. The three
  convolution operators remain FP32.
- Static INT8 uses signed per-channel QDQ quantization for `Conv`, `Gemm`, and
  `MatMul`, calibrated with all 829 per-example-normalized training windows.
- Calibration contains no validation or test rows.

Quantization is treated as an experiment. Smaller files are not declared
better unless validation quality and measured latency also satisfy the
predefined limits.

## Trade-off interpretation

- `dynamic_int8` failed p99 latency. Its validation macro F1 loss was `0.0000`, maximum class-recall loss was `0.0000`, p99 latency change was `6.4%`, and size reduction was `1.8%`.
- `static_int8` failed validation macro F1, per-class validation recall. Its validation macro F1 loss was `0.0250`, maximum class-recall loss was `0.1111`, p99 latency change was `-25.2%`, and size reduction was `55.4%`.

## Selection policy

A quantized variant must lose no more than `0.01` validation macro F1 or
`0.05` recall in any class, must be no more than 5% slower at p99, and must be
smaller than FP32. Selection uses validation metrics only; reused-test results
do not change the recommendation.

## Limits

- The test set has already been evaluated in earlier milestones and is not a
  pristine confirmatory set.
- Results describe ONNX Runtime CPU on the current Apple M2 development
  laptop, not the eventual underwater edge device.
- No retraining, quantization-aware training, pruning, float16 conversion, or
  dynamic input shape is included.
