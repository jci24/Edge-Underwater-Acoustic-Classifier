# Milestone 5 — Edge-Performance Evaluation

## Outcome

The unchanged Milestone 4 `class_vessel_balanced_sampling` CNN was exported
to ONNX and benchmarked at batch size one. Single-threaded ONNX Runtime is the
primary deployment condition; PyTorch eager is the reference.

| measurement | median ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| DSP preprocessing, decoded waveform | 3.674 | 4.530 | 7.246 |
| PyTorch model inference | 1.090 | 1.272 | 1.387 |
| ONNX Runtime model inference | 0.882 | 0.959 | 1.070 |
| PyTorch compute pipeline | 4.745 | 5.372 | 5.789 |
| ONNX compute pipeline | 4.492 | 5.033 | 5.915 |
| PyTorch full product pipeline | 6.954 | 10.558 | 12.076 |
| ONNX full product pipeline | 6.703 | 10.630 | 13.195 |

The DSP measurement excludes decoding and disk access. The full product
pipeline includes requested-interval decoding from a warm file cache,
preprocessing, and inference.

Secondary default-thread model-only results:

| runtime | configured threads | median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| PyTorch | 4 | 0.838 | 1.011 | 1.210 |
| ONNX Runtime | runtime default (`0`) | 0.368 | 0.554 | 0.693 |

## Engineering targets

| target | observed | limit | result |
|---|---:|---:|---|
| onnx_file_size | 116187.000 bytes | 5000000.000 bytes | PASS |
| onnx_single_thread_inference_p99 | 1.070 ms | 50.000 ms | PASS |
| onnx_full_pipeline_p99 | 13.195 ms | 500.000 ms | PASS |

## Model and export

- Parameters: 23,668
- PyTorch deployment state dict: 103,367 bytes
- ONNX model: 116,187 bytes
- Conv2d/Linear MACs: 23,989,504
- Approximate FLOPs: 47,979,008
- ONNX/PyTorch parity windows: 135/135
- Maximum absolute logit difference: 0.00000095

MACs exclude batch normalization, activation, and pooling work. Approximate
FLOPs use two operations per multiply-accumulate.

## Cold start and memory

Twenty fresh processes were measured per runtime.

| runtime | startup/import median ms | load median ms | first inference median ms | first full pipeline median ms | baseline RSS median MiB | load increase median MiB | full increase median MiB | maximum peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | 1018.9 | 2.9 | 1.807 | 6.889 | 293.5 | 1.5 | 21.9 | 315.8 |
| onnxruntime | 1111.9 | 17.7 | 1.044 | 7.268 | 275.4 | 18.6 | 39.8 | 333.6 |

RSS includes the Python process and required DSP/runtime libraries. The report
also preserves post-import baselines and load/full-pipeline increases so model
cost is not confused with the whole interpreter.

## Environment

- Machine: MacBook Air
- CPU: Apple M2
- Architecture: arm64
- CPU cores: 8 physical,
  8 logical
- RAM: 8 GB
- Operating system: macOS-26.5.2-arm64-arm-64bit-Mach-O
- Python: 3.14.6
- PyTorch / torchaudio / TorchCodec:
  2.10.0 /
  2.10.0 /
  0.10.0
- ONNX / ONNX Runtime / ONNX Script:
  1.22.0 /
  1.27.0 /
  0.7.1
- Primary threads: one PyTorch intra-op thread or one ONNX Runtime intra-op
  thread; ONNX execution is sequential with the CPU execution provider.

## Measurement limits

- Results come from the current Apple M2 development laptop, not the eventual underwater edge device.
- Default-thread results are secondary and do not replace the reproducible single-thread acceptance condition.
- Timing uses a warm model and warm file cache after 50 warm-up calls; cold-start measurements are reported separately.
- Fast execution does not repair the selected CNN's weak reused-test classification score from Milestone 4.
