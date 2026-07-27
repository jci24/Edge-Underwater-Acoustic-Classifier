# Waveform-to-spectrogram preprocessing

## Operating task

Each example represents one non-overlapping five-second interval from one
DeepShip recording. The model will produce one vessel-class prediction for
each interval.

The fixed class mapping is:

| Class | Label |
|---|---:|
| Cargo | 0 |
| Passengership | 1 |
| Tanker | 2 |
| Tug | 3 |

Incomplete final intervals are excluded rather than padded. The public subset
produces 1,118 complete windows and excludes 133 trailing seconds.

## Signal pipeline

The deterministic evaluation pipeline is:

```text
source interval
→ float32 waveform
→ channel mean
→ 16 kHz band-limited resampling
→ DC removal
→ exactly 80,000 samples
→ Hann STFT
→ power mel spectrogram
→ fixed-reference log power
→ normalization
→ [1, 64, 155] float32 tensor
```

Configuration:

| Setting | Value |
|---|---:|
| Sample rate | 16,000 Hz |
| Window | 5 seconds / 80,000 samples |
| FFT and window length | 1,024 samples |
| Hop | 512 samples |
| Mel bands | 64 |
| Frequency range | 10–8,000 Hz |
| Mel scale and normalization | Slaney |
| STFT centering | Disabled |
| Power | 2 |
| Log floor | `1e-10` |

A survey of 2,520 evenly distributed source segments found approximately
0.058% of total power above 8 kHz. This supports 16 kHz as the initial target
sample rate for the public subset. It does not prove that higher frequencies
will be irrelevant for the complete DeepShip dataset.

Waveforms are mean-centered, but they are not peak- or RMS-normalized. This
preserves relative level information. Inputs with an RMS below `1e-8` are
flagged as low-level but remain valid examples.

## Normalization

Two modes have the same tensor shape:

- `per_example` uses one mean and standard deviation across the example.
- `training_stats` uses one mean and standard deviation per mel band,
  calculated only from training windows.

Standard deviations are floored at `1e-6`. The training-statistics artifact
contains 64 means, 64 standard deviations, the contributing frame/window
counts, and the preprocessing configuration hash. `training_stats` is the
default model input; model evaluation must still compare both modes before
claiming that one is superior.

## Splits and traceability

Normalized vessel names are assigned within each class using a deterministic
70/15/15 policy and seed 42. Each class has at least one vessel in every split.
A vessel is rejected if it maps to conflicting classes, and validation fails
if a vessel appears in more than one split.

`data/manifests/deepship_windows.csv` records the source file, vessel/session
identifiers, class and label, split, sample rate, half-open start/end times, and
configuration hash for every window. Tensors are generated on demand and are
not duplicated on disk.

## Reproducibility boundary

Repeated CPU evaluation is bit-identical in the pinned local environment.
PyTorch does not promise identical results across releases, platforms, or CPU
and GPU backends, so package versions and preprocessing artifacts must remain
paired with experimental results.

## Milestone 1 validation

Validation against the public subset produced:

- 1,118 complete windows: 829 train, 154 validation, and 135 test
- 459 Cargo, 221 Passengership, 320 Tanker, and 118 Tug windows
- 43 vessel groups with no group appearing in multiple splits
- 829 training windows and 128,495 frames per mel band used for statistics
- zero NaN/Inf tensors and zero windows below the low-level threshold
- bit-identical repeated CPU evaluation
- 18 passing automated tests

The visual check covered one waveform and all three spectrogram states for
every class. It showed coherent time-frequency structure and no obvious
preprocessing failure. This is pipeline validation, not evidence of classifier
performance.
