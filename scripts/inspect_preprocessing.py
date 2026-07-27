#!/usr/bin/env python3
"""Save waveform and normalization comparisons for every vessel class."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from edge_underwater.dataset import decode_audio_interval
from edge_underwater.manifest import LABELS, read_csv_rows
from edge_underwater.preprocessing import (
    PreprocessingConfig,
    TrainingStatistics,
    WaveformToLogMel,
)


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
CONFIG_FILE = PROJECT_FOLDER / "data/preprocessing/config.json"
STATISTICS_FILE = PROJECT_FOLDER / "data/preprocessing/training_statistics.json"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"
OUTPUT_FILE = PROJECT_FOLDER / "data/plots/preprocessing_visual_check.png"


def draw_spectrogram(axis, tensor, title):
    image = axis.imshow(
        tensor.squeeze(0).numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Frame")
    axis.set_ylabel("Mel band")
    return image


def main():
    config = PreprocessingConfig.load(CONFIG_FILE)
    statistics = TrainingStatistics.load(STATISTICS_FILE, config)
    rows = read_csv_rows(MANIFEST_FILE)
    transform = WaveformToLogMel(config).eval()

    representative_rows = []
    for class_name in LABELS:
        representative_rows.append(
            next(row for row in rows if row["class"] == class_name)
        )

    figure, axes = plt.subplots(len(representative_rows), 4, figsize=(18, 12))

    for row_index, row in enumerate(representative_rows):
        waveform, sample_rate = decode_audio_interval(
            AUDIO_FOLDER / row["source_file"],
            float(row["start_seconds"]),
            float(row["end_seconds"]),
        )
        prepared_waveform = transform.prepare_waveform(waveform, sample_rate)
        log_mel, _, _ = transform.extract_log_mel(waveform, sample_rate)
        per_example = transform.normalize(log_mel, "per_example")
        training_stats = transform.normalize(
            log_mel,
            "training_stats",
            statistics,
        )

        time_seconds = (
            prepared_waveform.shape[1] / config.sample_rate_hz
        )
        time_axis = (
            torch.arange(prepared_waveform.shape[1]).numpy()
            / config.sample_rate_hz
        )
        axes[row_index, 0].plot(
            time_axis,
            prepared_waveform.squeeze(0).numpy(),
            linewidth=0.5,
        )
        axes[row_index, 0].set_xlim(0, time_seconds)
        axes[row_index, 0].set_title(f"{row['class']} waveform")
        axes[row_index, 0].set_xlabel("Seconds")
        axes[row_index, 0].set_ylabel("Amplitude")
        draw_spectrogram(axes[row_index, 1], log_mel, "Log-mel")
        draw_spectrogram(axes[row_index, 2], per_example, "Per-example")
        draw_spectrogram(axes[row_index, 3], training_stats, "Training stats")

    figure.suptitle("DeepShip preprocessing visual checks")
    figure.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved visual comparison to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
