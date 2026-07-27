"""On-demand DeepShip window dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from .manifest import read_csv_rows
from .preprocessing import (
    PreprocessingConfig,
    TrainingStatistics,
    WaveformToLogMel,
)


def decode_audio_interval(
    audio_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> tuple[torch.Tensor, int]:
    decoder = AudioDecoder(audio_path)
    samples = decoder.get_samples_played_in_range(
        start_seconds=start_seconds,
        stop_seconds=end_seconds,
    )
    return samples.data, samples.sample_rate


class DeepShipWindowDataset(Dataset):
    """Generate model-ready tensors from a traceable window manifest."""

    def __init__(
        self,
        manifest_path: Path,
        audio_root: Path,
        split: str | None = None,
        normalization: str = "training_stats",
        statistics: TrainingStatistics | None = None,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.config = config or PreprocessingConfig()
        self.audio_root = audio_root
        self.split = split
        self.normalization = normalization
        self.statistics = statistics
        self.transform = WaveformToLogMel(self.config).eval()
        self._decoder_path: Path | None = None
        self._decoder: AudioDecoder | None = None
        rows = read_csv_rows(manifest_path)

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if not rows:
            raise ValueError("No manifest rows matched the requested dataset.")
        if any(row["config_hash"] != self.config.config_hash for row in rows):
            raise ValueError("Manifest uses a different preprocessing config.")
        if normalization == "training_stats" and statistics is None:
            raise ValueError("training_stats normalization requires statistics.")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _decode_row(self, row: dict[str, str]) -> tuple[torch.Tensor, int]:
        audio_path = self.audio_root / row["source_file"]
        if self._decoder_path != audio_path:
            self._decoder = AudioDecoder(audio_path)
            self._decoder_path = audio_path
        if self._decoder is None:
            raise RuntimeError(f"Unable to create audio decoder for {audio_path}.")

        samples = self._decoder.get_samples_played_in_range(
            start_seconds=float(row["start_seconds"]),
            stop_seconds=float(row["end_seconds"]),
        )
        return samples.data, samples.sample_rate

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = self._decode_row(row)
        with torch.inference_mode():
            features, rms, low_level = self.transform(
                waveform,
                sample_rate,
                normalization=self.normalization,
                statistics=self.statistics,
            )

        return {
            "features": features,
            "label": int(row["label_index"]),
            "window_id": row["window_id"],
            "source_file": row["source_file"],
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "split": row["split"],
            "rms": rms,
            "low_level": low_level,
        }
