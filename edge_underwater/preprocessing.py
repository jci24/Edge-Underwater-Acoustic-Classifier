"""Deterministic waveform-to-log-mel preprocessing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchaudio
from torch import Tensor, nn


@dataclass(frozen=True)
class PreprocessingConfig:
    """All values that define the model input tensor."""

    sample_rate_hz: int = 16_000
    window_seconds: float = 5.0
    n_fft: int = 1_024
    win_length: int = 1_024
    hop_length: int = 512
    mel_bands: int = 64
    minimum_frequency_hz: float = 10.0
    maximum_frequency_hz: float = 8_000.0
    power: float = 2.0
    log_floor: float = 1e-10
    normalization_epsilon: float = 1e-6
    low_level_rms: float = 1e-8

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        if self.window_seconds <= 0:
            raise ValueError("Window duration must be positive.")
        if self.n_fft <= 0 or self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("STFT sizes must be positive.")
        if self.win_length > self.n_fft:
            raise ValueError("Window length cannot exceed FFT size.")
        if self.window_samples < self.n_fft:
            raise ValueError("Audio window must be at least one FFT frame.")
        if self.mel_bands <= 0:
            raise ValueError("Mel band count must be positive.")
        if not 0 <= self.minimum_frequency_hz < self.maximum_frequency_hz:
            raise ValueError("Frequency range must be increasing and non-negative.")
        if self.maximum_frequency_hz > self.sample_rate_hz / 2:
            raise ValueError("Maximum frequency cannot exceed Nyquist.")

    @property
    def window_samples(self) -> int:
        return round(self.sample_rate_hz * self.window_seconds)

    @property
    def output_frames(self) -> int:
        return 1 + (self.window_samples - self.n_fft) // self.hop_length

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.mel_bands, self.output_frames)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, output_path: Path) -> None:
        output = asdict(self)
        output["window_samples"] = self.window_samples
        output["output_shape"] = list(self.output_shape)
        output["config_hash"] = self.config_hash
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "PreprocessingConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        config_fields = cls.__dataclass_fields__
        config = cls(
            **{
                key: value
                for key, value in payload.items()
                if key in config_fields
            }
        )
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("Saved preprocessing config hash does not match its values.")
        return config


@dataclass(frozen=True)
class TrainingStatistics:
    """Per-mel-band mean and standard deviation fitted on training windows."""

    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    frame_count: int
    window_count: int
    config_hash: str

    def validate(self, config: PreprocessingConfig) -> None:
        if self.config_hash != config.config_hash:
            raise ValueError("Training statistics use a different preprocessing config.")
        if len(self.mean) != config.mel_bands:
            raise ValueError(f"Expected {config.mel_bands} training means.")
        if len(self.standard_deviation) != config.mel_bands:
            raise ValueError(f"Expected {config.mel_bands} training standard deviations.")
        if not all(math.isfinite(value) for value in self.mean):
            raise ValueError("Training means must be finite.")
        if not all(math.isfinite(value) for value in self.standard_deviation):
            raise ValueError("Training standard deviations must be finite.")
        if any(value <= 0 for value in self.standard_deviation):
            raise ValueError("Training standard deviations must be positive.")

    def save(self, output_path: Path) -> None:
        payload = {
            "mean": list(self.mean),
            "standard_deviation": list(self.standard_deviation),
            "frame_count": self.frame_count,
            "window_count": self.window_count,
            "config_hash": self.config_hash,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        input_path: Path,
        config: PreprocessingConfig,
    ) -> "TrainingStatistics":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        statistics = cls(
            mean=tuple(payload["mean"]),
            standard_deviation=tuple(payload["standard_deviation"]),
            frame_count=int(payload["frame_count"]),
            window_count=int(payload["window_count"]),
            config_hash=payload["config_hash"],
        )
        statistics.validate(config)
        return statistics


class RunningMelStatistics:
    """Accumulate training-only mel statistics without storing all tensors."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.total = torch.zeros(config.mel_bands, dtype=torch.float64)
        self.total_squared = torch.zeros(config.mel_bands, dtype=torch.float64)
        self.frame_count = 0
        self.window_count = 0

    def update(self, log_mel: Tensor, split: str) -> None:
        if split != "train":
            raise ValueError("Training statistics may only use training windows.")
        if tuple(log_mel.shape) != self.config.output_shape:
            raise ValueError(
                f"Expected tensor shape {self.config.output_shape}, "
                f"received {tuple(log_mel.shape)}."
            )
        if not torch.isfinite(log_mel).all():
            raise ValueError("Cannot fit statistics from NaN or infinite values.")

        values = log_mel.squeeze(0).to(dtype=torch.float64)
        self.total += values.sum(dim=1)
        self.total_squared += values.square().sum(dim=1)
        self.frame_count += values.shape[1]
        self.window_count += 1

    def finalize(self) -> TrainingStatistics:
        if self.window_count == 0:
            raise ValueError("No training windows were provided.")

        mean = self.total / self.frame_count
        variance = self.total_squared / self.frame_count - mean.square()
        standard_deviation = variance.clamp_min(0).sqrt()
        standard_deviation = standard_deviation.clamp_min(
            self.config.normalization_epsilon
        )

        return TrainingStatistics(
            mean=tuple(mean.tolist()),
            standard_deviation=tuple(standard_deviation.tolist()),
            frame_count=self.frame_count,
            window_count=self.window_count,
            config_hash=self.config.config_hash,
        )


class WaveformToLogMel(nn.Module):
    """Convert one audio interval into a fixed model-ready tensor."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        super().__init__()
        self.config = config or PreprocessingConfig()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate_hz,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.minimum_frequency_hz,
            f_max=self.config.maximum_frequency_hz,
            n_mels=self.config.mel_bands,
            window_fn=torch.hann_window,
            power=self.config.power,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        )

    def prepare_waveform(self, waveform: Tensor, input_sample_rate: int) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("Waveform must have shape [channels, samples].")
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise ValueError("Waveform cannot be empty.")
        if input_sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if not torch.isfinite(waveform).all():
            raise ValueError("Waveform contains NaN or infinite values.")

        waveform = waveform.to(dtype=torch.float32).mean(dim=0, keepdim=True)

        if input_sample_rate != self.config.sample_rate_hz:
            waveform = torchaudio.functional.resample(
                waveform,
                input_sample_rate,
                self.config.sample_rate_hz,
                resampling_method="sinc_interp_hann",
            )

        waveform = waveform - waveform.mean()
        expected_samples = self.config.window_samples

        if waveform.shape[1] > expected_samples:
            waveform = waveform[:, :expected_samples]
        elif waveform.shape[1] < expected_samples:
            missing_samples = expected_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, missing_samples))

        return waveform.contiguous()

    def extract_log_mel(
        self,
        waveform: Tensor,
        input_sample_rate: int,
    ) -> tuple[Tensor, float, bool]:
        waveform = self.prepare_waveform(waveform, input_sample_rate)
        rms = float(waveform.square().mean().sqrt().item())
        low_level = rms < self.config.low_level_rms
        power_spectrogram = self.mel_spectrogram(waveform)
        log_mel = 10.0 * torch.log10(
            power_spectrogram.clamp_min(self.config.log_floor)
        )

        if tuple(log_mel.shape) != self.config.output_shape:
            raise ValueError(
                f"Expected output shape {self.config.output_shape}, "
                f"received {tuple(log_mel.shape)}."
            )
        if not torch.isfinite(log_mel).all():
            raise ValueError("Log-mel tensor contains NaN or infinite values.")

        return log_mel, rms, low_level

    def normalize(
        self,
        log_mel: Tensor,
        mode: str,
        statistics: TrainingStatistics | None = None,
    ) -> Tensor:
        if mode == "none":
            return log_mel

        if mode == "per_example":
            mean = log_mel.mean()
            standard_deviation = log_mel.std(unbiased=False).clamp_min(
                self.config.normalization_epsilon
            )
            return (log_mel - mean) / standard_deviation

        if mode == "training_stats":
            if statistics is None:
                raise ValueError("training_stats normalization requires statistics.")
            statistics.validate(self.config)
            mean = torch.tensor(
                statistics.mean,
                dtype=log_mel.dtype,
                device=log_mel.device,
            ).view(1, -1, 1)
            standard_deviation = torch.tensor(
                statistics.standard_deviation,
                dtype=log_mel.dtype,
                device=log_mel.device,
            ).view(1, -1, 1)
            return (log_mel - mean) / standard_deviation

        raise ValueError(f"Unknown normalization mode: {mode}")

    def forward(
        self,
        waveform: Tensor,
        input_sample_rate: int,
        normalization: str = "training_stats",
        statistics: TrainingStatistics | None = None,
    ) -> tuple[Tensor, float, bool]:
        log_mel, rms, low_level = self.extract_log_mel(
            waveform,
            input_sample_rate,
        )
        features = self.normalize(log_mel, normalization, statistics)

        if not torch.isfinite(features).all():
            raise ValueError("Normalized tensor contains NaN or infinite values.")

        return features, rms, low_level
