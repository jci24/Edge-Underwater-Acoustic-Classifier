"""Deterministic hand-engineered features for five-second audio windows."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchaudio
from scipy.signal import find_peaks
from torch import Tensor

from .preprocessing import PreprocessingConfig, WaveformToLogMel


DEFAULT_FREQUENCY_BANDS = (
    (10.0, 50.0),
    (50.0, 100.0),
    (100.0, 250.0),
    (250.0, 500.0),
    (500.0, 1_000.0),
    (1_000.0, 2_000.0),
    (2_000.0, 4_000.0),
    (4_000.0, 8_000.0),
)


@dataclass(frozen=True)
class HandcraftedFeatureConfig:
    """Values that define the ordered handcrafted feature vector."""

    preprocessing_config_hash: str
    sample_rate_hz: int = 16_000
    n_fft: int = 1_024
    win_length: int = 1_024
    hop_length: int = 512
    minimum_frequency_hz: float = 10.0
    maximum_frequency_hz: float = 8_000.0
    mfcc_count: int = 20
    mel_bands: int = 64
    rolloff_fraction: float = 0.85
    frequency_bands_hz: tuple[tuple[float, float], ...] = DEFAULT_FREQUENCY_BANDS
    dominant_peak_count: int = 5
    peak_prominence_db: float = 3.0
    peak_separation_hz: float = 25.0
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if not self.preprocessing_config_hash:
            raise ValueError("A preprocessing configuration hash is required.")
        if self.sample_rate_hz <= 0 or self.n_fft <= 0:
            raise ValueError("Sample rate and FFT size must be positive.")
        if self.win_length > self.n_fft or self.hop_length <= 0:
            raise ValueError("STFT settings are invalid.")
        if self.mfcc_count <= 0 or self.mel_bands < self.mfcc_count:
            raise ValueError("MFCC and mel-band counts are invalid.")
        if not 0 < self.rolloff_fraction < 1:
            raise ValueError("Roll-off fraction must be between zero and one.")
        if self.dominant_peak_count <= 0:
            raise ValueError("Dominant peak count must be positive.")
        if len(self.feature_names) != 73:
            raise ValueError("The Milestone 2 feature contract must contain 73 values.")

    @classmethod
    def from_preprocessing(
        cls,
        preprocessing: PreprocessingConfig,
    ) -> "HandcraftedFeatureConfig":
        return cls(
            preprocessing_config_hash=preprocessing.config_hash,
            sample_rate_hz=preprocessing.sample_rate_hz,
            n_fft=preprocessing.n_fft,
            win_length=preprocessing.win_length,
            hop_length=preprocessing.hop_length,
            minimum_frequency_hz=preprocessing.minimum_frequency_hz,
            maximum_frequency_hz=preprocessing.maximum_frequency_hz,
            mel_bands=preprocessing.mel_bands,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = ["rms", "crest_factor", "zero_crossing_rate"]
        for statistic in ("mean", "std"):
            for name in (
                "spectral_centroid_hz",
                "spectral_bandwidth_hz",
                "spectral_rolloff_85_hz",
                "spectral_flatness",
            ):
                names.append(f"{name}_{statistic}")
        for low, high in self.frequency_bands_hz:
            names.append(f"band_energy_ratio_{int(low)}_{int(high)}_hz")
        for number in range(1, self.dominant_peak_count + 1):
            names.extend(
                (
                    f"dominant_peak_{number}_frequency_hz",
                    f"dominant_peak_{number}_relative_power",
                )
            )
        names.append("detected_peak_count")
        names.extend(f"mfcc_{number}_mean" for number in range(1, self.mfcc_count + 1))
        names.extend(f"mfcc_{number}_std" for number in range(1, self.mfcc_count + 1))
        names.extend(("spectral_flux_mean", "spectral_flux_std", "spectral_flux_max"))
        return tuple(names)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, output_path: Path) -> None:
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        payload["feature_count"] = len(self.feature_names)
        payload["config_hash"] = self.config_hash
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "HandcraftedFeatureConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in fields}
        values["frequency_bands_hz"] = tuple(
            tuple(band) for band in values["frequency_bands_hz"]
        )
        config = cls(**values)
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("Saved feature configuration hash does not match.")
        if payload.get("feature_names") != list(config.feature_names):
            raise ValueError("Saved feature order does not match the configuration.")
        return config


class HandcraftedFeatureExtractor:
    """Extract the fixed 73-feature vector in a documented order."""

    def __init__(
        self,
        config: HandcraftedFeatureConfig,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        self.config = config
        self.preprocessing = preprocessing or PreprocessingConfig()
        if config.preprocessing_config_hash != self.preprocessing.config_hash:
            raise ValueError("Feature and preprocessing configurations do not match.")
        expected = HandcraftedFeatureConfig.from_preprocessing(self.preprocessing)
        for name in (
            "sample_rate_hz",
            "n_fft",
            "win_length",
            "hop_length",
            "minimum_frequency_hz",
            "maximum_frequency_hz",
            "mel_bands",
        ):
            if getattr(config, name) != getattr(expected, name):
                raise ValueError(f"Feature setting {name} differs from preprocessing.")

        self.waveform_preparer = WaveformToLogMel(self.preprocessing).eval()
        self.window = torch.hann_window(config.win_length)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=config.sample_rate_hz,
            n_mfcc=config.mfcc_count,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs={
                "n_fft": config.n_fft,
                "win_length": config.win_length,
                "hop_length": config.hop_length,
                "f_min": config.minimum_frequency_hz,
                "f_max": config.maximum_frequency_hz,
                "n_mels": config.mel_bands,
                "window_fn": torch.hann_window,
                "power": 2.0,
                "center": False,
                "norm": "slaney",
                "mel_scale": "slaney",
            },
        )

    def _spectrum(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        complex_spectrum = torch.stft(
            waveform.squeeze(0),
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = complex_spectrum.abs().square()
        frequencies = torch.fft.rfftfreq(
            self.config.n_fft,
            d=1.0 / self.config.sample_rate_hz,
        )
        return power, frequencies

    def _spectral_frame_features(
        self,
        power: Tensor,
        frequencies: Tensor,
    ) -> dict[str, float]:
        epsilon = self.config.epsilon
        frame_energy = power.sum(dim=0).clamp_min(epsilon)
        centroid = (power * frequencies[:, None]).sum(dim=0) / frame_energy
        bandwidth = (
            power * (frequencies[:, None] - centroid[None, :]).square()
        ).sum(dim=0).div(frame_energy).sqrt()

        cumulative_power = power.cumsum(dim=0)
        rolloff_target = frame_energy * self.config.rolloff_fraction
        rolloff_bins = (cumulative_power >= rolloff_target[None, :]).to(
            torch.int64
        ).argmax(dim=0)
        rolloff = frequencies[rolloff_bins]

        safe_power = power.clamp_min(epsilon)
        flatness = safe_power.log().mean(dim=0).exp() / safe_power.mean(dim=0)

        values: dict[str, float] = {}
        for name, feature in (
            ("spectral_centroid_hz", centroid),
            ("spectral_bandwidth_hz", bandwidth),
            ("spectral_rolloff_85_hz", rolloff),
            ("spectral_flatness", flatness),
        ):
            values[f"{name}_mean"] = float(feature.mean())
            values[f"{name}_std"] = float(feature.std(unbiased=False))
        return values

    def _band_ratios(self, power: Tensor, frequencies: Tensor) -> dict[str, float]:
        average_power = power.mean(dim=1)
        useful = (
            (frequencies >= self.config.minimum_frequency_hz)
            & (frequencies <= self.config.maximum_frequency_hz)
        )
        total = float(average_power[useful].sum())
        denominator = max(total, self.config.epsilon)
        values = {}
        final_band = len(self.config.frequency_bands_hz) - 1
        for index, (low, high) in enumerate(self.config.frequency_bands_hz):
            if index == final_band:
                selected = (frequencies >= low) & (frequencies <= high)
            else:
                selected = (frequencies >= low) & (frequencies < high)
            name = f"band_energy_ratio_{int(low)}_{int(high)}_hz"
            values[name] = float(average_power[selected].sum()) / denominator
        return values

    def _dominant_peaks(
        self,
        power: Tensor,
        frequencies: Tensor,
    ) -> dict[str, float]:
        average_power = power.mean(dim=1)
        selected = (
            (frequencies >= self.config.minimum_frequency_hz)
            & (frequencies <= self.config.maximum_frequency_hz)
        )
        selected_power = average_power[selected]
        selected_frequencies = frequencies[selected]
        log_power = 10.0 * torch.log10(selected_power.clamp_min(self.config.epsilon))
        bin_width_hz = self.config.sample_rate_hz / self.config.n_fft
        minimum_bins = max(1, math.ceil(self.config.peak_separation_hz / bin_width_hz))
        peak_indexes, _ = find_peaks(
            log_power.cpu().numpy(),
            prominence=self.config.peak_prominence_db,
            distance=minimum_bins,
        )

        candidates = sorted(
            peak_indexes.tolist(),
            key=lambda index: (-float(selected_power[index]), index),
        )
        total_power = max(float(selected_power.sum()), self.config.epsilon)
        values: dict[str, float] = {}
        for number in range(1, self.config.dominant_peak_count + 1):
            if number <= len(candidates):
                index = candidates[number - 1]
                frequency = float(selected_frequencies[index])
                relative_power = float(selected_power[index]) / total_power
            else:
                frequency = 0.0
                relative_power = 0.0
            values[f"dominant_peak_{number}_frequency_hz"] = frequency
            values[f"dominant_peak_{number}_relative_power"] = relative_power
        values["detected_peak_count"] = float(len(candidates))
        return values

    def _mfcc_features(self, waveform: Tensor) -> dict[str, float]:
        coefficients = self.mfcc(waveform).squeeze(0)
        values = {}
        for index, value in enumerate(coefficients.mean(dim=1), start=1):
            values[f"mfcc_{index}_mean"] = float(value)
        for index, value in enumerate(
            coefficients.std(dim=1, unbiased=False),
            start=1,
        ):
            values[f"mfcc_{index}_std"] = float(value)
        return values

    def _flux_features(self, power: Tensor) -> dict[str, float]:
        normalized = power / power.sum(dim=0, keepdim=True).clamp_min(
            self.config.epsilon
        )
        differences = normalized[:, 1:] - normalized[:, :-1]
        flux = differences.square().sum(dim=0).sqrt()
        if flux.numel() == 0:
            flux = torch.zeros(1, dtype=power.dtype)
        return {
            "spectral_flux_mean": float(flux.mean()),
            "spectral_flux_std": float(flux.std(unbiased=False)),
            "spectral_flux_max": float(flux.max()),
        }

    def extract(self, waveform: Tensor, sample_rate: int) -> dict[str, float]:
        """Return one finite feature mapping in ``config.feature_names`` order."""

        with torch.inference_mode():
            prepared = self.waveform_preparer.prepare_waveform(waveform, sample_rate)
            rms = float(prepared.square().mean().sqrt())
            crest_factor = float(prepared.abs().max()) / max(rms, self.config.epsilon)
            samples = prepared.squeeze(0)
            zero_crossings = ((samples[:-1] * samples[1:]) < 0).to(torch.float32)

            values = {
                "rms": rms,
                "crest_factor": crest_factor,
                "zero_crossing_rate": float(zero_crossings.mean()),
            }
            power, frequencies = self._spectrum(prepared)
            values.update(self._spectral_frame_features(power, frequencies))
            values.update(self._band_ratios(power, frequencies))
            values.update(self._dominant_peaks(power, frequencies))
            values.update(self._mfcc_features(prepared))
            values.update(self._flux_features(power))

        if set(values) != set(self.config.feature_names):
            missing = set(self.config.feature_names) - set(values)
            extra = set(values) - set(self.config.feature_names)
            raise RuntimeError(f"Feature schema mismatch; missing={missing}, extra={extra}.")
        ordered = {name: values[name] for name in self.config.feature_names}
        if not all(math.isfinite(value) for value in ordered.values()):
            raise ValueError("Extracted features contain NaN or infinite values.")
        return ordered
