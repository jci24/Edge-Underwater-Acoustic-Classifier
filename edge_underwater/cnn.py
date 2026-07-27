"""Compact convolutional model for DeepShip log-mel windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SmallCnnConfig:
    """Architecture values that define the compact CNN."""

    input_channels: int = 1
    input_mel_bands: int = 64
    input_frames: int = 155
    convolution_channels: tuple[int, int, int] = (16, 32, 64)
    block_dropout: tuple[float, float, float] = (0.10, 0.15, 0.20)
    head_dropout: float = 0.25
    class_count: int = 4
    kernel_size: int = 3
    pooling_size: int = 2

    def __post_init__(self) -> None:
        if self.input_channels != 1:
            raise ValueError("Milestone 3 expects one input channel.")
        if self.input_mel_bands <= 0 or self.input_frames <= 0:
            raise ValueError("Input dimensions must be positive.")
        if len(self.convolution_channels) != 3 or len(self.block_dropout) != 3:
            raise ValueError("SmallCnn must contain exactly three convolution blocks.")
        if any(channel <= 0 for channel in self.convolution_channels):
            raise ValueError("Convolution channels must be positive.")
        if any(not 0 <= value < 1 for value in (*self.block_dropout, self.head_dropout)):
            raise ValueError("Dropout values must be in [0, 1).")
        if self.class_count != 4:
            raise ValueError("DeepShip has exactly four classes.")

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.input_channels, self.input_mel_bands, self.input_frames)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, output_path: Path) -> None:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["config_hash"] = self.config_hash
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, input_path: Path) -> "SmallCnnConfig":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        values = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__
        }
        values["convolution_channels"] = tuple(values["convolution_channels"])
        values["block_dropout"] = tuple(values["block_dropout"])
        config = cls(**values)
        if payload.get("config_hash") != config.config_hash:
            raise ValueError("Saved CNN configuration hash does not match.")
        return config


class ConvolutionBlock(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float,
        config: SmallCnnConfig,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=config.kernel_size,
                padding=config.kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(config.pooling_size),
            nn.Dropout2d(dropout),
        )


class SmallCnn(nn.Module):
    """Three convolution blocks, global pooling, and a four-class head."""

    def __init__(self, config: SmallCnnConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallCnnConfig()
        channels = (self.config.input_channels, *self.config.convolution_channels)
        self.blocks = nn.Sequential(
            *[
                ConvolutionBlock(
                    channels[index],
                    channels[index + 1],
                    self.config.block_dropout[index],
                    self.config,
                )
                for index in range(3)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.head_dropout),
            nn.Linear(self.config.convolution_channels[-1], self.config.class_count),
        )

    def _validate_features(self, features: Tensor) -> None:
        if features.ndim != 4:
            raise ValueError("CNN input must have shape [batch, channel, mel, frame].")
        if tuple(features.shape[1:]) != self.config.input_shape:
            raise ValueError(
                f"Expected input shape [batch, {self.config.input_shape}], "
                f"received {tuple(features.shape)}."
            )
        if (
            not torch.compiler.is_exporting()
            and not torch.isfinite(features).all()
        ):
            raise ValueError("CNN input contains NaN or infinite values.")

    def extract_embedding(self, features: Tensor) -> Tensor:
        """Return the deterministic pooled 64-value representation."""

        self._validate_features(features)
        pooled = self.global_pool(self.blocks(features))
        return torch.flatten(pooled, start_dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.head(self.extract_embedding(features))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
