import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from edge_underwater.features import (
    HandcraftedFeatureConfig,
    HandcraftedFeatureExtractor,
)
from edge_underwater.preprocessing import PreprocessingConfig


@pytest.fixture(scope="module")
def feature_tools():
    preprocessing = PreprocessingConfig()
    config = HandcraftedFeatureConfig.from_preprocessing(preprocessing)
    return config, HandcraftedFeatureExtractor(config, preprocessing)


def sine_wave(frequency_hz: float, amplitude: float = 1.0) -> torch.Tensor:
    samples = torch.arange(80_000, dtype=torch.float32)
    return (amplitude * torch.sin(2 * torch.pi * frequency_hz * samples / 16_000)).view(
        1, -1
    )


def test_feature_order_count_and_determinism(feature_tools):
    config, extractor = feature_tools
    waveform = sine_wave(440.0)

    first = extractor.extract(waveform, 16_000)
    second = extractor.extract(waveform, 16_000)

    assert len(first) == 73
    assert tuple(first) == config.feature_names
    assert first == second
    assert all(math.isfinite(value) for value in first.values())


def test_sine_wave_time_and_spectral_features(feature_tools):
    _, extractor = feature_tools
    features = extractor.extract(sine_wave(1_200.0, amplitude=0.5), 16_000)

    assert features["rms"] == pytest.approx(0.5 / math.sqrt(2), rel=0.01)
    assert features["zero_crossing_rate"] == pytest.approx(2_400 / 16_000, rel=0.02)
    assert features["spectral_centroid_hz_mean"] == pytest.approx(1_200, abs=20)
    assert features["dominant_peak_1_frequency_hz"] == pytest.approx(1_200, abs=16)
    assert features["band_energy_ratio_1000_2000_hz"] > 0.95


def test_mfcc_flux_and_silence_are_finite(feature_tools):
    config, extractor = feature_tools
    silence = extractor.extract(torch.zeros(1, 80_000), 16_000)

    assert len([name for name in config.feature_names if name.startswith("mfcc_")]) == 40
    assert silence["spectral_flux_mean"] == 0
    assert silence["spectral_flux_std"] == 0
    assert silence["spectral_flux_max"] == 0
    assert silence["detected_peak_count"] == 0
    assert all(math.isfinite(value) for value in silence.values())


def test_invalid_waveform_is_rejected(feature_tools):
    _, extractor = feature_tools
    waveform = torch.zeros(1, 80_000)
    waveform[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN"):
        extractor.extract(waveform, 16_000)


def test_committed_feature_table_matches_manifest_one_to_one():
    config = HandcraftedFeatureConfig.load(
        Path("data/features/handcrafted_config.json")
    )
    features = pd.read_csv("data/features/deepship_handcrafted_features.csv")
    manifest = pd.read_csv("data/manifests/deepship_windows.csv")

    assert len(features) == len(manifest) == 1_118
    assert features["window_id"].is_unique
    assert set(features["window_id"]) == set(manifest["window_id"])
    assert tuple(features.columns[-73:]) == config.feature_names
    assert set(features["config_hash"]) == {config.preprocessing_config_hash}
    assert set(features["feature_config_hash"]) == {config.config_hash}
