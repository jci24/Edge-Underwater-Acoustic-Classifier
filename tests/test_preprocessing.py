import pytest
import torch

from edge_underwater.preprocessing import (
    PreprocessingConfig,
    RunningMelStatistics,
    TrainingStatistics,
    WaveformToLogMel,
)


@pytest.fixture
def config():
    return PreprocessingConfig()


@pytest.fixture
def transform(config):
    return WaveformToLogMel(config).eval()


def test_output_shape_and_determinism(config, transform):
    generator = torch.Generator().manual_seed(7)
    waveform = torch.randn(1, 160_000, generator=generator)

    with torch.inference_mode():
        first, _, _ = transform(
            waveform,
            32_000,
            normalization="per_example",
        )
        second, _, _ = transform(
            waveform,
            32_000,
            normalization="per_example",
        )

    assert tuple(first.shape) == config.output_shape == (1, 64, 155)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_stereo_is_averaged_and_resampled(config, transform):
    left = torch.ones(160_000)
    right = -torch.ones(160_000)
    stereo = torch.stack([left, right])

    prepared = transform.prepare_waveform(stereo, 32_000)

    assert prepared.shape == (1, config.window_samples)
    assert torch.count_nonzero(prepared) == 0


@pytest.mark.parametrize("amplitude", [0.0, 1e-12])
def test_silence_and_low_level_are_finite(amplitude, config, transform):
    waveform = torch.full((1, config.window_samples), amplitude)

    features, rms, low_level = transform(
        waveform,
        config.sample_rate_hz,
        normalization="per_example",
    )

    assert tuple(features.shape) == config.output_shape
    assert torch.isfinite(features).all()
    assert low_level
    assert rms < config.low_level_rms


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_invalid_waveform_values_are_rejected(invalid_value, config, transform):
    waveform = torch.zeros(1, config.window_samples)
    waveform[0, 0] = invalid_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        transform(waveform, config.sample_rate_hz, normalization="per_example")


def test_waveform_is_cropped_or_padded_to_exact_length(config, transform):
    short = torch.zeros(1, config.window_samples - 3)
    long = torch.zeros(1, config.window_samples + 3)

    assert transform.prepare_waveform(short, config.sample_rate_hz).shape[1] == 80_000
    assert transform.prepare_waveform(long, config.sample_rate_hz).shape[1] == 80_000


def test_training_statistics_are_per_mel_band(config):
    accumulator = RunningMelStatistics(config)
    first = torch.zeros(config.output_shape)
    second = torch.ones(config.output_shape)

    accumulator.update(first, split="train")
    accumulator.update(second, split="train")
    statistics = accumulator.finalize()

    assert len(statistics.mean) == config.mel_bands
    assert len(statistics.standard_deviation) == config.mel_bands
    assert statistics.window_count == 2
    assert all(value == pytest.approx(0.5) for value in statistics.mean)
    assert all(
        value == pytest.approx(0.5)
        for value in statistics.standard_deviation
    )


def test_statistics_reject_validation_and_test_tensors(config):
    tensor = torch.zeros(config.output_shape)

    for split in ("validation", "test"):
        accumulator = RunningMelStatistics(config)
        with pytest.raises(ValueError, match="training windows"):
            accumulator.update(tensor, split=split)


def test_zero_variance_statistics_use_epsilon(config):
    accumulator = RunningMelStatistics(config)
    accumulator.update(torch.zeros(config.output_shape), split="train")

    statistics = accumulator.finalize()

    assert all(
        value == pytest.approx(config.normalization_epsilon)
        for value in statistics.standard_deviation
    )


def test_training_normalization_checks_config(config, transform):
    wrong_statistics = TrainingStatistics(
        mean=tuple(0.0 for _ in range(config.mel_bands)),
        standard_deviation=tuple(1.0 for _ in range(config.mel_bands)),
        frame_count=1,
        window_count=1,
        config_hash="wrong",
    )

    with pytest.raises(ValueError, match="different preprocessing config"):
        transform.normalize(
            torch.zeros(config.output_shape),
            "training_stats",
            wrong_statistics,
        )


def test_invalid_frequency_range_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        PreprocessingConfig(maximum_frequency_hz=8_001)
