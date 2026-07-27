import pytest
import torch
from torch import nn

from edge_underwater.cnn import SmallCnn, SmallCnnConfig
from edge_underwater.cnn_training import load_cnn_checkpoint


def test_small_cnn_shape_size_and_global_pooling():
    model = SmallCnn().eval()
    features = torch.zeros(2, 1, 64, 155)

    logits = model(features)

    assert logits.shape == (2, 4)
    assert model.parameter_count == 23_668
    assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)
    assert torch.isfinite(logits).all()


def test_small_cnn_embedding_is_finite_deterministic_and_64_values():
    model = SmallCnn().eval()
    features = torch.randn(3, 1, 64, 155)

    with torch.inference_mode():
        first = model.extract_embedding(features)
        second = model.extract_embedding(features)

    assert first.shape == (3, 64)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_small_cnn_evaluation_is_deterministic():
    torch.manual_seed(42)
    model = SmallCnn().eval()
    features = torch.randn(2, 1, 64, 155)

    with torch.inference_mode():
        first = model(features)
        second = model(features)

    assert torch.equal(first, second)


def test_small_cnn_rejects_wrong_shape_and_invalid_values():
    model = SmallCnn()
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.zeros(1, 1, 32, 155))

    features = torch.zeros(1, 1, 64, 155)
    features[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(features)


def test_checkpoint_round_trip(tmp_path):
    config = SmallCnnConfig()
    model = SmallCnn(config).eval()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            **config.__dict__,
        },
        "model_config_hash": config.config_hash,
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)

    loaded, _ = load_cnn_checkpoint(path)
    features = torch.randn(1, 1, 64, 155)
    with torch.inference_mode():
        assert torch.equal(model(features), loaded(features))
