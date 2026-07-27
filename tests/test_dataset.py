from pathlib import Path

import pytest
import torch

from edge_underwater.dataset import DeepShipWindowDataset
from edge_underwater.preprocessing import PreprocessingConfig


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_FOLDER / "data/manifests/deepship_windows.csv"
AUDIO_FOLDER = PROJECT_FOLDER / "data/raw/deepship"


@pytest.mark.skipif(
    not (AUDIO_FOLDER / "Cargo/103.wav").is_file(),
    reason="DeepShip raw audio is not available",
)
def test_real_window_is_traceable():
    dataset = DeepShipWindowDataset(
        manifest_path=MANIFEST_FILE,
        audio_root=AUDIO_FOLDER,
        normalization="none",
        config=PreprocessingConfig(),
    )

    item = dataset[0]

    assert tuple(item["features"].shape) == (1, 64, 155)
    assert torch.isfinite(item["features"]).all()
    assert item["source_file"]
    assert item["window_id"]
    assert item["end_seconds"] - item["start_seconds"] == 5.0
