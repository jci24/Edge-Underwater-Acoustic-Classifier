import torch

from edge_underwater.cnn_data import PreloadedLogMelStore
from edge_underwater.preprocessing import PreprocessingConfig, TrainingStatistics


def make_store(config):
    rows = []
    for index, split in enumerate(("train", "train", "validation", "test")):
        rows.append(
            {
                "window_id": f"window-{index}",
                "source_file": f"Cargo/{index}.wav",
                "class": "Cargo",
                "label_index": 0,
                "vessel_name": f"vessel {index}",
                "vessel_group": f"vessel-{index}",
                "session_identifier": f"session-{index}",
                "split": split,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "config_hash": config.config_hash,
            }
        )
    features = torch.stack(
        [
            torch.zeros(config.output_shape),
            torch.ones(config.output_shape),
            torch.full(config.output_shape, 2.0),
            torch.full(config.output_shape, 3.0),
        ]
    )
    return PreloadedLogMelStore(
        features=features,
        labels=torch.zeros(4, dtype=torch.long),
        rows=rows,
        config_hash=config.config_hash,
    )


def make_statistics(config):
    return TrainingStatistics(
        mean=tuple(0.5 for _ in range(config.mel_bands)),
        standard_deviation=tuple(0.5 for _ in range(config.mel_bands)),
        frame_count=310,
        window_count=2,
        config_hash=config.config_hash,
    )


def test_preloaded_store_normalizes_on_access_and_preserves_traceability():
    config = PreprocessingConfig()
    store = make_store(config)
    statistics = make_statistics(config)

    training_stats = store.subset("train", "training_stats", config, statistics)
    per_example = store.subset("validation", "per_example", config, statistics)

    assert len(training_stats) == 2
    assert training_stats[0]["window_id"] == "window-0"
    assert training_stats[0]["split"] == "train"
    assert torch.all(training_stats[0]["features"] == -1)
    assert torch.all(per_example[0]["features"] == 0)
    assert store.memory_bytes == store.features.nelement() * 4


def test_store_rejects_duplicate_window_ids():
    config = PreprocessingConfig()
    store = make_store(config)
    store.rows[1]["window_id"] = store.rows[0]["window_id"]

    try:
        store.__post_init__()
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Duplicate IDs were accepted.")
