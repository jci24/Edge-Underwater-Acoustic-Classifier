import torch
from torch.utils.data import Dataset

from edge_underwater.cnn import SmallCnnConfig
from edge_underwater.cnn_training import (
    CnnRunConfig,
    CnnTrainingConfig,
    class_weights_from_training_labels,
    load_cnn_checkpoint,
    select_primary_cnn,
    select_unweighted_normalization,
    train_cnn,
)


class TinyDataset(Dataset):
    def __init__(self, split, size=8):
        self.split = split
        generator = torch.Generator().manual_seed(7 if split == "train" else 8)
        self.features = torch.randn(size, 1, 64, 155, generator=generator)
        self.labels = torch.tensor([index % 4 for index in range(size)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "split": self.split,
            "window_id": f"{self.split}-{index}",
        }


def test_class_weights_use_training_counts():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])
    weights = class_weights_from_training_labels(labels)

    torch.testing.assert_close(weights, torch.tensor([0.5, 1.0, 2.0, 2.0]))


def test_training_records_history_stops_and_restores_checkpoint(tmp_path):
    model_config = SmallCnnConfig()
    training_config = CnnTrainingConfig(
        batch_size=4,
        maximum_epochs=2,
        patience=1,
        minimum_f1_improvement=1.0,
    )
    run_config = CnnRunConfig(
        name="test",
        normalization="training_stats",
        class_weighted=False,
        preprocessing_config_hash="preprocessing",
        model_config_hash=model_config.config_hash,
        training_config_hash=training_config.config_hash,
    )
    result = train_cnn(
        TinyDataset("train"),
        TinyDataset("validation"),
        model_config,
        training_config,
        run_config,
        tmp_path / "checkpoint.pt",
    )

    assert result.best_epoch == 1
    assert result.stopped_early
    assert len(result.history) == 2
    assert set(result.history[0]) == {
        "epoch",
        "training_loss",
        "training_accuracy",
        "training_macro_f1",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "learning_rate",
        "elapsed_seconds",
    }
    loaded, checkpoint = load_cnn_checkpoint(result.checkpoint_path)
    for name, value in result.model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name])
    assert checkpoint["selection_metric"] == "validation_macro_f1"


def test_training_rejects_rows_from_wrong_split(tmp_path):
    config = SmallCnnConfig()
    training = CnnTrainingConfig(maximum_epochs=1, patience=1)
    run = CnnRunConfig(
        "bad",
        "training_stats",
        False,
        "preprocessing",
        config.config_hash,
        training.config_hash,
    )
    try:
        train_cnn(
            TinyDataset("test"),
            TinyDataset("validation"),
            config,
            training,
            run,
            tmp_path / "bad.pt",
        )
    except ValueError as error:
        assert "Training dataset" in str(error)
    else:
        raise AssertionError("Test rows were accepted as training data.")


def test_selection_ties_prefer_training_stats_then_unweighted(tmp_path):
    class Result:
        def __init__(self, normalization, weighted, score):
            self.run_config = CnnRunConfig(
                normalization,
                normalization,
                weighted,
                "preprocessing",
                "model",
                "training",
            )
            self.best_validation_macro_f1 = score

    training_stats = Result("training_stats", False, 0.5)
    per_example = Result("per_example", False, 0.5)
    weighted = Result("weighted", True, 0.5)

    assert select_unweighted_normalization(training_stats, per_example) is training_stats
    assert select_primary_cnn(training_stats, weighted) is training_stats


def test_fixed_seed_repeats_weights_and_metrics(tmp_path):
    model_config = SmallCnnConfig()
    training_config = CnnTrainingConfig(
        batch_size=4,
        maximum_epochs=1,
        patience=1,
    )
    run_config = CnnRunConfig(
        "repeat",
        "training_stats",
        False,
        "preprocessing",
        model_config.config_hash,
        training_config.config_hash,
    )
    first = train_cnn(
        TinyDataset("train"),
        TinyDataset("validation"),
        model_config,
        training_config,
        run_config,
        tmp_path / "first.pt",
    )
    second = train_cnn(
        TinyDataset("train"),
        TinyDataset("validation"),
        model_config,
        training_config,
        run_config,
        tmp_path / "second.pt",
    )

    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
    deterministic_columns = set(first.history[0]) - {"elapsed_seconds"}
    assert {
        name: first.history[0][name] for name in deterministic_columns
    } == {
        name: second.history[0][name] for name in deterministic_columns
    }
