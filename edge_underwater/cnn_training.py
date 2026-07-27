"""Deterministic training utilities for the small CNN."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .cnn import SmallCnn, SmallCnnConfig


@dataclass(frozen=True)
class CnnTrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 50
    patience: int = 8
    minimum_f1_improvement: float = 1e-4
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.maximum_epochs <= 0 or self.patience <= 0:
            raise ValueError("Batch size, epochs, and patience must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Optimizer values are invalid.")
        if self.num_workers != 0:
            raise ValueError("Milestone 3 requires num_workers=0 for determinism.")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CnnRunConfig:
    name: str
    normalization: str
    class_weighted: bool
    preprocessing_config_hash: str
    model_config_hash: str
    training_config_hash: str

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class TrainingResult:
    model: SmallCnn
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_macro_f1: float
    stopped_early: bool
    checkpoint_path: Path
    run_config: CnnRunConfig


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def class_weights_from_training_labels(labels: Tensor, class_count: int = 4) -> Tensor:
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("Training labels must be a non-empty vector.")
    counts = torch.bincount(labels.to(torch.long), minlength=class_count)
    if len(counts) != class_count or (counts == 0).any():
        raise ValueError("Training labels must contain every class.")
    return labels.numel() / (class_count * counts.to(torch.float32))


def make_weighted_sampler(
    sampling_weights: Tensor,
    sample_count: int,
    seed: int,
) -> WeightedRandomSampler:
    if sampling_weights.ndim != 1 or len(sampling_weights) != sample_count:
        raise ValueError("Sampling weights must align with the requested sample count.")
    if not torch.isfinite(sampling_weights).all() or (sampling_weights <= 0).any():
        raise ValueError("Sampling weights must be finite and positive.")
    return WeightedRandomSampler(
        sampling_weights.to(torch.float64),
        num_samples=sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _loader(
    dataset: Dataset,
    config: CnnTrainingConfig,
    shuffle: bool,
    sampling_weights: Tensor | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if sampling_weights is not None:
        if shuffle:
            raise ValueError("Weighted sampling and shuffle cannot both be enabled.")
        sampler = make_weighted_sampler(
            sampling_weights,
            sample_count=len(dataset),
            seed=config.seed,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        generator=generator,
    )


def _run_epoch(
    model: SmallCnn,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: AdamW | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            features = batch["features"].to(torch.float32)
            targets = batch["label"].to(torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())

    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    return {
        "loss": total_loss / len(labels),
        "accuracy": float((label_array == prediction_array).mean()),
        "macro_f1": float(
            f1_score(label_array, prediction_array, labels=(0, 1, 2, 3), average="macro")
        ),
    }


def train_cnn(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    model_config: SmallCnnConfig,
    training_config: CnnTrainingConfig,
    run_config: CnnRunConfig,
    checkpoint_path: Path,
    sampling_weights: Tensor | None = None,
) -> TrainingResult:
    if any(
        str(training_dataset[index]["split"]) != "train"
        for index in range(len(training_dataset))
    ):
        raise ValueError("Training dataset contains non-training rows.")
    if any(
        str(validation_dataset[index]["split"]) != "validation"
        for index in range(len(validation_dataset))
    ):
        raise ValueError("Validation dataset contains non-validation rows.")

    set_deterministic_seed(training_config.seed)
    model = SmallCnn(model_config)
    if sampling_weights is not None:
        if len(sampling_weights) != len(training_dataset):
            raise ValueError("Sampling weights must align with training rows.")
        if not torch.isfinite(sampling_weights).all() or (sampling_weights <= 0).any():
            raise ValueError("Sampling weights must be finite and positive.")
    training_loader = _loader(
        training_dataset,
        training_config,
        shuffle=sampling_weights is None,
        sampling_weights=sampling_weights,
    )
    validation_loader = _loader(validation_dataset, training_config, shuffle=False)
    if run_config.class_weighted:
        labels = torch.tensor(
            [int(training_dataset[index]["label"]) for index in range(len(training_dataset))]
        )
        class_weights = class_weights_from_training_labels(labels)
    else:
        class_weights = None
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    best_model_state: dict[str, Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    stopped_early = False

    for epoch in range(1, training_config.maximum_epochs + 1):
        started = time.perf_counter()
        training_metrics = _run_epoch(
            model,
            training_loader,
            loss_function,
            optimizer,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            loss_function,
            optimizer=None,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_metrics["loss"],
                "training_accuracy": training_metrics["accuracy"],
                "training_macro_f1": training_metrics["macro_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        improved = (
            validation_metrics["macro_f1"]
            > best_macro_f1 + training_config.minimum_f1_improvement
        )
        if improved:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= training_config.patience:
            stopped_early = True
            break

    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    model.load_state_dict(best_model_state)
    checkpoint = {
        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "run_config": asdict(run_config),
        "model_config_hash": model_config.config_hash,
        "training_config_hash": training_config.config_hash,
        "run_config_hash": run_config.config_hash,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "class_names": ["Cargo", "Passengership", "Tanker", "Tug"],
        "selection_metric": "validation_macro_f1",
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_macro_f1=best_macro_f1,
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
        run_config=run_config,
    )


def load_cnn_checkpoint(checkpoint_path: Path) -> tuple[SmallCnn, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_values = dict(checkpoint["model_config"])
    model_values["convolution_channels"] = tuple(model_values["convolution_channels"])
    model_values["block_dropout"] = tuple(model_values["block_dropout"])
    model_config = SmallCnnConfig(**model_values)
    if checkpoint["model_config_hash"] != model_config.config_hash:
        raise ValueError("Checkpoint model configuration hash does not match.")
    model = SmallCnn(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def select_unweighted_normalization(
    training_stats_result: TrainingResult,
    per_example_result: TrainingResult,
) -> TrainingResult:
    """Select by validation macro F1, preferring training stats on a tie."""

    if training_stats_result.run_config.class_weighted:
        raise ValueError("training_stats candidate must be unweighted.")
    if per_example_result.run_config.class_weighted:
        raise ValueError("per_example candidate must be unweighted.")
    if training_stats_result.run_config.normalization != "training_stats":
        raise ValueError("First candidate must use training_stats.")
    if per_example_result.run_config.normalization != "per_example":
        raise ValueError("Second candidate must use per_example.")
    if (
        per_example_result.best_validation_macro_f1
        > training_stats_result.best_validation_macro_f1
    ):
        return per_example_result
    return training_stats_result


def select_primary_cnn(
    unweighted_result: TrainingResult,
    weighted_result: TrainingResult,
) -> TrainingResult:
    """Select by validation macro F1, preferring unweighted on a tie."""

    if unweighted_result.run_config.class_weighted:
        raise ValueError("First primary candidate must be unweighted.")
    if not weighted_result.run_config.class_weighted:
        raise ValueError("Second primary candidate must be class weighted.")
    if (
        weighted_result.best_validation_macro_f1
        > unweighted_result.best_validation_macro_f1
    ):
        return weighted_result
    return unweighted_result
