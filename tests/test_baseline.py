import numpy as np
import pandas as pd
import pytest

from edge_underwater.baseline import (
    ORDERED_CLASSES,
    assert_probability_contract,
    classification_metrics,
    logistic_pipeline,
    training_sample_weights,
    validate_vessel_splits,
)
from edge_underwater.features import HandcraftedFeatureConfig
from edge_underwater.preprocessing import PreprocessingConfig


@pytest.fixture
def feature_config():
    return HandcraftedFeatureConfig.from_preprocessing(PreprocessingConfig())


@pytest.fixture
def model_table(feature_config):
    random = np.random.default_rng(42)
    rows = []
    split_sizes = {"train": 12, "validation": 4, "test": 4}
    for label, class_name in enumerate(ORDERED_CLASSES):
        for split, size in split_sizes.items():
            for index in range(size):
                row = {
                    "window_id": f"{split}-{label}-{index}",
                    "source_file": f"{class_name}/{split}-{index // 2}.wav",
                    "class": class_name,
                    "label_index": label,
                    "vessel_group": f"{class_name}-{split}-{index // 4}",
                    "split": split,
                    "start_seconds": float(index * 5),
                    "end_seconds": float(index * 5 + 5),
                }
                values = random.normal(size=len(feature_config.feature_names))
                values[label] += 4
                row.update(dict(zip(feature_config.feature_names, values, strict=True)))
                rows.append(row)
    return pd.DataFrame(rows)


def test_training_weights_balance_classes_and_vessels(model_table):
    training = model_table.loc[model_table["split"] == "train"]
    weights = training_sample_weights(training)
    weighted = training.assign(weight=weights)

    class_totals = weighted.groupby("label_index")["weight"].sum()
    vessel_totals = weighted.groupby(["label_index", "vessel_group"])["weight"].sum()

    assert class_totals.max() == pytest.approx(class_totals.min())
    for _, totals in vessel_totals.groupby(level=0):
        assert totals.max() == pytest.approx(totals.min())


def test_pipeline_fit_is_unchanged_by_non_training_values(model_table, feature_config):
    training = model_table.loc[model_table["split"] == "train"]
    weights = training_sample_weights(training)
    x = training.loc[:, feature_config.feature_names]
    y = training["label_index"]
    first = logistic_pipeline(0.1).fit(x, y, classifier__sample_weight=weights)

    changed = model_table.copy()
    changed.loc[changed["split"] != "train", feature_config.feature_names] = 1e12
    changed_training = changed.loc[changed["split"] == "train"]
    second = logistic_pipeline(0.1).fit(
        changed_training.loc[:, feature_config.feature_names],
        changed_training["label_index"],
        classifier__sample_weight=training_sample_weights(changed_training),
    )

    np.testing.assert_array_equal(
        first.named_steps["variance"].get_support(),
        second.named_steps["variance"].get_support(),
    )
    np.testing.assert_array_equal(
        first.named_steps["scaler"].mean_,
        second.named_steps["scaler"].mean_,
    )
    np.testing.assert_array_equal(
        first.named_steps["classifier"].coef_,
        second.named_steps["classifier"].coef_,
    )


def test_probabilities_and_metrics_have_fixed_four_class_contract(
    model_table,
    feature_config,
):
    training = model_table.loc[model_table["split"] == "train"]
    test = model_table.loc[model_table["split"] == "test"]
    model = logistic_pipeline(0.1).fit(
        training.loc[:, feature_config.feature_names],
        training["label_index"],
        classifier__sample_weight=training_sample_weights(training),
    )
    probabilities = model.predict_proba(test.loc[:, feature_config.feature_names])
    repeated = model.predict_proba(test.loc[:, feature_config.feature_names])

    assert_probability_contract(model, probabilities)
    np.testing.assert_array_equal(probabilities, repeated)
    metrics = classification_metrics(
        test["label_index"].to_numpy(),
        model.predict(test.loc[:, feature_config.feature_names]),
    )
    assert np.asarray(metrics["confusion_matrix"]).shape == (4, 4)
    assert set(metrics["per_class"]) == set(ORDERED_CLASSES)
    assert sum(value["support"] for value in metrics["per_class"].values()) == len(test)


def test_vessels_are_disjoint_and_training_guard_rejects_other_splits(model_table):
    validate_vessel_splits(model_table)
    with pytest.raises(ValueError, match="training"):
        training_sample_weights(model_table)
