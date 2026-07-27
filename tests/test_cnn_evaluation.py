import pandas as pd
import pytest

from edge_underwater.cnn_evaluation import (
    select_error_review_rows,
    validate_error_review,
    write_error_review,
)


def make_predictions():
    rows = []
    for index in range(30):
        true_label = index % 4
        predicted = (true_label + 1 + index // 12) % 4
        probabilities = [0.05, 0.05, 0.05, 0.05]
        probabilities[predicted] = 0.75 + index / 1000
        probabilities[true_label] = 1 - sum(probabilities)
        rows.append(
            {
                "window_id": f"window-{index:02d}",
                "source_file": f"Class/{index % 6}.wav",
                "class": ("Cargo", "Passengership", "Tanker", "Tug")[true_label],
                "label_index": true_label,
                "predicted_class": (
                    "Cargo",
                    "Passengership",
                    "Tanker",
                    "Tug",
                )[predicted],
                "predicted_label_index": predicted,
                "start_seconds": float(index * 5),
                "end_seconds": float(index * 5 + 5),
                "probability_Cargo": probabilities[0],
                "probability_Passengership": probabilities[1],
                "probability_Tanker": probabilities[2],
                "probability_Tug": probabilities[3],
            }
        )
    return pd.DataFrame(rows)


def test_error_review_selection_is_deterministic_and_diverse():
    predictions = make_predictions()
    first = select_error_review_rows(predictions)
    second = select_error_review_rows(predictions)

    assert first["window_id"].tolist() == second["window_id"].tolist()
    assert len(first) == 20
    assert first["window_id"].is_unique
    assert first["class"].nunique() == 4
    assert first["source_file"].nunique() == 6
    assert (first["class"] != first["predicted_class"]).all()


def test_error_review_is_non_overwriting_and_validates_blank_template(tmp_path):
    output = tmp_path / "review.csv"
    write_error_review(select_error_review_rows(make_predictions()), output)

    result = validate_error_review(output)
    assert result == {"row_count": 20, "completed_count": 0, "complete": False}
    with pytest.raises(FileExistsError):
        write_error_review(select_error_review_rows(make_predictions()), output)


def test_error_review_uses_all_available_errors_when_fewer_than_twenty():
    selected = select_error_review_rows(make_predictions().head(3))

    assert len(selected) == 3
    assert (selected["class"] != selected["predicted_class"]).all()


def test_completed_error_review_enforces_controlled_vocabulary(tmp_path):
    output = tmp_path / "completed.csv"
    write_error_review(select_error_review_rows(make_predictions()), output)
    rows = pd.read_csv(output)
    rows["noise_present"] = "yes"
    rows["noise_type"] = "broadband;tonal"
    rows["vessel_audibility"] = "partly_masked"
    rows["ambiguity"] = "low_signal_to_noise"
    rows["confidence"] = "medium"
    rows["notes"] = "Steady tone with intermittent broadband noise."
    rows.to_csv(output, index=False)

    result = validate_error_review(output, require_complete=True)

    assert result == {"row_count": 20, "completed_count": 20, "complete": True}
