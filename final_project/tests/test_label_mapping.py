from __future__ import annotations

from src.label_mapping import map_to_sentiment, normalize_label, normalize_task_label


def test_surprice_is_canonicalized() -> None:
    assert normalize_label("Surprice") == "Surprise"


def test_list_and_repeated_labels_are_normalized() -> None:
    assert normalize_label("['Joy', 'Joy', 'Sad']") == "Joy"


def test_emotion_to_sentiment_mapping() -> None:
    assert map_to_sentiment("Joy") == "Positive"
    assert map_to_sentiment("Sad") == "Negative"
    assert map_to_sentiment("Surprise") == "Neutral"
    assert normalize_task_label("Angry", task="emotion") == "Angry"
