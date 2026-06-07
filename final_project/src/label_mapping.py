"""Label normalization utilities for SentiUrdu-1M.

The raw `Category` column contains several shapes:

- leading/trailing spaces, e.g. " Joy"
- list-like strings, e.g. "['Joy', 'Sad']"
- comma-separated repeated labels, e.g. "Joy , Joy"
- misspellings, especially "Surprice"

This module normalizes those variants to canonical emotion labels and maps
canonical emotions to sentiment labels.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


EMOTION_LABELS = ["Joy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

RAW_TO_CANONICAL_EMOTION = {
    "Joy": "Joy",
    "Sad": "Sad",
    "Sadness": "Sad",
    "Angry": "Angry",
    "Anger": "Angry",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Surprice": "Surprise",
    "Surprise": "Surprise",
}

EMOTION_TO_SENTIMENT = {
    "Joy": "Positive",
    "Sad": "Negative",
    "Angry": "Negative",
    "Fear": "Negative",
    "Disgust": "Negative",
    "Surprise": "Neutral",
}

TIE_BREAK_ORDER = ["Joy", "Sad", "Disgust", "Angry", "Fear", "Surprise"]


def _distribution(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).items()
    }


def get_emotion_label_map() -> dict[str, int]:
    """Return canonical emotion label-to-id mapping."""
    return {label: idx for idx, label in enumerate(EMOTION_LABELS)}


def get_sentiment_label_map() -> dict[str, int]:
    """Return canonical sentiment label-to-id mapping."""
    return {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


def _extract_raw_tokens(label: object) -> list[str]:
    if not isinstance(label, str) or not label.strip():
        return []

    value = label.strip()
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            return []

    return [token.strip().strip("'\"") for token in re.split(r",\s*", value) if token.strip()]


def parse_category_labels(label: object) -> list[str]:
    """Parse a raw category cell into canonical emotion labels."""
    canonical: list[str] = []
    for token in _extract_raw_tokens(label):
        normalized = RAW_TO_CANONICAL_EMOTION.get(token.strip())
        if normalized:
            canonical.append(normalized)
    return canonical


def normalize_label(label: object) -> Optional[str]:
    """Normalize raw category/emotion labels into one canonical emotion label."""
    labels = parse_category_labels(label)
    if not labels:
        return None

    counts = Counter(labels)
    highest_count = max(counts.values())
    tied = [item for item, count in counts.items() if count == highest_count]
    if len(tied) == 1:
        return tied[0]

    return min(
        tied,
        key=lambda item: TIE_BREAK_ORDER.index(item)
        if item in TIE_BREAK_ORDER
        else len(TIE_BREAK_ORDER),
    )


def map_to_sentiment(label: object) -> Optional[str]:
    """Map raw or canonical emotion labels to sentiment classes."""
    emotion = label if label in EMOTION_LABELS else normalize_label(label)
    if emotion is None:
        return None
    return EMOTION_TO_SENTIMENT.get(emotion)


def normalize_task_label(label: object, task: str = "sentiment") -> Optional[str]:
    """Normalize a raw label for the selected task."""
    emotion = normalize_label(label)
    if emotion is None:
        return None
    if task == "emotion":
        return emotion
    if task == "sentiment":
        return EMOTION_TO_SENTIMENT.get(emotion)
    raise ValueError(f"Unsupported task: {task!r}. Expected 'sentiment' or 'emotion'.")


def find_unknown_label_tokens(labels: Iterable[object]) -> dict[str, int]:
    """Count raw label tokens that are not recognized as canonical emotions."""
    unknown: Counter[str] = Counter()
    for label in labels:
        for token in _extract_raw_tokens(label):
            if token and token not in RAW_TO_CANONICAL_EMOTION:
                unknown[token] += 1
    return dict(sorted(unknown.items(), key=lambda item: (-item[1], item[0])))


def validate_labels(df: pd.DataFrame, label_column: str) -> None:
    """Print warnings for unknown raw category tokens."""
    unknown = find_unknown_label_tokens(df[label_column].dropna())
    if not unknown:
        print("No unknown label tokens found.")
        return

    print(f"Unknown label tokens found: {len(unknown)}")
    for token, count in list(unknown.items())[:25]:
        print(f"  {token!r}: {count}")


def build_label_mapping_summary(
    df: pd.DataFrame,
    label_column: str,
    task: str,
) -> dict[str, Any]:
    """Build a JSON-serializable summary of label normalization."""
    normalized = df[label_column].map(normalize_label)
    task_labels = df[label_column].map(lambda value: normalize_task_label(value, task))
    unknown_tokens = find_unknown_label_tokens(df[label_column].dropna())

    return {
        "task": task,
        "raw_label_column": label_column,
        "canonical_emotion_labels": EMOTION_LABELS,
        "canonical_sentiment_labels": SENTIMENT_LABELS,
        "emotion_to_sentiment": EMOTION_TO_SENTIMENT,
        "raw_to_canonical_emotion": RAW_TO_CANONICAL_EMOTION,
        "non_null_raw_labels": int(df[label_column].notna().sum()),
        "normalized_emotion_rows": int(normalized.notna().sum()),
        "task_label_rows": int(task_labels.notna().sum()),
        "normalized_emotion_distribution": _distribution(normalized),
        "task_label_distribution": _distribution(task_labels),
        "unknown_label_tokens": unknown_tokens,
    }


def save_label_mapping_summary(
    df: pd.DataFrame,
    label_column: str,
    task: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Save a label-normalization summary JSON file."""
    summary = build_label_mapping_summary(df, label_column, task)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
