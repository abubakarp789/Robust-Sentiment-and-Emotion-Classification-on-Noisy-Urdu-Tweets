"""Configurable Urdu tweet preprocessing for the final project.

The functions in this module are migrated from the stable Assignment 3
pipeline and kept deliberately model-agnostic. The most important design
choice is emoji removal: SentiUrdu-1M labels are influenced by emoji-based
weak supervision, so emojis must not remain as model features.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

import emoji
import pandas as pd


DEFAULT_PREPROCESSING_CONFIG: dict[str, bool | int] = {
    "normalize_unicode": True,
    "remove_urls": True,
    "remove_mentions": True,
    "clean_hashtags": True,
    "remove_emojis": True,
    "remove_numbers": True,
    "remove_punctuation": True,
    "normalize_whitespace": True,
    "min_text_length": 2,
}


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\S+)")
NUMBER_RE = re.compile(r"[\d\u0660-\u0669\u06F0-\u06F9]+")
ASCII_PUNCT_RE = re.compile(r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]""")
URDU_ARABIC_PUNCT_RE = re.compile(r"[\u060C\u061B\u061F\u066A\u066B\u066C\u06D4]+")
WHITESPACE_RE = re.compile(r"\s+")

# Small but useful Urdu/Arabic script normalizations. NFC handles combining
# marks; these replacements handle common codepoint variants seen in Urdu text.
CHARACTER_VARIANTS = str.maketrans(
    {
        "\u064A": "\u06CC",  # Arabic Yeh -> Farsi/Urdu Yeh
        "\u0649": "\u06CC",  # Alef Maksura -> Yeh
        "\u0643": "\u06A9",  # Arabic Kaf -> Keheh
        "\u06C0": "\u06C1",  # Heh with Yeh above -> Heh goal
    }
)


def _merged_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_PREPROCESSING_CONFIG)
    if config:
        merged.update(config)
    return merged


def normalize_unicode(text: str) -> str:
    """Normalize Urdu/Arabic Unicode variants."""
    return unicodedata.normalize("NFC", text).translate(CHARACTER_VARIANTS)


def remove_urls(text: str) -> str:
    """Remove URLs from tweet text."""
    return URL_RE.sub("", text)


def remove_mentions(text: str) -> str:
    """Remove @user mentions."""
    return MENTION_RE.sub("", text)


def clean_hashtags(text: str) -> str:
    """Remove # while preserving the hashtag text as a regular token."""
    return HASHTAG_RE.sub(r"\1", text)


def remove_emojis(text: str) -> str:
    """Remove emojis to prevent weak-label leakage."""
    return emoji.replace_emoji(text, replace="")


def remove_numbers(text: str) -> str:
    """Remove Western and Eastern Arabic-Indic numeric tokens."""
    return NUMBER_RE.sub("", text)


def remove_punctuation(text: str) -> str:
    """Remove ASCII and Urdu/Arabic punctuation while preserving Urdu text."""
    text = ASCII_PUNCT_RE.sub(" ", text)
    text = URDU_ARABIC_PUNCT_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""
    return WHITESPACE_RE.sub(" ", text).strip()


def preprocess_text(text: object, config: Mapping[str, Any] | None = None) -> str:
    """Run the full configurable preprocessing pipeline.

    Non-string, missing, or empty values are converted to an empty string so
    callers can safely apply this function to a pandas column.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cfg = _merged_config(config)
    cleaned = text

    if cfg.get("normalize_unicode", True):
        cleaned = normalize_unicode(cleaned)
    if cfg.get("remove_urls", True):
        cleaned = remove_urls(cleaned)
    if cfg.get("remove_mentions", True):
        cleaned = remove_mentions(cleaned)
    if cfg.get("clean_hashtags", True):
        cleaned = clean_hashtags(cleaned)
    if cfg.get("remove_emojis", True):
        cleaned = remove_emojis(cleaned)
    if cfg.get("remove_numbers", True):
        cleaned = remove_numbers(cleaned)
    if cfg.get("remove_punctuation", True):
        cleaned = remove_punctuation(cleaned)
    if cfg.get("normalize_whitespace", True):
        cleaned = normalize_whitespace(cleaned)

    return cleaned


def preprocess_series(series: pd.Series, config: Mapping[str, Any] | None = None) -> pd.Series:
    """Apply preprocessing to a pandas Series with UTF-8-safe string handling."""
    return series.astype(object).map(lambda value: preprocess_text(value, config))


def token_count(text: object) -> int:
    """Return whitespace token count for already-cleaned text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def is_valid_clean_text(text: object, min_text_length: int = 1) -> bool:
    """Check whether cleaned text has at least `min_text_length` tokens."""
    return token_count(text) >= min_text_length
