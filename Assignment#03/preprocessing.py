"""
Urdu tweet preprocessing utilities — Assignment 3.

This module is the runtime version of the 8-step pipeline developed and justified
in Assignment 1 (Milestone 1). It exposes both the individual step functions
(useful for ablation and visualisation in `01_dataset_analysis.ipynb`) and the
fully composed `preprocess_tweet` pipeline used by every model in
`02_model_implementation.ipynb`.

Pipeline order (each step is mathematically a transformation t_i : str -> str):

    1. normalize_unicode        — NFC normalisation of Arabic-script codepoints
    2. remove_urls              — strip http/https/www links
    3. remove_mentions          — strip @username handles
    4. clean_hashtags           — drop "#" character but keep the word
    5. remove_emojis            — CRITICAL: prevents emoji→label leakage
    6. remove_numbers           — strip Western and Eastern Arabic-Indic digits
    7. remove_punctuation       — strip ASCII + Urdu/Arabic punctuation
    8. normalize_whitespace     — collapse runs of whitespace to a single space

`T(x) = t_8 ∘ t_7 ∘ … ∘ t_1(x)` is therefore equivalent to `preprocess_tweet(x)`.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections import Counter
from typing import List, Optional

import emoji
import pandas as pd

# Re-exported from config to avoid import cycles when this module is used
# stand-alone (e.g. by a different notebook). Keys are the *canonical* emotion
# names (after normalisation by `parse_category`).
EMOTION_TO_SENTIMENT_DEFAULT = {
    "Joy":      "Positive",
    "Sad":      "Negative",
    "Angry":    "Negative",
    "Fear":     "Negative",
    "Disgust":  "Negative",
    "Surprise": "Neutral",
}

# Raw → canonical emotion name (handles surface variants and misspellings).
_EMOTION_NORMALISE = {
    "Joy": "Joy",
    "Sad": "Sad",
    "Angry": "Angry",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Surprice": "Surprise",
    "Surprise": "Surprise",
}


# ---------------------------------------------------------------------------
# Step 1 — Unicode normalisation (NFC)
# ---------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Apply NFC normalisation to unify equivalent Unicode representations."""
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Step 2 — URL removal
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def remove_urls(text: str) -> str:
    """Remove all HTTP/HTTPS URLs and www-links."""
    return _URL_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 3 — Mention (@username) removal
# ---------------------------------------------------------------------------
_MENTION_RE = re.compile(r"@\w+")


def remove_mentions(text: str) -> str:
    """Remove Twitter @username mentions."""
    return _MENTION_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 4 — Hashtag symbol cleanup (keep the word)
# ---------------------------------------------------------------------------
_HASHTAG_RE = re.compile(r"#(\S+)")


def clean_hashtags(text: str) -> str:
    """Remove '#' symbol but keep the hashtag word as a regular token."""
    return _HASHTAG_RE.sub(r"\1", text)


# ---------------------------------------------------------------------------
# Step 5 — Emoji removal (label-leakage prevention)
# ---------------------------------------------------------------------------
def remove_emojis(text: str) -> str:
    """Strip all emoji characters.

    The SentiUrdu-1M labels were derived from emoticons used in the original
    tweets. Keeping emojis as features would leak the labelling heuristic into
    the input, producing artificially high accuracy that does not reflect
    Urdu language understanding.
    """
    return emoji.replace_emoji(text, replace="")


# ---------------------------------------------------------------------------
# Step 6 — Number removal (Western + Eastern digits)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[\d٠-٩۰-۹]")


def remove_numbers(text: str) -> str:
    """Remove Western digits (0-9) and Eastern Arabic-Indic digits (٠-٩, ۰-۹)."""
    return _NUM_RE.sub("", text)


# ---------------------------------------------------------------------------
# Step 7 — Punctuation removal (ASCII + Urdu/Arabic)
# ---------------------------------------------------------------------------
_ASCII_PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')
_URDU_PUNCT_RE = re.compile(r"[۔،؟؛٪٫٬]+")


def remove_punctuation(text: str) -> str:
    """Remove ASCII punctuation and Urdu/Arabic punctuation marks."""
    text = _ASCII_PUNCT_RE.sub("", text)
    text = _URDU_PUNCT_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# Step 8 — Whitespace normalisation
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_tweet(text: object) -> str:
    """Apply the full 8-step Urdu tweet preprocessing pipeline.

    Accepts non-string input safely (returns "") so it can be applied with
    `pd.Series.apply` over columns that may contain NaN.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = clean_hashtags(text)
    text = remove_emojis(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text


# Vectorised convenience for pandas
def preprocess_series(s: pd.Series) -> pd.Series:
    """Apply the full pipeline to a pandas Series of raw tweets."""
    return s.astype(object).map(preprocess_tweet)


# ---------------------------------------------------------------------------
# Whitespace tokeniser (used by CNN / BiLSTM vocab building)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Whitespace tokenisation — adequate for already-cleaned Urdu tweets."""
    if not text:
        return []
    return text.split()


# ---------------------------------------------------------------------------
# Label derivation utilities
# ---------------------------------------------------------------------------
def parse_category(category: object) -> List[str]:
    """Split a raw `Category` cell into a list of canonical emotion labels.

    The SentiUrdu-1M CSV stores the column in several inconsistent shapes:
        "Joy", " Joy", "['Joy']", "['Joy', 'Sad']", "Joy , Joy", "Sad, Joy"
    Plus the misspelling "Surprice" for Surprise.

    Returns an empty list for missing / unparseable cells. Output values are
    drawn from the canonical set
        {"Joy", "Sad", "Angry", "Fear", "Disgust", "Surprise"}.
    """
    if not isinstance(category, str) or not category.strip():
        return []
    c = category.strip()
    raw_tokens: List[str]
    if c.startswith("["):
        try:
            raw_tokens = [str(x).strip() for x in ast.literal_eval(c)]
        except (ValueError, SyntaxError):
            raw_tokens = []
    else:
        raw_tokens = [t.strip() for t in re.split(r",\s*", c) if t.strip()]
    canonical = [_EMOTION_NORMALISE[t] for t in raw_tokens if t in _EMOTION_NORMALISE]
    return canonical


def majority_emotion(category: object) -> Optional[str]:
    """Reduce a (possibly multi-label) `Category` cell to a single emotion.

    We use majority vote with ties broken by frequency in the corpus
    (Joy > Sad > Disgust > Angry > Fear > Surprise). Returns None if the cell
    has no recognised emotion token.
    """
    labels = parse_category(category)
    if not labels:
        return None
    counts = Counter(labels)
    top = counts.most_common()
    max_count = top[0][1]
    tied = [lab for lab, n in top if n == max_count]
    if len(tied) == 1:
        return tied[0]
    order = ["Joy", "Sad", "Disgust", "Angry", "Fear", "Surprise"]
    return min(tied, key=lambda x: order.index(x) if x in order else len(order))


def derive_sentiment_from_category(
    category: object,
    mapping: Optional[dict] = None,
) -> Optional[str]:
    """Map a (possibly messy) `Category` cell to a 3-class sentiment label.

    Performs label normalisation via `majority_emotion` first, so this works
    correctly on the raw column shapes documented above.
    Returns None for cells with no recognised emotion.
    """
    emo = majority_emotion(category)
    if emo is None:
        return None
    mapping = mapping or EMOTION_TO_SENTIMENT_DEFAULT
    return mapping.get(emo)


_EMO_TOKEN_RE = re.compile(r"'([A-Z _]+) , (-?\d+(?:\.\d+)?)'")
_POS_SET = {
    "SMILING FACE WITH SMILING EYES",
    "FACE WITH TEARS OF JOY",
    "SMILING FACE WITH HEART-SHAPED EYES",
    "HEAVY BLACK HEART",
    "TWO HEARTS",
    "RED HEART",
    "FOLDED HANDS",
    "SPARKLES",
    "ROSE",
    "BLOSSOM",
    "GRINNING FACE",
    "GRINNING FACE WITH SMILING EYES",
    "BEAMING FACE WITH SMILING EYES",
    "WHITE HEART SUIT",
    "LOVE LETTER",
    "SMILING FACE WITH HALO",
    "WINKING FACE",
}
_NEG_SET = {
    "POUTING FACE",
    "ANGRY FACE",
    "CRYING FACE",
    "LOUDLY CRYING FACE",
    "DISAPPOINTED FACE",
    "BROKEN HEART",
    "FACE WITH STEAM FROM NOSE",
    "FACE SCREAMING IN FEAR",
    "WORRIED FACE",
    "FEARFUL FACE",
    "SAD BUT RELIEVED FACE",
    "WEARY FACE",
    "TIRED FACE",
}


def derive_sentiment_from_emotions(emotions: object) -> Optional[str]:
    """Derive a 3-class sentiment from the raw `Emotions` column.

    The column stores strings like:
        "['SMILING FACE WITH SMILING EYES , 0.644', 'SPARKLES , 0.351']"

    We sum the confidence scores of positive vs. negative emoji names and
    return the majority class. Returns "Neutral" if both sides are zero.
    """
    if not isinstance(emotions, str) or not emotions:
        return None

    pos = 0.0
    neg = 0.0
    for name, score in _EMO_TOKEN_RE.findall(emotions):
        try:
            s = float(score)
        except ValueError:
            continue
        if name in _POS_SET:
            pos += s
        elif name in _NEG_SET:
            neg += s

    if pos == 0.0 and neg == 0.0:
        return "Neutral"
    if pos >= neg:
        return "Positive"
    return "Negative"


def build_sentiment_label(row: pd.Series) -> Optional[str]:
    """Map the SentiUrdu-1M `Category` value to a 3-class sentiment label.

    Following the SentiUrdu-1M paper convention, the sentiment task uses the
    same ~533K rows that have a non-null `Category`. The (~515K) NaN rows are
    intentionally dropped because the available emoji-only fallback signal is
    too noisy to support a reliable label (it would assign a default
    "Neutral" to any tweet whose emojis are not in our positive/negative
    sets, badly skewing the distribution).
    """
    return derive_sentiment_from_category(row.get("Category"))


__all__ = [
    "normalize_unicode",
    "remove_urls",
    "remove_mentions",
    "clean_hashtags",
    "remove_emojis",
    "remove_numbers",
    "remove_punctuation",
    "normalize_whitespace",
    "preprocess_tweet",
    "preprocess_series",
    "tokenize",
    "parse_category",
    "majority_emotion",
    "derive_sentiment_from_category",
    "derive_sentiment_from_emotions",
    "build_sentiment_label",
    "EMOTION_TO_SENTIMENT_DEFAULT",
]
