"""
Centralised configuration for Assignment 3.

All notebooks import their paths, seeds and shared hyperparameters from here so that
changes in one place propagate everywhere.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent  # "NLP Project"

DATA_CSV = WORKSPACE_ROOT / "Assignment#01" / "Urdu Tweets Dataset.csv"
DATA_XLSX = WORKSPACE_ROOT / "Assignment#01" / "Urdu Tweets Dataset.xlsx"

OUTPUTS = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
MODEL_DIR = OUTPUTS / "models"
RESULTS_DIR = OUTPUTS / "results"
CACHE_DIR = OUTPUTS / "cache"
EMBED_DIR = OUTPUTS / "embeddings"
REPORT_DIR = PROJECT_ROOT / "report"

for _p in (OUTPUTS, FIG_DIR, MODEL_DIR, RESULTS_DIR, CACHE_DIR, EMBED_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix the seed for python random, numpy and (if available) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional during pure-EDA notebooks
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms slow training; keep cuDNN benchmark on for speed.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["Joy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
EMOTION2ID = {lab: i for i, lab in enumerate(EMOTION_LABELS)}
ID2EMOTION = {i: lab for lab, i in EMOTION2ID.items()}

# Raw label normalisation — the SentiUrdu-1M CSV has surface variants:
#   "Joy" / " Joy" / "['Joy']" / "Joy , Joy"
# Misspellings: "Surprice" → "Surprise"
EMOTION_RAW_TO_CANONICAL = {
    "Joy": "Joy",
    "Sad": "Sad",
    "Angry": "Angry",
    "Fear": "Fear",
    "Disgust": "Disgust",
    "Surprice": "Surprise",
    "Surprise": "Surprise",
}

# Mapping used when deriving 3-class sentiment from the canonical emotion label.
EMOTION_TO_SENTIMENT = {
    "Joy":      "Positive",
    "Sad":      "Negative",
    "Angry":    "Negative",
    "Fear":     "Negative",
    "Disgust":  "Negative",
    "Surprise": "Neutral",
}
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
SENT2ID = {lab: i for i, lab in enumerate(SENTIMENT_LABELS)}
ID2SENT = {i: lab for lab, i in SENT2ID.items()}


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
SPLIT_RATIO = (0.70, 0.15, 0.15)  # train / val / test
STRATIFY = True


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------
TFIDF = dict(
    ngram_range=(1, 2),
    max_features=50_000,
    min_df=3,
    sublinear_tf=True,
)

LR = dict(
    solver="saga",
    C=1.0,
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    random_state=SEED,
)

SVC_PARAMS = dict(
    C=1.0,
    max_iter=2000,
    class_weight="balanced",
    random_state=SEED,
)


# ---------------------------------------------------------------------------
# Deep learning (CNN, BiLSTM)
# ---------------------------------------------------------------------------
EMBED_DIM = 300
MAX_LEN = 64        # 95th percentile of Urdu tweet token counts is ~50
VOCAB_SIZE = 80_000
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

CNN = dict(
    filter_sizes=(3, 4, 5),
    num_filters=128,
    dropout=0.5,
    lr=1e-3,
    batch_size=256,
    epochs=8,
    patience=2,
)

BILSTM = dict(
    hidden_size=256,
    num_layers=2,
    dropout=0.4,
    bidirectional=True,
    attention=True,
    lr=1e-3,
    batch_size=256,
    epochs=8,
    patience=2,
)


# ---------------------------------------------------------------------------
# Transformer fine-tuning
# ---------------------------------------------------------------------------
TRANSFORMER_MAX_LEN = 96

TRANSFORMERS = {
    "mbert": dict(
        model_name="bert-base-multilingual-cased",
        lr=2e-5,
        epochs=3,
        batch_size=32,
        grad_accum=1,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
    ),
    "xlm-r": dict(
        model_name="xlm-roberta-base",
        lr=2e-5,
        epochs=3,
        batch_size=32,
        grad_accum=1,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
    ),
    "urdu-roberta": dict(
        model_name="urduhack/roberta-urdu-small",
        lr=3e-5,
        epochs=3,
        batch_size=64,
        grad_accum=1,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
    ),
}


# ---------------------------------------------------------------------------
# fastText Urdu embeddings (downloaded by 02_model_implementation.ipynb)
# ---------------------------------------------------------------------------
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ur.300.vec.gz"
FASTTEXT_PATH = EMBED_DIR / "cc.ur.300.vec.gz"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIG_DPI = 150
FIG_FORMAT = "png"          # change to "pdf" or "svg" if needed
PALETTE = "viridis"
