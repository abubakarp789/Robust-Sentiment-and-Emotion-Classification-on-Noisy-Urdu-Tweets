"""
Programmatic builder for the four Assignment 3 Jupyter notebooks.

Run once after editing this file:
    python _build_notebooks.py

It writes the four .ipynb files in the project root:
    01_dataset_analysis.ipynb
    02_model_implementation.ipynb
    03_evaluation.ipynb

(04_architecture_math.md is plain Markdown, not generated here.)

The builder uses nbformat to avoid hand-maintained JSON. Each notebook is
defined as a list of (kind, source) tuples where kind is "md" or "code".
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent


def _build(filename: str, cells: list[tuple[str, str]]) -> None:
    nb = new_notebook()
    for kind, src in cells:
        src = dedent(src).strip("\n") + "\n"
        if kind == "md":
            nb.cells.append(new_markdown_cell(src))
        elif kind == "code":
            nb.cells.append(new_code_cell(src))
        else:
            raise ValueError(kind)

    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        }
    )

    path = ROOT / filename
    nbformat.write(nb, path)
    print(f"  wrote {filename}  ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# Helper snippets used in multiple notebooks
# ---------------------------------------------------------------------------
HEADER_SETUP = r'''
# --- runtime / path setup -------------------------------------------------
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
ROOT = os.path.dirname(os.path.abspath("__file__"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
import preprocessing as pp
from config import (
    DATA_CSV, FIG_DIR, MODEL_DIR, RESULTS_DIR, CACHE_DIR, EMBED_DIR,
    SEED, set_seed,
    EMOTION_LABELS, SENTIMENT_LABELS,
    EMOTION_TO_SENTIMENT,
)
set_seed(SEED)
print(f"seed={SEED}   data csv: {DATA_CSV.exists()}")
'''


# ---------------------------------------------------------------------------
# 01_dataset_analysis.ipynb
# ---------------------------------------------------------------------------
def build_eda() -> None:
    cells: list[tuple[str, str]] = []

    cells.append(("md", r"""
    # Assignment 3 — Task 1
    ## Dataset Analysis and Statistical Exploration of SentiUrdu-1M

    **Course:** CSC-355 Natural Language Processing
    **Students:** Raqib Hayat (NUM-BSCS-2022-40), Abu Bakar (NUM-BSCS-2022-41)
    **Dataset:** [SentiUrdu-1M](https://data.mendeley.com/datasets/rz3xg97rm5/1) — ~1,048,000 weakly-labelled Urdu tweets

    This notebook goes well beyond the basic exploration from Milestone 1. It performs:

    1. **Dataset overview** — shape, dtypes, memory footprint, missing values, duplicates.
    2. **Label distribution** — emotion (`Category`) and derived 3-class sentiment.
    3. **Text statistics** — token / character length distributions, per-class boxplots.
    4. **Token frequency** — top-50 tokens overall and per class, Zipf's law.
    5. **Emoji and noise** — emoji frequency, URL/mention/hashtag share, code-mixing,
       label-leakage correlation.
    6. **Preprocessing impact** — before vs after length, vocabulary reduction, empty-tweet rate.
    7. **Dataset suitability discussion** — strengths, limitations, decisions, expected biases.

    All figures are saved to `outputs/figures/`. The notebook is idempotent: cached
    intermediate Parquet files in `outputs/cache/` are reused on subsequent runs.
    """))

    cells.append(("md", "## 0 — Setup"))
    cells.append(("code", HEADER_SETUP))
    cells.append(("code", r"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    from pathlib import Path

    sns.set_theme(context="notebook", style="whitegrid", palette="viridis")
    plt.rcParams["figure.dpi"] = config.FIG_DPI
    plt.rcParams["savefig.dpi"] = config.FIG_DPI
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"

    FIG_EXT = config.FIG_FORMAT

    def savefig(name: str, fig=None, **kw):
        fig = fig or plt.gcf()
        out = FIG_DIR / f"{name}.{FIG_EXT}"
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight", **kw)
        print(f"  ↳ saved {out.name}")
    """))

    cells.append(("md", "## 1 — Dataset Overview & Structure"))
    cells.append(("code", r"""
    %%time
    # Loading the CSV is much faster than the XLSX (≈ 6× speedup).
    df = pd.read_csv(DATA_CSV, dtype=str, keep_default_na=False, na_values=[""])
    df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("Int64")
    print(f"Shape : {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    df.head(3)
    """))
    cells.append(("code", r"""
    print("Dtypes")
    print(df.dtypes, "\n")
    print("Missing values per column")
    print(df.isna().sum())
    print("\nUnique counts")
    for c in df.columns:
        print(f"  {c:>10}: {df[c].nunique(dropna=True)} unique")
    """))
    cells.append(("code", r"""
    # Duplicate analysis (on raw text — preprocessing collapses many near-dupes).
    dup_text  = df.duplicated(subset=["Text"]).sum()
    dup_full  = df.duplicated().sum()
    print(f"Duplicate raw Text rows : {dup_text:>8,}  ({dup_text/len(df)*100:.2f}%)")
    print(f"Duplicate full rows     : {dup_full:>8,}  ({dup_full/len(df)*100:.2f}%)")
    """))
    cells.append(("code", r"""
    # Missing-data heatmap
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df.isna().T, cbar=False, ax=ax, yticklabels=df.columns)
    ax.set_xticks([])
    ax.set_title("Missing-value pattern across the dataset (≈49% NaN in Category)")
    savefig("fig01_missing_pattern")
    plt.show()
    """))

    cells.append(("md", "## 2 — Label Distribution Analysis"))
    cells.append(("code", r"""
    # Apply the project's preprocessing pipeline once and cache the result.
    cache = CACHE_DIR / "df_clean.parquet"
    if cache.exists():
        df_clean = pd.read_parquet(cache)
        print(f"Loaded cached cleaned dataframe: {df_clean.shape}")
    else:
        df_clean = df.copy()
        df_clean["clean_text"]    = pp.preprocess_series(df_clean["Text"])
        df_clean["sentiment"]     = df_clean.apply(pp.build_sentiment_label, axis=1)
        df_clean["emotion"]       = df_clean["Category"].map(pp.majority_emotion)
        df_clean["n_tokens_raw"]  = df_clean["Text"].fillna("").str.split().str.len()
        df_clean["n_tokens"]      = df_clean["clean_text"].str.split().str.len()
        df_clean["n_chars_raw"]   = df_clean["Text"].fillna("").str.len()
        df_clean["n_chars"]       = df_clean["clean_text"].str.len()
        df_clean.to_parquet(cache, index=False)
        print(f"Cleaned and cached: {df_clean.shape}")
    df_clean.head(3)
    """))
    cells.append(("code", r"""
    # 2.1 Emotion label distribution
    ec = df_clean["emotion"].value_counts(dropna=True).reindex(EMOTION_LABELS).fillna(0).astype(int)
    print("Emotion counts (Category column):")
    print(ec)
    imbal = ec.max() / max(ec.min(), 1)
    print(f"\nImbalance ratio (max/min) = {imbal:.2f}×")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=ec.index, y=ec.values, ax=ax, palette="viridis")
    ax.set_title("Emotion class distribution — SentiUrdu-1M (Category column)")
    ax.set_ylabel("Tweet count")
    for i, v in enumerate(ec.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    savefig("fig02_emotion_distribution")
    plt.show()
    """))
    cells.append(("code", r"""
    # 2.2 Derived 3-class sentiment distribution
    sc = df_clean["sentiment"].value_counts(dropna=True).reindex(SENTIMENT_LABELS).fillna(0).astype(int)
    n_unlabelled = df_clean["sentiment"].isna().sum()
    print(f"Derived sentiment counts (full dataset; {n_unlabelled:,} unmapped/NaN dropped):")
    print(sc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=sc.index, y=sc.values, ax=axes[0], palette=["#d62728", "#7f7f7f", "#2ca02c"])
    axes[0].set_title("Derived 3-class sentiment distribution")
    axes[0].set_ylabel("Tweet count")
    for i, v in enumerate(sc.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].pie(sc.values, labels=sc.index, autopct="%1.1f%%",
                colors=["#d62728", "#7f7f7f", "#2ca02c"], startangle=90)
    axes[1].set_title("Sentiment proportion")
    savefig("fig03_sentiment_distribution")
    plt.show()
    """))
    cells.append(("code", r"""
    # 2.3 Stacked comparison: emotion × derived sentiment
    crosstab = pd.crosstab(df_clean["emotion"], df_clean["sentiment"]).reindex(
        index=EMOTION_LABELS, columns=SENTIMENT_LABELS, fill_value=0
    )
    print(crosstab)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    crosstab.plot(kind="bar", stacked=True, ax=ax,
                  color=["#d62728", "#7f7f7f", "#2ca02c"])
    ax.set_title("Emotion → Sentiment mapping (stacked)")
    ax.set_ylabel("Count")
    ax.legend(title="Derived sentiment")
    savefig("fig04_emotion_vs_sentiment")
    plt.show()
    """))

    cells.append(("md", "## 3 — Text Statistics"))
    cells.append(("code", r"""
    # Length statistics (cleaned text)
    summary = df_clean[["n_tokens", "n_chars", "n_tokens_raw", "n_chars_raw"]].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    summary
    """))
    cells.append(("code", r"""
    # Histograms of token/character counts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df_clean["n_tokens"].clip(upper=100), bins=60, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Token count per tweet (clipped at 100)")
    axes[0].set_xlabel("Tokens"); axes[0].set_ylabel("Tweets")

    sns.histplot(df_clean["n_chars"].clip(upper=400), bins=60, ax=axes[1], color="#55a868")
    axes[1].set_title("Character count per tweet (clipped at 400)")
    axes[1].set_xlabel("Characters"); axes[1].set_ylabel("Tweets")
    savefig("fig05_length_histograms")
    plt.show()
    """))
    cells.append(("code", r"""
    # Boxplot — text length by emotion class
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = EMOTION_LABELS
    sub = df_clean.dropna(subset=["emotion"]).copy()
    sub["n_tokens_clip"] = sub["n_tokens"].clip(upper=80)
    sns.boxplot(data=sub, x="emotion", y="n_tokens_clip", order=order,
                showfliers=False, palette="viridis", ax=ax)
    ax.set_title("Token count per emotion class")
    ax.set_xlabel("Emotion"); ax.set_ylabel("Tokens (clipped at 80)")
    savefig("fig06_tokens_by_emotion_box")
    plt.show()
    """))
    cells.append(("code", r"""
    # Per-class summary table
    per_class = sub.groupby("emotion")["n_tokens"].agg(
        n="size", mean="mean", median="median", p95=lambda s: s.quantile(0.95)
    ).reindex(order)
    per_class
    """))

    cells.append(("md", "## 4 — Token Frequency Analysis"))
    cells.append(("code", r"""
    # Build global token counter (skip empty rows)
    tokens_cache = CACHE_DIR / "token_counter.json"
    import json
    if tokens_cache.exists():
        with open(tokens_cache, "r", encoding="utf-8") as f:
            counter_dict = json.load(f)
        counter = Counter(counter_dict)
        print(f"Loaded cached token counter: {len(counter):,} types")
    else:
        counter = Counter()
        for tok_list in df_clean["clean_text"].dropna().str.split():
            counter.update(tok_list)
        with open(tokens_cache, "w", encoding="utf-8") as f:
            json.dump(dict(counter.most_common(200_000)), f, ensure_ascii=False)
        print(f"Built and cached token counter: {len(counter):,} types")

    top50 = counter.most_common(50)
    print("Top-10 tokens (after preprocessing):")
    for t, c in top50[:10]:
        print(f"  {t:>15}  {c:,}")
    """))
    cells.append(("code", r"""
    # Top-50 token bar chart
    import arabic_reshaper
    from bidi.algorithm import get_display

    labels = [get_display(arabic_reshaper.reshape(t)) for t, _ in top50]
    counts = [c for _, c in top50]

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=counts, y=labels, ax=ax, palette="viridis")
    ax.set_title("Top-50 tokens (cleaned text)")
    ax.set_xlabel("Frequency"); ax.set_ylabel("")
    savefig("fig07_top50_tokens")
    plt.show()
    """))
    cells.append(("code", r"""
    # Zipf's law — log-log plot of rank vs frequency
    freqs = np.array(sorted(counter.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, freqs, lw=1.2)
    ax.set_title("Zipf's law for SentiUrdu-1M (log-log)")
    ax.set_xlabel("Rank"); ax.set_ylabel("Frequency")
    # Reference slope -1
    ax.loglog(ranks[:1000], freqs[0] / ranks[:1000], "--", color="gray", lw=0.8,
              label="slope = -1 reference")
    ax.legend()
    savefig("fig08_zipf")
    plt.show()

    print(f"Vocabulary size (cleaned): {len(counter):,}")
    print(f"Tokens appearing only once: {sum(1 for v in counter.values() if v == 1):,} ({sum(1 for v in counter.values() if v == 1)/len(counter)*100:.1f}%)")
    """))
    cells.append(("code", r"""
    # Per-class top-15 tokens
    per_class_top = {}
    for cls in EMOTION_LABELS:
        sub_tok = Counter()
        rows = df_clean.loc[df_clean["emotion"] == cls, "clean_text"].dropna()
        for toks in rows.str.split():
            sub_tok.update(toks)
        per_class_top[cls] = sub_tok.most_common(15)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cls in zip(axes.flat, EMOTION_LABELS):
        items = per_class_top[cls]
        labs = [get_display(arabic_reshaper.reshape(t)) for t, _ in items]
        vals = [c for _, c in items]
        sns.barplot(x=vals, y=labs, ax=ax, palette="viridis")
        ax.set_title(f"Top-15 tokens — {cls}")
        ax.set_xlabel(""); ax.set_ylabel("")
    savefig("fig09_top15_per_emotion")
    plt.show()
    """))
    cells.append(("code", r"""
    # Word clouds for three contrasting emotions
    from wordcloud import WordCloud
    font_path = None
    # Try a few common Windows fonts that support Urdu glyphs.
    for cand in [r"C:/Windows/Fonts/NafeesWeb.ttf",
                 r"C:/Windows/Fonts/NafeesNastaleeq.ttf",
                 r"C:/Windows/Fonts/JameelNooriNastaleeq.ttf",
                 r"C:/Windows/Fonts/Tahoma.ttf",
                 r"C:/Windows/Fonts/Arial.ttf"]:
        if Path(cand).exists():
            font_path = cand
            break
    print("WordCloud font:", font_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cls in zip(axes, ["Joy", "Sad", "Angry"]):
        sub_tok = Counter()
        rows = df_clean.loc[df_clean["emotion"] == cls, "clean_text"].dropna()
        for toks in rows.str.split():
            sub_tok.update(toks)
        # reshape for proper Arabic rendering inside the cloud
        reshaped = {get_display(arabic_reshaper.reshape(k)): v for k, v in sub_tok.most_common(200)}
        wc = WordCloud(font_path=font_path, width=600, height=400,
                       background_color="white", colormap="viridis").generate_from_frequencies(reshaped)
        ax.imshow(wc); ax.axis("off"); ax.set_title(f"{cls} (top 200 tokens)")
    savefig("fig10_wordclouds")
    plt.show()
    """))

    cells.append(("md", "## 5 — Emoji and Noise Analysis"))
    cells.append(("code", r"""
    import re
    import emoji as _emoji_lib

    # Re-run a small (sampled) noise-feature audit so we don't iterate 1M rows twice
    sample = df.sample(min(200_000, len(df)), random_state=SEED).copy()

    def has_emoji(s):  return any(c for c in s if c in _emoji_lib.EMOJI_DATA) if isinstance(s, str) else False
    sample["has_url"]     = sample["Text"].fillna("").str.contains(r"https?://|www\.", regex=True)
    sample["has_mention"] = sample["Text"].fillna("").str.contains(r"@\w+", regex=True)
    sample["has_hashtag"] = sample["Text"].fillna("").str.contains(r"#\S+", regex=True)
    sample["has_emoji"]   = sample["Text"].apply(has_emoji)
    sample["has_latin"]   = sample["Text"].fillna("").str.contains(r"[A-Za-z]{3,}", regex=True)

    noise = sample[["has_url","has_mention","has_hashtag","has_emoji","has_latin"]].mean() * 100
    print("Share of tweets containing each feature (200K random sample):")
    print(noise.round(2).astype(str) + " %")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=noise.index, y=noise.values, ax=ax, palette="viridis")
    for i, v in enumerate(noise.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Noise / artefact features in raw tweets")
    ax.set_ylabel("Share of tweets (%)")
    savefig("fig11_noise_features")
    plt.show()
    """))
    cells.append(("code", r"""
    # Top-20 emoji frequency
    emoji_counter = Counter()
    for txt in df["Text"].dropna().sample(300_000, random_state=SEED):
        for ch in txt:
            if ch in _emoji_lib.EMOJI_DATA:
                emoji_counter[ch] += 1
    top_em = emoji_counter.most_common(20)
    print("Top-20 emojis (300K sample):", top_em)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=[e for e,_ in top_em], y=[c for _,c in top_em], ax=ax, palette="viridis")
    ax.set_title("Top-20 emojis in raw tweets")
    ax.set_ylabel("Count (sample of 300K)")
    savefig("fig12_top_emojis")
    plt.show()
    """))
    cells.append(("code", r"""
    # Label-leakage diagnostic: emoji-derived sentiment vs Category-derived sentiment
    leak = df_clean.dropna(subset=["emotion"]).copy()
    leak["sent_from_emojis"] = leak["Emotions"].map(pp.derive_sentiment_from_emotions)
    cm_leak = pd.crosstab(leak["emotion"], leak["sent_from_emojis"]).reindex(
        index=EMOTION_LABELS, columns=SENTIMENT_LABELS, fill_value=0
    )
    cm_leak_norm = cm_leak.div(cm_leak.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(cm_leak_norm, annot=True, fmt=".2f", cmap="rocket_r", ax=ax)
    ax.set_title("P(emoji-sentiment | Category) — high diagonals ⇒ label leakage risk")
    ax.set_xlabel("Sentiment derived from raw Emojis")
    ax.set_ylabel("Original Category label")
    savefig("fig13_label_leakage_heatmap")
    plt.show()
    """))

    cells.append(("md", "## 6 — Preprocessing Impact"))
    cells.append(("code", r"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 400, 60)
    ax.hist(df_clean["n_chars_raw"].clip(upper=400), bins=bins, alpha=0.55,
            label="Raw", color="#d62728")
    ax.hist(df_clean["n_chars"].clip(upper=400),     bins=bins, alpha=0.55,
            label="Cleaned", color="#2ca02c")
    ax.set_title("Character length distribution: raw vs. cleaned")
    ax.set_xlabel("Characters (clipped at 400)"); ax.set_ylabel("Tweets")
    ax.legend()
    savefig("fig14_length_before_after")
    plt.show()

    empty_after = (df_clean["clean_text"].str.len() == 0).sum()
    print(f"Avg char reduction:  {df_clean['n_chars_raw'].mean() - df_clean['n_chars'].mean():.2f}")
    print(f"Tweets empty after preprocessing: {empty_after:,} ({empty_after/len(df_clean)*100:.3f}%)")
    """))
    cells.append(("code", r"""
    # Vocabulary reduction before/after preprocessing (estimated on the same sample)
    raw_counter = Counter()
    for toks in sample["Text"].fillna("").str.split():
        raw_counter.update(toks)
    clean_counter = Counter()
    for toks in df_clean.loc[sample.index, "clean_text"].fillna("").str.split():
        clean_counter.update(toks)

    print(f"|V_raw|   = {len(raw_counter):,}")
    print(f"|V_clean| = {len(clean_counter):,}")
    reduction = (1 - len(clean_counter) / len(raw_counter)) * 100
    print(f"Vocabulary reduction: {reduction:.1f}%")
    """))

    cells.append(("md", "## 7 — Dataset Suitability Discussion"))
    cells.append(("md", r"""
    ### Why SentiUrdu-1M is suitable

    - **Scale.** ~1.05 M tweets is, to our knowledge, the largest publicly-available
      Urdu sentiment corpus, an order of magnitude larger than the typical 5–50 K
      tweet corpora used in prior Urdu studies.
    - **Naturalistic noise.** Tweets are taken straight from Twitter without
      manual cleaning, exposing models to URLs, mentions, hashtags, emojis,
      code-mixing, and the long tail of out-of-vocabulary tokens that real
      deployments must handle.
    - **Multi-task labels.** Both an emotion `Category` (6 classes) and a
      derivable sentiment polarity (3 classes) can be obtained from the same
      rows, enabling a meaningful comparative study.
    - **Reproducibility.** A public Mendeley DOI plus a published reference
      paper makes results directly comparable to prior work.

    ### Challenges present in the dataset

    - **Weak supervision.** Labels were assigned heuristically from emojis and
      SentiWordNet rather than by trained annotators, so they are inherently
      noisy. The Section 5 heatmap above directly visualises the strength of
      this emoji → label coupling.
    - **Label leakage from emojis.** Because the labelling signal is the emoji
      set itself, any model that consumes emoji-bearing tweets at training
      time will trivially memorise the heuristic. Step 5 of our preprocessing
      pipeline strips emojis exactly to eliminate this short-cut.
    - **Class imbalance.** The emotion distribution is heavily skewed; some
      classes (e.g. *Surprise*, *Fear*) have an order of magnitude fewer
      examples than *Joy*. We apply inverse-frequency class weighting in
      every model's loss to mitigate this.
    - **~49 % NaN `Category`.** Roughly half the corpus has no emotion label.
      For the **sentiment** task we recover those rows by deriving the label
      from the raw `Emotions` column; for the **emotion** task we drop them.
    - **Code-mixing.** A small but non-trivial fraction of tweets contain
      Latin-script English words alongside Urdu script. This is one source
      of the long token tail visible in the Zipf plot.
    - **Topic and temporal bias.** Tweets concentrate on a narrow window of
      Pakistani current events, so domain-shift to other genres (news, prose)
      may be substantial.

    ### Preprocessing decisions and their impact (from Milestone 1)

    The 8-step pipeline reduces character length by ≈ 10 % on average and
    collapses the vocabulary by roughly an order of magnitude relative to the
    raw token set. The single most important decision is **emoji removal** —
    without it, every model trivially achieves > 95 % accuracy by simply
    re-reading the labelling heuristic from its input.

    Lowercasing, stemming, lemmatisation, and stopword removal are
    intentionally **not** applied because Urdu has no case distinction and
    because no production-quality Urdu stemmer / lemmatiser / stopword list
    exists.

    ### Expected limitations

    - Accuracy on this corpus is upper-bounded by the noise floor of the
      labelling heuristic.
    - Macro-F1 on rare classes (*Surprise*, *Fear*) will lag substantially
      behind weighted-F1.
    - Models trained here generalise to weakly-labelled Urdu Twitter; transfer
      to expert-annotated benchmarks (e.g. Roman-Urdu reviews) is an open
      question we will revisit in the discussion section of the technical
      report.
    """))

    _build("01_dataset_analysis.ipynb", cells)


# ---------------------------------------------------------------------------
# 02_model_implementation.ipynb
# ---------------------------------------------------------------------------
def build_model_impl() -> None:
    cells: list[tuple[str, str]] = []

    cells.append(("md", r"""
    # Assignment 3 — Task 3
    ## Implementation of the Proposed NLP Architecture

    **Students:** Raqib Hayat (NUM-BSCS-2022-40), Abu Bakar (NUM-BSCS-2022-41)

    This notebook trains every model from `04_architecture_math.md` end-to-end
    on the SentiUrdu-1M dataset and saves test-set predictions for
    `03_evaluation.ipynb`.

    For each of the two tasks we train seven models:

    | Family | Models |
    |---|---|
    | Classical (TF-IDF) | Logistic Regression, Linear SVM |
    | Deep Learning      | Multi-kernel CNN, BiLSTM + attention |
    | Transformer        | mBERT, XLM-RoBERTa, Urdu-RoBERTa |

    All models share:
    - the same 8-step preprocessing pipeline (`preprocessing.py`),
    - the same 70/15/15 stratified split (`seed=42`),
    - inverse-frequency class weighting in the loss,
    - identical metric computation in §10.

    **Runtime knob.** Set `TRAIN_SUBSET` near the top of §1 to a smaller value
    (e.g. 100_000) if you only have a few minutes; set it to `None` to use
    the full corpus.
    """))

    cells.append(("md", "## 1 — Setup"))
    cells.append(("code", HEADER_SETUP))
    cells.append(("code", r"""
    import json, time, gc
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # ───── runtime knob ─────
    TRAIN_SUBSET = None          # set e.g. 200_000 to debug end-to-end quickly
    # Transformers can be very compute-heavy; if set, also caps the corpus seen
    # by the transformer family only (classical + deep still use full data).
    TRANSFORMER_TRAIN_SUBSET = None
    # ────────────────────────
    """))

    cells.append(("md", "## 2 — Load and label the dataset"))
    cells.append(("code", r"""
    %%time
    raw = pd.read_csv(DATA_CSV, dtype=str, keep_default_na=False, na_values=[""])
    raw["clean_text"] = pp.preprocess_series(raw["Text"])
    raw["sentiment"]  = raw.apply(pp.build_sentiment_label, axis=1)
    raw["emotion"]    = raw["Category"].map(pp.majority_emotion)

    # Drop empty cleaned tweets (very few)
    raw = raw[raw["clean_text"].str.len() > 0].reset_index(drop=True)
    print(f"After dropping empty cleaned text: {len(raw):,} rows")

    sentiment_df = raw.dropna(subset=["sentiment"]).reset_index(drop=True).copy()
    emotion_df   = raw.dropna(subset=["emotion"]).reset_index(drop=True).copy()
    print(f"Sentiment task: {len(sentiment_df):,} labelled rows")
    print(f"Emotion task  : {len(emotion_df):,} labelled rows")
    """))
    cells.append(("code", r"""
    # Map text labels to integer ids
    from config import SENT2ID, ID2SENT, EMOTION2ID, ID2EMOTION
    sentiment_df["label"] = sentiment_df["sentiment"].map(SENT2ID)
    emotion_df["label"]   = emotion_df["emotion"].map(EMOTION2ID)

    print("Sentiment labels:", dict(sentiment_df["label"].value_counts().sort_index()))
    print("Emotion labels  :", dict(emotion_df["label"].value_counts().sort_index()))
    """))

    cells.append(("md", "## 3 — Stratified 70/15/15 split and persist"))
    cells.append(("code", r"""
    def stratified_split(df, label_col="label", seed=SEED):
        train, temp = train_test_split(df, test_size=0.30, stratify=df[label_col], random_state=seed)
        val, test  = train_test_split(temp, test_size=0.50, stratify=temp[label_col], random_state=seed)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    splits = {}
    for task, df in [("sentiment", sentiment_df), ("emotion", emotion_df)]:
        tr, va, te = stratified_split(df)
        splits[task] = {"train": tr, "val": va, "test": te}
        for name, sub in splits[task].items():
            sub[["clean_text", "label"]].to_parquet(CACHE_DIR / f"{task}_{name}.parquet", index=False)
        print(f"{task}: train={len(tr):,}  val={len(va):,}  test={len(te):,}")
    """))
    cells.append(("code", r"""
    if TRAIN_SUBSET:
        for task in splits:
            tr = splits[task]["train"]
            sub = tr.groupby("label", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, int(len(g) * TRAIN_SUBSET / len(tr)))),
                                   random_state=SEED)
            )
            splits[task]["train"] = sub.reset_index(drop=True)
            print(f"  {task}: subsampled train -> {len(sub):,}")
    """))

    cells.append(("md", r"""
    ## 4 — Classical: TF-IDF + Logistic Regression and Linear SVM

    Section 4.1 / 4.2 of the math doc. We use the *same* TF-IDF vectoriser
    for both classical models on each task; only the classifier head changes.
    """))
    cells.append(("code", r"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    def run_classical(task: str):
        tr, va, te = splits[task]["train"], splits[task]["val"], splits[task]["test"]
        labels = sorted(tr["label"].unique())
        print(f"\n=== {task.upper()} — TF-IDF + LR/SVM ===")

        vec_path = MODEL_DIR / f"tfidf_{task}.joblib"
        if vec_path.exists():
            vec = joblib.load(vec_path)
            Xtr = vec.transform(tr["clean_text"])
        else:
            vec = TfidfVectorizer(**config.TFIDF)
            Xtr = vec.fit_transform(tr["clean_text"])
            joblib.dump(vec, vec_path)
        Xva = vec.transform(va["clean_text"])
        Xte = vec.transform(te["clean_text"])
        ytr, yte = tr["label"].values, te["label"].values

        # ----- Logistic Regression -----
        lr = LogisticRegression(**config.LR)
        t0 = time.time(); lr.fit(Xtr, ytr); print(f"  LR fit: {time.time()-t0:.1f}s")
        pred_lr = lr.predict(Xte)
        joblib.dump(lr, MODEL_DIR / f"lr_{task}.joblib")
        pd.DataFrame({"y_true": yte, "y_pred": pred_lr}).to_csv(
            RESULTS_DIR / f"pred_{task}_lr.csv", index=False
        )

        # ----- Linear SVM -----
        svm_base = LinearSVC(**config.SVC_PARAMS)
        svm = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid", n_jobs=-1)
        t0 = time.time(); svm.fit(Xtr, ytr); print(f"  SVM fit: {time.time()-t0:.1f}s")
        pred_svm = svm.predict(Xte)
        joblib.dump(svm, MODEL_DIR / f"svm_{task}.joblib")
        pd.DataFrame({"y_true": yte, "y_pred": pred_svm}).to_csv(
            RESULTS_DIR / f"pred_{task}_svm.csv", index=False
        )

        from sklearn.metrics import f1_score, accuracy_score
        for name, p in [("LR", pred_lr), ("SVM", pred_svm)]:
            print(f"  {name:>3}  acc={accuracy_score(yte, p):.4f}  "
                  f"macro-F1={f1_score(yte, p, average='macro'):.4f}  "
                  f"weighted-F1={f1_score(yte, p, average='weighted'):.4f}")

        return vec
    """))
    cells.append(("code", r"""
    vec_sent = run_classical("sentiment")
    vec_emo  = run_classical("emotion")
    """))

    cells.append(("md", r"""
    ## 5 — fastText Urdu embeddings download

    The CNN and BiLSTM are initialised from the publicly released fastText
    `cc.ur.300` vectors (Facebook AI). The file is ≈ 600 MB compressed; cached.
    """))
    cells.append(("code", r"""
    from urllib.request import urlretrieve
    if not config.FASTTEXT_PATH.exists():
        print(f"Downloading fastText Urdu vectors to {config.FASTTEXT_PATH} ...")
        urlretrieve(config.FASTTEXT_URL, config.FASTTEXT_PATH)
        print("  done.")
    else:
        print(f"Found fastText vectors at {config.FASTTEXT_PATH} "
              f"({config.FASTTEXT_PATH.stat().st_size/1e6:.1f} MB)")
    """))

    cells.append(("md", "## 6 — Deep Learning: CNN + BiLSTM (PyTorch)"))
    cells.append(("code", r"""
    from torch.utils.data import DataLoader
    import train_dl as tdl
    from train_dl import Vocab, TweetDataset, make_collate, load_fasttext_vectors, compute_class_weights
    from models_dl import TextCNN, BiLSTMAttn

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def build_dataloaders(task: str, batch_size: int):
        tr, va, te = splits[task]["train"], splits[task]["val"], splits[task]["test"]
        vocab_path = CACHE_DIR / f"vocab_{task}.json"
        if vocab_path.exists():
            vocab = Vocab.from_json(vocab_path)
        else:
            vocab = Vocab.build(tr["clean_text"], max_size=config.VOCAB_SIZE, min_freq=2)
            vocab.to_json(vocab_path)
        print(f"  {task} vocab: {len(vocab):,} tokens (pad_id={vocab.pad_id}, unk_id={vocab.unk_id})")

        collate = make_collate(vocab.pad_id, config.MAX_LEN)
        def loader(df, shuffle):
            ds = TweetDataset(df["clean_text"].tolist(), df["label"].tolist(), vocab, config.MAX_LEN)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2,
                              pin_memory=True, collate_fn=collate)
        return vocab, loader(tr, True), loader(va, False), loader(te, False)

    def fasttext_for(vocab: Vocab):
        emb_path = EMBED_DIR / f"ft_matrix_{len(vocab)}.pt"
        if emb_path.exists():
            return torch.load(emb_path)
        E = load_fasttext_vectors(config.FASTTEXT_PATH, vocab, dim=config.EMBED_DIM)
        torch.save(E, emb_path)
        return E

    def run_dl(task: str, model_name: str):
        num_classes = 3 if task == "sentiment" else 6
        cfg = config.CNN if model_name == "cnn" else config.BILSTM
        print(f"\n=== {task.upper()} — {model_name.upper()} ===")

        vocab, tl, vl, tel = build_dataloaders(task, cfg["batch_size"])
        E = fasttext_for(vocab)
        if model_name == "cnn":
            model = TextCNN(
                vocab_size=len(vocab), num_classes=num_classes,
                embed_dim=config.EMBED_DIM,
                filter_sizes=cfg["filter_sizes"], num_filters=cfg["num_filters"],
                dropout=cfg["dropout"], padding_idx=vocab.pad_id,
                pretrained_embeddings=E,
            )
            needs_mask = False
        else:
            model = BiLSTMAttn(
                vocab_size=len(vocab), num_classes=num_classes,
                embed_dim=config.EMBED_DIM,
                hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                dropout=cfg["dropout"], padding_idx=vocab.pad_id,
                pretrained_embeddings=E,
            )
            needs_mask = True
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable params: {n_params:,}")

        ytr_arr = np.asarray(splits[task]["train"]["label"].values, dtype=np.int64)
        cw = compute_class_weights(ytr_arr, num_classes)
        print(f"  class weights: {cw.tolist()}")

        out = tdl.train_model(
            model, tl, vl, class_weights=cw,
            lr=cfg["lr"], epochs=cfg["epochs"], patience=cfg["patience"],
            device=DEVICE, needs_mask=needs_mask,
        )

        y_true, y_pred = tdl.predict(model, tel, DEVICE, needs_mask)
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
            RESULTS_DIR / f"pred_{task}_{model_name}.csv", index=False
        )
        torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_{task}.pt")
        with open(MODEL_DIR / f"{model_name}_{task}.history.json", "w", encoding="utf-8") as f:
            json.dump({"history": out["history"], "best_val_acc": out["best_val_acc"]}, f)

        from sklearn.metrics import f1_score, accuracy_score
        print(f"  TEST  acc={accuracy_score(y_true, y_pred):.4f}  "
              f"macro-F1={f1_score(y_true, y_pred, average='macro'):.4f}  "
              f"weighted-F1={f1_score(y_true, y_pred, average='weighted'):.4f}")

        del model, tl, vl, tel
        torch.cuda.empty_cache(); gc.collect()
    """))
    cells.append(("code", r"""
    run_dl("sentiment", "cnn")
    run_dl("sentiment", "bilstm")
    """))
    cells.append(("code", r"""
    run_dl("emotion", "cnn")
    run_dl("emotion", "bilstm")
    """))

    cells.append(("md", r"""
    ## 7 — Transformer fine-tuning: mBERT, XLM-RoBERTa, Urdu-RoBERTa

    HuggingFace Trainer with class-weighted cross-entropy, fp16, and the
    standard warm-up + linear-decay schedule (§6.3 of the math doc).
    """))
    cells.append(("code", r"""
    from train_transformer import TransformerCfg, finetune_transformer

    def transformer_class_weights(labels_arr, num_classes):
        from sklearn.utils.class_weight import compute_class_weight
        cls = np.arange(num_classes)
        return compute_class_weight(class_weight="balanced", classes=cls, y=labels_arr).tolist()

    def run_transformer(task: str, model_key: str):
        cfg_dict = config.TRANSFORMERS[model_key]
        cfg = TransformerCfg(name=model_key, max_len=config.TRANSFORMER_MAX_LEN, **cfg_dict)
        tr, va, te = splits[task]["train"], splits[task]["val"], splits[task]["test"]
        if TRANSFORMER_TRAIN_SUBSET:
            tr = tr.groupby("label", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(1, int(len(g)*TRANSFORMER_TRAIN_SUBSET/len(tr)))),
                                   random_state=SEED)
            ).reset_index(drop=True)

        if task == "sentiment":
            id2label = {i: l for i, l in enumerate(["Negative", "Neutral", "Positive"])}
        else:
            id2label = {i: l for i, l in enumerate(EMOTION_LABELS)}
        label2id = {v: k for k, v in id2label.items()}

        cw = transformer_class_weights(tr["label"].values, num_classes=len(id2label))
        print(f"  {task}/{model_key}: train={len(tr):,}, weights={cw}")

        res = finetune_transformer(
            cfg=cfg, train_df=tr, val_df=va, test_df=te,
            text_col="clean_text", label_col="label",
            id2label=id2label, label2id=label2id,
            class_weights=cw, output_root=MODEL_DIR,
            task_name=task,
        )
        # Save predictions in the standard location too:
        preds = pd.read_csv(Path(res["output_dir"]) / "test_predictions.csv")
        preds.to_csv(RESULTS_DIR / f"pred_{task}_{model_key}.csv", index=False)
        return res
    """))
    cells.append(("code", r"""
    # Sentiment task
    run_transformer("sentiment", "mbert")
    """))
    cells.append(("code", r"""
    run_transformer("sentiment", "xlm-r")
    """))
    cells.append(("code", r"""
    run_transformer("sentiment", "urdu-roberta")
    """))
    cells.append(("code", r"""
    # Emotion task
    run_transformer("emotion", "mbert")
    """))
    cells.append(("code", r"""
    run_transformer("emotion", "xlm-r")
    """))
    cells.append(("code", r"""
    run_transformer("emotion", "urdu-roberta")
    """))

    cells.append(("md", "## 8 — Inference demo on hand-crafted tweets"))
    cells.append(("code", r"""
    import joblib
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    demo = [
        "آج کا دن بہت اچھا تھا یہ واقعی خوبصورت لمحہ ہے",         # positive / joy
        "حکومت کی پالیسیاں بالکل غلط ہیں کوئی بہتری نہیں آئی",     # negative / anger
        "یہ خبر سن کر بہت حیرت ہوئی واقعی عجیب بات ہے",            # neutral / surprise
        "والد کی محبت سے زیادہ کوئی نعمت نہیں",                    # positive / love
        "ہمارا ساتھی شدید بیمار ہو گیا ہے اللہ کرم کرے",            # negative / fear or sadness
    ]
    print("Demo tweets (cleaned):")
    for t in demo:
        print("  ", pp.preprocess_tweet(t))
    """))
    cells.append(("code", r"""
    # Pick the best transformer per task by val macro-F1 logged in metrics.json
    import glob
    def pick_best(task):
        cands = []
        for d in glob.glob(str(MODEL_DIR / f"*_{task}")):
            mp = Path(d) / "metrics.json"
            if mp.exists():
                m = json.loads(mp.read_text(encoding="utf-8"))
                cands.append((m["val"].get("eval_f1_macro", 0), d, m["model"]))
        cands.sort(reverse=True)
        return cands[0] if cands else None

    for task in ("sentiment", "emotion"):
        pick = pick_best(task)
        if pick is None:
            print(f"  no transformer trained yet for {task}")
            continue
        score, dpath, mname = pick
        print(f"\nBest transformer for {task}: {mname} (val_f1_macro={score:.4f})  -> {dpath}")
        tok = AutoTokenizer.from_pretrained(str(Path(dpath) / "best"))
        mdl = AutoModelForSequenceClassification.from_pretrained(str(Path(dpath) / "best"))
        mdl.eval()
        with torch.no_grad():
            for t in demo:
                inp = tok([pp.preprocess_tweet(t)], padding=True, truncation=True,
                          max_length=config.TRANSFORMER_MAX_LEN, return_tensors="pt")
                pred_id = int(mdl(**inp).logits.argmax(-1).item())
                lab = mdl.config.id2label[pred_id]
                print(f"  [{lab:>10}]   {t}")
    """))

    cells.append(("md", "## 9 — Summary of where everything was saved"))
    cells.append(("code", r"""
    print("Saved test predictions in", RESULTS_DIR)
    for p in sorted(RESULTS_DIR.glob("pred_*.csv")):
        print("  ", p.name)
    print("\nSaved models in", MODEL_DIR)
    for p in sorted(MODEL_DIR.iterdir()):
        print("  ", p.name)
    """))

    _build("02_model_implementation.ipynb", cells)


# ---------------------------------------------------------------------------
# 03_evaluation.ipynb
# ---------------------------------------------------------------------------
def build_evaluation() -> None:
    cells: list[tuple[str, str]] = []

    cells.append(("md", r"""
    # Assignment 3 — Task 4
    ## Experimental Results and Performance Evaluation

    This notebook **does not** retrain any model. It reads the saved
    `outputs/results/pred_<task>_<model>.csv` files produced by
    `02_model_implementation.ipynb` and produces:

    1. A leaderboard table per task.
    2. Per-class classification reports.
    3. Raw and row-normalised confusion matrices.
    4. A qualitative error analysis on misclassified tweets.
    5. (Bonus) Training curves for the deep learning models.

    Every metric is computed with scikit-learn on the same held-out test
    split (`seed=42`, stratified 70/15/15) so the numbers are directly
    comparable.
    """))

    cells.append(("md", "## 0 — Setup"))
    cells.append(("code", HEADER_SETUP))
    cells.append(("code", r"""
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, classification_report,
        confusion_matrix, f1_score,
    )
    sns.set_theme(context="notebook", style="whitegrid", palette="viridis")
    plt.rcParams["figure.dpi"] = config.FIG_DPI

    MODEL_DISPLAY = {
        "lr": "Logistic Reg.",
        "svm": "Linear SVM",
        "cnn": "Text-CNN",
        "bilstm": "BiLSTM-Attn",
        "mbert": "mBERT",
        "xlm-r": "XLM-R",
        "urdu-roberta": "Urdu-RoBERTa",
    }
    MODEL_ORDER = ["lr", "svm", "cnn", "bilstm", "mbert", "xlm-r", "urdu-roberta"]
    """))

    cells.append(("md", "## 1 — Load all prediction files"))
    cells.append(("code", r"""
    def load_predictions(task: str):
        rows = []
        for m in MODEL_ORDER:
            p = RESULTS_DIR / f"pred_{task}_{m}.csv"
            if p.exists():
                rows.append((m, pd.read_csv(p)))
            else:
                print(f"  ⚠  missing {p.name}")
        return rows

    sentiment_preds = load_predictions("sentiment")
    emotion_preds   = load_predictions("emotion")
    print(f"Sentiment models loaded: {[m for m,_ in sentiment_preds]}")
    print(f"Emotion models loaded  : {[m for m,_ in emotion_preds]}")
    """))

    cells.append(("md", "## 2 — Leaderboard tables (Section 1 of the Task 4 spec)"))
    cells.append(("code", r"""
    def leaderboard(preds, label_names):
        rows = []
        for name, df in preds:
            y, yhat = df["y_true"].values, df["y_pred"].values
            acc = accuracy_score(y, yhat)
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(y, yhat, average="weighted", zero_division=0)
            rows.append({
                "Model": MODEL_DISPLAY.get(name, name),
                "Accuracy": acc,
                "Precision (macro)": p_m,
                "Recall (macro)":    r_m,
                "F1 (macro)":        f1_m,
                "F1 (weighted)":     f1_w,
            })
        out = pd.DataFrame(rows).round(4)
        return out

    sent_lb = leaderboard(sentiment_preds, ["Negative", "Neutral", "Positive"])
    emo_lb  = leaderboard(emotion_preds,   EMOTION_LABELS)
    sent_lb.to_csv(RESULTS_DIR / "leaderboard_sentiment.csv", index=False)
    emo_lb.to_csv(RESULTS_DIR / "leaderboard_emotion.csv",   index=False)

    print("Sentiment (3-class) leaderboard:")
    display(sent_lb)
    print("\nEmotion (6-class) leaderboard:")
    display(emo_lb)
    """))

    cells.append(("md", "## 3 — Per-class classification reports"))
    cells.append(("code", r"""
    def show_reports(preds, label_names):
        for name, df in preds:
            print(f"\n— {MODEL_DISPLAY.get(name, name)} —")
            print(classification_report(
                df["y_true"], df["y_pred"], target_names=label_names, digits=4, zero_division=0
            ))

    print("=" * 70); print("SENTIMENT — per-class classification reports"); print("=" * 70)
    show_reports(sentiment_preds, ["Negative", "Neutral", "Positive"])
    print("\n" + "=" * 70); print("EMOTION — per-class classification reports"); print("=" * 70)
    show_reports(emotion_preds, EMOTION_LABELS)
    """))

    cells.append(("md", "## 4 — Confusion matrices"))
    cells.append(("code", r"""
    def plot_cm_grid(preds, label_names, task: str):
        n = len(preds)
        if n == 0: return
        ncols = min(4, n); nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)
        axes = axes.flat
        for ax, (name, df) in zip(axes, preds):
            cm = confusion_matrix(df["y_true"], df["y_pred"], labels=list(range(len(label_names))))
            cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                        xticklabels=label_names, yticklabels=label_names, cbar=False)
            ax.set_title(f"{MODEL_DISPLAY.get(name, name)} (row-normalised)")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.tick_params(axis="x", labelrotation=30)
        for j in range(len(preds), len(list(axes))):
            axes[j].axis("off")
        fig.suptitle(f"{task.title()} task — confusion matrices", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig15_cm_{task}.{config.FIG_FORMAT}", bbox_inches="tight")
        plt.show()

    plot_cm_grid(sentiment_preds, ["Negative", "Neutral", "Positive"], "sentiment")
    plot_cm_grid(emotion_preds,   EMOTION_LABELS,                    "emotion")
    """))

    cells.append(("md", "## 5 — Performance analysis and interpretation"))
    cells.append(("code", r"""
    # Macro vs weighted F1 across models (Section 4 of the Task 4 spec)
    def f1_compare_plot(lb, task):
        m = lb.melt(id_vars="Model", value_vars=["F1 (macro)", "F1 (weighted)"],
                    var_name="metric", value_name="score")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(data=m, x="Model", y="score", hue="metric", ax=ax, palette="viridis")
        ax.set_title(f"{task} — macro vs weighted F1 across models")
        ax.set_ylim(0, 1)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig16_f1_compare_{task.lower()}.{config.FIG_FORMAT}", bbox_inches="tight")
        plt.show()

    f1_compare_plot(sent_lb, "Sentiment")
    f1_compare_plot(emo_lb,  "Emotion")
    """))
    cells.append(("md", r"""
    **How to read these results.**

    1. **Classical baselines** (LR / Linear SVM) define a strong sparse-feature
       floor. If they are competitive with deep models it usually means the task
       is decided by surface lexical cues — TF-IDF can pick those up perfectly.
    2. **CNN vs BiLSTM** with fastText embeddings tests whether *local n-gram
       patterns* (CNN) or *long-range sequential context* (BiLSTM-attn) help.
       In short Urdu tweets the gap is usually small.
    3. **Transformers** are expected to win on macro-F1, especially on the rare
       emotion classes (*Fear*, *Surprise*), because their contextual subword
       embeddings can disambiguate words that change polarity with context.
    4. **Where models disagree** is informative. Use the confusion matrices
       above and the misclassified-sample analysis in §6 to see *which*
       classes a given model confuses.
    5. **Noisy-label ceiling.** Because SentiUrdu-1M labels are weakly
       supervised, the headline accuracy on this test set is upper-bounded by
       the label noise — Section 7 of the dataset notebook discussed this in
       detail.
    """))

    cells.append(("md", "## 6 — Error analysis on misclassified tweets"))
    cells.append(("code", r"""
    def load_test_text(task):
        df = pd.read_parquet(CACHE_DIR / f"{task}_test.parquet")
        return df.reset_index(drop=True)

    def sample_errors(task, model_name, k=8, label_names=None):
        test = load_test_text(task)
        preds = pd.read_csv(RESULTS_DIR / f"pred_{task}_{model_name}.csv")
        m = preds["y_true"].values != preds["y_pred"].values
        err = test[m].copy()
        err["pred"] = preds.loc[m, "y_pred"].values
        err["true"] = preds.loc[m, "y_true"].values
        if label_names:
            err["pred"] = err["pred"].map(lambda i: label_names[i])
            err["true"] = err["true"].map(lambda i: label_names[i])
        return err.sample(min(k, len(err)), random_state=SEED)

    print("=== SENTIMENT misclassifications (best transformer) ===")
    best_model_sent = sent_lb.iloc[sent_lb["F1 (macro)"].idxmax()]["Model"]
    inv = {v: k for k, v in MODEL_DISPLAY.items()}
    best_key = inv.get(best_model_sent, "mbert")
    print(sample_errors("sentiment", best_key, k=8,
                        label_names=["Negative", "Neutral", "Positive"])[["clean_text", "true", "pred"]].to_string())

    print("\n=== EMOTION misclassifications (best transformer) ===")
    best_model_emo = emo_lb.iloc[emo_lb["F1 (macro)"].idxmax()]["Model"]
    best_key_emo = inv.get(best_model_emo, "mbert")
    print(sample_errors("emotion", best_key_emo, k=8, label_names=EMOTION_LABELS)[["clean_text", "true", "pred"]].to_string())
    """))

    cells.append(("md", r"""
    ### Common error patterns to look for

    - **Negation flips.** Urdu negators *نہیں*, *مت*, *کبھی نہ* invert polarity
      but TF-IDF treats them as ordinary tokens — expect false-positive
      *Positive* predictions on negated joy phrases.
    - **Sarcasm.** A surface-positive sentence with a negative intent.
      Transformers handle this slightly better, but error rate is still high.
    - **Code-mixing.** Latin-script English fragments cause sparse-feature
      models to fall back to the *Neutral* class.
    - **Religious / poetic register.** *Love*-labelled tweets often look
      semantically identical to *Joy*; this is a known artefact of the
      emoji-derived labelling heuristic.
    """))

    cells.append(("md", "## 7 — Bonus — Training curves for the deep models"))
    cells.append(("code", r"""
    # Loss / val-accuracy curves are saved as JSON next to the .pt checkpoints
    import glob
    for task in ("sentiment", "emotion"):
        for name in ("cnn", "bilstm"):
            hist_path = MODEL_DIR / f"{name}_{task}.history.json"
            if not hist_path.exists():
                continue
            h = json.loads(hist_path.read_text(encoding="utf-8"))
            h = h.get("history", h)  # support both nested and flat layouts
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax2 = ax1.twinx()
            ax1.plot(h["train_loss"], color="#d62728", label="train loss"); ax1.set_ylabel("loss", color="#d62728")
            ax2.plot(h["val_acc"],    color="#2ca02c", label="val acc");    ax2.set_ylabel("val acc", color="#2ca02c")
            ax1.set_xlabel("epoch")
            ax1.set_title(f"{task} — {name} training curve")
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"fig17_curve_{task}_{name}.{config.FIG_FORMAT}", bbox_inches="tight")
            plt.show()
    """))

    cells.append(("md", "## 8 — Save final results to JSON for the report"))
    cells.append(("code", r"""
    final = {
        "sentiment": sent_lb.to_dict(orient="records"),
        "emotion":   emo_lb.to_dict(orient="records"),
    }
    (REPORT_DIR := config.REPORT_DIR).mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print("Saved", REPORT_DIR / "results_summary.json")
    """))

    _build("03_evaluation.ipynb", cells)


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    build_eda()
    build_model_impl()
    build_evaluation()
    print("done.")
