"""Generate the publication-ready architecture diagram in SVG, PDF, and PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "background": "#FBFCFE",
    "ink": "#243247",
    "muted": "#65758B",
    "line": "#9AABBE",
    "shadow": "#DCE3EC",
    "input_fill": "#EAF3FC",
    "input_edge": "#86AED2",
    "clean_fill": "#EDF4FA",
    "clean_edge": "#91ABC3",
    "label_fill": "#FFF7DF",
    "label_edge": "#D8BA69",
    "classical_fill": "#EDF5FD",
    "classical_edge": "#8DB4D8",
    "neural_fill": "#EAF8F4",
    "neural_edge": "#82BCAE",
    "transformer_fill": "#F5EFFA",
    "transformer_edge": "#B69BCB",
    "eval_fill": "#EDF1F6",
    "eval_edge": "#8396AD",
    "note_fill": "#FFF2F3",
    "note_edge": "#D9A1A8",
    "accent": "#5F7898",
}


def rounded_box(ax, x, y, w, h, fill, edge, radius=0.16, shadow=True, lw=1.25):
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.07, y - 0.07),
                w,
                h,
                boxstyle=f"round,pad=0.02,rounding_size={radius}",
                linewidth=0,
                facecolor=COLORS["shadow"],
                alpha=0.42,
                zorder=1,
            )
        )
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(box)
    return box


def stage_badge(ax, x, y, number, edge):
    ax.add_patch(Circle((x, y), 0.22, facecolor="white", edgecolor=edge, linewidth=1.2, zorder=5))
    ax.text(
        x,
        y,
        number,
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=6,
    )


def card(ax, x, y, w, h, number, eyebrow, title, lines, fill, edge, title_size=11.2):
    rounded_box(ax, x, y, w, h, fill, edge)
    stage_badge(ax, x + 0.36, y + h - 0.36, number, edge)
    ax.text(
        x + 0.68,
        y + h - 0.28,
        eyebrow.upper(),
        ha="left",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["muted"],
        zorder=5,
    )
    ax.text(
        x + 0.32,
        y + h - 0.72,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=5,
    )
    start_y = y + h - 1.16
    spacing = 0.34 if len(lines) == 1 else min(0.34, (h - 1.46) / (len(lines) - 1))
    for idx, line in enumerate(lines):
        cy = start_y - idx * spacing
        ax.add_patch(Circle((x + 0.39, cy + 0.01), 0.035, facecolor=edge, edgecolor="none", zorder=5))
        ax.text(
            x + 0.54,
            cy,
            line,
            ha="left",
            va="center",
            fontsize=7.7 if len(lines) >= 4 else 7.9,
            color=COLORS["ink"],
            zorder=5,
        )


def arrow(ax, start, end, color=None, lw=1.45, mutation=11, zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color or COLORS["line"],
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def line(ax, xs, ys, color=None, lw=1.35, zorder=2):
    ax.plot(xs, ys, color=color or COLORS["line"], linewidth=lw, solid_capstyle="round", zorder=zorder)


def build():
    fig, ax = plt.subplots(figsize=(16, 9.6), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        8,
        9.68,
        "Leak-Free Urdu Sentiment and Emotion Classification Pipeline",
        ha="center",
        va="center",
        fontsize=20.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        8,
        9.30,
        "Controlled data preparation, three model families, and unified class-sensitive evaluation",
        ha="center",
        va="center",
        fontsize=9.6,
        color=COLORS["muted"],
    )
    line(ax, [1.0, 15.0], [9.05, 9.05], color="#DDE5EE", lw=0.9)

    top_y, top_h = 6.46, 2.22
    input_x, input_w = 0.55, 3.75
    clean_x, clean_w = 5.05, 5.05
    label_x, label_w = 10.85, 4.60

    card(
        ax,
        input_x,
        top_y,
        input_w,
        top_h,
        "01",
        "Input data",
        "SentiUrdu-1M",
        ["Raw Urdu tweets", "Weak emotion categories", "Emoji-confidence metadata"],
        COLORS["input_fill"],
        COLORS["input_edge"],
    )
    card(
        ax,
        clean_x,
        top_y,
        clean_w,
        top_h,
        "02",
        "Leakage-aware preprocessing",
        "Deterministic Cleaning",
        [
            "NFC normalization",
            "Remove URLs and mentions",
            "Retain hashtag text",
            "Remove emojis and digits",
            "Clean punctuation and whitespace",
        ],
        COLORS["clean_fill"],
        COLORS["clean_edge"],
    )
    card(
        ax,
        label_x,
        top_y,
        label_w,
        top_h,
        "03",
        "Targets and partitions",
        "Label Preparation",
        [
            r"298 raw forms $\rightarrow$ 6 emotions",
            r"Emotions $\rightarrow$ 3-class sentiment",
            "Stratified split: 70 / 15 / 15",
        ],
        COLORS["label_fill"],
        COLORS["label_edge"],
    )

    arrow(ax, (input_x + input_w + 0.08, top_y + top_h / 2), (clean_x - 0.12, top_y + top_h / 2))
    arrow(ax, (clean_x + clean_w + 0.08, top_y + top_h / 2), (label_x - 0.12, top_y + top_h / 2))

    branch_y, branch_h, branch_w = 3.58, 2.12, 4.25
    branch_xs = [0.55, 5.88, 11.20]
    branch_centers = [x + branch_w / 2 for x in branch_xs]

    # Clean distribution bus from label preparation into the three model families.
    label_center = label_x + label_w / 2
    bus_y = 6.08
    line(ax, [label_center, label_center], [top_y, bus_y])
    line(ax, [branch_centers[0], branch_centers[2]], [bus_y, bus_y])
    ax.add_patch(Circle((label_center, bus_y), 0.065, facecolor=COLORS["accent"], edgecolor="white", linewidth=0.8, zorder=4))
    for center in branch_centers:
        arrow(ax, (center, bus_y), (center, branch_y + branch_h + 0.04), mutation=10)

    card(
        ax,
        branch_xs[0],
        branch_y,
        branch_w,
        branch_h,
        "04A",
        "Sparse lexical baseline",
        "Classical Models",
        ["TF-IDF (1-2 grams)", "Logistic Regression", "Linear SVM"],
        COLORS["classical_fill"],
        COLORS["classical_edge"],
    )
    card(
        ax,
        branch_xs[1],
        branch_y,
        branch_w,
        branch_h,
        "04B",
        "Static embeddings + sequence models",
        "Neural Models",
        ["300-d Urdu fastText", "Text-CNN", "BiLSTM + Attention"],
        COLORS["neural_fill"],
        COLORS["neural_edge"],
    )
    card(
        ax,
        branch_xs[2],
        branch_y,
        branch_w,
        branch_h,
        "04C",
        "Contextual subword encoders",
        "Transformer Models",
        ["WordPiece / SentencePiece", "mBERT", "XLM-R", "Urdu-RoBERTa"],
        COLORS["transformer_fill"],
        COLORS["transformer_edge"],
    )

    eval_x, eval_y, eval_w, eval_h = 2.05, 1.08, 11.90, 1.72
    eval_top = eval_y + eval_h
    converge_y = 3.15
    eval_targets = [4.35, 8.0, 11.65]
    for center, target in zip(branch_centers, eval_targets):
        line(ax, [center, center], [branch_y, converge_y])
        arrow(ax, (center, converge_y), (target, eval_top + 0.04), mutation=10)

    rounded_box(
        ax,
        eval_x,
        eval_y,
        eval_w,
        eval_h,
        COLORS["eval_fill"],
        COLORS["eval_edge"],
        radius=0.20,
        lw=1.45,
    )
    stage_badge(ax, eval_x + 0.42, eval_y + eval_h - 0.40, "05", COLORS["eval_edge"])
    ax.text(
        eval_x + 0.77,
        eval_y + eval_h - 0.32,
        "UNIFIED OPTIMIZATION AND ASSESSMENT",
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=COLORS["muted"],
    )
    ax.text(
        8,
        eval_y + 0.97,
        "Class-Weighted Training and Evaluation",
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["ink"],
    )

    metrics = [
        ("Accuracy", 3.10, 1.20),
        ("Macro / weighted Precision", 5.20, 2.45),
        ("Recall", 7.52, 1.18),
        ("F1-score", 8.95, 1.18),
        ("Confusion matrices", 10.62, 1.80),
        ("Qualitative error analysis", 12.68, 2.15),
    ]
    for text, cx, width in metrics:
        rounded_box(
            ax,
            cx - width / 2,
            eval_y + 0.28,
            width,
            0.38,
            "#FFFFFF",
            "#C8D2DE",
            radius=0.11,
            shadow=False,
            lw=0.8,
        )
        ax.text(cx, eval_y + 0.47, text, ha="center", va="center", fontsize=7.2, color=COLORS["ink"], zorder=5)

    note_x, note_y, note_w, note_h = 2.05, 0.18, 11.90, 0.58
    rounded_box(
        ax,
        note_x,
        note_y,
        note_w,
        note_h,
        COLORS["note_fill"],
        COLORS["note_edge"],
        radius=0.15,
        shadow=False,
        lw=1.0,
    )
    ax.add_patch(Circle((note_x + 0.34, note_y + note_h / 2), 0.115, facecolor="#F7D9DD", edgecolor=COLORS["note_edge"], linewidth=0.9, zorder=5))
    ax.text(note_x + 0.34, note_y + note_h / 2 - 0.005, "!", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#A85D68", zorder=6)
    ax.text(
        note_x + 0.58,
        note_y + note_h / 2,
        "Leakage Prevention Note",
        ha="left",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#934F5A",
        zorder=5,
    )
    ax.text(
        note_x + 2.95,
        note_y + note_h / 2,
        "Emojis are removed before feature extraction to prevent leakage from the weak-labeling heuristic.",
        ha="left",
        va="center",
        fontsize=8.1,
        color=COLORS["ink"],
        zorder=5,
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    for extension, kwargs in {
        "svg": {},
        "pdf": {"metadata": {"Title": "Leak-Free Urdu Sentiment and Emotion Classification Pipeline"}},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(
            OUT_DIR / f"system_architecture.{extension}",
            facecolor=COLORS["background"],
            bbox_inches="tight",
            pad_inches=0.08,
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    build()
