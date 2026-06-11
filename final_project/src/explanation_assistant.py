"""Lightweight rule-based and template-based explanation assistant.

Helps interpret model predictions, error patterns, performance characteristics, and project-level insights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def explain_prediction(
    text: str,
    true_label: str | None = None,
    predicted_label: str | None = None,
    confidence: float | None = None,
    model_name: str | None = None,
) -> str:
    """Generate a human-readable explanation of a model prediction."""
    parts = []
    model_disp = model_name or "The model"
    
    if predicted_label:
        conf_str = f" with {confidence:.2f} confidence" if confidence is not None else ""
        parts.append(f"{model_disp} predicted the sentiment as **{predicted_label}**{conf_str}.")
    
    if true_label and predicted_label:
        if true_label == predicted_label:
            parts.append("This matches the ground truth label, indicating a correct classification.")
        else:
            parts.append(f"This is an error; the actual ground truth label is **{true_label}**.")

    # Rule-based text characteristics
    tokens = str(text).split()
    length = len(tokens)
    
    parts.append(f"The input text contains {length} tokens.")
    
    # Check for sentiment indicators (in Urdu, e.g. نہیں, اچھا, برا)
    negations = ["نہ", "نہیں", "بغیر", "علاوہ"]
    has_negation = any(neg in tokens for neg in negations)
    if has_negation:
        parts.append("Note: The text contains negation words (e.g., 'نہیں'), which can invert polarity and complicate classification.")

    if length <= 4:
        parts.append("The text is short, which often leads to low context and ambiguity.")
    elif length >= 25:
        parts.append("The text is relatively long and may contain multiple clauses or conflicting sentiments.")

    return " ".join(parts)


def explain_error(row: Mapping[str, Any] | pd.Series) -> str:
    """Generate an explanation for a misclassified example using error metrics."""
    true_label = row.get("true_label")
    predicted_label = row.get("predicted_label")
    confidence = row.get("confidence")
    length = row.get("text_length")
    
    parts = [
        f"Error Analysis: True label is **{true_label}**, but model predicted **{predicted_label}** "
        f"(confidence: {confidence:.2f}, text length: {length} tokens)."
    ]
    
    causes = []
    
    # 1. Minority-class confusion
    if true_label == "Neutral":
        causes.append("minority-class confusion (Neutral class represents <1% of the dataset)")
        
    # 2. High-confidence model error
    if row.get("is_high_confidence_error") or (confidence is not None and confidence >= 0.80 and not row.get("is_correct", True)):
        causes.append("high-confidence model error, possibly indicating label noise or complex sarcastic context")
        
    # 3. Short text ambiguity
    if row.get("is_short_text_error") or (length is not None and length <= 3):
        causes.append("short ambiguous tweet lacking sufficient semantic context")
        
    # 4. Polarity ambiguity or swap
    if {true_label, predicted_label} == {"Positive", "Negative"}:
        causes.append("polarity ambiguity or swap (confusing strong positive expressions with negative ones)")
        
    # General factors
    causes.append("class imbalance and potential weak-label noise from the emoji-based SentiUrdu-1M dataset construction")
    
    parts.append("Likely contributing factors: " + "; ".join(causes) + ".")
    return " ".join(parts)


def generate_model_summary(model_name: str, metrics: Mapping[str, Any]) -> str:
    """Generate a plain-English summary of a model's strengths and weaknesses."""
    test_metrics = metrics.get("test", {})
    macro_f1 = test_metrics.get("macro_f1", 0.0)
    acc = test_metrics.get("accuracy", 0.0)
    per_class = test_metrics.get("per_class", {})
    
    neutral_f1 = per_class.get("Neutral", {}).get("f1", 0.0)
    negative_f1 = per_class.get("Negative", {}).get("f1", 0.0)
    positive_f1 = per_class.get("Positive", {}).get("f1", 0.0)
    
    summary = (
        f"### Model Summary for {model_name}\n"
        f"- **Overall Performance**: Achieves a Test Accuracy of {acc:.4f} and a Test Macro-F1 of {macro_f1:.4f}.\n"
        f"- **Class-Specific Strengths**: "
    )
    
    strengths = []
    if positive_f1 > 0.80:
        strengths.append(f"highly effective on the majority Positive class (F1: {positive_f1:.4f})")
    if negative_f1 > 0.80:
        strengths.append(f"robust on the Negative class (F1: {negative_f1:.4f})")
        
    if strengths:
        summary += ", ".join(strengths) + ".\n"
    else:
        summary += "moderate performance across majority polarity classes.\n"
        
    summary += f"- **Weaknesses**: Struggled significantly with the minority 'Neutral' class (F1: {neutral_f1:.4f}). This is primarily due to class imbalance."
    return summary


def generate_final_insight(comparison_df: pd.DataFrame) -> str:
    """Generate a final project-level insight based on the model comparison leaderboard."""
    if comparison_df.empty:
        return "No leaderboard data available to generate project insights."
        
    best_row = comparison_df.iloc[0]
    best_name = best_row["model_name"]
    best_f1 = best_row["test_macro_f1"]
    
    svm_row = comparison_df[comparison_df["model_name"] == "linear_svm"]
    
    insight = (
        f"### Project-Level Insights\n"
        f"- The top-performing model on the leaderboard is **{best_name}** with a Test Macro-F1 of **{best_f1:.4f}**.\n"
    )
    
    if not svm_row.empty:
        svm_f1 = svm_row.iloc[0]["test_macro_f1"]
        if best_name == "linear_svm":
            insight += (
                f"- The statistical Linear SVM remains the strongest model in this study ({svm_f1:.4f} Macro-F1). "
                f"This suggests that simple n-gram representation is a highly competitive baseline for this noisy dataset."
            )
        else:
            diff = best_f1 - svm_f1
            insight += (
                f"- **{best_name}** outperformed the Linear SVM baseline by a margin of **{diff:.4f}** in Macro-F1. "
                f"This highlights the value of pre-trained contextual representations or advanced deep architectures."
            )
    else:
        insight += f"- **{best_name}** represents the current state of the art in this classification suite."
        
    return insight


def export_explanation_samples(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
    model_name: str,
    max_samples: int = 5,
) -> None:
    """Export a set of sample explanations (correct, wrong, high confidence errors)."""
    samples = []
    
    # Correct predictions
    correct = predictions_df[predictions_df["is_correct"]].head(max_samples)
    for _, row in correct.iterrows():
        samples.append({
            "text": row["clean_text"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "is_correct": True,
            "explanation": explain_prediction(
                row["clean_text"], row["true_label"], row["predicted_label"], row["confidence"], model_name
            )
        })
        
    # Errors
    errors = predictions_df[~predictions_df["is_correct"]].head(max_samples)
    for _, row in errors.iterrows():
        # Build a row dictionary matching the explain_error expected columns
        row_dict = dict(row)
        row_dict["is_high_confidence_error"] = float(row["confidence"]) >= 0.80
        row_dict["is_short_text_error"] = int(row["text_length"]) <= 3
        row_dict["is_minority_class_error"] = row["true_label"] == "Neutral"
        
        samples.append({
            "text": row["clean_text"],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "is_correct": False,
            "explanation": explain_error(row_dict)
        })
        
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
