"""Artifact-backed Streamlit demo for Urdu sentiment and emotion classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from inference import load_inference_model, preprocess_input, resolve_model_run


TASK_OPTIONS = {"Sentiment": "sentiment", "Emotion": "emotion"}
MODEL_OPTIONS = {
    "TF-IDF + Linear SVM": "linear_svm",
    "TF-IDF + Logistic Regression": "logistic_regression",
    "TF-IDF + Multinomial NB": "multinomial_nb",
    "Text-CNN": "text_cnn",
    "BiLSTM + Attention": "bilstm_attention",
    "mBERT": "mbert",
    "XLM-RoBERTa": "xlm_roberta",
    "Urdu-RoBERTa": "urdu_roberta",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def available_model_options(task: str) -> dict[str, str]:
    available: dict[str, str] = {}
    for display, model_key in MODEL_OPTIONS.items():
        try:
            resolve_model_run(PROJECT_ROOT, task, model_key)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            continue
        available[display] = model_key
    return available


@st.cache_resource(show_spinner=False)
def cached_model(task: str, model_key: str, project_root: str):
    return load_inference_model(model_key, Path(project_root), task=task)


def load_selected_model(task: str, model_key: str):
    """Load the requested task model, falling back to that task's Linear SVM."""
    try:
        return cached_model(task, model_key, str(PROJECT_ROOT)), model_key, None
    except Exception as selected_error:
        if model_key == "linear_svm":
            return None, model_key, f"Linear SVM could not be loaded: {selected_error}"
        try:
            fallback = cached_model(task, "linear_svm", str(PROJECT_ROOT))
            return fallback, "linear_svm", (
                f"{model_key} could not be loaded; the {task} Linear SVM was used instead."
            )
        except Exception as fallback_error:
            return None, model_key, (
                f"Selected model failed ({selected_error}); fallback failed ({fallback_error})."
            )


def format_model_name(model_key: str) -> str:
    for display, key in MODEL_OPTIONS.items():
        if key == model_key:
            return display
    return model_key.replace("_", " ").title()


def render_prediction(task: str, result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    label = prediction["predicted_label"]
    score = float(prediction["score"])
    score_kind = prediction["score_kind"]
    if result.get("warning"):
        st.warning(result["warning"])
    st.success(f"Predicted {task}: **{label}**")
    score_col, model_col = st.columns(2)
    score_label = "Probability" if score_kind == "probability" else "Decision score"
    score_col.metric(score_label, f"{score:.4f}")
    model_col.metric("Model used", format_model_name(result["model_key"]))
    probabilities = prediction.get("probabilities", [])
    labels = prediction.get("labels", [])
    if probabilities and len(probabilities) == len(labels):
        chart = pd.DataFrame({"Class": labels, "Score": probabilities}).set_index("Class")
        st.bar_chart(chart)
    if score_kind == "decision_score":
        st.caption("The SVM value is a normalized margin for comparison, not a calibrated probability.")


def main() -> None:
    st.set_page_config(
        page_title="Urdu Sentiment and Emotion Classification",
        page_icon="NLP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.title("Urdu NLP Project")
        selected_task_display = st.selectbox("Classification task", list(TASK_OPTIONS))
        task = TASK_OPTIONS[selected_task_display]

    results_dir = PROJECT_ROOT / "outputs" / task / "results"
    split_summary = read_json(results_dir / "split_summary.json")
    aggregate = read_json(results_dir / "aggregate_metrics.json")
    leaderboard = read_csv(results_dir / "model_comparison_leaderboard.csv")
    selected_summary = aggregate.get("selected_model", {})
    options = available_model_options(task)

    with st.sidebar:
        st.markdown(f"**Task:** {selected_task_display}")
        st.markdown("**Dataset:** SentiUrdu-1M")
        st.markdown("**Split:** Group-safe 70/15/15")
        st.markdown(
            f"**Selected model:** {format_model_name(selected_summary.get('model_name', 'linear_svm'))}"
        )
        st.divider()
        for label, key in (("Train", "train_size"), ("Validation", "validation_size"), ("Test", "test_size")):
            value = split_summary.get(key)
            st.metric(label, f"{int(value):,}" if value is not None else "Unavailable")
        st.caption(
            "Labels are weakly supervised. No human-gold evaluation is claimed. "
            "Models are ranked by validation macro-F1."
        )

    st.title("Group-Safe Urdu Sentiment and Emotion Classification")
    st.caption("Separate task pipelines, repeated seeds, saved checkpoints, and validation-only selection")

    st.markdown("## Official Benchmark")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Task", selected_task_display)
    metric_cols[1].metric("Retained rows", f"{split_summary.get('rows_after_filtering', 0):,}")
    metric_cols[2].metric(
        "Validation Macro-F1",
        f"{selected_summary.get('validation_macro_f1_mean', 0):.4f}",
    )
    metric_cols[3].metric(
        "Test Macro-F1",
        f"{selected_summary.get('test_macro_f1_mean', 0):.4f}",
    )
    interval = selected_summary.get("bootstrap_95", {})
    if interval:
        st.info(
            "Selected-model test Macro-F1 bootstrap 95% interval: "
            f"{interval.get('lower_95', 0):.4f} to {interval.get('upper_95', 0):.4f}."
        )

    st.markdown("## Interactive Inference")
    input_col, control_col = st.columns([3, 2])
    with input_col:
        user_text = st.text_area(
            "Enter an Urdu or Roman Urdu tweet",
            height=140,
            placeholder="Enter text for classification...",
            key=f"urdu_tweet_input_{task}",
        )
        clean_input = str(user_text).strip()
    with control_col:
        selected_display = st.selectbox("Saved model", list(options) or ["No model available"])
        selected_key = options.get(selected_display)
        classify_button = st.button(
            f"Classify {selected_task_display}", type="primary", width="stretch"
        )

    preview = ""
    if clean_input:
        try:
            preview = str(preprocess_input(clean_input, PROJECT_ROOT, task=task)).strip()
        except Exception as exc:
            st.warning(f"Preprocessing preview is unavailable: {exc}")

    if classify_button:
        if not clean_input:
            st.warning("Please enter some text before classifying.")
            st.session_state.pop("last_inference", None)
        elif not preview:
            st.warning("The input became empty after preprocessing. Please enter a longer Urdu tweet.")
            st.session_state.pop("last_inference", None)
        elif selected_key is None:
            st.error("No completed model artifact is available for this task.")
        else:
            model, actual_key, warning = load_selected_model(task, selected_key)
            if model is None:
                st.error(warning or "The model could not be loaded.")
            else:
                prediction = model.predict(preview)
                st.session_state["last_inference"] = {
                    "task": task,
                    "raw_text": clean_input,
                    "clean_text": preview,
                    "model_key": actual_key,
                    "prediction": prediction,
                    "warning": warning,
                }

    st.markdown("## Preprocessing Preview")
    preview_cols = st.columns(2)
    preview_cols[0].code(clean_input or "[Enter text above]", language=None)
    preview_cols[1].code(preview or "[Cleaned text appears here]", language=None)

    st.markdown("## Prediction")
    inference = st.session_state.get("last_inference")
    if inference and inference.get("task") == task:
        render_prediction(task, inference)
    else:
        st.caption("Run classification to display the saved model's prediction.")

    st.markdown("## Validation-Ranked Leaderboard")
    if leaderboard.empty:
        st.warning("Aggregate leaderboard is unavailable.")
    else:
        columns = [
            "rank",
            "model_family",
            "model_name",
            "seed_count",
            "validation_macro_f1_mean",
            "validation_macro_f1_std",
            "test_macro_f1_mean",
            "test_macro_f1_std",
        ]
        st.dataframe(
            leaderboard[columns].style.format(
                {
                    "validation_macro_f1_mean": "{:.4f}",
                    "validation_macro_f1_std": "{:.4f}",
                    "test_macro_f1_mean": "{:.4f}",
                    "test_macro_f1_std": "{:.4f}",
                }
            ).highlight_max(subset=["validation_macro_f1_mean"], color="#d9ead3"),
            hide_index=True,
            width="stretch",
        )

    st.markdown("## Method and Limitations")
    st.markdown(
        "- Duplicate-linked tweet IDs and normalized texts are confined to one split.\n"
        "- Conflicting duplicate-label groups are excluded from the official benchmark.\n"
        "- Classical and neural models use three seeds; Transformers use one resource-constrained seed.\n"
        "- Transformer runs use 50,000 training examples and one epoch.\n"
        "- Labels remain weakly supervised because independent human annotators were unavailable.\n"
        "- Predictions must not be treated as ground truth or used for decisions about individuals."
    )


if __name__ == "__main__":
    main()
