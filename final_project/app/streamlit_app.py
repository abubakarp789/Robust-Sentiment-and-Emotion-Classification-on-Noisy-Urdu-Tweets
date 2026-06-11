"""Presentation-focused Streamlit demo for Urdu tweet sentiment classification."""

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

try:
    from explanation_assistant import explain_prediction
    from inference import load_inference_model, preprocess_input
    from utils import load_config
except ImportError:
    sys.path.append(str(PROJECT_ROOT))
    from src.explanation_assistant import explain_prediction
    from src.inference import load_inference_model, preprocess_input
    from src.utils import load_config


MODEL_OPTIONS = {
    "TF-IDF + Linear SVM (Best Overall)": "linear_svm",
    "TF-IDF + Logistic Regression": "logistic_regression",
    "TF-IDF + Multinomial NB": "multinomial_nb",
    "XLM-RoBERTa (Transformer)": "xlm_roberta",
    "mBERT (Transformer)": "mbert",
}

FIGURES = [
    ("final_model_family_comparison.png", "Final model-family comparison"),
    ("baseline_model_accuracy_vs_macro_f1.png", "Baseline accuracy versus Macro-F1"),
    ("baseline_linear_svm_confusion_heatmap.png", "Linear SVM confusion matrix"),
    ("neural_vs_baseline_macro_f1.png", "Neural models versus the baseline"),
    ("transformer_vs_baseline_neural_macro_f1.png", "Transformer, neural, and baseline comparison"),
]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact, returning an empty mapping when it is unavailable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV artifact without allowing a missing or malformed file to crash the app."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def model_artifact_available(model_key: str, models_dir: Path) -> bool:
    """Return whether the minimum saved files for a model are present."""
    if model_key in {"linear_svm", "logistic_regression", "multinomial_nb"}:
        return (models_dir / f"baseline_{model_key}.joblib").is_file()

    model_dir = models_dir / f"transformer_{model_key}" / "best"
    weight_files = ("model.safetensors", "pytorch_model.bin")
    return model_dir.is_dir() and any((model_dir / name).is_file() for name in weight_files)


@st.cache_resource(show_spinner=False)
def cached_model(model_key: str, project_root: str):
    """Cache expensive model loading across Streamlit reruns."""
    return load_inference_model(model_key, Path(project_root))


def load_selected_model(model_key: str):
    """Load a selected model and fall back to Linear SVM when necessary."""
    try:
        return cached_model(model_key, str(PROJECT_ROOT)), model_key, None
    except Exception as selected_error:
        if model_key == "linear_svm":
            return None, model_key, f"Linear SVM could not be loaded: {selected_error}"
        try:
            fallback = cached_model("linear_svm", str(PROJECT_ROOT))
            warning = (
                f"The selected model ({model_key}) could not be loaded. "
                "Linear SVM was used instead."
            )
            return fallback, "linear_svm", warning
        except Exception as fallback_error:
            return None, model_key, (
                f"The selected model failed to load ({selected_error}), and the Linear SVM "
                f"fallback also failed ({fallback_error})."
            )


def format_model_name(model_key: str) -> str:
    names = {
        "linear_svm": "TF-IDF + Linear SVM",
        "logistic_regression": "TF-IDF + Logistic Regression",
        "multinomial_nb": "TF-IDF + Multinomial NB",
        "xlm_roberta": "XLM-RoBERTa",
        "mbert": "mBERT",
    }
    return names.get(model_key, model_key.replace("_", " ").title())


def render_sidebar(
    split_summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    available_options: dict[str, str],
) -> None:
    with st.sidebar:
        st.title("Urdu NLP Project")
        st.caption("Final semester project demo")
        st.markdown("**Dataset:** SentiUrdu-1M")
        st.markdown("**Task:** Three-class sentiment classification")
        st.markdown("**Best model:** TF-IDF + Linear SVM")

        st.divider()
        st.subheader("Saved Split Sizes")
        split_rows = {
            "Train": split_summary.get("train_size", 0),
            "Validation": split_summary.get("validation_size", 0),
            "Test": split_summary.get("test_size", 0),
        }
        for name, count in split_rows.items():
            st.metric(name, f"{int(count):,}" if count else "Unavailable")

        st.subheader("Class Distribution")
        distribution = split_summary.get("class_distribution_before_split", {})
        if distribution:
            dist_df = pd.DataFrame(
                {"Class": distribution.keys(), "Rows": distribution.values()}
            ).set_index("Class")
            st.bar_chart(dist_df, horizontal=True)
        else:
            st.warning("Class distribution artifact is unavailable.")

        st.subheader("Quick Leaderboard")
        if not leaderboard.empty:
            columns = ["model_name", "test_macro_f1", "test_accuracy"]
            st.dataframe(
                leaderboard[columns].head(5).rename(
                    columns={
                        "model_name": "Model",
                        "test_macro_f1": "Macro-F1",
                        "test_accuracy": "Accuracy",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.warning("Leaderboard artifact is unavailable.")

        unavailable = [name for name in MODEL_OPTIONS if name not in available_options]
        if unavailable:
            st.caption("Unavailable checkpoints: " + ", ".join(unavailable))

        st.subheader("How to Use This Demo")
        st.markdown(
            "1. Enter an Urdu or Roman Urdu tweet.\n"
            "2. Choose an available saved model.\n"
            "3. Classify and inspect preprocessing, probabilities, and explanation.\n"
            "4. Review the saved evaluation and error-analysis evidence below."
        )


def render_prediction(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    label = prediction["predicted_label"]
    confidence = float(prediction["confidence"])
    model_name = format_model_name(result["model_key"])

    if result.get("warning"):
        st.warning(result["warning"])

    status = st.success if label == "Positive" else st.error if label == "Negative" else st.info
    status(f"Predicted sentiment: **{label}**")
    metric_col, model_col = st.columns(2)
    metric_col.metric("Confidence", f"{confidence:.4f}")
    model_col.metric("Model used", model_name)

    probabilities = prediction.get("probabilities", [])
    if len(probabilities) == 3:
        chart = pd.DataFrame(
            {"Class": ["Negative", "Neutral", "Positive"], "Probability": probabilities}
        ).set_index("Class")
        st.bar_chart(chart)
    else:
        st.warning("A complete three-class probability distribution was not returned.")


def main() -> None:
    st.set_page_config(
        page_title="Urdu Sentiment Classification",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config(PROJECT_ROOT / "config.yaml")
    results_dir = PROJECT_ROOT / config["outputs"]["results_dir"]
    figures_dir = PROJECT_ROOT / config["outputs"]["figures_dir"]
    models_dir = PROJECT_ROOT / config["outputs"]["models_dir"]
    error_dir = PROJECT_ROOT / config["outputs"]["error_analysis_dir"]

    split_summary = read_json(results_dir / "split_summary.json")
    leaderboard = read_csv(results_dir / "model_comparison_leaderboard.csv")
    error_summary = read_json(error_dir / "baseline_error_summary.json")
    high_confidence_errors = read_csv(
        error_dir / "baseline_linear_svm_test_high_confidence_wrong.csv"
    )
    available_options = {
        display: key
        for display, key in MODEL_OPTIONS.items()
        if model_artifact_available(key, models_dir)
    }
    if not available_options:
        available_options = {"TF-IDF + Linear SVM (Best Overall)": "linear_svm"}

    render_sidebar(split_summary, leaderboard, available_options)

    st.title("Leakage-Aware Urdu Tweet Sentiment Classification")
    st.caption("Artifact-backed final project review: inference, comparison, and limitations")

    st.markdown("## 1. Project Overview")
    st.write(
        "This project compares statistical, neural, and multilingual transformer models for "
        "Negative, Neutral, and Positive classification of Urdu tweets. The review app reads "
        "saved artifacts only; it does not retrain models or regenerate data splits."
    )
    st.info(
        "**Best Final Model: TF-IDF + Linear SVM**  \n"
        "**Test Macro-F1: 0.5040**  \n"
        "**Test Accuracy: 0.8531**  \n"
        "**Neutral F1: 0.1303**"
    )

    st.markdown("## 2. Dataset and Task Information")
    dataset_cols = st.columns(4)
    dataset_cols[0].metric("Filtered rows", f"{split_summary.get('rows_after_filtering', 0):,}")
    dataset_cols[1].metric("Train", f"{split_summary.get('train_size', 0):,}")
    dataset_cols[2].metric("Validation", f"{split_summary.get('validation_size', 0):,}")
    dataset_cols[3].metric("Test", f"{split_summary.get('test_size', 0):,}")
    st.write(
        "Labels are mapped to three sentiment classes. Emojis are removed during preprocessing "
        "because the source labels were weakly supervised using emoji signals, so retaining them "
        "would create label leakage."
    )

    st.markdown("## 3. Interactive Model Inference")
    input_col, control_col = st.columns([3, 2])
    with input_col:
        user_text = st.text_area(
            "Enter an Urdu tweet",
            height=150,
            placeholder="یہاں اردو ٹویٹ لکھیں یا پیسٹ کریں...",
            key="urdu_tweet_input",
        )
        clean_input = str(user_text).strip()
    with control_col:
        selected_display = st.selectbox(
            "Select an available saved model",
            list(available_options.keys()),
            index=0,
        )
        selected_key = available_options[selected_display]
        st.caption("Linear SVM is the default because it has the strongest saved test Macro-F1.")
        classify_button = st.button("Classify Sentiment", type="primary", width="stretch")

    preview = ""
    if clean_input:
        try:
            preview = str(preprocess_input(clean_input, PROJECT_ROOT)).strip()
        except Exception as exc:
            st.warning(f"Preprocessing preview is unavailable: {exc}")

    if classify_button:
        if not clean_input:
            st.warning("Please enter some text before classifying.")
            st.session_state.pop("last_inference", None)
        elif not preview:
            st.warning("The input became empty after preprocessing. Please enter a longer Urdu tweet.")
            st.session_state.pop("last_inference", None)
        else:
            model, actual_key, warning = load_selected_model(selected_key)
            if model is None:
                st.error(warning or "No inference model could be loaded.")
                st.session_state.pop("last_inference", None)
            else:
                try:
                    prediction = model.predict(preview)
                    st.session_state["last_inference"] = {
                        "raw_text": clean_input,
                        "clean_text": preview,
                        "model_key": actual_key,
                        "prediction": prediction,
                        "warning": warning,
                    }
                except Exception as exc:
                    st.error(f"Prediction failed without changing any saved artifacts: {exc}")
                    st.session_state.pop("last_inference", None)

    st.markdown("## 4. Preprocessing Preview")
    if clean_input:
        preview_cols = st.columns(2)
        preview_cols[0].markdown("**Original input**")
        preview_cols[0].code(clean_input, language=None)
        preview_cols[1].markdown("**Model input after cleaning**")
        preview_cols[1].code(preview or "[Empty after preprocessing]", language=None)
    else:
        st.caption("Enter text above to preview Unicode normalization, URL/mention removal, and emoji removal.")

    inference_result = st.session_state.get("last_inference")
    st.markdown("## 5. Prediction Result")
    if inference_result:
        render_prediction(inference_result)
    else:
        st.caption("A prediction will appear here after successful classification.")

    st.markdown("## 6. Explanation Assistant")
    if inference_result:
        prediction = inference_result["prediction"]
        explanation = explain_prediction(
            text=inference_result["clean_text"],
            predicted_label=prediction["predicted_label"],
            confidence=float(prediction["confidence"]),
            model_name=format_model_name(inference_result["model_key"]),
        )
        st.info(explanation)
    else:
        st.caption("The assistant summarizes a completed prediction and known text-level risk factors.")
    st.caption(
        "The explanation assistant is rule-based and template-based. It is not a generative LLM "
        "and does not provide a causal explanation of the model."
    )

    st.markdown("## 7. Model Comparison Leaderboard")
    if not leaderboard.empty:
        display_columns = [
            "model_family",
            "model_name",
            "test_accuracy",
            "test_macro_f1",
            "negative_f1",
            "neutral_f1",
            "positive_f1",
        ]
        st.dataframe(
            leaderboard[display_columns].style.format(
                {column: "{:.4f}" for column in display_columns if column.endswith(("accuracy", "f1"))}
            ).highlight_max(subset=["test_macro_f1"], color="#d9ead3"),
            hide_index=True,
            width="stretch",
        )
    else:
        st.warning("Model comparison leaderboard is missing or unreadable.")

    st.markdown(
        "- Linear SVM has the best saved test Macro-F1.\n"
        "- Multinomial NB has the highest accuracy, but its much lower Macro-F1 shows that accuracy "
        "is misleading under severe imbalance.\n"
        "- Neither neural model exceeded Linear SVM.\n"
        "- Transformers were limited to a 50,000-example training subset and one epoch.\n"
        "- Neutral remains the most difficult class across model families."
    )

    st.markdown("## 8. Final Model Decision")
    st.success(
        "Linear SVM was selected because Macro-F1 is more reliable than accuracy for this highly "
        "imbalanced dataset. It achieved the strongest overall class-balanced result while retaining "
        "competitive accuracy."
    )

    st.markdown("## 9. Error Analysis Summary")
    svm_test = error_summary.get("models", {}).get("linear_svm", {}).get("test", {})
    if svm_test:
        neutral_error = svm_test.get("class_errors", {}).get("Neutral", {})
        top_pair = (svm_test.get("confusion_pairs") or [{}])[0]
        error_cols = st.columns(4)
        error_cols[0].metric("Test errors", f"{svm_test.get('total_errors', 0):,}")
        error_cols[1].metric("Error rate", f"{svm_test.get('error_rate', 0):.2%}")
        error_cols[2].metric("Neutral error rate", f"{neutral_error.get('error_rate', 0):.2%}")
        error_cols[3].metric(
            "High-confidence errors",
            f"{len(high_confidence_errors):,}" if not high_confidence_errors.empty else "Unavailable",
        )
        st.write(
            f"The most common confusion is **{top_pair.get('true_label', 'Unknown')} → "
            f"{top_pair.get('predicted_label', 'Unknown')}**, with "
            f"**{int(top_pair.get('count', 0)):,}** saved test cases. Positive/Negative polarity "
            "swaps dominate the error set, while Neutral has the highest class-specific error rate."
        )
    else:
        st.warning("Baseline error-analysis summary is missing or unreadable.")

    st.markdown("## 10. Visual Results")
    for index in range(0, len(FIGURES), 2):
        columns = st.columns(2)
        for column, (filename, caption) in zip(columns, FIGURES[index : index + 2]):
            figure_path = figures_dir / filename
            with column:
                st.markdown(f"**{caption}**")
                if figure_path.is_file():
                    st.image(str(figure_path), caption=caption, width="stretch")
                else:
                    st.warning(f"Figure not found: {filename}")

    st.markdown("## 11. Ethics and Limitations")
    st.markdown(
        "- **Weak-label bias:** Emoji-derived labels can be noisy even after emoji leakage is removed.\n"
        "- **Minority-class fairness:** Neutral represents far below 1% of filtered rows and has weak F1.\n"
        "- **Language variation:** Urdu script, Roman Urdu, English code-mixing, sarcasm, and dialects "
        "remain difficult.\n"
        "- **Compute constraints:** Transformers used 50,000 training examples and one epoch; neural "
        "models used randomly initialized embeddings.\n"
        "- **Responsible use:** Predictions should not be used as ground truth for surveillance, "
        "moderation penalties, or decisions about individuals."
    )

    st.markdown("## 12. How to Interpret the Results")
    st.write(
        "Macro-F1 gives equal importance to Negative, Neutral, and Positive performance, so it is the "
        "primary selection metric. Accuracy is dominated by the majority Positive class. A single "
        "tweet prediction is a model estimate, not a verified sentiment label; confidence is useful "
        "for comparison but is not a guarantee of correctness or calibration."
    )


if __name__ == "__main__":
    main()
