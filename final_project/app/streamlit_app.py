"""Streamlit app for Urdu Tweet Sentiment and Emotion Classification."""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure src/ is in the python path for streamlit imports
app_dir = Path(__file__).resolve().parent
project_root = app_dir.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from inference import load_inference_model, preprocess_input
    from explanation_assistant import explain_prediction
    from utils import load_config
except ImportError:
    # Fallback to local import if executed differently
    sys.path.append(str(project_root))
    from src.inference import load_inference_model, preprocess_input
    from src.explanation_assistant import explain_prediction
    from src.utils import load_config


MODEL_OPTIONS = {
    "TF-IDF + Linear SVM (Best Overall)": "linear_svm",
    "TF-IDF + Logistic Regression": "logistic_regression",
    "TF-IDF + Multinomial NB": "multinomial_nb",
    "XLM-RoBERTa (Transformer)": "xlm_roberta",
    "mBERT (Transformer)": "mbert",
}


def main() -> None:
    st.set_page_config(
        page_title="Urdu Sentiment Classification",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Leakage-Aware Urdu Tweet Sentiment Classification")
    st.caption("Milestone 5 Deployment Demo — Classifiers & Explanation Assistant")

    # Selected Best Model Banner
    st.info(
        "🏆 **Selected Best Model**: **TF-IDF + Linear SVM** (Test Macro-F1: `0.5040`, Neutral F1: `0.1303`, Accuracy: `0.8531`). "
        "Under the leakage-preventing pipeline (with emojis removed), it outperforms deep learning and transformer "
        "models trained under resource constraints."
    )

    # Load configuration
    config = load_config(project_root / "config.yaml")
    results_dir = project_root / config["outputs"]["results_dir"]
    leaderboard_path = results_dir / "model_comparison_leaderboard.csv"

    # Sidebar
    with st.sidebar:
        st.header("Project Information")
        st.write("**Dataset**: SentiUrdu-1M (stratified splits)")
        st.write("**Target Task**: Sentiment (Negative, Neutral, Positive)")
        
        # Display leaderboard summary in sidebar
        if leaderboard_path.exists():
            st.subheader("Leaderboard Summary")
            lb_df = pd.read_csv(leaderboard_path)
            # Show simplified columns
            st.dataframe(
                lb_df[["model_name", "test_macro_f1", "neutral_f1"]].rename(
                    columns={
                        "model_name": "Model",
                        "test_macro_f1": "Macro-F1",
                        "neutral_f1": "Neutral-F1",
                    }
                ),
                hide_index=True,
            )
        else:
            st.info("Leaderboard file not found. Run evaluations to generate.")
            
        st.header("Ethics & Limitations")
        st.markdown(
            "- **Weak Supervision Bias**: SentiUrdu-1M labels are derived from emoji heuristics. Emoji removal is enforced to prevent label leakage.\n"
            "- **Minority Class Performance**: The 'Neutral' class remains extremely challenging due to severe class imbalance (<1% of data).\n"
            "- **Code-Mixing**: Tweets contain Roman Urdu and English terms, introducing noise."
        )

    # Main area columns
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Interactive Model Inference")
        tweet_text = st.text_area(
            "Enter an Urdu tweet:",
            height=140,
            placeholder="یہاں اردو ٹویٹ لکھیں یا پیسٹ کریں...",
        )

        selected_display = st.selectbox("Select model for prediction:", list(MODEL_OPTIONS.keys()))
        model_key = MODEL_OPTIONS[selected_display]

        predict_button = st.button("Classify Sentiment", type="primary")

    with col2:
        st.subheader("Inference Result & Analysis")
        
        if predict_button:
            if not tweet_text.strip():
                st.warning("Please enter some text before classifying.")
            else:
                # 1. Preprocess
                clean_text = preprocess_input(tweet_text, project_root)
                
                st.markdown("**Preprocessed Text (Cleaned):**")
                st.code(clean_text if clean_text.strip() else "[Empty after emoji/punctuation cleaning]")

                # 2. Predict with fallback
                model_wrapper = None
                try:
                    # Attempt to load selected model
                    model_wrapper = load_inference_model(model_key, project_root)
                    st.success(f"Loaded {selected_display} successfully.")
                except Exception as e:
                    st.warning(
                        f"Could not load selected model ({model_key}). "
                        f"Falling back to the baseline Linear SVM model."
                    )
                    try:
                        # Fallback to Linear SVM
                        model_wrapper = load_inference_model("linear_svm", project_root)
                        st.info("Loaded baseline Linear SVM successfully.")
                    except Exception as fallback_error:
                        st.error(f"Failed to load baseline Linear SVM fallback: {fallback_error}")

                if model_wrapper:
                    # Run inference
                    res = model_wrapper.predict(clean_text)
                    pred_label = res["predicted_label"]
                    confidence = res["confidence"]
                    
                    # Sentiment color display
                    color = "red" if pred_label == "Negative" else "green" if pred_label == "Positive" else "gray"
                    st.markdown(f"### Predicted Sentiment: :{color}[{pred_label}]")
                    st.metric("Confidence Score", f"{confidence:.4f}")
                    
                    # Display probability distribution
                    probs_df = pd.DataFrame({
                        "Class": ["Negative", "Neutral", "Positive"],
                        "Probability": res["probabilities"]
                    })
                    st.bar_chart(probs_df.set_index("Class"))
                    
                    # 3. Explanation
                    st.markdown("### Explanation Assistant")
                    explanation = explain_prediction(
                        text=clean_text,
                        predicted_label=pred_label,
                        confidence=confidence,
                        model_name=model_wrapper.model_key if hasattr(model_wrapper, "model_key") else model_key
                    )
                    st.info(explanation)
                    
                    st.caption(
                        "Note: Explanations are rule-based/template-driven to outline text characteristics "
                        "(length, polarity indicator terms, negations) and highlight known error classes."
                    )
                else:
                    st.error("No inference model could be loaded. Please ensure model checkpoints exist.")

    # Bottom comparison section
    st.subheader("Model Comparison Leaderboard")
    if leaderboard_path.exists():
        comp_df = pd.read_csv(leaderboard_path)
        # Apply style highlighting to the top model
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: rgba(112, 173, 71, 0.3)' if v else '' for v in is_max]
            
        st.dataframe(
            comp_df.style.apply(highlight_max, subset=["test_macro_f1"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Leaderboard details not available yet.")


if __name__ == "__main__":
    main()
