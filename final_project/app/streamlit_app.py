"""Placeholder Streamlit app for the final NLP semester project.

This layout is intentionally non-functional for model inference right now.
The next implementation step is to connect `src.inference` to saved model
artifacts after the final project structure is stable.
"""

from __future__ import annotations

import streamlit as st


MODEL_OPTIONS = [
    "TF-IDF + Logistic Regression",
    "TF-IDF + Linear SVM",
    "Text-CNN",
    "BiLSTM-Attention",
    "mBERT",
    "XLM-RoBERTa",
    "Urdu-RoBERTa",
]


def main() -> None:
    st.set_page_config(
        page_title="Urdu Sentiment and Emotion Classification",
        layout="wide",
    )

    st.title("Leakage-Aware Urdu Tweet Sentiment and Emotion Classification")
    st.caption("Final semester project placeholder app")

    tweet_text = st.text_area(
        "Enter an Urdu tweet",
        height=140,
        placeholder="Paste an Urdu tweet here...",
    )

    selected_model = st.selectbox("Select model", MODEL_OPTIONS)

    if st.button("Predict", type="primary"):
        st.subheader("Prediction Result")
        st.info("Placeholder prediction: Positive / Joy")

        st.subheader("Confidence Score")
        st.progress(72)
        st.write("Placeholder confidence: 0.72")

        st.subheader("Preprocessing Output")
        if tweet_text.strip():
            st.code("Preprocessed text will appear here after inference is implemented.")
        else:
            st.code("No input text provided.")

        st.subheader("Model Explanation")
        st.write(
            "A human-readable explanation will appear here after the inference "
            "and explanation modules are connected."
        )

        st.subheader("Error Analysis Note")
        st.write(
            "If the prediction is incorrect or low-confidence, this section will "
            "summarize likely error causes such as negation, sarcasm, code-mixing, "
            "weak-label noise, or minority-class confusion."
        )

    with st.sidebar:
        st.header("Project Info")
        st.write("Task: Urdu sentiment and emotion classification")
        st.write("Dataset: SentiUrdu-1M")
        st.write("Status: Placeholder UI")
        st.write("Next step: connect saved model artifacts")


if __name__ == "__main__":
    main()
