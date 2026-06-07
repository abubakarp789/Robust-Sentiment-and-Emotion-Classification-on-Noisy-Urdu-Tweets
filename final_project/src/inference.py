"""Inference stubs for the final NLP semester project.

These functions are placeholders. They should be connected to the saved
preprocessing pipeline and trained model artifacts after the final project
repository structure is stable.
"""

from __future__ import annotations

from typing import Any, Dict


def load_model(model_name: str, model_path: str | None = None) -> Any:
    """Load a trained model artifact.

    Args:
        model_name: Human-readable model identifier.
        model_path: Optional path to a saved model checkpoint.

    Returns:
        Loaded model object.
    """
    raise NotImplementedError("Model loading is not implemented yet.")


def preprocess_input(text: str) -> str:
    """Preprocess a single input tweet for inference."""
    raise NotImplementedError("Input preprocessing is not implemented yet.")


def predict_text(model: Any, text: str) -> Dict[str, Any]:
    """Predict sentiment or emotion for a single text input."""
    raise NotImplementedError("Text prediction is not implemented yet.")


def explain_prediction(
    text: str,
    prediction: Dict[str, Any],
    model_name: str | None = None,
) -> str:
    """Generate a human-readable explanation for a prediction."""
    raise NotImplementedError("Prediction explanation is not implemented yet.")
