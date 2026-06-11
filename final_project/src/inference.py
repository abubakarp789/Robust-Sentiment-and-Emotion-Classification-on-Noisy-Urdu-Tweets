"""Inference pipeline for trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .preprocessing import preprocess_text
    from .train_baseline import resolve_project_path
    from .utils import load_config
except ImportError:
    from preprocessing import preprocess_text
    from train_baseline import resolve_project_path
    from utils import load_config


class InferenceModel:
    """Wrapper class for uniform model inference."""

    def __init__(self, model_type: str, model_obj: Any, label_mapping: dict | None = None, tokenizer: Any = None):
        self.model_type = model_type
        self.model_obj = model_obj
        self.label_mapping = label_mapping or {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.tokenizer = tokenizer

    def predict(self, text: str) -> dict[str, Any]:
        """Predict sentiment and confidence score for preprocessed text."""
        # Ensure text is not empty
        clean_text = text if text.strip() else "ٹویٹ"
        
        if self.model_type == "transformer":
            inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=96)
            self.model_obj.eval()
            with torch.no_grad():
                logits = self.model_obj(**inputs).logits
                probs = torch.softmax(logits, dim=-1).numpy()[0]
                pred_idx = int(probs.argmax())
                confidence = float(probs[pred_idx])
                pred_label = self.label_mapping[pred_idx]
            return {"predicted_label": pred_label, "confidence": confidence, "probabilities": probs.tolist()}
            
        elif self.model_type == "baseline":
            pred_label = str(self.model_obj.predict([clean_text])[0])

            if hasattr(self.model_obj, "predict_proba"):
                probs = np.asarray(self.model_obj.predict_proba([clean_text]))[0]
            elif hasattr(self.model_obj, "decision_function"):
                decisions = np.asarray(self.model_obj.decision_function([clean_text]))
                if decisions.ndim == 1:
                    decisions = np.column_stack([-decisions, decisions])
                probs = softmax(decisions, axis=1)[0]
            else:
                raise TypeError("Baseline model exposes neither predict_proba nor decision_function.")
            confidence = float(probs.max())

            return {"predicted_label": pred_label, "confidence": confidence, "probabilities": probs.tolist()}
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def load_inference_model(model_name: str, project_root: str | Path = ".") -> InferenceModel:
    """Load a trained model and return an InferenceModel wrapper.

    Args:
        model_name: Name of the model to load (e.g. 'linear_svm', 'mbert', 'xlm_roberta')
        project_root: Root directory of the final_project

    Returns:
        InferenceModel instance.
    """
    root = Path(project_root).resolve()
    config = load_config(root / "config.yaml")
    models_dir = resolve_project_path(root, config["outputs"]["models_dir"])
    
    # 1. Transformer Models
    if model_name in ("mbert", "xlm_roberta"):
        model_path = models_dir / f"transformer_{model_name}" / "best"
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer model path not found: {model_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model_obj = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Load label mapping
        mapping_path = models_dir / "transformer_label_mapping.json"
        label_mapping = None
        if mapping_path.exists():
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            label_mapping = {int(k): v for k, v in payload["id_to_label"].items()}
            
        return InferenceModel("transformer", model_obj, label_mapping, tokenizer)
        
    # 2. Baseline Models
    elif model_name in ("linear_svm", "logistic_regression", "multinomial_nb"):
        filename = f"baseline_{model_name}.joblib"
        model_path = models_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Baseline model path not found: {model_path}")
            
        model_obj = joblib.load(model_path)
        return InferenceModel("baseline", model_obj)
        
    else:
        raise ValueError(f"Unsupported inference model: {model_name}")


def preprocess_input(text: str, project_root: str | Path = ".") -> str:
    """Preprocess a single raw Urdu text for inference."""
    root = Path(project_root).resolve()
    config = load_config(root / "config.yaml")
    return preprocess_text(text, config.get("preprocessing"))
