"""Task-aware inference for baseline, neural, and Transformer run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .models_dl import BiLSTMAttention, TextCNN
    from .neural_utils import encode_text, load_checkpoint
    from .preprocessing import preprocess_text
    from .utils import load_config
except ImportError:
    from models_dl import BiLSTMAttention, TextCNN
    from neural_utils import encode_text, load_checkpoint
    from preprocessing import preprocess_text
    from utils import load_config


MODEL_FAMILIES = {
    "logistic_regression": "baseline",
    "linear_svm": "baseline",
    "multinomial_nb": "baseline",
    "text_cnn": "neural",
    "bilstm_attention": "neural",
    "mbert": "transformer",
    "xlm_roberta": "transformer",
    "urdu_roberta": "transformer",
}
NEURAL_CLASSES = {"text_cnn": TextCNN, "bilstm_attention": BiLSTMAttention}


class InferenceModel:
    """Uniform inference wrapper with explicit score semantics."""

    def __init__(
        self,
        model_type: str,
        model_obj: Any,
        label_mapping: dict[int, str] | None = None,
        tokenizer: Any = None,
        vocab: dict[str, int] | None = None,
        max_sequence_length: int = 96,
        device: torch.device | None = None,
    ) -> None:
        self.model_type = model_type
        self.model_obj = model_obj
        self.label_mapping = label_mapping or {}
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.device = device or torch.device("cpu")

    def predict(self, text: str) -> dict[str, Any]:
        """Predict one cleaned text and identify probability versus decision score."""
        clean_text = text.strip() or "tweet"
        if self.model_type == "transformer":
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            self.model_obj.eval()
            with torch.no_grad():
                logits = self.model_obj(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            prediction_id = int(probabilities.argmax())
            return self._probability_result(prediction_id, probabilities)

        if self.model_type == "neural":
            if self.vocab is None:
                raise ValueError("Neural inference requires a vocabulary")
            encoded = encode_text(clean_text, self.vocab, self.max_sequence_length)
            inputs = torch.tensor([encoded], dtype=torch.long, device=self.device)
            mask = inputs.ne(self.vocab["<pad>"])
            self.model_obj.eval()
            with torch.no_grad():
                logits = self.model_obj(inputs, mask)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            prediction_id = int(probabilities.argmax())
            return self._probability_result(prediction_id, probabilities)

        if self.model_type == "baseline":
            predicted_label = str(self.model_obj.predict([clean_text])[0])
            labels = [str(value) for value in self.model_obj.classes_]
            if hasattr(self.model_obj, "predict_proba"):
                probabilities = np.asarray(self.model_obj.predict_proba([clean_text]))[0]
                score_kind = "probability"
            elif hasattr(self.model_obj, "decision_function"):
                decisions = np.asarray(self.model_obj.decision_function([clean_text]))
                if decisions.ndim == 1:
                    decisions = np.column_stack([-decisions, decisions])
                probabilities = softmax(decisions, axis=1)[0]
                score_kind = "decision_score"
            else:
                raise TypeError("Baseline model exposes neither probabilities nor margins")
            predicted_index = labels.index(predicted_label)
            score = float(probabilities[predicted_index])
            return {
                "predicted_label": predicted_label,
                "score": score,
                "confidence": score,
                "score_kind": score_kind,
                "probabilities": probabilities.tolist(),
                "labels": labels,
            }
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _probability_result(self, prediction_id: int, probabilities: np.ndarray) -> dict[str, Any]:
        label = self.label_mapping[prediction_id]
        score = float(probabilities[prediction_id])
        labels = [self.label_mapping[index] for index in range(len(probabilities))]
        return {
            "predicted_label": label,
            "score": score,
            "confidence": score,
            "score_kind": "probability",
            "probabilities": probabilities.tolist(),
            "labels": labels,
        }


def resolve_model_run(
    project_root: str | Path,
    task: str,
    model_name: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Resolve a model run, choosing its best validation seed when omitted."""
    root = Path(project_root).resolve()
    family = MODEL_FAMILIES.get(model_name)
    if family is None:
        raise ValueError(f"Unsupported inference model: {model_name}")
    model_root = root / "outputs" / task / "runs" / family / model_name
    metrics_name = f"{family}_metrics.json"
    candidates: list[tuple[float, float, int, Path]] = []
    for path in model_root.glob(f"seed_*/results/{metrics_name}"):
        run_seed = int(path.parents[1].name.removeprefix("seed_"))
        if seed is not None and run_seed != seed:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["models"][model_name]["validation"]
        candidates.append(
            (float(validation["macro_f1"]), float(validation["accuracy"]), run_seed, path.parents[1])
        )
    if not candidates:
        raise FileNotFoundError(f"No completed run found for {task}/{model_name}")
    validation_macro_f1, validation_accuracy, selected_seed, run_root = max(candidates)
    return {
        "task": task,
        "family": family,
        "model_name": model_name,
        "seed": selected_seed,
        "run_root": run_root,
        "validation_macro_f1": validation_macro_f1,
        "validation_accuracy": validation_accuracy,
    }


def load_inference_model(
    model_name: str,
    project_root: str | Path = ".",
    task: str = "sentiment",
    seed: int | None = None,
) -> InferenceModel:
    """Load the best-validation run for a task/model pair."""
    root = Path(project_root).resolve()
    run = resolve_model_run(root, task, model_name, seed)
    run_root = Path(run["run_root"])
    models_dir = run_root / "models"
    config = load_config(root / f"config_{task}.yaml")
    family = run["family"]

    if family == "baseline":
        model_path = models_dir / f"baseline_{model_name}.joblib"
        return InferenceModel("baseline", joblib.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if family == "neural":
        vocab = json.loads((models_dir / "neural_vocab.json").read_text(encoding="utf-8"))
        mapping_payload = json.loads(
            (models_dir / "neural_label_mapping.json").read_text(encoding="utf-8")
        )
        label_mapping = {
            int(key): value for key, value in mapping_payload["id_to_label"].items()
        }
        checkpoint_path = models_dir / f"neural_{model_name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_kwargs = checkpoint["metadata"]["model_kwargs"]
        model, _ = load_checkpoint(
            checkpoint_path, NEURAL_CLASSES[model_name], model_kwargs, device
        )
        return InferenceModel(
            "neural",
            model,
            label_mapping=label_mapping,
            vocab=vocab,
            max_sequence_length=int(config["neural_models"]["data"]["max_sequence_length"]),
            device=device,
        )

    model_path = models_dir / f"transformer_{model_name}" / "best"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    ).to(device)
    mapping_payload = json.loads(
        (models_dir / "transformer_label_mapping.json").read_text(encoding="utf-8")
    )
    label_mapping = {int(key): value for key, value in mapping_payload["id_to_label"].items()}
    return InferenceModel(
        "transformer",
        model,
        label_mapping=label_mapping,
        tokenizer=tokenizer,
        max_sequence_length=int(config["transformer_models"]["data"]["max_sequence_length"]),
        device=device,
    )


def preprocess_input(
    text: str,
    project_root: str | Path = ".",
    task: str = "sentiment",
) -> str:
    """Apply the official preprocessing configuration for one task."""
    root = Path(project_root).resolve()
    config = load_config(root / f"config_{task}.yaml")
    return preprocess_text(text, config["preprocessing"])
