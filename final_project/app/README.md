# App Folder

## Purpose

This folder contains the interactive user interface deployment for the project. It provides a Streamlit application allowing users to input Urdu tweets, preprocess them, select various models (including classical machine learning, deep neural networks, and transformer architectures), see predictions, and receive generative explanations for classifications and error analysis in real time.

## Contents

| File/Folder | Description |
|---|---|
| [streamlit_app.py](streamlit_app.py) | Streamlit dashboard application file |

## How It Is Used

The Streamlit application loads trained model checkpoints from the output directories and provides an interactive web UI.
- It uses inference utilities from [../src/inference.py](../src/inference.py) to preprocess text and query predictions.
- It uses the explanation assistant from [../src/explanation_assistant.py](../src/explanation_assistant.py) to provide template-based/rule-based GenAI explanations of classification predictions.
- It reads the overall model leaderboard from [../outputs/results/model_comparison_leaderboard.csv](../outputs/results/model_comparison_leaderboard.csv) to display a model performance comparison table.

To run the Streamlit application:
```powershell
streamlit run app\streamlit_app.py
```

## Related Files

- [../README.md](../README.md)
- [../config.yaml](../config.yaml)
- [../src/inference.py](../src/inference.py)
- [../src/explanation_assistant.py](../src/explanation_assistant.py)
- [../outputs/results/model_comparison_leaderboard.csv](../outputs/results/model_comparison_leaderboard.csv)

## Notes

- The Streamlit app relies on trained model checkpoints under `outputs/models/`. It includes a robust fallback system: if transformer or neural models are not found on the local filesystem, it defaults to the best selected classical model (`baseline_linear_svm.joblib`).
- The application file should be edited manually to add UI features or modify visualization states.
