# Numbered Entry Points

This folder is the project's command layer. Its small numbered entry points run the reusable implementation in `src/` in the correct pipeline order. They resolve `final_project/config.yaml` automatically, so they can be launched from either the repository root or the `final_project` folder.

The files here intentionally do not contain model architectures or training algorithms. Keeping commands in `scripts/` and implementation in `src/` avoids duplication and makes the submission easier to inspect.

- `01_prepare_data.py`: preprocess, label, split, and save summaries.
- `02_train_classical.py`: train TF-IDF baselines.
- `03_train_neural.py`: train Text-CNN/BiLSTM with optional sample mode.
- `04_train_transformers.py`: train locally available Transformer models with optional sample mode.
- `05_evaluate_models.py`: recompute metrics from saved predictions.
- `06_generate_visualizations.py`: rebuild evidence-backed charts.

## Folder Boundary

- Edit reusable preprocessing, model, evaluation, and plotting logic in `src/`.
- Use the numbered files here to execute the complete workflow.
- Do not merge these folders: they have separate responsibilities and together satisfy the source-code and training-script deliverables.
