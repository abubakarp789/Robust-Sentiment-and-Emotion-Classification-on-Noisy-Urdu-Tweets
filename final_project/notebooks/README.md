# Analysis Notebooks

## Purpose

This folder contains Jupyter Notebooks that document and explore the project milestones. These notebooks are designed for analysis, visualization, and validation, rather than model training.

## Contents

| File/Folder | Description |
|---|---|
| [01_dataset_analysis.ipynb](01_dataset_analysis.ipynb) | Exploratory Data Analysis, preprocessing examples, label normalization, and split checks |
| [02_baseline_models.ipynb](02_baseline_models.ipynb) | Performance analysis of classical machine learning baselines |
| [03_neural_models.ipynb](03_neural_models.ipynb) | Performance analysis and training curve visualizations of deep neural networks (Text-CNN, BiLSTM-Attention) |
| [04_transformer_models.ipynb](04_transformer_models.ipynb) | Performance analysis of pre-trained transformer fine-tuning (mBERT, XLM-RoBERTa) |
| [05_error_analysis.ipynb](05_error_analysis.ipynb) | Error analysis notebook exploring confusion patterns and misclassified samples |

## How It Is Used

These notebooks are used for presentation, interpretation, and reproducing results:
- **No Retraining**: To ensure quick reproducibility and avoid expensive computational loads, the notebooks load pre-generated prediction csv files and metrics from [../outputs/results/](../outputs/results/) and [../outputs/predictions/](../outputs/predictions/).
- **Figures**: The plotting code inside notebooks matches the scripts in `src/` and references figures stored in [../outputs/figures/](../outputs/figures/).

To start Jupyter and browse the notebooks:
```powershell
jupyter notebook notebooks
```

## Related Files

- [../README.md](../README.md)
- [../outputs/results/](../outputs/results/)
- [../outputs/figures/](../outputs/figures/)

## Notes

- Any changes to core logic (preprocessing, training, evaluation) should be done inside the `src/` directory. The notebooks should only be edited to update presentation formatting or add specific plots.
