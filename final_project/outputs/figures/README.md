# Figures Folder

## Purpose

This folder stores generated visualization plots, confusion matrices, confidence distributions, and model performance comparisons as PNG images.

## Contents

This folder contains various comparison and matrix plots. Key figures include:

| File/Folder | Description |
|---|---|
| [baseline_model_macro_f1_comparison.png](baseline_model_macro_f1_comparison.png) | Comparison plot of F1 scores across classical models |
| [baseline_linear_svm_confusion_heatmap.png](baseline_linear_svm_confusion_heatmap.png) | Confusion matrix heatmap for the deployed Linear SVM |
| [final_model_family_comparison.png](final_model_family_comparison.png) | F1 comparison across model families (ML, Neural, Transformers) |
| [baseline_linear_svm_confidence_distribution.png](baseline_linear_svm_confidence_distribution.png) | Histogram of SVM confidence scores for correct vs incorrect cases |

## How It Is Used

The plots in this folder are generated programmatically:
- **Generation**: Created by plotting scripts (such as `src/plot_baseline_errors.py`, `src/plot_neural_results.py`, and `src/plot_transformer_results.py`) using `matplotlib` and `seaborn`.
- **Reference**: These figures describe the packaged sentiment rerun and are used by the notebooks, documentation, and Streamlit application. Final-report dual-task figures are stored separately under `../report_snapshot/`.

## Related Files

- [../README.md](../README.md)
- [../../notebooks/](../../notebooks/)
- [../../docs/results_analysis.md](../../docs/results_analysis.md)

## Notes

- All figures are generated programmatically and should not be modified using image editors. If plots need to be styled or updated, modify the corresponding plotting script in `src/` and run it.
