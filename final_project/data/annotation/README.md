# Annotation Support Folder

## Purpose

This folder contains the files for an optional manual annotation study. It is intended to support a future human-verified clean test subset to evaluate model alignment against ground-truth human annotations.

## Contents

| File/Folder | Description |
|---|---|
| [annotation_sample.csv](annotation_sample.csv) | Balanced sample of tweets for manual labeling |
| [annotation_readme.md](annotation_readme.md) | Description and guidelines for annotation |

## How It Is Used

The sample dataset is generated using [../../src/create_annotation_sample.py](../../src/create_annotation_sample.py).
- **Sample Size**: `300` tweets
- **Class Distribution**: Exactly balanced (`100` Positive, `100` Negative, `100` Neutral)
- **Training/Evaluation**: This sample is **NOT** used for training or evaluation in the current project, and it has no impact on the reported metrics.
- **Future Use**: It will serve as a high-quality human-verified gold standard subset for model testing.

## Related Files

- [../README.md](../README.md)
- [annotation_readme.md](annotation_readme.md)
- [../../src/create_annotation_sample.py](../../src/create_annotation_sample.py)

## Notes

- **Warning**: Do not modify `annotation_sample.csv` unless you are actively performing manual annotation. Columns `manual_label` and `annotator_notes` are left empty by default and should only be populated during human review.
