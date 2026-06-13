# Annotation Support Folder

## Purpose

This folder contains an optional candidate sample for a future manual annotation study. It is not a gold test set in its current unlabeled state.

## Contents

| File/Folder | Description |
|---|---|
| [annotation_sample.csv](annotation_sample.csv) | Balanced sample of tweets for manual labeling |
| [annotation_readme.md](annotation_readme.md) | Generated status note confirming that the sample is not used in current metrics |

## How It Is Used

The sample dataset is generated using [../../src/create_annotation_sample.py](../../src/create_annotation_sample.py).
- **Sample Size**: `300` tweets
- **Class Distribution**: Exactly balanced (`100` Positive, `100` Negative, `100` Neutral)
- **Training/Evaluation**: This sample is **NOT** used for training or evaluation in the current project, and it has no impact on the reported metrics.
- **Future Use**: After independent annotation and agreement checks, it could support a cleaner evaluation subset.

## Related Files

- [../README.md](../README.md)
- [annotation_readme.md](annotation_readme.md)
- [../../src/create_annotation_sample.py](../../src/create_annotation_sample.py)

## Notes

- **Warning**: Do not modify `annotation_sample.csv` unless you are actively performing manual annotation. Columns `manual_label` and `annotator_notes` are left empty by default and should only be populated during human review.
