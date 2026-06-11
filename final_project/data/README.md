# Data Organization

## Purpose

This folder organizes the data pipeline assets for the SentiUrdu-1M sentiment and emotion classification project. It maintains a clean separation of concern between raw input data, intermediate processed/cleaned outputs, final evaluation splits, and annotation samples.

## Contents

| File/Folder | Description |
|---|---|
| [raw/](raw/) | Folder containing the raw source dataset |
| [processed/](processed/) | Folder containing the preprocessed and label-normalized dataset |
| [splits/](splits/) | Folder containing the train, validation, and test CSV splits |
| [annotation/](annotation/) | Folder containing an optional annotation sample for manual verification |

## How It Is Used

The files in this folder are part of the dataset preparation pipeline:
1. **Raw Input**: SentiUrdu-1M dataset is loaded from [raw/](raw/).
2. **Preprocessing & Mapping**: The data is preprocessed to remove noise/emojis and labels are normalized, saving the result into [processed/](processed/).
3. **Splitting**: Stratified splits are generated and stored in [splits/](splits/) for reproducible model training and testing.
4. **Annotation**: A balanced sample of 300 rows is separated in [annotation/](annotation/) for human validation. Note that this annotation sample is completely optional and is not used in training or evaluation of reported metrics.

The pipeline is controlled by settings in [../config.yaml](../config.yaml) and executed via `src/create_splits.py`.

## Related Files

- [../README.md](../README.md)
- [../config.yaml](../config.yaml)
- [raw/README.md](raw/README.md)
- [processed/README.md](processed/README.md)
- [splits/README.md](splits/README.md)
- [annotation/README.md](annotation/README.md)

## Notes

- **Warning**: Do not manually modify datasets in `raw/`, `processed/`, or `splits/`. If the preprocessing rules change, update [../config.yaml](../config.yaml) or `src/preprocessing.py` and run `src/create_splits.py` to regenerate all assets.
