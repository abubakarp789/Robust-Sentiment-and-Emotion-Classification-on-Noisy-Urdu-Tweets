# Saved Data Splits

## Purpose

This folder contains the final, immutable stratified train, validation, and test splits. Storing these partitions as static files ensures that all models (baseline ML, deep neural, and transformers) are trained and evaluated on exactly the same data for fair comparison.

## Contents

| File/Folder | Description |
|---|---|
| [train.csv](train.csv) | Training split dataset |
| [validation.csv](validation.csv) | Validation split dataset |
| [test.csv](test.csv) | Test split dataset |

## How It Is Used

These files are the inputs for all training and evaluation scripts in the codebase:
- **Split Sizes**:
  - **Train**: `362,576` rows
  - **Validation**: `77,695` rows
  - **Test**: `77,695` rows
- They are generated from the processed dataset by [../../src/create_splits.py](../../src/create_splits.py).
- All models use [train.csv](train.csv) for fitting/training, [validation.csv](validation.csv) for hyperparameter tuning/early stopping, and [test.csv](test.csv) for reporting final performance metrics.

## Related Files

- [../README.md](../README.md)
- [../../src/create_splits.py](../../src/create_splits.py)

## Notes

> [!IMPORTANT]
> Do not manually edit these files under any circumstances. If they need to be regenerated (for example, if preprocessing settings change in `config.yaml`), run `python src/create_splits.py` to overwrite them.
