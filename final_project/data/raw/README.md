# Raw Data Folder

## Purpose

This folder contains the original raw source dataset for the SentiUrdu-1M project. Keeping a copy of the raw dataset inside this directory ensures the project is completely self-contained and reproducible.

## Contents

| File/Folder | Description |
|---|---|
| [Urdu Tweets Dataset.csv](Urdu%20Tweets%20Dataset.csv) | Copied raw SentiUrdu-1M dataset CSV |

## How It Is Used

The raw dataset `Urdu Tweets Dataset.csv` contains the original unlabeled and weakly-labeled tweets collected from Twitter.
- **Expected Rows**: `1,048,000`
- **Expected Columns**: `Id`, `Text`, `Category`
- This file is read as the entry point by the split-creation script [../../src/create_splits.py](../../src/create_splits.py).
- The path configurations are defined in [../../config.yaml](../../config.yaml).

## Related Files

- [../README.md](../README.md)
- [../../config.yaml](../../config.yaml)
- [../../src/create_splits.py](../../src/create_splits.py)

## Notes

> [!WARNING]
> The raw CSV file `Urdu Tweets Dataset.csv` is extremely large (~219 MB) and should not be modified manually. All updates to preprocessing or cleaning rules must be implemented programmatically in the source scripts.
