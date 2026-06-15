# Processed Data

Task-specific datasets are stored in `sentiment/dataset.csv` (446,745 rows) and `emotion/dataset.csv` (446,640 rows). They contain `id`, `raw_text`, `clean_text`, source labels, canonical labels, task labels, and token length. Generate them through `src/create_splits.py` with the corresponding task config.
