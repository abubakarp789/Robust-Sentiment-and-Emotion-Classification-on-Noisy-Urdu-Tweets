# Dataset Description

## Identity and Local Source

Dataset: SentiUrdu-1M<br>
Local experimental file: `data/raw/Urdu Tweets Dataset.csv`<br>
Repository origin: retained from the semester project dataset supplied in an earlier milestone

The project literature links SentiUrdu-1M to its public data article, but this package uses only the local repository copy.

## Raw CSV Facts

- Rows: 1,048,000
- Columns: `Id`, `Text`, `Emotions`, `Category`
- Missing `Category`: 514,571
- Non-null `Category`: 533,429

`Emotions` contains weak-supervision signals. `Category` is the noisy raw target field used by the project.

## Canonical Classes Before Text Filtering

| Emotion | Count |
|---|---:|
| Joy | 459,728 |
| Sad | 50,417 |
| Disgust | 17,083 |
| Angry | 2,802 |
| Fear | 1,847 |
| Surprise | 1,552 |

Derived sentiment counts are Positive 459,728, Negative 72,149, and Neutral 1,552. The largest-to-smallest ratio is about 296:1.

## Preprocessing Impact and Two Snapshots

### Assignment 3 / Assignment 4 snapshot

- Empty cleaned rows removed: 1,172
- Final labeled rows: 532,661
- Train: 372,862
- Validation: 79,899
- Test: 79,900
- Tasks: sentiment and emotion

### Packaged final_project snapshot

- Filter: at least two cleaned tokens
- Empty/short rows removed: 15,463
- Final labeled rows: 517,966
- Train: 362,576
- Validation: 77,695
- Test: 77,695
- Task saved in current splits: sentiment

The packaged split class counts are stored in `outputs/results/split_summary.json`. These two snapshots are both verified from repository files but are not interchangeable.

## Split Policy

- Ratio: 70/15/15
- Random seed: 42
- Stratified by selected task label
- Shared saved rows for every model within a given snapshot
- Training-only vocabulary/feature fitting

## Weak-Label Limitations

- Labels are not fully human verified.
- Emoji/lexicon heuristics can disagree with sarcasm, negation, poetic language, or context.
- Rare classes provide limited evidence for learning and evaluation.
- Derived sentiment collapses several emotions into Negative and uses only Surprise for Neutral.
- Metrics measure agreement with noisy labels, not guaranteed human sentiment truth.

## Privacy and Use

The dataset contains social-media text and may include sensitive, offensive, or identifying content. The project should be used for coursework and aggregate research inspection, not automated surveillance or high-stakes decisions.
