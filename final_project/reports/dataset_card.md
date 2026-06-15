# Dataset Card: Group-Safe SentiUrdu-1M Benchmark

## Source

SentiUrdu-1M contains 1,048,000 noisy Urdu tweets. Category labels are weakly supervised; 514,571 rows do not contain the required Category label.

## Official Task Datasets

| Task | Retained rows | Train | Validation | Test | Classes |
|---|---:|---:|---:|---:|---:|
| Sentiment | 446,745 | 312,710 | 67,016 | 67,019 | 3 |
| Emotion | 446,640 | 312,643 | 66,995 | 67,002 | 6 |

## Data Quality Controls

- Emoji-derived shortcut cues are removed from text.
- Tweet IDs and normalized texts define connected duplicate groups.
- Conflicting-label groups are excluded.
- Exact normalized-text duplicates are reduced to one row.
- Split validators require zero ID and text overlap.

## Limitations

Labels remain weakly supervised and may be wrong. No human-gold evaluation is claimed. The data represent Pakistani Twitter discourse and should not be assumed to generalize to formal Urdu, private messages, or other regions.
