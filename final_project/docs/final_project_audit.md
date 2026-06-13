# Final Project Audit

## Scope and Verdict

This audit checks the self-contained `final_project/` package against the professor's six stages, seven named deliverables, eight required report sections, CCP characteristics, and 50-mark rubric.

The package is ready for source inspection, data-pipeline review, saved-result evaluation, final-report assessment, and a live Linear SVM demonstration. Evidence coverage does not guarantee awarded marks.

## Evidence Coverage

| Area | Status | Primary evidence |
|---|---|---|
| Problem, objectives, scope, and feasibility | Ready | Root README and final report |
| Literature review and research gap | Ready | Final report Related Work and references |
| Architecture and methodology | Ready | Final report architecture; `docs/methodology.md`; modular source |
| Data preparation | Ready for packaged sentiment task | Raw/processed/split data and data-pipeline code |
| Classical model training and inference | Ready | Training code, predictions, metrics, and three `.joblib` models |
| Neural experiments | Evidence available with constraint | Training code, predictions, metrics, histories, vocabulary, and label map; `.pt` weights absent |
| Transformer experiments | Evidence available with constraint | Training code, predictions, metrics, histories, and tokenizer/config files; model weights absent |
| Six-class emotion evaluation | Ready as report evidence | `outputs/report_snapshot/` and final report |
| Evaluation and optimization | Ready | Seven-model comparisons, per-class metrics, confusion matrices, error analysis |
| Visualizations | Ready | Packaged figures and final-report snapshot figures |
| Final report | Ready | `outputs/reports/final_nlp_project_report.pdf` |
| Live demonstration | Ready with packaged baseline | Streamlit app and Linear SVM artifact |

## Result Consistency

Two verified snapshots are intentionally preserved:

1. **Final-report snapshot:** 532,661 rows, sentiment and emotion tasks, Urdu fastText neural setup, and seven-model dual-task leaderboards under `outputs/report_snapshot/`.
2. **Packaged sentiment rerun:** 517,966 rows after a two-token minimum, sentiment task, random neural embeddings, and seven-model leaderboard under `outputs/results/`.

Documents identify the snapshot before quoting a result. The two sets of metrics must not be merged.

## Known Constraints

- Labels are weakly supervised and highly imbalanced.
- The optional 300-row annotation sample is not manually labeled and is not a gold test set.
- Evaluation uses one fixed seed without confidence intervals or significance tests.
- Neural and Transformer model weights are absent, so exact deep-checkpoint inference is unavailable.
- Packaged Transformer training requires locally cached pretrained model resources.
- The reliable demonstration path is the included TF-IDF + Linear SVM pipeline.

## Documentation Review

- Folder READMEs describe actual files, schemas, and artifact availability.
- Prediction documentation uses the real fields `raw_text`, `is_correct`, and `text_length`.
- Data documentation includes all raw columns and correctly defines `text_length` as token count.
- App documentation describes deterministic explanations and the baseline fallback accurately.
- Submission documents use self-contained paths inside `final_project/`.
- The upload manifest distinguishes required evidence, optional milestone history, and excluded local files.

## Final Judgment

Every professor requirement has a corresponding artifact or transparent limitation. The final report has all required research-paper sections, the implementation covers the complete NLP lifecycle, and the saved evidence supports quantitative and qualitative analysis without requiring expensive retraining during grading.
