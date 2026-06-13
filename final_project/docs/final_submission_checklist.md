# Final Submission Checklist

## Overall Status

**All six rubric components, totaling 50 available marks, are mapped to concrete repository evidence.** This is a coverage statement, not a prediction of awarded marks.

The project contains every deliverable named in the brief: source code, data pipeline, training scripts, evaluation scripts, visualizations, documentation, repository history, final report, and a live baseline demonstration.

## 1. Problem Identification and Proposal

Rubric weight: 4 marks<br>
Status: **Ready**

Evidence:

- `final_project/README.md`
- `final_project/docs/methodology.md`
- `final_project/outputs/reports/final_nlp_project_report.pdf`

Criteria covered: relevant real-world Urdu NLP problem, explicit objectives, defined scope, local-data feasibility, ambiguity, scale, and measurable performance objectives.

## 2. Literature Review and Research Gap Analysis

Rubric weight: 5 marks<br>
Status: **Ready**

Evidence:

- Related Work, research-gap discussion, and references in `final_project/outputs/reports/final_nlp_project_report.pdf`
- Literature-gap summary in `final_project/docs/professor_requirements_alignment.md`

Criteria covered: traditional ML, neural architectures, Transformers, methodology comparison, limitations, references, and the leakage/fair-evaluation research gap.

## 3. System Design and Methodology

Rubric weight: 6 marks<br>
Status: **Ready**

Evidence:

- System-architecture figure embedded in the final report
- `final_project/docs/methodology.md`
- `final_project/src/preprocessing.py`
- `final_project/src/label_mapping.py`
- `final_project/src/create_splits.py`
- `final_project/config.yaml`

Criteria covered: system architecture, data flow, preprocessing, leakage control, features, model branches, training strategy, evaluation methodology, technology choices, and alternatives.

## 4. Implementation and Development

Rubric weight: 18 marks<br>
Status: **Ready**

Evidence:

- Complete modules under `final_project/src/`
- Stable commands under `final_project/scripts/`
- Raw, processed, split, and annotation data under `final_project/data/`
- Classical, neural, and Transformer training implementations
- Saved model predictions, training histories, metrics, and classical model files
- Regression tests under `final_project/tests/`
- Streamlit application under `final_project/app/`

Criteria covered: dataset preparation, preprocessing, feature extraction, model development, training, validation, testing, class weighting, early stopping, mixed precision support, sample modes, modularity, and graceful missing-resource errors.

Submission note: large neural and Transformer weight files are not included. Their training code, saved predictions, metrics, histories, and configuration evidence are included, while the complete Linear SVM artifact supports the live demo. This limitation must remain visible during grading.

## 5. Evaluation, Optimization, and Analysis

Rubric weight: 8 marks<br>
Status: **Ready**

Evidence:

- `final_project/outputs/report_snapshot/`
- `final_project/outputs/results/`
- `final_project/outputs/error_analysis/`
- `final_project/outputs/figures/`
- `final_project/docs/results_analysis.md`

Criteria covered: seven-model comparison, sentiment and emotion evaluation, accuracy, macro precision/recall/F1, weighted-F1, per-class reports, confusion matrices, class weighting, early stopping, quantitative comparison, qualitative errors, and improvement discussion.

Research limitations are correctly disclosed: weak labels, rare classes, no completed gold test set, and no repeated-seed confidence intervals. These are threats to validity, not missing rubric deliverables.

## 6. Final Report and Demonstration

Rubric weight: 9 marks<br>
Status: **Ready**

Evidence:

- `final_project/outputs/reports/final_nlp_project_report.pdf`
- `final_project/app/streamlit_app.py`
- `final_project/docs/demonstration_guide.md`
- `final_project/docs/demo_script.md`
- `final_project/docs/presentation_outline.md`
- `final_project/docs/ethics_and_limitations.md`

The final report includes Abstract, Introduction, Related Work, Proposed Methodology, Dataset and Experimental Setup, Results and Discussion, Conclusion, and References. The included Linear SVM model provides a reliable local demonstration after installing the listed requirements.

## Final Hand-In Checks

- [x] Final report is included inside the self-contained submission folder.
- [x] Sentiment and emotion report leaderboards are packaged in `outputs/report_snapshot/`.
- [x] Source, data, scripts, evaluation, figures, docs, and tests are present.
- [x] Markdown links and project validators pass.
- [x] Saved predictions reproduce the packaged result metrics.
- [ ] On the presentation machine, install `requirements.txt` using Python 3.11 or 3.12.
- [ ] Launch Streamlit once and rehearse the Linear SVM demo before presenting.
- [ ] Confirm whether the department separately requires similarity or AI-use declarations.
- [ ] Add the source-code repository URL to the LMS/cover sheet if a URL field is provided.

## Important Presentation Rule

Quote Assignment 4 dual-task results from `outputs/report_snapshot/` when discussing the research paper. Quote packaged sentiment rerun results from `outputs/results/` when demonstrating the runnable final-project pipeline. Do not merge the two snapshots.
