# Professor Requirements Alignment

## Course Metadata

| Field | Submission evidence |
|---|---|
| University | Namal University Mianwali |
| Department | Department of Computer Science |
| Course | CSC-355 Natural Language Processing |
| Instructor | Dr. Muzamil Ahmed |
| Session / Semester | 2022-2026, 8th Semester |
| Total Marks | 50 |
| Submission Date in brief | May 20, 2026 |
| CLOs | CLO-2, CLO-3, CLO-4 |

## Stage 1: Problem Identification and Proposal

Professor requirement: a real-world NLP problem with ambiguity, large text data, alternatives, and measurable objectives.

Project evidence:

- Urdu sentiment and emotion classification on 1,048,000 noisy tweets.
- Ambiguity from sarcasm, negation, code-mixing, spelling variation, short context, and weak labels.
- Alternatives include TF-IDF models, CNN, BiLSTM-Attention, mBERT, XLM-R, and Urdu-RoBERTa.
- Metrics include accuracy, macro precision/recall/F1, weighted-F1, per-class results, and confusion matrices.
- Proposal, scope, objectives, and feasibility are consolidated in the final report and root README.

Alignment: **Complete**.

## Stage 2: Literature Review and Problem Analysis

Professor requirement: review traditional, neural, and Transformer methods; compare them; identify limitations and a gap.

Project evidence:

- The final report synthesizes traditional, neural, Transformer, weak-supervision, and emoji-aware research and includes a complete reference list.
- Separate analysis of classical/lexicon, neural, Transformer, weak-supervision, and emoji-fusion methods.
- Gap: large weakly labeled Urdu data is rarely compared under shared preprocessing, splits, leakage control, and class-balanced metrics.

Alignment: **Complete**.

## Stage 3: System Design and Methodology

Professor requirement: architecture, data flow, preprocessing, representations, models, training, evaluation, justification, and alternatives.

Project evidence:

- System-architecture diagram in the final report.
- Eight-stage cleaning pipeline and explicit emoji leakage prevention.
- Six-class emotion normalization and three-class sentiment mapping.
- TF-IDF, word-embedding neural, and Transformer representation branches.
- Shared 70/15/15 stratified protocol with seed 42 and macro-F1 selection.
- Alternatives and trade-offs documented in `methodology.md` and the report.

Alignment: **Complete**.

## Stage 4: Implementation and Experimental Development

Professor requirement: dataset preparation, preprocessing, features, models, training, validation, testing, and optimization using appropriate frameworks.

Project evidence:

- Python implementation with pandas, scikit-learn, PyTorch, and Hugging Face.
- Raw/processed/split datasets and deterministic label mapping.
- Classical, neural, and Transformer training scripts.
- Validation-based early stopping, class weighting, gradient clipping, mixed precision support, and sample modes.
- Saved predictions, histories, reports, figures, and a runnable classical checkpoint.
- Automated tests and project validators.

Alignment: **Complete**.

## Required Deliverables

| Deliverable | Evidence | Status |
|---|---|---|
| Complete Source Code | `src/`, `app/`, notebooks | Present |
| Data Processing Pipeline | `scripts/01_prepare_data.py`, preprocessing/label/split modules | Present |
| Model Training Scripts | `scripts/02-04_*.py` and training modules | Present |
| Evaluation Scripts | `scripts/05_evaluate_models.py`, evaluation/error modules | Present |
| Visualizations | `outputs/figures/`, generation script, report figures | Present |
| Documentation | README, `docs/`, notebook and dataset READMEs | Present |
| Source Code Repository | Git repository plus the self-contained `final_project/` package | Present locally; submit repository URL |

## Stage 5: Evaluation, Analysis, and Optimization

Professor requirement: rigorous experiments, appropriate metrics, multiple models, quantitative and qualitative analysis.

Project evidence:

- Seven model variants across three model families.
- Both sentiment and emotion leaderboards in `outputs/report_snapshot/`.
- Per-class reports, confusion matrices, error samples, training histories, and model-family comparison figures.
- Optimization includes class weighting, smoothed Transformer weights, early stopping, gradient clipping, and mixed precision support.
- Analysis explains majority-class collapse, rare-class errors, sarcasm, negation, code-mixing, and weak-label noise.

Alignment: **Complete**.

## Stage 6: Final Report and Demonstration

Professor requirement: research-paper report and live demonstration.

Project evidence:

- IEEE-style final PDF with all eight required report sections.
- Submission copy under `outputs/reports/`.
- Streamlit interface with Urdu input, leak-free preprocessing, prediction, confidence display, model leaderboard, and explanation assistant.
- Included Linear SVM checkpoint supports the dependable demo path.
- Speaking script, demo guide, slide outline, and likely Q&A are included.

Alignment: **Ready**, with dependency installation and rehearsal remaining as presentation-machine checks.

## CCP Characteristics

| CCP characteristic | Project evidence |
|---|---|
| Large/noisy dataset | 1.048 million Urdu tweets with missing/noisy weak labels |
| Language ambiguity | Negation, sarcasm, code-mixing, spelling variation, short context |
| Competing objectives | Accuracy, macro-F1, minority recall, leakage control, compute cost |
| Significant computation | Neural and Transformer training with recorded GPU execution |
| Alternative approaches | Classical, neural, multilingual, and Urdu-specific models |
| Engineering decisions | Emoji removal, label mapping, fixed splits, class weighting |
| Trade-off analysis | Accuracy versus macro-F1 and compute versus performance |
| Experimental evidence | Predictions, metrics, confusion matrices, figures, error analysis |

## Final Judgment

The package provides evidence for every stage, named deliverable, required report section, CCP characteristic, and rubric component in the professor's brief. This is an evidence-coverage judgment, not a guarantee of awarded marks. Scientific limitations remain transparently documented and should be discussed during the viva.
