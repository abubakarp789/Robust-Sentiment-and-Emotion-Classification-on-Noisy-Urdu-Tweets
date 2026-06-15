# Independent Rubric and Gap Analysis

Audit date: June 15, 2026  
Project: Leakage-Aware Urdu Tweet Sentiment and Emotion Classification  
Rubric: CSC-355 Natural Language Processing Design Project (50 marks)

## 1. Executive Verdict

The repository is a substantial technical project rather than a report-only submission. It includes a 1,048,000-row source dataset, deterministic Urdu tweet preprocessing, label normalization, fixed train/validation/test files, seven evaluated models across three model families, saved model weights, predictions, metrics, figures, notebooks, automated tests, a Streamlit application, and report material.

The current package is stronger than the June 14 audit supplied with this request. Neural checkpoints and mBERT/XLM-R model weights are now present, all seven test prediction files reproduce their reported metrics, all project validators pass, and the Streamlit app starts successfully. The earlier statements that deep checkpoints were missing and that the app could not be executed are therefore stale for this workspace.

The main unresolved issue is experimental validity. The code creates row-level random stratified splits without grouping duplicate tweet IDs or normalized text. As a result, exact content appears in multiple splits even though the repository repeatedly describes the pipeline as leakage-free.

| Independent split check | Verified result |
|---|---:|
| Packaged test rows | 77,695 |
| Test rows whose exact `clean_text` occurs in training | 13,327 |
| Percentage of test rows seen in training | 17.153% |
| Unique cleaned texts shared by train and test | 11,136 |
| Tweet IDs shared by train and test | 445 |
| Cleaned-text groups with conflicting labels | 1,756 |
| Rows in conflicting duplicate-text groups | 12,625 |

For the packaged Linear SVM, macro-F1 is 0.6573 on test rows whose exact text occurs in training and 0.4521 on unseen test text. This does not imply deliberate leakage or false metrics. It demonstrates that row-level splitting materially inflates the reported evaluation and invalidates the broad "leakage-free" claim.

Strict technical estimate: **41.0/50**.  
Likely artifact-first estimate: **46-48/50**, because nearly every requested implementation artifact exists.  
The instructor determines the final grade.

## 2. Verification Performed

The audit inspected the root assignments and the self-contained `final_project/` package.

Verified repository facts:

- 218 files under `final_project/`.
- 1,048,000-row raw CSV.
- 517,966-row processed sentiment dataset.
- 362,576 / 77,695 / 77,695 train, validation, and test rows.
- Three classical model artifacts are present.
- Two neural checkpoints are present, each approximately 59 MB.
- mBERT and XLM-R checkpoints are present, approximately 711 MB and 1.11 GB respectively.
- Seven model prediction files cover the complete validation and test sets.
- Five artifact-backed analysis notebooks are present.
- Git remote, history, and Git LFS tracking are present.
- The latest commit in this workspace is dated June 11, 2026.

Commands and checks completed successfully:

```text
python -m pytest tests -q
35 passed

python src/validate_pipeline.py --config config.yaml
passed

python src/validate_readme_links.py
passed

python src/validate_notebooks.py
passed

python src/validate_final_project.py --config config.yaml
passed

Streamlit /_stcore/health
ok

Linear SVM Urdu inference smoke test
passed
```

All seven packaged test prediction files independently reproduce the accuracy and macro-F1 values in the leaderboard.

Coverage was not measured because `pytest-cov` is not installed and is not listed in `requirements.txt`. Test count alone should not be presented as a coverage percentage.

## 3. Rubric Scorecard

| Rubric component | Maximum | Strict estimate | Assessment |
|---|---:|---:|---|
| Problem Identification and Proposal | 4 | 4.0 | Strong and relevant CCP |
| Literature Review and Research Gap | 5 | 4.5 | Strong assignment evidence; final package summary is too compressed |
| System Design and Methodology | 6 | 4.5 | Broad design; duplicate-safe splitting is missing |
| Implementation and Development | 18 | 16.0 | Complete multi-family implementation and artifacts; task and inference gaps remain |
| Evaluation, Optimization, and Analysis | 8 | 4.5 | Rich artifacts; leakage and selection protocol materially weaken claims |
| Final Report and Demonstration | 9 | 7.5 | Report and runnable demo exist; final exports and scientific alignment need work |
| **Total** | **50** | **41.0** | **Strong project requiring evaluation cleanup** |

## 4. Rubric-by-Rubric Analysis

### 4.1 Problem Identification and Proposal - 4.0/4

Implemented well:

- Real-world Urdu social-media sentiment and emotion problem.
- Large, noisy, weakly labeled corpus with ambiguity, code-mixing, spelling variation, sarcasm, and class imbalance.
- Multiple viable solution families: sparse classical models, neural networks, and Transformers.
- Measurable objectives using accuracy, macro-F1, weighted-F1, per-class metrics, and error analysis.
- Clear engineering motivation for emoji removal because labels were partly generated from emoji heuristics.
- Appropriate framing as a Complex Computational Problem.

Recommended presentation improvement:

- Add one primary research question and two or three secondary questions.
- Define success criteria before the result section, especially minority-class recall and macro-F1 targets.

### 4.2 Literature Review and Research Gap - 4.5/5

Implemented well:

- The earlier coursework folders contain the reviewed research-paper collection.
- The earlier technical report includes 20 references and comparative discussion.
- Classical, neural, Transformer, multilingual, weak-supervision, Urdu-specific, and Twitter-oriented approaches are represented.
- The research gap is connected to emoji-derived shortcut leakage, shared evaluation, class imbalance, and low-resource Urdu modeling.

Remaining gaps:

- `final_project/reports/final_report.md` reduces the literature review to four short bullets and only four references.
- The final package does not include a related-work comparison table with dataset, task, sample size, model, metric, result, and limitation.
- Search databases, keywords, date range, inclusion criteria, and exclusion criteria are not recorded.
- The final report claims a 20-paper review without carrying the corresponding citations into its own bibliography.

To reach full marks:

- Move the strongest Assignment 2/3 literature evidence into the final report.
- Add a 10-15 paper comparison matrix.
- Add a short search protocol and explicit research questions.

### 4.3 System Design and Methodology - 4.5/6

Implemented well:

- Modular pipeline for preprocessing, label mapping, splitting, training, evaluation, error analysis, inference, plotting, validation, and deployment.
- Deterministic seed configuration.
- Training-only fitting for TF-IDF vocabulary and neural vocabulary.
- Validation macro-F1 checkpoint selection for neural models.
- Configurable class weights, early stopping, gradient clipping, and mixed precision.
- Saved split and label-distribution metadata.

Critical methodology gap:

- `src/create_splits.py` applies `train_test_split` directly to rows. It neither removes duplicate texts nor groups tweet IDs/text hashes before splitting.

Measured consequences:

- 11,136 unique normalized texts occur in both train and test.
- 13,327 test rows have exact normalized text already in train.
- 445 tweet IDs occur in both train and test.
- 1,756 normalized-text groups contain conflicting labels.

Other methodology gaps:

- `data/processed/processed_sentiment_dataset.csv` is hard-coded even though the configuration exposes a task setting.
- Sentiment and emotion do not have isolated configurations and output directories.
- The final package implements and reports only the sentiment experiment while the title and abstract still claim sentiment and emotion classification.
- There is no data hash, config hash, git commit, or environment snapshot recorded for each run.
- There is no written policy for duplicate texts with conflicting weak labels.

Required correction:

1. Normalize text before splitting.
2. Create a group key from valid tweet ID and normalized-text hash.
3. Keep each duplicate group entirely within one split.
4. Drop exact repeated rows unless frequency is an explicit modeling feature.
5. Exclude or manually adjudicate conflicting-label text groups.
6. Add automated assertions for zero cross-split ID and text overlap.
7. Regenerate all official results after the split change.

### 4.4 Implementation and Development - 16.0/18

Implemented well:

- Complete Python source structure with reusable modules and CLI scripts.
- Urdu/Arabic Unicode normalization and social-media cleaning.
- Noisy label canonicalization and sentiment mapping.
- Three classical baselines: Logistic Regression, Linear SVM, and Multinomial NB.
- Two neural models: Text-CNN and BiLSTM with additive attention.
- Two packaged Transformers: mBERT and XLM-RoBERTa.
- Saved model weights, tokenizers, vocabularies, label mappings, histories, predictions, metrics, confusion matrices, and figures.
- Streamlit app with model selection, prediction, preprocessing display, leaderboard, figures, and explanation output.
- Git LFS is used for large datasets and artifacts.
- 35 automated tests and multiple repository validators pass.

Remaining implementation gaps:

- `src/inference.py` supports classical and Transformer models but not the packaged neural checkpoints.
- The Streamlit demo therefore cannot demonstrate Text-CNN or BiLSTM despite shipping their weights.
- The final package has no complete emotion training outputs, checkpoints, evaluation, or dual-task inference path.
- `urdu_roberta` is disabled and has no packaged final-project checkpoint.
- Transformer training resolves remote model names with `from_pretrained` and is not fully offline reproducible from a fresh machine.
- Dependencies use broad lower bounds; there is no lockfile or exact environment export.
- There is no CI workflow.
- There is no automated group-leakage regression test.
- No explicit repository license or model card was found. A dataset card is present.

To reach full marks:

- Implement neural inference from the saved vocabulary, mapping, model kwargs, and checkpoints.
- Either narrow the project claim to sentiment or package the full emotion experiment separately.
- Add exact dependency locking and run metadata with hashes.
- Add CI and small synthetic smoke training tests for every model family.
- Add a license and model card.

### 4.5 Evaluation, Optimization, and Analysis - 4.5/8

Implemented well:

- Seven-model comparison across classical, neural, and Transformer families.
- Validation and test predictions are saved for every model.
- Accuracy, macro precision/recall/F1, weighted-F1, per-class metrics, confusion matrices, and error tables are present.
- All reported test metrics were independently reproduced from saved predictions.
- The project correctly emphasizes macro-F1 under severe imbalance.
- Error analysis identifies Neutral-class failure and Positive/Negative confusion.
- Optimization measures include class weighting, early stopping, gradient clipping, mixed precision, and training subsets for constrained Transformers.

Critical evaluation gaps:

1. **Duplicate leakage:** 17.153% of test rows contain exact normalized text seen during training.
2. **Test-based model ranking:** `src/compare_models.py` sorts the final leaderboard by test macro-F1, and the report selects the deployed model from test performance. Model selection should use validation only.
3. **No human gold test:** the 300-row annotation sample exists, but all `manual_label` cells are empty.
4. **Single seed:** no mean, standard deviation, confidence interval, or significance analysis is reported.
5. **Unequal budgets:** Transformers use 50,000 training rows and one epoch; neural and classical models use different budgets. These are resource-constrained comparisons, not controlled architecture comparisons.
6. **No central ablation study:** emoji removal, class weights, text-length threshold, word/character features, and random/pretrained embeddings are not directly compared.
7. **Uncalibrated SVM confidence:** softmax is applied to SVM decision margins and displayed as confidence, but this is not a calibrated probability.
8. **No external or cross-domain test:** conclusions rely on one weakly labeled Twitter corpus.

Independent SVM result:

| Test subset | Rows | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Exact text seen in training | 13,327 | 0.9049 | 0.6573 |
| Text not seen in training | 64,368 | 0.8424 | 0.4521 |

To reach full marks:

- Rebuild group-safe splits and rerun every official result.
- Rank and select models using validation macro-F1 only, then evaluate the chosen model once on test.
- Complete two-annotator Urdu labeling and agreement analysis.
- Run at least three seeds and report mean, standard deviation, and bootstrap confidence intervals.
- Add controlled ablations for the project’s central design decisions.
- Calibrate the deployed classifier or rename confidence to an uncalibrated decision score.

### 4.6 Final Report and Demonstration - 7.5/9

Implemented well:

- The Markdown report includes the required major sections.
- Results tables match the packaged prediction artifacts.
- Dataset card, ethics document, evaluation summary, slide outline, demo script, export guide, and submission checklist are present.
- The Streamlit application starts successfully in the current environment.
- Linear SVM Urdu inference was smoke-tested successfully.
- The app exposes classical and Transformer inference with fallback behavior.

Remaining gaps:

- No final-project PDF report is present; only the older Assignment 3 technical-report PDF exists.
- No PPTX or HTML presentation deck is present; only a slide outline exists.
- The final report bibliography has only four references.
- The report and app call the evaluation leakage-free even though duplicate leakage remains.
- The report title/abstract claim emotion classification, but the final package’s official experiment and demo are sentiment-only.
- Neural checkpoints cannot be selected in the app.
- The app labels uncalibrated SVM margin-derived values as confidence.
- The app’s full interactive user path was not browser-automated during this audit; startup health and direct inference were verified.

Administrative risk:

- The professor-provided submission date is May 20, 2026.
- This audit is dated June 15, 2026.
- The latest commit in this workspace is June 11, 2026.
- Confirm that post-deadline updates are accepted before presenting them as the submitted version.

## 5. Required Deliverables Matrix

| Required deliverable | Current evidence | Assessment |
|---|---|---|
| Complete Source Code | `src/`, `app/`, tests, scripts in assignments | Present; emotion and neural inference are incomplete in final package |
| Data Processing Pipeline | preprocessing, label mapping, processed data, fixed splits | Present; duplicate-safe splitting missing |
| Model Training Scripts | classical, neural, Transformer trainers | Present |
| Evaluation Scripts | metrics, reports, matrices, predictions, error analysis | Present and internally reproducible |
| Visualizations | 17 PNG figures and notebook views | Present |
| Documentation | README tree, reports, folder guides, notebooks | Strong but final report bibliography is weak |
| Source Code Repository | Git remote, history, Git LFS | Present |
| Final Report | Markdown final report and older Assignment 3 PDF | Partially complete; export current final report to PDF |
| Live Demonstration | Streamlit app and model artifacts | Runnable for classical/Transformer sentiment inference |

## 6. Highest-Priority Improvement Plan

### Priority 0: Correct Evaluation Validity

1. Implement group-safe splitting by tweet ID and normalized-text hash.
2. Define and document a policy for conflicting duplicate labels.
3. Add split-overlap tests that fail on any shared ID or normalized text.
4. Rerun all seven official model evaluations.
5. Replace every table, figure, abstract result, conclusion, and demo leaderboard with regenerated outputs.
6. Remove the phrase "leakage-free" until both target leakage and duplicate leakage are controlled.

### Priority 1: Establish a Clean Selection Protocol

1. Freeze design choices using validation macro-F1.
2. Rank the leaderboard by validation macro-F1.
3. Select one final model before opening test results.
4. Evaluate the selected model on test once.
5. Record run command, config, data hash, git commit, seed, versions, hardware, sample size, epochs, and runtime.

### Priority 2: Strengthen Scientific Evidence

1. Label the existing 300-row sample with two Urdu-proficient annotators.
2. Add annotation rules and adjudication guidance.
3. Report Cohen’s kappa and gold-label model performance.
4. Run three to five seeds for the main models.
5. Add bootstrap 95% confidence intervals.
6. Add ablations for emoji removal, class weights, word vs character TF-IDF, and random vs pretrained embeddings.

### Priority 3: Align Scope, Package, and Demo

1. Choose one honest scope:
   - sentiment-only final project; or
   - separate complete sentiment and emotion pipelines.
2. Use task-specific configs and output directories.
3. Implement neural inference and expose it in Streamlit.
4. Calibrate Linear SVM with `CalibratedClassifierCV`, or display "decision score" instead of confidence.
5. Export the current report to PDF and the slide outline to PPTX/PDF.
6. Include the full literature review and references in the final report.

### Priority 4: Engineering Quality

1. Add a pinned lockfile or environment export.
2. Add CI for tests, validation, links, and a synthetic smoke pipeline.
3. Install `pytest-cov`, establish a baseline, and raise core-module coverage.
4. Add artifact SHA-256 hashes.
5. Add a model card, license, dataset-use statement, and required AI-use declaration.

## 7. Recommended Final Experiment Matrix

| Experiment | Training policy | Seeds | Official evaluation |
|---|---|---:|---|
| Majority baseline | Group-safe train | deterministic | Validation + final test |
| Logistic Regression | Full group-safe train | 5 | Gold + weak-label test |
| Linear SVM | Full group-safe train | 5 | Gold + weak-label test |
| Character SVM ablation | Same budget | 3 | Weak-label test |
| Text-CNN random embeddings | Same neural budget | 3 | Gold + weak-label test |
| Text-CNN pretrained embeddings | Same neural budget | 3 | Gold + weak-label test |
| BiLSTM-Attention | Same neural budget | 3 | Gold + weak-label test |
| mBERT | Fixed documented budget | 3 | Gold + weak-label test |
| XLM-R | Same Transformer budget | 3 | Gold + weak-label test |

If full Transformer reruns are computationally infeasible, label them as resource-constrained pilot experiments rather than direct budget-matched comparisons.

## 8. Final Submission Checklist

### Must Fix Before Claiming Full Compliance

- [ ] Remove cross-split ID and normalized-text leakage.
- [ ] Rerun official metrics on group-safe splits.
- [ ] Select models using validation metrics only.
- [ ] Update all report and demo claims from regenerated outputs.
- [ ] Align the project title and abstract with the implemented task scope.
- [ ] Replace "leakage-free" with a precise, verified statement.

### Strongly Recommended for 50/50

- [ ] Complete gold annotation and inter-annotator agreement.
- [ ] Run repeated seeds and confidence intervals.
- [ ] Add controlled ablation studies.
- [ ] Add neural inference and full demo coverage.
- [ ] Calibrate displayed probability/confidence values.
- [ ] Export the current final report and presentation.
- [ ] Carry the full literature review into the final report.
- [ ] Add environment locking, CI, coverage, hashes, license, and model card.
- [ ] Confirm acceptance of updates made after May 20, 2026.

## 9. Final Judgment

The project satisfies the professor’s requested breadth and demonstrates substantial implementation skill. It investigates a genuine complex NLP problem, compares multiple model paradigms, preserves extensive artifacts, and includes a runnable demonstration. The saved results are internally consistent and independently reproducible from the prediction files.

The project is not yet scientifically defensible as fully leakage-free. The fastest route to a top score is not another architecture. It is a group-safe split, validation-only model selection, a human-labeled test subset, repeated-seed uncertainty, controlled ablations, and exact alignment between scope, report, artifacts, and demo.
