# Dual-Task Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible group-safe sentiment and emotion benchmark for all eight models, then regenerate the demo and final documentation from official artifacts.

**Architecture:** Extend the existing config-driven pipeline with task-isolated paths, group-safe split generation, seed-specific runs, and result aggregation. Preserve existing model implementations where possible, add uniform neural inference, and make reports consume aggregate artifacts rather than hand-entered numbers.

**Tech Stack:** Python 3.11, pandas, scikit-learn, PyTorch, Hugging Face Transformers, Streamlit, pytest, Markdown/PDF/PPTX tooling.

---

### Task 1: Task-Isolated Configuration

**Files:**
- Create: `config_sentiment.yaml`
- Create: `config_emotion.yaml`
- Modify: `src/utils.py`
- Test: `tests/test_task_configuration.py`

- [ ] Write tests proving sentiment and emotion resolve to distinct processed, split, model, result, prediction, figure, and error-analysis directories.
- [ ] Run `python -m pytest tests/test_task_configuration.py -q` and verify the new tests fail.
- [ ] Add both task configs and minimal path helpers.
- [ ] Run the focused test and the full existing suite.

### Task 2: Group-Safe Split Generation

**Files:**
- Modify: `src/create_splits.py`
- Modify: `src/validate_pipeline.py`
- Create: `tests/test_group_safe_splits.py`

- [ ] Write synthetic tests with shared IDs, shared normalized text, and conflicting labels.
- [ ] Verify tests fail because current row-level splitting leaks groups.
- [ ] Implement connected duplicate grouping, conflicting-group exclusion, deterministic deduplication, stratified group assignment, hashes, and overlap assertions.
- [ ] Verify the focused and full test suites pass.
- [ ] Generate official sentiment and emotion splits and validate zero overlap.

### Task 3: Seed-Specific Training Outputs

**Files:**
- Modify: `src/train_baseline.py`
- Modify: `src/train_neural.py`
- Modify: `src/train_transformer.py`
- Create: `src/run_experiments.py`
- Create: `tests/test_run_isolation.py`

- [ ] Write tests proving seed/model/task runs resolve to unique paths and preserve metadata.
- [ ] Verify tests fail with the shared current output paths.
- [ ] Add run context arguments and an orchestrator for seeds 42/52/62 and Transformer seed 42.
- [ ] Enable Urdu-RoBERTa in both task configs.
- [ ] Verify focused and full tests.

### Task 4: Validation-Only Aggregation

**Files:**
- Modify: `src/compare_models.py`
- Create: `src/aggregate_experiments.py`
- Create: `tests/test_experiment_aggregation.py`

- [ ] Write tests proving ranking uses validation macro-F1 even when test ordering differs.
- [ ] Write tests for mean, sample standard deviation, final-model selection, and bootstrap intervals.
- [ ] Verify tests fail against the current test-sorted leaderboard.
- [ ] Implement aggregation and official test evaluation metadata.
- [ ] Verify focused and full tests.

### Task 5: Dual-Task Multi-Family Inference

**Files:**
- Modify: `src/inference.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/test_streamlit_app.py`
- Create: `tests/test_inference.py`

- [ ] Write tests for baseline, neural, and Transformer artifact loading by task.
- [ ] Write tests ensuring SVM output is labeled as a decision score.
- [ ] Verify tests fail because neural/task-aware inference is absent.
- [ ] Implement uniform task-aware inference and app controls.
- [ ] Verify focused tests and app compilation.

### Task 6: Run the Official Benchmark

**Files:**
- Generated: `data/processed/{sentiment,emotion}/`
- Generated: `data/splits/{sentiment,emotion}/`
- Generated: `outputs/{sentiment,emotion}/`

- [ ] Run all baseline models for seeds 42, 52, and 62 on both tasks.
- [ ] Run Text-CNN and BiLSTM for seeds 42, 52, and 62 on both tasks.
- [ ] Run mBERT, XLM-RoBERTa, and Urdu-RoBERTa for seed 42 on both tasks.
- [ ] Aggregate results and generate official leaderboards, confidence intervals, confusion matrices, error analyses, and figures.
- [ ] Independently recompute every reported metric from predictions.

### Task 7: Generate Current Documentation

**Files:**
- Modify: `README.md`
- Modify: `reports/final_report.md`
- Modify: `reports/final_evaluation_summary.md`
- Modify: `reports/dataset_card.md`
- Modify: `reports/demo_script.md`
- Modify: `reports/slides_outline.md`
- Create: `reports/model_card.md`
- Create: `reports/experiment_manifest.json`
- Create: `src/generate_reports.py`
- Test: `tests/test_generated_reports.py`

- [ ] Write tests that compare report numbers and selected models to aggregate artifacts.
- [x] Verify regression tests reject stale single-task documents.
- [ ] Generate all documentation from official artifacts and include the full bibliography.
- [ ] Export the final report and presentation artifacts.
- [ ] Verify report consistency tests and link validation.

### Task 8: Final Verification

**Files:**
- Modify: `src/validate_final_project.py`
- Modify: `reports/final_submission_checklist.md`

- [ ] Extend validation to both tasks, all eight models, split overlap, run metadata, report consistency, and demo artifacts.
- [ ] Run the full test suite.
- [ ] Run all project validators.
- [ ] Launch Streamlit and verify its health endpoint.
- [ ] Smoke-test one baseline, neural, and Transformer prediction per task.
- [ ] Run `git diff --check` and review the final diff for stale claims or generated inconsistencies.
