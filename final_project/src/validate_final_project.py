"""Project-wide validator for SentiUrdu-1M Sentiment Classification.

Checks the existence and validity of data splits, model checkpoints, prediction outputs,
evaluation summaries, figures, Streamlit deployment, and final reports.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path
import pandas as pd

try:
    from validate_notebooks import validate_notebooks
    from validate_pipeline import validate_data_assets
except ImportError:
    from src.validate_notebooks import validate_notebooks
    from src.validate_pipeline import validate_data_assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_success(msg: str) -> None:
    print(f"[SUCCESS] {msg}")


def print_warning(msg: str) -> None:
    print(f"[WARNING] {msg}")


def print_failure(msg: str) -> None:
    print(f"[FAILURE] {msg}")


def check_splits(data_cfg: dict) -> bool:
    print("\n--- Checking Data Splits ---")
    split_dir = PROJECT_ROOT / data_cfg.get("output_dir", "data/splits")
    splits = ["train.csv", "validation.csv", "test.csv"]
    all_ok = True

    for split in splits:
        path = split_dir / split
        if not path.exists():
            print_failure(f"Split file missing: {path}")
            all_ok = False
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8")
            if df.empty:
                print_failure(f"Split file is empty: {path}")
                all_ok = False
            else:
                print_success(f"{split} loaded successfully ({len(df):,} rows).")
                # Check columns
                required = ["clean_text", "task_label"]
                for col in required:
                    if col not in df.columns:
                        print_failure(f"Column '{col}' missing in split {split}")
                        all_ok = False
        except Exception as e:
            print_failure(f"Error reading split {split}: {e}")
            all_ok = False

    return all_ok


def check_data_assets(config: dict) -> bool:
    print("\n--- Checking Organized Data Assets ---")
    try:
        validate_data_assets(config)
        print_success("Raw, processed, annotation, and data README assets are valid.")
        return True
    except Exception as exc:
        print_failure(f"Data asset validation failed: {exc}")
        return False


def check_models(config: dict) -> bool:
    print("\n--- Checking Model Checkpoints ---")
    models_dir = PROJECT_ROOT / config["outputs"].get("models_dir", "outputs/models")
    all_ok = True

    # 1. Baselines
    baselines = ["baseline_linear_svm.joblib", "baseline_logistic_regression.joblib", "baseline_multinomial_nb.joblib"]
    for model in baselines:
        path = models_dir / model
        if not path.exists():
            print_failure(f"Baseline model missing: {path}")
            all_ok = False
        else:
            print_success(f"Baseline model found: {model} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Neural Models
    neural_models = ["neural_bilstm_attention.pt", "neural_text_cnn.pt", "neural_vocab.json", "neural_label_mapping.json"]
    for model in neural_models:
        path = models_dir / model
        if not path.exists():
            print_failure(f"Neural model file missing: {path}")
            all_ok = False
        else:
            print_success(f"Neural model file found: {model} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. Transformers
    transformers = ["transformer_mbert", "transformer_xlm_roberta"]
    for tf_model in transformers:
        best_dir = models_dir / tf_model / "best"
        if not best_dir.exists():
            # Check if main folder exists at least
            main_dir = models_dir / tf_model
            if not main_dir.exists():
                print_failure(f"Transformer model folder missing: {tf_model}")
                all_ok = False
            else:
                print_warning(f"Transformer model 'best' folder missing, but main folder exists: {tf_model}")
        else:
            print_success(f"Transformer model found: {tf_model}/best")

    return all_ok


def check_results() -> bool:
    print("\n--- Checking Evaluation Results & Leaderboard ---")
    results_dir = PROJECT_ROOT / "outputs/results"
    all_ok = True

    # 1. Leaderboard
    leaderboard_path = results_dir / "model_comparison_leaderboard.csv"
    if not leaderboard_path.exists():
        print_failure("Leaderboard file missing: model_comparison_leaderboard.csv")
        all_ok = False
    else:
        try:
            df = pd.read_csv(leaderboard_path)
            print_success(f"Leaderboard file found with {len(df)} models.")
            # Check for best model
            best_model = df.iloc[0]["model_name"]
            print_success(f"Rank #1 Model on Leaderboard: {best_model} (Macro-F1: {df.iloc[0]['test_macro_f1']:.4f})")
        except Exception as e:
            print_failure(f"Error reading leaderboard: {e}")
            all_ok = False

    # 2. Final Summary JSON
    summary_path = results_dir / "final_evaluation_summary.json"
    if not summary_path.exists():
        print_failure("Evaluation summary JSON missing: final_evaluation_summary.json")
        all_ok = False
    else:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            print_success("final_evaluation_summary.json parsed successfully.")
            # Validate top-level keys
            expected_keys = ["project_name", "final_model_selected", "headline_metric", "split_sizes", "best_models", "rankings"]
            for key in expected_keys:
                if key not in summary:
                    print_failure(f"Missing key '{key}' in final_evaluation_summary.json")
                    all_ok = False
        except Exception as e:
            print_failure(f"Error parsing final_evaluation_summary.json: {e}")
            all_ok = False

    return all_ok


def check_predictions() -> bool:
    print("\n--- Checking Model Predictions ---")
    pred_dir = PROJECT_ROOT / "outputs/predictions"
    all_ok = True

    # 7 models, test & validation each
    models = ["baseline_linear_svm", "baseline_logistic_regression", "baseline_multinomial_nb",
              "neural_bilstm_attention", "neural_text_cnn", "transformer_mbert", "transformer_xlm_roberta"]
    splits = ["test", "validation"]

    for model in models:
        for split in splits:
            filename = f"{model}_{split}_predictions.csv"
            path = pred_dir / filename
            if not path.exists():
                print_failure(f"Prediction file missing: {filename}")
                all_ok = False
            else:
                print_success(f"Prediction file found: {filename}")

    return all_ok


def check_error_analysis() -> bool:
    print("\n--- Checking Error Analysis ---")
    err_dir = PROJECT_ROOT / "outputs/error_analysis"
    all_ok = True

    # 1. Summary JSON
    sum_path = err_dir / "baseline_error_summary.json"
    if not sum_path.exists():
        print_failure(f"Baseline error summary missing: {sum_path.name}")
        all_ok = False
    else:
        print_success(f"Baseline error summary found: {sum_path.name}")

    # 2. Explanation samples
    exp_path = err_dir / "explanation_samples.json"
    if not exp_path.exists():
        print_failure(f"Explanation samples JSON missing: {exp_path.name}")
        all_ok = False
    else:
        print_success(f"Explanation samples found: {exp_path.name}")

    return all_ok


def check_figures() -> bool:
    print("\n--- Checking Generated Figures ---")
    fig_dir = PROJECT_ROOT / "outputs/figures"
    all_ok = True

    key_figures = [
        "baseline_model_macro_f1_comparison.png",
        "neural_model_macro_f1_comparison.png",
        "transformer_model_macro_f1_comparison.png",
        "final_model_family_comparison.png",
        "baseline_linear_svm_confusion_heatmap.png",
    ]

    for fig in key_figures:
        path = fig_dir / fig
        if not path.exists():
            print_failure(f"Key figure missing: {fig}")
            all_ok = False
        else:
            print_success(f"Key figure found: {fig}")

    return all_ok


def check_deployment() -> bool:
    print("\n--- Checking Deployment App ---")
    app_path = PROJECT_ROOT / "app/streamlit_app.py"
    if not app_path.exists():
        print_failure("Streamlit app file missing: app/streamlit_app.py")
        return False

    try:
        py_compile.compile(str(app_path), doraise=True)
        print_success("app/streamlit_app.py compiled successfully with no syntax errors.")
        return True
    except py_compile.PyCompileError as e:
        print_failure(f"Streamlit app compilation failed: {e}")
        return False


def check_notebooks(config_path: str | Path) -> bool:
    print("\n--- Checking Analysis Notebooks ---")
    return validate_notebooks(config_path, verbose=True)


def check_reports() -> bool:
    print("\n--- Checking Submission Reports ---")
    reports_dir = PROJECT_ROOT / "reports"
    all_ok = True

    required_reports = [
        "final_report.md",
        "dataset_card.md",
        "ethics_and_limitations.md",
        "slides_outline.md",
        "demo_script.md",
    ]

    for r in required_reports:
        path = reports_dir / r
        if not path.exists():
            print_failure(f"Required report missing: {r}")
            all_ok = False
        else:
            print_success(f"Report found: {r}")

    # Validate final_report.md sections
    report_path = reports_dir / "final_report.md"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        required_headers = [
            "## Abstract",
            "## 1. Introduction",
            "## 2. Problem Statement",
            "## 3. Motivation",
            "## 4. Dataset Description",
            "## 5. Literature Review Summary",
            "## 6. Methodology",
            "## 7. Experimental Setup",
            "## 8. Results",
            "## 9. Error Analysis",
            "## 10. Ethical Considerations",
            "## 11. Deployment",
            "## 12. Limitations",
            "## 13. Future Work",
            "## 14. Conclusion",
            "## References"
        ]
        
        print("\nChecking report headers in final_report.md:")
        for header in required_headers:
            if header not in content:
                print_failure(f"Missing header: '{header}'")
                all_ok = False
            else:
                print_success(f"Header found: '{header}'")

    return all_ok


def check_readmes_and_links() -> bool:
    print("\n--- Checking Folder READMEs and Link Validator ---")
    
    # 1. Check validate_readme_links.py exists
    validator_path = PROJECT_ROOT / "src" / "validate_readme_links.py"
    if not validator_path.exists():
        print_failure("Link validator script missing: src/validate_readme_links.py")
        return False
    print_success("src/validate_readme_links.py exists.")

    # 2. Check all required READMEs exist and are not empty
    required_readmes = [
        "README.md",
        "app/README.md",
        "data/README.md",
        "data/raw/README.md",
        "data/processed/README.md",
        "data/splits/README.md",
        "data/annotation/README.md",
        "notebooks/README.md",
        "outputs/README.md",
        "outputs/models/README.md",
        "outputs/predictions/README.md",
        "outputs/results/README.md",
        "outputs/error_analysis/README.md",
        "outputs/figures/README.md",
        "reports/README.md",
        "src/README.md",
        "tests/README.md",
    ]
    
    all_readmes_ok = True
    for relative_path in required_readmes:
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            print_failure(f"Required README missing: {relative_path}")
            all_readmes_ok = False
        elif path.read_text(encoding="utf-8").strip() == "":
            print_failure(f"Required README is empty: {relative_path}")
            all_readmes_ok = False
        else:
            # For root README, check if it highlights best model
            if relative_path == "README.md":
                content = path.read_text(encoding="utf-8")
                best_model_mention = "Best final model" in content or "Best Final Model" in content
                if not best_model_mention:
                    print_failure("README.md does not explicitly mention the best overall model (Linear SVM).")
                    all_readmes_ok = False
                else:
                    print_success("Root README.md is valid and highlights the best selected model.")
            else:
                print_success(f"README valid: {relative_path}")
                
    if not all_readmes_ok:
        return False

    # 3. Run relative link check
    try:
        try:
            from validate_readme_links import scan_project
        except ImportError:
            from src.validate_readme_links import scan_project
        
        errors = scan_project()
        if errors:
            print_failure("Markdown relative link validation failed. Broken links found:")
            for filepath, broken_list in errors.items():
                print(f"  File: {filepath}")
                for link_raw, reason in broken_list:
                    print(f"    - Link: '{link_raw}' -> {reason}")
            return False
        else:
            print_success("Markdown relative link validation passed successfully.")
    except Exception as e:
        print_failure(f"Error running markdown link validation: {e}")
        return False

    return True


def run_validation(config_path: str | Path | None = None) -> None:
    config_path = Path(config_path or PROJECT_ROOT / "config.yaml").resolve()
    if not config_path.exists():
        print_failure(f"Config path missing: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            import yaml
            config = yaml.safe_load(f)
    except Exception as e:
        print_failure(f"Error parsing config.yaml: {e}")
        sys.exit(1)

    all_checks = []

    all_checks.append(check_data_assets(config))
    all_checks.append(check_splits(config["data"]))
    all_checks.append(check_models(config))
    all_checks.append(check_results())
    all_checks.append(check_predictions())
    all_checks.append(check_error_analysis())
    all_checks.append(check_figures())
    all_checks.append(check_deployment())
    all_checks.append(check_notebooks(config_path))
    all_checks.append(check_reports())
    all_checks.append(check_readmes_and_links())

    print("\n==================================================")
    if all(all_checks):
        print_success("ALL PROJECT VALIDATIONS PASSED SUCCESSFULLY! Ready for final submission.")
        sys.exit(0)
    else:
        print_failure("SOME PROJECT VALIDATIONS FAILED. Please address the errors above.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final submission artifacts.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    run_validation(args.config)


if __name__ == "__main__":
    main()
