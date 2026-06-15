"""Run the complete final-submission validation suite."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

try:
    from .utils import load_config
    from .validate_notebooks import validate_notebooks
    from .validate_official_benchmark import validate_official_benchmark
    from .validate_pipeline import validate_data_assets
    from .validate_professor_requirements import validate_requirements
    from .validate_readme_links import scan_project
except ImportError:
    from utils import load_config
    from validate_notebooks import validate_notebooks
    from validate_official_benchmark import validate_official_benchmark
    from validate_pipeline import validate_data_assets
    from validate_professor_requirements import validate_requirements
    from validate_readme_links import scan_project


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_data_assets(config: dict) -> bool:
    try:
        validate_data_assets(config)
        return True
    except Exception as exc:
        print(f"[FAILURE] Data assets: {exc}")
        return False


def check_deployment() -> bool:
    try:
        py_compile.compile(str(PROJECT_ROOT / "app" / "streamlit_app.py"), doraise=True)
        print("[SUCCESS] Streamlit app compiles.")
        return True
    except py_compile.PyCompileError as exc:
        print(f"[FAILURE] Streamlit app: {exc}")
        return False


def check_notebooks(config_path: str | Path) -> bool:
    return validate_notebooks(config_path, verbose=True)


def run_validation(config_path: str | Path) -> bool:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    checks = [
        check_data_assets(config),
        validate_official_benchmark(PROJECT_ROOT)["valid"],
        check_deployment(),
        check_notebooks(PROJECT_ROOT / "config.yaml"),
        not scan_project(),
        bool(validate_requirements()["ready"]),
    ]
    if all(checks):
        print("[SUCCESS] All final project validations passed.")
    else:
        print("[FAILURE] One or more final project validations failed.")
    return all(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_sentiment.yaml")
    args = parser.parse_args()
    raise SystemExit(0 if run_validation(args.config) else 1)


if __name__ == "__main__":
    main()
