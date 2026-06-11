"""Validator script for local relative links in project Markdown files."""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "Assignment#01",
    "Assignment#02",
    "Assignment#03",
}

# Regex to match markdown links: [text](url) or ![alt](url)
LINK_PATTERN = re.compile(r"!?\[([^\]]*?)\]\(([^)]+?)\)")


def is_external_or_ignored(url: str) -> bool:
    """Check if the URL is an external link, mailto, or anchor-only link."""
    # Ignore anchor-only links
    if url.startswith("#"):
        return True
    # Ignore web protocols
    if url.startswith(("http://", "https://", "mailto:", "ftp://")):
        return True
    return False


def is_absolute_link(url: str) -> bool:
    """Check if the URL is an absolute path (discouraged)."""
    if url.startswith("file:///"):
        return True
    # Check for Windows drive letter like C:\ or D:/
    if len(url) > 1 and url[1] == ":" and url[0].isalpha():
        return True
    # Check for root paths
    if url.startswith(("/", "\\")):
        return True
    return False


def validate_markdown_file(filepath: Path) -> list[tuple[str, str]]:
    """Scan a markdown file and return a list of broken links (link_raw, error_reason)."""
    broken_links = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [("<read_file>", f"Error reading file: {e}")]

    matches = LINK_PATTERN.findall(content)
    md_dir = filepath.parent

    for text, url_raw in matches:
        # Clean URL
        url_clean = url_raw.strip()
        # Split anchor or query params
        url_clean = url_clean.split("#")[0].split("?")[0]

        # Ignore empty, external, or anchor-only
        if not url_clean or is_external_or_ignored(url_clean):
            continue

        # Check for absolute links
        if is_absolute_link(url_clean):
            broken_links.append((url_raw, "Absolute path not allowed (must be relative)"))
            continue

        # Decode URL encoding (e.g. %20 -> space)
        decoded_path = unquote(url_clean)

        # Build target path
        target_path = (md_dir / decoded_path).resolve()

        # Check if file/directory exists
        if not target_path.exists():
            broken_links.append((url_raw, f"Target path does not exist: {decoded_path}"))

    return broken_links


def scan_project() -> dict[str, list[tuple[str, str]]]:
    """Find all markdown files, check their links, and return a dict of errors."""
    errors = {}
    md_files = list(PROJECT_ROOT.glob("**/*.md"))

    for filepath in md_files:
        # Check exclusions
        parts = filepath.relative_to(PROJECT_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue

        broken = validate_markdown_file(filepath)
        if broken:
            relative_name = str(filepath.relative_to(PROJECT_ROOT))
            errors[relative_name] = broken

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local relative links in Markdown files.")
    parser.add_argument("--verbose", action="store_true", help="Print all checked files.")
    args = parser.parse_args()

    print("=== Scanning Markdown Files for Broken Relative Links ===")
    md_files = list(PROJECT_ROOT.glob("**/*.md"))
    scanned_files = []
    total_links_checked = 0

    errors = {}

    for filepath in md_files:
        parts = filepath.relative_to(PROJECT_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in parts):
            continue

        scanned_files.append(filepath)
        broken = validate_markdown_file(filepath)
        
        # Count links checked in this file
        try:
            content = filepath.read_text(encoding="utf-8")
            matches = LINK_PATTERN.findall(content)
            for text, url_raw in matches:
                url_clean = url_raw.strip().split("#")[0].split("?")[0]
                if url_clean and not is_external_or_ignored(url_clean):
                    total_links_checked += 1
        except Exception:
            pass

        if broken:
            relative_name = str(filepath.relative_to(PROJECT_ROOT))
            errors[relative_name] = broken

    print(f"Total markdown files scanned: {len(scanned_files)}")
    print(f"Total local relative links checked: {total_links_checked}")

    if errors:
        print("\n[FAILURE] Broken links found:")
        for filepath, broken_list in errors.items():
            print(f"\nFile: {filepath}")
            for link_raw, reason in broken_list:
                print(f"  - Link: '{link_raw}' -> {reason}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All relative links are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
