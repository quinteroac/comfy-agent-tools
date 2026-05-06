from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

IGNORED_PREFIXES = (
    "outputs/",
    ".venv/",
    ".pytest_cache/",
)
IGNORED_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".wav",
)


def run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def staged_paths() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def needs_changelog(paths: list[str]) -> bool:
    meaningful = []
    for path in paths:
        if path == "CHANGELOG.md":
            continue
        if path.startswith(IGNORED_PREFIXES):
            continue
        if path.endswith(IGNORED_SUFFIXES):
            continue
        meaningful.append(path)
    return bool(meaningful)


def ensure_changelog() -> None:
    if CHANGELOG.exists():
        return
    CHANGELOG.write_text(
        "# Changelog\n\n"
        "All notable changes to this project will be documented in this file.\n\n"
        "## [Unreleased]\n\n"
        "### Added\n\n",
        encoding="utf-8",
    )


def add_entry(entry: str, section: str) -> None:
    ensure_changelog()
    text = CHANGELOG.read_text(encoding="utf-8")
    unreleased = "## [Unreleased]"
    heading = f"### {section}"
    bullet = f"- {entry.strip()}\n"

    if not entry.strip():
        raise SystemExit("Entry cannot be empty.")

    if unreleased not in text:
        text = text.rstrip() + f"\n\n{unreleased}\n\n{heading}\n\n{bullet}"
        CHANGELOG.write_text(text, encoding="utf-8")
        return

    if heading in text:
        insert_at = text.index(heading) + len(heading)
        insert_at = text.index("\n", insert_at) + 1
        while text[insert_at : insert_at + 1] == "\n":
            insert_at += 1
        text = text[:insert_at] + bullet + text[insert_at:]
        CHANGELOG.write_text(text, encoding="utf-8")
        return

    insert_at = text.index(unreleased) + len(unreleased)
    insert_at = text.index("\n", insert_at) + 1
    section_text = f"\n{heading}\n\n{bullet}"
    text = text[:insert_at] + section_text + text[insert_at:]
    CHANGELOG.write_text(text, encoding="utf-8")


def check_staged() -> int:
    try:
        paths = staged_paths()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return 1

    if not paths or not needs_changelog(paths):
        return 0

    if "CHANGELOG.md" in paths:
        return 0

    sys.stderr.write(
        "CHANGELOG.md is not staged.\n\n"
        "Add a short user-visible entry before committing, for example:\n\n"
        '  uv run python scripts/update_changelog.py --entry "Describe the change"\n'
        "  git add CHANGELOG.md\n\n"
        "Use --no-verify only for truly internal changes.\n"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maintain CHANGELOG.md entries.")
    parser.add_argument("--entry", help="Add an entry under [Unreleased].")
    parser.add_argument(
        "--section",
        default="Changed",
        choices=["Added", "Changed", "Fixed", "Removed", "Deprecated", "Security"],
        help="Changelog section for --entry.",
    )
    parser.add_argument(
        "--check-staged",
        action="store_true",
        help="Fail if staged project changes do not include CHANGELOG.md.",
    )
    args = parser.parse_args(argv)

    if args.entry:
        add_entry(args.entry, args.section)
    if args.check_staged:
        return check_staged()
    if not args.entry and not args.check_staged:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
