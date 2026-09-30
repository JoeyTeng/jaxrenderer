"""Compare Ruff diagnostics between a baseline and candidate checkout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

DIRECTORIES = ("assets", "renderer", "examples", "test_resources", "tests", "tools")
MAX_EXAMPLES = 10


Diagnostic = tuple[str, str, str, str]


def parse_diagnostics(
    output: str, root: Path, locations: dict[Diagnostic, tuple[int, int]] | None = None
) -> Counter[Diagnostic]:
    """Return diagnostic counts keyed by stable, row-independent fields."""
    try:
        records = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"Ruff returned invalid JSON: {error}") from error
    if not isinstance(records, list):
        raise ValueError("Ruff JSON output must be a list")

    result: Counter[Diagnostic] = Counter()
    source_lines: dict[Path, list[str]] = {}
    for record in records:
        try:
            filename = Path(record["filename"])
            if not filename.is_absolute():
                filename = root / filename
            filename = filename.resolve()
            path = filename.relative_to(root.resolve()).as_posix()
            code = record["code"]
            message = record["message"]
            row = record["location"]["row"]
            column = record["location"]["column"]
            if not isinstance(code, str) or not isinstance(message, str):
                raise TypeError
            if not isinstance(row, int) or not isinstance(column, int):
                raise TypeError
            if row < 1 or column < 1:
                raise ValueError
            if filename not in source_lines:
                source_lines[filename] = filename.read_text(
                    encoding="utf-8"
                ).splitlines()
            lines = source_lines[filename]
            if row > len(lines):
                raise ValueError
            source = lines[row - 1].strip()
        except (KeyError, TypeError, ValueError, OSError, UnicodeError) as error:
            raise ValueError("Ruff JSON contains a malformed diagnostic") from error
        result[(path, code, message, source)] += 1
        if locations is not None:
            locations[(path, code, message, source)] = (row, column)
    return result


def run_ruff(
    root: Path,
    executable: str,
    locations: dict[Diagnostic, tuple[int, int]] | None = None,
) -> Counter[Diagnostic]:
    directories = [name for name in DIRECTORIES if (root / name).is_dir()]
    if not directories:
        raise RuntimeError(f"Ruff found no project directories in {root}")
    # Apply one fixed policy to both trees, independent of candidate configuration.
    command = [
        executable,
        "check",
        "--isolated",
        "--no-respect-gitignore",
        "--select",
        "E4,E7,E9,F",
        "--ignore",
        "F722,F821",
        "--target-version",
        "py39",
        "--output-format",
        "json",
        *directories,
    ]
    completed = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode not in (0, 1):
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "no diagnostic output"
        )
        raise RuntimeError(
            f"Ruff failed for {root} (exit {completed.returncode}): {detail}"
        )
    if "No Python files found under the given path(s)" in completed.stderr:
        raise RuntimeError(f"Ruff found no Python files in non-empty scope: {root}")
    return parse_diagnostics(completed.stdout, root, locations)


def compare(
    baseline: Counter, candidate: Counter
) -> list[tuple[tuple[str, str, str, str], int]]:
    return sorted(
        (key, count - baseline[key])
        for key, count in candidate.items()
        if count > baseline[key]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    args = parser.parse_args(argv)
    for label, root in (
        ("baseline", args.base_root),
        ("candidate", args.candidate_root),
    ):
        if not root.is_dir():
            parser.error(f"{label} root is not a directory: {root}")
    base_root = args.base_root.resolve()
    candidate_root = args.candidate_root.resolve()
    executable = shutil.which("ruff")
    if executable is None:
        print("Ruff executable was not found on PATH", file=sys.stderr)
        return 2
    executable = str(Path(executable).resolve())
    candidate_locations: dict[Diagnostic, tuple[int, int]] = {}
    try:
        baseline = run_ruff(base_root, executable)
        candidate = run_ruff(candidate_root, executable, candidate_locations)
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(
        f"Ruff lint diagnostics: baseline {sum(baseline.values())}, "
        f"candidate {sum(candidate.values())}."
    )
    new_issues = compare(baseline, candidate)
    if not new_issues:
        print("Ruff lint gate passed: no new diagnostics.")
        return 0
    total = sum(count for _, count in new_issues)
    print(f"Ruff lint gate failed: {total} new diagnostic(s).", file=sys.stderr)
    shown = 0
    for key, count in new_issues:
        if shown >= MAX_EXAMPLES:
            break
        path, code, message, _source = key
        row, column = candidate_locations.get(key, (0, 0))
        print(
            f"{path}:{row}:{column}: {code} {message} ({count} additional)",
            file=sys.stderr,
        )
        shown += 1
    if len(new_issues) > MAX_EXAMPLES:
        print(
            f"... and {len(new_issues) - MAX_EXAMPLES} more distinct diagnostic(s)",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
