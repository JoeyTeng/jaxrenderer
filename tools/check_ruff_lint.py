"""Compare Ruff diagnostics between a baseline and candidate checkout."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

MAX_EXAMPLES = 10


Diagnostic = tuple[str, str, str, str]


def resolve_scope(root: Path, path: str) -> Path:
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Ruff scope escapes project root: {path}") from error
    return resolved


def has_python_files(path: Path) -> bool:
    if path.is_file():
        return path.suffix in {".py", ".pyi"}
    return any(
        candidate.is_file() and candidate.suffix in {".py", ".pyi"}
        for candidate in path.rglob("*")
    )


def validate_scopes(roots: tuple[Path, Path], paths: list[str]) -> None:
    """Require every requested scope to contain Python files in some revision."""
    for path in paths:
        found = False
        for root in roots:
            resolved = resolve_scope(root, path)
            if not resolved.exists():
                continue
            if not resolved.is_file() and not resolved.is_dir():
                raise RuntimeError(
                    f"Ruff scope is not a file or directory in {root}: {path}"
                )
            found |= has_python_files(resolved)
        if not found:
            raise RuntimeError(
                f"Ruff scope contains no Python files in either revision: {path}"
            )


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
    paths: list[str],
    locations: dict[Diagnostic, tuple[int, int]] | None = None,
) -> Counter[Diagnostic]:
    active_paths = []
    for path in paths:
        resolved = resolve_scope(root, path)
        if not resolved.exists():
            continue
        if not resolved.is_file() and not resolved.is_dir():
            raise RuntimeError(
                f"Ruff scope is not a file or directory in {root}: {path}"
            )
        if not has_python_files(resolved):
            continue
        active_paths.append(path)
    if not active_paths:
        return Counter()
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
        *active_paths,
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
        raise RuntimeError(f"Ruff found no Python files in scope: {root}")
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
    parser.add_argument(
        "--paths",
        required=True,
        nargs="+",
        metavar="PATH",
        help="repository-relative files or directories to scan",
    )
    args = parser.parse_args(argv)
    for path in args.paths:
        scoped_path = Path(path)
        if scoped_path.is_absolute() or ".." in scoped_path.parts:
            parser.error(
                f"scope must be repository-relative and cannot traverse: {path}"
            )
        if not path or path == ".":
            parser.error("scope must name a file or directory within the repository")
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
        validate_scopes((base_root, candidate_root), args.paths)
        baseline = run_ruff(base_root, executable, args.paths)
        candidate = run_ruff(
            candidate_root, executable, args.paths, candidate_locations
        )
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
