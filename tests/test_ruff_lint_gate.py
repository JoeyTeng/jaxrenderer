import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from tools import check_ruff_lint as gate


def diagnostic(root: Path, row: int = 1, message: str = "undefined name") -> dict:
    source_path = root / "renderer" / "module.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["pass"] * row
    lines[row - 1] = "    missing_name"
    source_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "filename": str(source_path),
        "code": "F821",
        "message": message,
        "location": {"row": row, "column": 3},
    }


def test_candidate_only_issue_fails_even_when_total_matches(tmp_path: Path) -> None:
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    old_issue = diagnostic(base, message="old issue")
    new_issue = diagnostic(candidate)

    with patch.object(gate.shutil, "which", return_value="ruff"):
        with patch.object(
            gate,
            "run_ruff",
            side_effect=[
                gate.parse_diagnostics(json.dumps([old_issue]), base),
                gate.parse_diagnostics(json.dumps([new_issue]), candidate),
            ],
        ):
            assert (
                gate.main(
                    ["--base-root", str(base), "--candidate-root", str(candidate)]
                )
                == 1
            )


def test_duplicate_multiplicity_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    item = diagnostic(root)
    parsed = gate.parse_diagnostics(json.dumps([item, item]), root)
    assert sum(parsed.values()) == 2
    assert list(parsed.values()) == [2]


def test_row_changes_do_not_create_new_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    first = gate.parse_diagnostics(json.dumps([diagnostic(root, row=2)]), root)
    shifted = gate.parse_diagnostics(json.dumps([diagnostic(root, row=90)]), root)
    assert gate.compare(first, shifted) == []


def test_malformed_output_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        gate.parse_diagnostics("{bad", tmp_path)


def test_ruff_error_exit_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "renderer").mkdir()
    completed = type(
        "Result", (), {"returncode": 2, "stderr": "configuration error", "stdout": ""}
    )()
    with patch.object(gate.subprocess, "run", return_value=completed):
        with pytest.raises(RuntimeError, match="configuration error"):
            gate.run_ruff(root, "ruff")


def test_ruff_uses_fixed_isolated_policy(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "renderer").mkdir(parents=True)
    completed = type("Result", (), {"returncode": 0, "stderr": "", "stdout": "[]"})()
    with patch.object(gate.subprocess, "run", return_value=completed) as run:
        gate.run_ruff(root, "ruff")

    command = run.call_args.args[0]
    assert "--isolated" in command
    assert "--no-respect-gitignore" in command
    assert command[command.index("--select") + 1] == "E4,E7,E9,F"
    assert command[command.index("--ignore") + 1] == "F722,F821"
    assert command[command.index("--target-version") + 1] == "py39"
    assert command[-1] == "renderer"


def test_relative_roots_are_resolved_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "candidate").mkdir()
    monkeypatch.chdir(tmp_path)
    with patch.object(gate.shutil, "which", return_value="ruff"):
        with patch.object(gate, "run_ruff", return_value=gate.Counter()) as run:
            assert (
                gate.main(["--base-root", "base", "--candidate-root", "candidate"]) == 0
            )

    assert run.call_args_list[0].args[0] == (tmp_path / "base").resolve()


def test_real_ruff_ignores_candidate_rule_configuration(tmp_path: Path) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    base = tmp_path / "base"
    candidate = tmp_path / "candidate"
    for root in (base, candidate):
        (root / "renderer").mkdir(parents=True)
    (candidate / "pyproject.toml").write_text(
        '[tool.ruff]\ninclude = ["renderer/**/*.py"]\n'
        '[tool.ruff.lint]\nselect = ["F"]\nignore = ["F401"]\n',
        encoding="utf-8",
    )
    (base / "renderer" / "module.py").write_text("import os\n", encoding="utf-8")
    (candidate / "renderer" / "module.py").write_text(
        "import os\nimport sys\n", encoding="utf-8"
    )

    baseline = gate.run_ruff(base, str(ruff))
    current = gate.run_ruff(candidate, str(ruff))
    assert sum(baseline.values()) == 1
    assert sum(current.values()) == 2
    assert sum(count for _, count in gate.compare(baseline, current)) == 1


def test_real_ruff_scans_files_ignored_by_candidate_gitignore(tmp_path: Path) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    candidate = tmp_path / "candidate"
    renderer = candidate / "renderer"
    renderer.mkdir(parents=True)
    (candidate / ".gitignore").write_text("renderer/module.py\n", encoding="utf-8")
    (renderer / "module.py").write_text("import os\n", encoding="utf-8")

    diagnostics = gate.run_ruff(candidate, str(ruff))

    assert sum(diagnostics.values()) == 1
    assert next(iter(diagnostics))[0] == "renderer/module.py"


def test_nonempty_scope_without_python_files_fails(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "renderer").mkdir(parents=True)
    completed = type(
        "Result",
        (),
        {
            "returncode": 0,
            "stderr": "warning: No Python files found under the given path(s)",
            "stdout": "[]",
        },
    )()
    with patch.object(gate.subprocess, "run", return_value=completed):
        with pytest.raises(RuntimeError, match="no Python files"):
            gate.run_ruff(root, "ruff")
