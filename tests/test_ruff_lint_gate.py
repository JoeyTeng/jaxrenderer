"""Runtime behaviour tests for the repository's Ruff lint gate."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]
LINT_SCOPE = ["assets", "renderer", "examples", "test_resources", "tests", "tools"]


def test_ruff_includes_python_and_stub_files_in_every_lint_scope(
    tmp_path: Path,
) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    _ = (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    expected: set[str] = set()
    for scope in LINT_SCOPE:
        for relative in (
            "module.py",
            "module.pyi",
            "nested/module.py",
            "nested/module.pyi",
        ):
            path = tmp_path / scope / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _ = path.write_text("", encoding="utf-8")
            expected.add(str(path.resolve()))

    result = subprocess.run(
        [
            str(ruff),
            "check",
            "--show-files",
            "--no-respect-gitignore",
            *LINT_SCOPE,
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert expected.issubset(set(result.stdout.splitlines()))


def test_real_ruff_rejects_a_lint_violation_in_a_stub_file(tmp_path: Path) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    _ = (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    stub = tmp_path / "assets" / "lint_error.pyi"
    stub.parent.mkdir()
    _ = stub.write_text(
        "def duplicate() -> None: ...\ndef duplicate() -> None: ...\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(ruff), "check", "--no-respect-gitignore", "assets"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "F811" in result.stdout
    assert "lint_error.pyi" in result.stdout


def test_real_ruff_checks_gitignored_python_files_with_override(
    tmp_path: Path,
) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    _ = (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / ".git").mkdir()
    _ = (tmp_path / ".gitignore").write_text("assets/hidden.py\n", encoding="utf-8")
    source = tmp_path / "assets" / "hidden.py"
    source.parent.mkdir()
    _ = source.write_text("import os\n", encoding="utf-8")
    ignored = subprocess.run(
        [str(ruff), "check", "assets"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    checked = subprocess.run(
        [str(ruff), "check", "--no-respect-gitignore", "assets"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert ignored.returncode == 0
    assert "No Python files found" in ignored.stdout + ignored.stderr
    assert checked.returncode == 1
    assert "F401" in checked.stdout
    assert "hidden.py" in checked.stdout
