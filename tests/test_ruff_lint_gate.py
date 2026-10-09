"""Contract tests for the repository's direct Ruff lint gate."""

from pathlib import Path
import re
import subprocess
import sys
import tomllib

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
LINT_SCOPE = ["assets", "renderer", "examples", "test_resources", "tests", "tools"]


def load_checks_workflow() -> dict:
    value = yaml.load(
        (WORKFLOWS / "checks.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(value, dict)
    return value


def test_ci_runs_direct_ruff_check_on_the_full_candidate_scope() -> None:
    workflow = load_checks_workflow()
    lint = workflow["jobs"]["lint"]
    checkout = next(
        step
        for step in lint["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    lint_step = next(
        step for step in lint["steps"] if step.get("name") == "Check Ruff lint"
    )

    assert checkout["with"]["ref"] == (
        "${{ inputs.commit || github.event.pull_request.head.sha || github.sha }}"
    )
    assert (
        sum(
            step.get("uses", "").startswith("actions/checkout@")
            for step in lint["steps"]
        )
        == 1
    )
    assert lint_step["run"].split() == [
        "uv",
        "run",
        "--no-sync",
        "--python",
        "3.14",
        "ruff",
        "check",
        "--no-respect-gitignore",
        *LINT_SCOPE,
    ]
    assert not any("baseline" in str(step).lower() for step in lint["steps"])


def test_ruff_gate_keeps_the_repository_rule_selection_and_ignores() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ruff = pyproject["tool"]["ruff"]
    assert ruff["lint"]["select"] == ["E4", "E7", "E9", "F", "I"]
    assert ruff["lint"]["ignore"] == ["F722", "F821"]
    assert ruff["include"] == [
        f"{scope}/**/*.{suffix}" for scope in LINT_SCOPE for suffix in ("py", "pyi")
    ]


def test_pre_commit_runs_normal_ruff_check_with_existing_hooks_preserved() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    local_hooks = next(
        repo["hooks"] for repo in config["repos"] if repo["repo"] == "local"
    )
    hooks_by_id = {hook["id"]: hook for hook in local_hooks}

    assert hooks_by_id["ruff-check"]["entry"] == (
        "uv run --no-sync ruff check --no-respect-gitignore"
    )
    assert hooks_by_id["ruff-imports"]["entry"].endswith(
        "ruff check --no-respect-gitignore --select I --fix"
    )
    assert hooks_by_id["ruff-format"]["entry"] == (
        "uv run --no-sync ruff format --no-respect-gitignore"
    )
    for hook_id in ("ruff-imports", "ruff-check", "ruff-format"):
        pattern = re.compile(hooks_by_id[hook_id]["files"])
        assert pattern.fullmatch("renderer/module.py")
        assert pattern.fullmatch("renderer/stubs/module.pyi")
        assert not pattern.fullmatch("typings/module.pyi")


def test_real_ruff_rejects_a_lint_violation_in_a_stub_file(tmp_path: Path) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    stub = tmp_path / "assets" / "lint_error.pyi"
    stub.parent.mkdir()
    stub.write_text(
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
    (tmp_path / "pyproject.toml").write_text(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("assets/hidden.py\n", encoding="utf-8")
    source = tmp_path / "assets" / "hidden.py"
    source.parent.mkdir()
    source.write_text("import os\n", encoding="utf-8")
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
