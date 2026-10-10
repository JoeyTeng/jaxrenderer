"""Contract tests for the blocking core Pyright gate."""

from pathlib import Path
import tomllib
from typing import Any, cast

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CORE_PYRIGHT = "uv run --no-sync --python 3.14 pyright --warnings renderer/types.py"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_core_pyright_gate_is_strict_required_and_candidate_scoped() -> None:
    checks = load_yaml(WORKFLOWS / "checks.yml")["jobs"]
    lint = checks["lint"]
    checkouts = [
        step
        for step in lint["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    gate = named_step(lint, "Run strict Pyright on core types")

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == (
        "${{ inputs.commit || github.event.pull_request.head.sha || github.sha }}"
    )
    assert gate["run"] == CORE_PYRIGHT
    assert "continue-on-error" not in gate
    assert "continue-on-error" not in lint


def test_full_repository_pyright_remains_visible_and_advisory() -> None:
    check = load_yaml(WORKFLOWS / "checks.yml")["jobs"]["check"]
    full_pyright = named_step(check, "Run pyright")

    assert full_pyright["if"] == "matrix.python-version == '3.14'"
    assert full_pyright["continue-on-error"] == "true"
    assert full_pyright["run"] == (
        'uv run --no-sync --python "${{ matrix.python-version }}" pyright'
    )


def test_pyright_uses_the_project_uv_environment_without_weakening_strictness() -> None:
    pyright = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["pyright"]

    assert pyright["typeCheckingMode"] == "strict"
    assert pyright["venvPath"] == "."
    assert pyright["venv"] == ".venv"
    assert pyright["pythonVersion"] == "3.14"


def test_pre_commit_runs_the_same_core_gate_on_config_or_target_changes() -> None:
    config = load_yaml(ROOT / ".pre-commit-config.yaml")
    hooks = next(repo["hooks"] for repo in config["repos"] if repo["repo"] == "local")
    core_hook = next(hook for hook in hooks if hook["id"] == "pyright-core-types")
    full_hook = next(hook for hook in hooks if hook["id"] == "pyright")

    assert core_hook["entry"] == CORE_PYRIGHT
    assert core_hook["files"] == r"^(pyproject\.toml|renderer/types\.py)$"
    assert core_hook["pass_filenames"] == "false"
    assert full_hook["stages"] == ["manual"]


def test_release_reuses_the_same_required_gate_on_the_frozen_candidate() -> None:
    release = load_yaml(WORKFLOWS / "pypi.yml")["jobs"]
    checks = load_yaml(WORKFLOWS / "checks.yml")

    assert release["cpu"]["uses"] == "./.github/workflows/checks.yml"
    assert release["cpu"]["with"]["commit"] == "${{ needs.prepare.outputs.head_sha }}"
    assert checks["on"]["workflow_call"]["inputs"]["commit"]["required"] == "true"
