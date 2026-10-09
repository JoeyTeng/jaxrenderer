"""Contract tests for the repository's direct Ruff lint gate."""

from pathlib import Path
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
        *LINT_SCOPE,
    ]
    assert not any("baseline" in str(step).lower() for step in lint["steps"])


def test_ruff_gate_keeps_the_repository_rule_selection_and_ignores() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ruff = pyproject["tool"]["ruff"]
    assert ruff["lint"]["select"] == ["E4", "E7", "E9", "F", "I"]
    assert ruff["lint"]["ignore"] == ["F722", "F821"]


def test_pre_commit_runs_normal_ruff_check_with_existing_hooks_preserved() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    local_hooks = next(
        repo["hooks"] for repo in config["repos"] if repo["repo"] == "local"
    )
    hooks_by_id = {hook["id"]: hook for hook in local_hooks}

    assert hooks_by_id["ruff-check"]["entry"] == "uv run --no-sync ruff check"
    assert hooks_by_id["ruff-imports"]["entry"].endswith("ruff check --select I --fix")
    assert hooks_by_id["ruff-format"]["entry"] == "uv run --no-sync ruff format"
