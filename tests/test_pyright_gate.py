"""Contract tests for the blocking core Pyright gate."""

from pathlib import Path
import re
import shlex
import tomllib
from typing import Any, cast

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_core_pyright_gate_is_strict_required_and_candidate_scoped() -> None:
    checks = load_yaml(WORKFLOWS / "checks.yml")["jobs"]
    lint = checks["lint"]
    checkouts = [
        step
        for step in lint["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    gate = next(
        step
        for step in lint["steps"]
        if "pyright" in shlex.split(step.get("run", ""))
        and "renderer/types.py" in shlex.split(step.get("run", ""))
    )
    command = shlex.split(gate["run"])

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == (
        "${{ inputs.commit || github.event.pull_request.head.sha || github.sha }}"
    )
    assert "--warnings" in command
    assert gate.get("continue-on-error", "false") == "false"
    assert lint.get("continue-on-error", "false") == "false"


def test_pyright_configuration_is_strict() -> None:
    pyright = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["pyright"]

    assert pyright["typeCheckingMode"] == "strict"


def test_pre_commit_runs_the_same_core_gate_on_config_or_target_changes() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(config, dict)
    hooks = next(repo["hooks"] for repo in config["repos"] if repo["repo"] == "local")
    core_hook = next(hook for hook in hooks if hook["id"] == "pyright-core-types")
    full_hook = next(hook for hook in hooks if hook["id"] == "pyright")

    pattern = re.compile(core_hook["files"])
    assert pattern.search("pyproject.toml")
    assert pattern.search("renderer/types.py")
    assert not pattern.search("tests/unrelated.py")
    assert core_hook["pass_filenames"] is False
    assert full_hook["stages"] == ["manual"]


def test_release_reuses_the_same_required_gate_on_the_frozen_candidate() -> None:
    release = load_yaml(WORKFLOWS / "pypi.yml")["jobs"]
    checks = load_yaml(WORKFLOWS / "checks.yml")

    assert release["cpu"]["uses"] == "./.github/workflows/checks.yml"
    assert release["cpu"]["with"]["commit"] == "${{ needs.prepare.outputs.head_sha }}"
    assert checks["on"]["workflow_call"]["inputs"]["commit"]["required"] == "true"
