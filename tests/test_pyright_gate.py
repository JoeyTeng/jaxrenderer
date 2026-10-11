"""Contract tests for the blocking core Pyright gate."""

from collections import Counter
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


def direct_command_tokens(command: str, tool: str) -> list[str]:
    """Return argv for a simple uv invocation of the expected tool."""
    assert "\n" not in command and "\r" not in command
    assert not re.search(r"[|;&<>`]|\$\(", command)
    tokens = shlex.split(command)
    assert tokens[:2] == ["uv", "run"]
    tokens = tokens[2:]
    while tokens and tokens[0] in {"--no-sync", "--python"}:
        option = tokens.pop(0)
        if option == "--python":
            assert tokens
            tokens.pop(0)
    assert tokens and tokens[0] == tool
    return tokens


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
    command = direct_command_tokens(gate["run"], "pyright")

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["ref"] == (
        "${{ inputs.commit || github.event.pull_request.head.sha || github.sha }}"
    )
    assert Counter(command) == Counter(["pyright", "--warnings", "renderer/types.py"])
    assert gate.get("continue-on-error", "false") == "false"
    assert lint.get("continue-on-error", "false") == "false"

    full_pyright = next(
        step
        for step in checks["check"]["steps"]
        if "pyright" in shlex.split(step.get("run", ""))
        and "renderer/types.py" not in shlex.split(step.get("run", ""))
    )
    assert direct_command_tokens(full_pyright["run"], "pyright") == ["pyright"]
    assert re.fullmatch(
        r"\s*matrix\.python-version\s*==\s*'3\.14'\s*",
        full_pyright["if"],
    )
    assert full_pyright["continue-on-error"] == "true"


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
    core_command = direct_command_tokens(core_hook["entry"], "pyright")

    pattern = re.compile(core_hook["files"])
    assert pattern.search("pyproject.toml")
    assert pattern.search("renderer/types.py")
    assert not pattern.search("tests/unrelated.py")
    assert core_hook["pass_filenames"] is False
    assert Counter(core_command) == Counter(
        ["pyright", "--warnings", "renderer/types.py"]
    )
    assert full_hook["stages"] == ["manual"]
