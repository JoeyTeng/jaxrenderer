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


def supported_python_version(version: str) -> bool:
    return re.fullmatch(r"3\.(?:12|13|14)(?:\.\d+)?", version) is not None


def direct_command_tokens(
    command: str, tool: str, expected_python: str | None = None
) -> list[str]:
    """Return argv for a simple uv invocation of the expected tool."""
    assert "\n" not in command and "\r" not in command
    assert not re.search(r"[|;&<>`]|\$\(", command)
    tokens = shlex.split(command)
    assert tokens[:2] == ["uv", "run"]
    tokens = tokens[2:]
    python_options = 0
    while tokens and tokens[0] in {"--no-sync", "--python"}:
        option = tokens.pop(0)
        if option == "--python":
            python_options += 1
            assert tokens
            python_version = tokens.pop(0)
            matrix_version = "${{ matrix.python-version }}"
            if python_version == matrix_version:
                assert expected_python == matrix_version
            else:
                assert supported_python_version(python_version)
            if expected_python is not None:
                if expected_python == matrix_version:
                    assert python_version == matrix_version
                else:
                    expected_parts = expected_python.split(".")
                    actual_parts = python_version.split(".")
                    assert actual_parts[:2] == expected_parts[:2]
    assert python_options <= 1
    if expected_python is not None:
        assert python_options == 1
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
    lint_python = next(
        step
        for step in lint["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )["with"]["python-version"]
    command = direct_command_tokens(gate["run"], "pyright", lint_python)

    assert len(checkouts) == 1
    assert checkouts[0]["with"]["repository"] == (
        "${{ github.event.pull_request.head.repo.full_name || github.repository }}"
    )
    assert checkouts[0]["with"]["ref"] == (
        "${{ inputs.commit || github.event.pull_request.head.sha || github.sha }}"
    )
    assert Counter(command) == Counter(["pyright", "--warnings", "renderer/types.py"])
    assert gate.get("continue-on-error", "false") == "false"
    assert lint.get("continue-on-error", "false") == "false"

    check = checks["check"]
    matrix_python_versions = check["strategy"]["matrix"]["python-version"]
    assert matrix_python_versions
    assert all(supported_python_version(version) for version in matrix_python_versions)
    check_python = next(
        step
        for step in check["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )["with"]["python-version"]
    assert check_python == "${{ matrix.python-version }}"
    sync_step = next(
        step
        for step in check["steps"]
        if shlex.split(step.get("run", ""))[:2] == ["uv", "sync"]
    )
    sync_command = sync_step["run"].strip()
    assert "\n" not in sync_command and "\r" not in sync_command
    sync_tokens = shlex.split(sync_command)
    assert sync_tokens[:2] == ["uv", "sync"]
    assert sync_tokens[sync_tokens.index("--python") + 1] == check_python
    assert Counter(sync_tokens) == Counter(
        [
            "uv",
            "sync",
            "--locked",
            "--all-groups",
            "--python",
            "${{ matrix.python-version }}",
        ]
    )
    python_check = next(
        step
        for step in check["steps"]
        if "python" in shlex.split(step.get("run", ""))
        and "--version" in shlex.split(step.get("run", ""))
    )
    assert direct_command_tokens(python_check["run"], "python", check_python) == [
        "python",
        "--version",
    ]
    pytest_step = next(
        step for step in check["steps"] if "pytest" in shlex.split(step.get("run", ""))
    )
    pytest_tokens = direct_command_tokens(pytest_step["run"], "pytest", check_python)
    assert Counter(pytest_tokens[1:]) == Counter(
        ["tests/", "--import-mode", "importlib"]
    )
    assert pytest_tokens[pytest_tokens.index("--import-mode") + 1] == "importlib"
    full_pyright = next(
        step
        for step in check["steps"]
        if "pyright" in shlex.split(step.get("run", ""))
        and "renderer/types.py" not in shlex.split(step.get("run", ""))
    )
    assert direct_command_tokens(full_pyright["run"], "pyright", check_python) == [
        "pyright"
    ]
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
    assert "venvPath" in pyright and "venv" in pyright
    assert (ROOT / pyright["venvPath"] / pyright["venv"]).resolve() == (
        ROOT / ".venv"
    ).resolve()
    global_python = pyright["pythonVersion"]
    assert supported_python_version(global_python)
    assert all(
        supported_python_version(environment.get("pythonVersion", global_python))
        for environment in pyright.get("executionEnvironments", [])
    )


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
