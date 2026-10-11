"""Contract tests for the repository's direct Ruff lint gate."""

from collections import Counter
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tomllib

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
LINT_SCOPE = ["assets", "renderer", "examples", "test_resources", "tests", "tools"]


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
                assert re.fullmatch(r"3\.(?:12|13|14)(?:\.\d+)?", python_version)
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
    lint_steps = lint["steps"]
    lint_step_index, lint_step = next(
        (index, step)
        for index, step in enumerate(lint_steps)
        if (tokens := shlex.split(step.get("run", "")))
        and "ruff" in tokens
        and tokens[tokens.index("ruff") + 1] == "check"
        and "--no-respect-gitignore" in tokens
        and "--select" not in tokens
    )
    pyright_step_index = next(
        index
        for index, step in enumerate(lint_steps)
        if "pyright" in shlex.split(step.get("run", ""))
        and "renderer/types.py" in shlex.split(step.get("run", ""))
    )
    assert lint_step_index < pyright_step_index

    checkout_steps = [
        step
        for step in lint_steps[:lint_step_index]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout_steps
    checkout_options = checkout_steps[0]["with"]
    assert (ROOT / checkout_options.get("path", ".")).resolve() == ROOT.resolve()
    assert not checkout_options.get("sparse-checkout")
    lint_python = next(
        step
        for step in lint_steps
        if step.get("uses", "").startswith("actions/setup-python@")
    )["with"]["python-version"]

    for index, step in enumerate(lint_steps[:pyright_step_index]):
        if index == lint_step_index:
            continue
        action = step.get("uses", "")
        if action:
            assert action.startswith(
                ("actions/checkout@", "actions/setup-python@", "astral-sh/setup-uv@")
            )
            assert "run" not in step
            continue

        command = step["run"].strip()
        assert "\n" not in command and "\r" not in command
        assert not re.search(r"[|;&<>`]|\$\(", command)
        tokens = shlex.split(command)
        if tokens[:2] == ["uv", "sync"]:
            assert tokens[tokens.index("--python") + 1] == lint_python
            assert Counter(tokens[2:]) == Counter(
                ["--locked", "--all-groups", "--python", lint_python]
            )
            continue

        tokens = direct_command_tokens(command, "ruff", lint_python)
        if tokens[1] == "check" and "--select" in tokens[2:]:
            select_index = tokens.index("--select")
            assert tokens[select_index + 1] == "I"
            assert Counter(
                token
                for index, token in enumerate(tokens[2:], start=2)
                if index not in (select_index, select_index + 1)
            ) == Counter(["--no-respect-gitignore", *LINT_SCOPE])
        else:
            assert tokens[1] == "format"
            assert Counter(tokens[2:]) == Counter(
                ["--no-respect-gitignore", "--check", *LINT_SCOPE]
            )

    tokens = direct_command_tokens(lint_step["run"], "ruff", lint_python)
    assert tokens[1] == "check"
    assert Counter(tokens[2:]) == Counter(["--no-respect-gitignore", *LINT_SCOPE])


def test_ruff_configuration_keeps_required_rules_and_intentional_ignores() -> None:
    ruff = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "ruff"
    ]["lint"]

    assert {"E4", "E7", "E9", "F", "I"}.issubset(ruff["select"])
    assert set(ruff["ignore"]) == {"F722", "F821"}


def test_ruff_includes_python_and_stub_files_in_every_lint_scope(
    tmp_path: Path,
) -> None:
    ruff = Path(sys.executable).with_name("ruff")
    assert ruff.is_file()
    (tmp_path / "pyproject.toml").write_text(
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
            path.write_text("", encoding="utf-8")
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


def test_pre_commit_runs_normal_ruff_check_with_existing_hooks_preserved() -> None:
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    local_hooks = next(
        repo["hooks"] for repo in config["repos"] if repo["repo"] == "local"
    )
    hooks_by_id = {hook["id"]: hook for hook in local_hooks}

    import_tokens = direct_command_tokens(hooks_by_id["ruff-imports"]["entry"], "ruff")
    lint_tokens = direct_command_tokens(hooks_by_id["ruff-check"]["entry"], "ruff")
    format_tokens = direct_command_tokens(hooks_by_id["ruff-format"]["entry"], "ruff")

    assert import_tokens[1] == "check"
    select_index = import_tokens.index("--select")
    assert import_tokens[select_index + 1] == "I"
    assert Counter(
        token
        for index, token in enumerate(import_tokens[2:], start=2)
        if index not in (select_index, select_index + 1)
    ) == Counter(["--no-respect-gitignore", "--fix"])

    assert lint_tokens[1] == "check"
    assert Counter(lint_tokens[2:]) == Counter(["--no-respect-gitignore"])

    assert format_tokens[1] == "format"
    assert Counter(format_tokens[2:]) == Counter(["--no-respect-gitignore"])

    for hook_id in ("ruff-imports", "ruff-check", "ruff-format"):
        pattern = re.compile(hooks_by_id[hook_id]["files"])
        for scope in LINT_SCOPE:
            for relative in (
                "module.py",
                "module.pyi",
                "nested/module.py",
                "nested/module.pyi",
            ):
                assert pattern.fullmatch(f"{scope}/{relative}")
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
