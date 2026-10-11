"""Contract tests for the manual delayed TPU result collector."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import shlex
from typing import Any, cast

import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


def load_workflow() -> dict[str, Any]:
    """Load GitHub Actions YAML without YAML 1.1 boolean coercion."""
    value = yaml.load(
        (WORKFLOWS / "collect-tpu.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def step_with_id(job: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("id") == step_id)


def step_running_module(job: dict[str, Any], module: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if f"-m {module}" in step.get("run", ""))


def assert_finish_invocation(step: dict[str, Any]) -> None:
    command = step["run"].strip()
    assert "\n" not in command and "\r" not in command
    arguments = shlex.split(command)
    assert arguments[:4] == [
        "python",
        "-m",
        "tools.tpu_collection_gate",
        "finish",
    ]
    options = arguments[4:]
    assert len(options) == 6
    assert Counter(zip(options[::2], options[1::2], strict=True)) == Counter(
        {
            ("--source-dir", "collection-source"): 1,
            ("--output-dir", "collection-artifacts"): 1,
            ("--provider-success", "$PROVIDER_SUCCESS"): 1,
        }
    )
    assert "continue-on-error" not in step


def locked_controller_install(job: dict[str, Any]) -> dict[str, Any]:
    return next(
        step
        for step in job["steps"]
        if step.get("run", "").strip().startswith("uv sync ")
    )


def assert_locked_controller_sync(job: dict[str, Any]) -> None:
    steps = job["steps"]
    setup_python_step = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    setup_uv_step = next(
        step for step in steps if step.get("uses", "").startswith("astral-sh/setup-uv@")
    )
    install_step = locked_controller_install(job)
    assert steps.index(setup_python_step) < steps.index(install_step)
    assert steps.index(setup_uv_step) < steps.index(install_step)
    command = install_step["run"].strip()
    assert "\n" not in command and "\r" not in command
    arguments = shlex.split(command)
    assert arguments[:2] == ["uv", "sync"]
    options = arguments[2:]
    python_indexes = [
        index for index, value in enumerate(options) if value == "--python"
    ]
    assert len(python_indexes) == 1 and python_indexes[0] + 1 < len(options)
    python_index = python_indexes[0]
    python_version = options[python_index + 1]
    assert re.fullmatch(r"\d+\.\d+(?:\.\d+)?", python_version)
    version_parts = tuple(int(part) for part in python_version.split("."))
    assert version_parts[0] == 3 and 12 <= version_parts[1] <= 14
    setup_python = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/setup-python@")
    )["with"]["python-version"]
    setup_version = tuple(int(part) for part in setup_python.split("."))
    assert version_parts[:2] == setup_version[:2]
    remaining = options[:python_index] + options[python_index + 2 :]
    assert Counter(remaining) == Counter(
        {"--locked": 1, "--only-group": 1, "ci-kaggle": 1}
    )
    assert remaining[remaining.index("--only-group") + 1] == "ci-kaggle"


def test_collector_is_manual_read_only_and_serialised_by_source_attempt() -> None:
    workflow = load_workflow()

    assert set(workflow["on"]) == {"workflow_dispatch"}
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"source_run_id", "source_attempt", "commit"}
    assert all(value["required"] == "true" for value in inputs.values())
    assert inputs["source_attempt"]["default"] == "1"
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {
        "group": "tpu-collection-${{ inputs.source_run_id }}-${{ inputs.source_attempt }}",
        "cancel-in-progress": "false",
    }
    assert set(workflow["jobs"]) == {"collect"}
    job = workflow["jobs"]["collect"]
    assert job["environment"] == "kaggle-tpu"
    assert "uv publish" not in str(job)
    assert "PYPI_API_TOKEN" not in str(job)
    assert "UV_PUBLISH_TOKEN" not in str(job)


def test_collector_binds_original_evidence_before_provider_status_query() -> None:
    job = load_workflow()["jobs"]["collect"]
    steps = job["steps"]
    prepare = step_with_id(job, "prepare")
    provider = step_with_id(job, "provider")
    finish = step_running_module(job, "tools.tpu_collection_gate finish")
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    install = locked_controller_install(job)

    assert steps.index(prepare) < steps.index(install) < steps.index(provider)
    assert steps.index(provider) < steps.index(finish) < steps.index(upload)
    job_timeout = int(job["timeout-minutes"])
    provider_timeout = int(provider["timeout-minutes"])
    assert 0 < provider_timeout <= 5
    assert job_timeout <= 15
    assert job_timeout - provider_timeout >= 10
    assert prepare["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "SOURCE_RUN_ID": "${{ inputs.source_run_id }}",
        "SOURCE_ATTEMPT": "${{ inputs.source_attempt }}",
        "SOURCE_COMMIT": "${{ inputs.commit }}",
    }
    assert "tools.tpu_collection_gate prepare" in prepare["run"]
    assert '--source-run "$SOURCE_RUN_ID"' in prepare["run"]
    assert '--source-attempt "$SOURCE_ATTEMPT"' in prepare["run"]
    assert '--commit "$SOURCE_COMMIT"' in prepare["run"]
    assert "--output-dir collection-source" in prepare["run"]
    assert provider["env"] == {
        "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
        "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
    }
    assert ".venv/bin/python -u -m tools.kaggle_collect" in provider["run"]
    assert "tools.kaggle_ci" not in provider["run"]
    assert "--accelerator" not in provider["run"]
    assert "--binding collection-source/binding.json" in provider["run"]
    assert (
        "--resume-state collection-source/kaggle-controller-state.json"
        in provider["run"]
    )
    assert "GH_TOKEN" not in provider["env"]

    assert finish["if"] == "always()"
    assert_finish_invocation(finish)
    assert finish["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "PROVIDER_SUCCESS": "${{ steps.provider.outcome == 'success' }}",
    }
    assert "continue-on-error" not in provider
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert "collection-source/" in upload["with"]["path"]
    assert "collection-artifacts/" in upload["with"]["path"]
    assert int(upload["with"]["retention-days"]) >= 30


def test_collector_checkout_and_dependency_install_use_trusted_locked_code() -> None:
    job = load_workflow()["jobs"]["collect"]
    checkout = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert_locked_controller_sync(job)
