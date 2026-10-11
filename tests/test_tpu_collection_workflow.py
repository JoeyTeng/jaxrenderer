"""Contract tests for the manual delayed TPU result collector."""

from __future__ import annotations

from pathlib import Path
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


def locked_controller_install(job: dict[str, Any]) -> dict[str, Any]:
    return next(
        step
        for step in job["steps"]
        if step.get("run", "").strip().startswith("uv sync ")
    )


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
    assert "tools.tpu_collection_gate finish" in finish["run"]
    assert finish["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "PROVIDER_SUCCESS": "${{ steps.provider.outcome == 'success' }}",
    }
    assert "continue-on-error" not in provider
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert "collection-source/" in upload["with"]["path"]
    assert "collection-artifacts/" in upload["with"]["path"]


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
    install = locked_controller_install(job)
    install_args = shlex.split(install["run"])
    assert install_args[:2] == ["uv", "sync"]
    assert "--locked" in install_args
    only_groups = [
        install_args[index + 1]
        for index, argument in enumerate(install_args[:-1])
        if argument == "--only-group"
    ]
    assert only_groups == ["ci-kaggle"]
