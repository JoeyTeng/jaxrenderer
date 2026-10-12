"""Contract tests for the manual delayed TPU result collector."""

from __future__ import annotations

from pathlib import Path
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
    setup_indexes = [steps.index(setup_python_step), steps.index(setup_uv_step)]
    install_index = next(
        index
        for index, step in enumerate(steps)
        if "--locked" in step.get("run", "") and "ci-kaggle" in step.get("run", "")
    )
    assert install_index > max(setup_indexes)


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
    assert "PYPI_API_TOKEN" not in str(job)
    assert "UV_PUBLISH_TOKEN" not in str(job)


def test_collector_binds_original_evidence_before_provider_status_query() -> None:
    job = load_workflow()["jobs"]["collect"]
    prepare = step_with_id(job, "prepare")
    provider = step_with_id(job, "provider")
    finish = next(
        step
        for step in job["steps"]
        if step.get("if") == "always()" and "PROVIDER_SUCCESS" in step.get("env", {})
    )
    steps = job["steps"]
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    install = next(
        step
        for step in job["steps"]
        if "--locked" in step.get("run", "") and "ci-kaggle" in step.get("run", "")
    )

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
    assert provider["env"] == {
        "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
        "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
    }
    assert "GH_TOKEN" not in provider["env"]

    assert finish["if"] == "always()"
    assert "continue-on-error" not in finish
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
