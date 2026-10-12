"""Contract tests for the scheduled delayed TPU result collector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from tools.tpu_collection_periodic import MAX_RUNTIME_SECONDS
import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


def load_workflow() -> dict[str, Any]:
    """Load GitHub Actions YAML without YAML 1.1 boolean coercion."""
    value = yaml.load(
        (WORKFLOWS / "collect-tpu-periodic.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def step_with_id(job: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("id") == step_id)


def artifact_upload_for_path(job: dict[str, Any], path: str) -> dict[str, Any]:
    return next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
        and step.get("with", {}).get("path") == path
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
    setup_indexes = [steps.index(setup_python_step), steps.index(setup_uv_step)]
    install_index = next(
        index
        for index, step in enumerate(steps)
        if "--locked" in step.get("run", "") and "ci-kaggle" in step.get("run", "")
    )
    assert install_index > max(setup_indexes)


def test_periodic_workflow_is_master_only_read_only_and_serialised() -> None:
    workflow = load_workflow()

    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {
        "group": "tpu-periodic-collection",
        "cancel-in-progress": "false",
    }

    discover = workflow["jobs"]["discover"]
    assert discover["if"] == (
        "github.repository == 'JoeyTeng/jaxrenderer' && "
        "github.ref == 'refs/heads/master'"
    )
    assert int(discover["timeout-minutes"]) * 60 >= MAX_RUNTIME_SECONDS
    assert discover["outputs"] == {
        "has_sources": "${{ steps.discover.outputs.has_sources }}",
        "matrix": "${{ steps.discover.outputs.matrix }}",
    }
    assert set(workflow["jobs"]) == {"discover", "collect"}


def test_discovery_is_read_only_no_source_job_and_retains_its_evidence() -> None:
    workflow = load_workflow()
    discover = workflow["jobs"]["discover"]
    step = step_with_id(discover, "discover")
    upload = artifact_upload_for_path(discover, "periodic-discovery/")

    assert step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "KAGGLE" not in str(discover)
    assert "secrets." not in str(discover)
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "periodic-discovery/"
    assert int(upload["with"]["retention-days"]) >= 30

    collect = workflow["jobs"]["collect"]
    assert collect["needs"] == "discover"
    assert collect["if"] == "needs.discover.outputs.has_sources == 'true'"
    assert int(collect["timeout-minutes"]) * 60 - MAX_RUNTIME_SECONDS >= 5 * 60
    assert collect["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "1",
        "matrix": "${{ fromJSON(needs.discover.outputs.matrix) }}",
    }


def test_collection_uses_original_attempt_lock_and_trusted_locked_controller() -> None:
    workflow = load_workflow()
    collect = workflow["jobs"]["collect"]

    assert collect["environment"] == "kaggle-tpu"
    assert collect["concurrency"] == {
        "group": "tpu-collection-${{ matrix.source_run_id }}-${{ matrix.source_attempt }}",
        "cancel-in-progress": "false",
    }

    checkout = next(
        step
        for step in collect["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert_locked_controller_sync(collect)


def test_collection_uploads_all_evidence_and_only_terminal_outcomes_get_marker() -> (
    None
):
    workflow = load_workflow()
    collect = workflow["jobs"]["collect"]
    step = step_with_id(collect, "collect")
    evidence = artifact_upload_for_path(collect, "periodic-artifacts/")
    terminal = artifact_upload_for_path(
        collect, "periodic-artifacts/periodic-terminal.json"
    )

    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
        "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
        "SOURCE_RUN_ID": "${{ matrix.source_run_id }}",
        "SOURCE_ATTEMPT": "${{ matrix.source_attempt }}",
        "SOURCE_COMMIT": "${{ matrix.commit }}",
    }
    assert "workflow_dispatch" not in step["run"]
    assert "submit" not in step["run"].lower()

    assert evidence["if"] == "always()"
    evidence_name = evidence["with"]["name"]
    assert "${{ matrix.source_run_id }}" in evidence_name
    assert "${{ matrix.source_attempt }}" in evidence_name
    assert not evidence_name.startswith("tpu-periodic-terminal-")
    assert evidence["with"]["path"] == "periodic-artifacts/"
    assert int(evidence["with"]["retention-days"]) >= 30

    assert terminal["if"] == "always() && steps.collect.outputs.terminal == 'true'"
    assert terminal["with"]["name"] == (
        "tpu-periodic-terminal-${{ matrix.source_run_id }}-"
        "${{ matrix.source_attempt }}-${{ matrix.commit }}"
    )
    assert terminal["with"]["path"] == "periodic-artifacts/periodic-terminal.json"
    assert int(terminal["with"]["retention-days"]) >= 30
    assert "PYPI_API_TOKEN" not in str(workflow)
    assert "UV_PUBLISH_TOKEN" not in str(workflow)
    assert "release-tpu.yml" not in str(workflow)
