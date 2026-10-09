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


def step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_collector_is_manual_read_only_and_serialised_by_source_attempt() -> None:
    workflow = load_workflow()

    assert workflow["name"] == "Collect delayed TPU results"
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
    assert job["timeout-minutes"] == "15"
    assert job["environment"] == "kaggle-tpu"
    assert "PyPI" not in str(job)
    assert "uv publish" not in str(job)
    assert "PYPI_API_TOKEN" not in str(job)
    assert "UV_PUBLISH_TOKEN" not in str(job)


def test_collector_binds_original_evidence_before_provider_status_query() -> None:
    job = load_workflow()["jobs"]["collect"]
    steps = job["steps"]
    names = [step.get("name", step.get("uses")) for step in steps]
    prepare = step_named(
        job, "Bind the original publishing run and retrieve its saved state"
    )
    provider = step_named(job, "Collect the saved Kaggle TPU result")
    finish = step_named(job, "Validate the collected result and write its receipt")
    upload = step_named(job, "Upload TPU collection evidence")

    assert names.index(prepare["name"]) < names.index(
        "Install the locked Kaggle controller"
    )
    assert names.index("Install the locked Kaggle controller") < names.index(
        provider["name"]
    )
    assert (
        names.index(provider["name"])
        < names.index(finish["name"])
        < names.index(upload["name"])
    )
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
    assert provider["timeout-minutes"] == "5"
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
    assert upload["with"]["retention-days"] == "30"


def test_collector_checkout_and_dependency_install_use_trusted_locked_code() -> None:
    job = load_workflow()["jobs"]["collect"]
    checkout = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    setup_python = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    install = step_named(job, "Install the locked Kaggle controller")

    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert setup_python["with"]["python-version"] == "3.14"
    assert install["run"] == "uv sync --locked --only-group ci-kaggle --python 3.14"
