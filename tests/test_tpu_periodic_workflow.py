"""Contract tests for the scheduled delayed TPU result collector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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


def named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_periodic_workflow_is_master_only_read_only_and_serialised() -> None:
    workflow = load_workflow()

    assert workflow["name"] == "Collect delayed TPU results periodically"
    assert workflow["on"] == {
        "schedule": [{"cron": "17,47 * * * *"}],
        "workflow_dispatch": "",
    }
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
    assert discover["timeout-minutes"] == "10"
    assert discover["outputs"] == {
        "has_sources": "${{ steps.discover.outputs.has_sources }}",
        "matrix": "${{ steps.discover.outputs.matrix }}",
    }
    assert set(workflow["jobs"]) == {"discover", "collect"}


def test_discovery_is_read_only_no_source_job_and_retains_its_evidence() -> None:
    workflow = load_workflow()
    discover = workflow["jobs"]["discover"]
    step = named_step(discover, "Find eligible timed-out TPU source runs")
    upload = named_step(discover, "Upload periodic discovery evidence")

    assert step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "tools.tpu_collection_periodic discover" in step["run"]
    assert "--output-dir periodic-discovery" in step["run"]
    assert '--github-output "$GITHUB_OUTPUT"' in step["run"]
    assert "KAGGLE" not in str(discover)
    assert "secrets." not in str(discover)
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "periodic-discovery/"
    assert upload["with"]["retention-days"] == "30"

    collect = workflow["jobs"]["collect"]
    assert collect["needs"] == "discover"
    assert collect["if"] == "needs.discover.outputs.has_sources == 'true'"
    assert collect["strategy"] == {
        "fail-fast": "false",
        "max-parallel": "1",
        "matrix": "${{ fromJSON(needs.discover.outputs.matrix) }}",
    }


def test_collection_uses_original_attempt_lock_and_trusted_locked_controller() -> None:
    workflow = load_workflow()
    collect = workflow["jobs"]["collect"]

    assert collect["timeout-minutes"] == "15"
    assert collect["environment"] == "kaggle-tpu"
    assert collect["concurrency"] == {
        "group": "tpu-collection-${{ matrix.source_run_id }}-${{ matrix.source_attempt }}",
        "cancel-in-progress": "false",
    }
    assert collect["strategy"]["max-parallel"] == "1"

    checkout = next(
        step
        for step in collect["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    python = next(
        step
        for step in collect["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    uv = next(
        step
        for step in collect["steps"]
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    )
    install = named_step(collect, "Install the locked Kaggle controller")
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": "false",
    }
    assert python["with"]["python-version"] == "3.14"
    assert uv["with"]["version"] == "0.12.20"
    assert install["run"] == "uv sync --locked --only-group ci-kaggle --python 3.14"


def test_collection_uploads_all_evidence_and_only_terminal_outcomes_get_marker() -> (
    None
):
    workflow = load_workflow()
    collect = workflow["jobs"]["collect"]
    step = named_step(collect, "Collect and validate the delayed TPU result")
    evidence = named_step(collect, "Upload periodic collection evidence")
    terminal = named_step(collect, "Record terminal collection marker")

    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
        "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
        "SOURCE_RUN_ID": "${{ matrix.source_run_id }}",
        "SOURCE_ATTEMPT": "${{ matrix.source_attempt }}",
        "SOURCE_COMMIT": "${{ matrix.commit }}",
    }
    assert "tools.tpu_collection_periodic collect" in step["run"]
    assert '--source-run "$SOURCE_RUN_ID"' in step["run"]
    assert '--source-attempt "$SOURCE_ATTEMPT"' in step["run"]
    assert '--commit "$SOURCE_COMMIT"' in step["run"]
    assert "--output-dir periodic-artifacts" in step["run"]
    assert '--github-output "$GITHUB_OUTPUT"' in step["run"]
    assert "workflow_dispatch" not in step["run"]
    assert "submit" not in step["run"].lower()

    assert evidence["if"] == "always()"
    assert evidence["with"]["name"] == (
        "tpu-periodic-${{ matrix.commit }}-${{ matrix.source_run_id }}-"
        "${{ matrix.source_attempt }}-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert evidence["with"]["path"] == "periodic-artifacts/"
    assert evidence["with"]["retention-days"] == "30"

    assert terminal["if"] == "always() && steps.collect.outputs.terminal == 'true'"
    assert terminal["with"]["name"] == (
        "tpu-periodic-terminal-${{ matrix.source_run_id }}-"
        "${{ matrix.source_attempt }}-${{ matrix.commit }}"
    )
    assert terminal["with"]["path"] == "periodic-artifacts/periodic-terminal.json"
    assert terminal["with"]["retention-days"] == "30"
    assert "PyPI" not in str(workflow)
    assert "PYPI_API_TOKEN" not in str(workflow)
    assert "uv publish" not in str(workflow)
    assert "release-tpu.yml" not in str(workflow)
