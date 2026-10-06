"""Contract tests for the release publication workflow graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
FROZEN_SHA = "${{ needs.prepare.outputs.head_sha }}"


def load_workflow(name: str) -> dict[str, Any]:
    """Load GitHub Actions YAML without YAML 1.1 boolean coercion."""
    value = yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert isinstance(value, dict)
    return value


def step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def checkout_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]


def assert_hard_gates_fail_closed(
    jobs: dict[str, dict[str, Any]], job_names: tuple[str, ...]
) -> None:
    for job_name in job_names:
        job = jobs[job_name]
        assert "continue-on-error" not in job
        assert "if" not in job


def test_publish_waits_for_every_release_gate_and_cannot_skip_failed_needs() -> None:
    workflow = load_workflow("pypi.yml")
    jobs = workflow["jobs"]
    publish = jobs["publish"]

    assert set(publish["needs"]) == {"prepare", "cpu", "build", "gpu", "tpu"}
    # GitHub's implicit success() condition blocks failed, skipped, or cancelled
    # dependencies. An explicit condition is safe only when it preserves that gate.
    assert publish.get("if") in (None, "${{ success() }}")
    assert jobs["gpu"]["needs"] == ["prepare", "cpu", "build"]
    assert jobs["tpu"]["needs"] == ["prepare", "cpu", "build"]
    assert_hard_gates_fail_closed(
        jobs, ("prepare", "cpu", "build", "gpu", "tpu", "publish")
    )


def test_release_checkouts_and_reusable_workflows_use_the_frozen_commit() -> None:
    release = load_workflow("pypi.yml")["jobs"]

    for checkout in checkout_steps(release["prepare"]):
        assert checkout["with"]["ref"] == "${{ github.sha }}"
    for job_name in ("build", "publish"):
        assert checkout_steps(release[job_name])[0]["with"]["ref"] == FROZEN_SHA

    for job_name in ("cpu", "gpu", "tpu"):
        assert release[job_name]["with"]["commit"] == FROZEN_SHA
    assert (
        release["cpu"]["with"]["lint-base"] == "${{ needs.prepare.outputs.lint_base }}"
    )

    for workflow_name, binding_name in (
        ("release-gpu.yml", "gpu-release-binding.json"),
        ("release-tpu.yml", "tpu-release-binding.json"),
    ):
        child = load_workflow(workflow_name)["jobs"]
        provider_job_name = "gpu" if "gpu" in child else "tpu"
        assert_hard_gates_fail_closed(child, ("prepare", provider_job_name))
        prepare = child["prepare"]
        freeze = step_named(
            prepare, "Freeze the release candidate and workflow attempt"
        )
        assert freeze["env"]["RELEASE_COMMIT"] == "${{ inputs.commit }}"
        provider_checkouts = checkout_steps(child[provider_job_name])
        assert provider_checkouts[0]["with"]["ref"] == "${{ github.sha }}"
        assert binding_name in str(child)


def test_cpu_reuse_includes_the_full_matrix_render_and_minimum_numpy_gates() -> None:
    checks = load_workflow("checks.yml")["jobs"]
    assert_hard_gates_fail_closed(
        checks,
        (
            "lint",
            "check",
            "macos-render-regression",
            "linux-render-regression",
            "numpy-minimum",
        ),
    )

    for job_name, job in checks.items():
        refs = [step["with"]["ref"] for step in checkout_steps(job)]
        if job_name == "lint":
            assert refs[0].startswith("${{ inputs.commit || ")
            assert refs[1].startswith("${{ inputs.lint-base || ")
        else:
            assert refs and all(ref.startswith("${{ inputs.commit || ") for ref in refs)

    matrix = checks["check"]
    assert matrix["strategy"]["matrix"]["python-version"] == ["3.12", "3.13", "3.14"]
    assert any("pytest tests/" in step.get("run", "") for step in matrix["steps"])

    for job_name in ("macos-render-regression", "linux-render-regression"):
        runs = "\n".join(step.get("run", "") for step in checks[job_name]["steps"])
        assert "tests/render_regression.py tests/test_smoke_grad.py" in runs

    minimum = checks["numpy-minimum"]
    runs = "\n".join(step.get("run", "") for step in minimum["steps"])
    assert '"numpy==2.1.3"' in runs
    assert "pytest tests/ --import-mode importlib" in runs
    assert "tests/render_regression.py tests/test_smoke_grad.py" in runs


def test_build_is_published_once_from_the_same_attempt_artifact() -> None:
    jobs = load_workflow("pypi.yml")["jobs"]
    build = jobs["build"]
    publish = jobs["publish"]

    assert sum("uv build" in step.get("run", "") for step in build["steps"]) == 1
    retained = next(
        step
        for step in build["steps"]
        if step.get("name") == "Retain the validated distributions"
    )
    artifact = "release-dist-${{ github.run_attempt }}"
    assert retained["with"]["name"] == artifact

    downloaded = next(
        step
        for step in publish["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
        and step.get("with", {}).get("name") == artifact
    )
    assert downloaded["with"]["path"] == "dist/"
    assert not any("uv build" in step.get("run", "") for step in publish["steps"])

    steps = publish["steps"]
    recheck_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Recheck the tag and frozen workflow attempt"
    )
    distribution_download_index = steps.index(downloaded)
    publication_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Publish the validated distributions"
    )
    assert recheck_index < distribution_download_index < publication_index
    assert (
        step_named(publish, "Publish the validated distributions")["run"]
        == "uv publish"
    )


def test_pypi_token_exists_only_in_the_publication_step() -> None:
    workflow = load_workflow("pypi.yml")
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "PyPI"

    token_steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "PYPI_API_TOKEN" in str(step)
    ]
    assert all(
        "PYPI_API_TOKEN" not in str(job.get("env", {}))
        for job in workflow["jobs"].values()
    )
    publication = step_named(publish, "Publish the validated distributions")
    assert token_steps == [publication]
    assert publication["env"] == {"UV_PUBLISH_TOKEN": "${{ secrets.PYPI_API_TOKEN }}"}
