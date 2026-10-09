"""Contract tests for the release publication workflow graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools import kaggle_ci
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
    assert publish["if"] == "${{ success() && github.event_name == 'release' }}"
    assert jobs["gpu"]["needs"] == ["prepare", "cpu", "build"]
    assert jobs["tpu"]["needs"] == ["prepare", "cpu", "build"]
    assert jobs["gpu"]["secrets"] == "inherit"
    assert jobs["tpu"]["secrets"] == "inherit"
    assert {
        job_name for job_name, job in jobs.items() if job.get("secrets") == "inherit"
    } == {"gpu", "tpu"}
    assert_hard_gates_fail_closed(jobs, ("prepare", "cpu", "build", "gpu", "tpu"))


def test_manual_rehearsal_shares_gates_and_cannot_publish() -> None:
    workflow = load_workflow("pypi.yml")
    jobs = workflow["jobs"]
    rehearsal = jobs["rehearsal"]

    assert workflow["on"]["workflow_dispatch"] in (None, {}, "")
    assert set(rehearsal["needs"]) == {"prepare", "cpu", "build", "gpu", "tpu"}
    assert (
        rehearsal["if"]
        == "${{ success() && github.event_name == 'workflow_dispatch' }}"
    )
    assert "environment" not in rehearsal
    assert all("PYPI_API_TOKEN" not in str(step) for step in rehearsal["steps"])
    assert not any(
        step.get("uses", "").startswith("actions/upload-artifact@")
        or "uv build" in step.get("run", "")
        or "uv publish" in step.get("run", "")
        for step in rehearsal["steps"]
    )

    gate_jobs = jobs
    assert jobs["prepare"]["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert jobs["prepare"]["steps"][1]["uses"].startswith("actions/setup-python@")
    freeze = step_named(jobs["prepare"], "Freeze and validate the candidate")
    assert freeze["env"]["RELEASE_COMMIT"] == "${{ github.sha }}"
    assert "workflow_dispatch" not in str(freeze["env"])
    for job_name in ("cpu", "build", "gpu", "tpu"):
        if "with" in gate_jobs[job_name]:
            assert gate_jobs[job_name]["with"]["commit"] == FROZEN_SHA

    steps = rehearsal["steps"]
    binding_recheck = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Recheck the frozen workflow attempt"
    )
    downloads = [
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(downloads) == 2 and binding_recheck < downloads[1]
    assert all(
        steps[index]["with"]["name"]
        in (
            "release-binding-${{ github.run_attempt }}",
            "release-dist-${{ github.run_attempt }}",
        )
        for index in downloads
    )
    summary = step_named(rehearsal, "Summarise the validated release rehearsal")
    assert all(
        key in summary["env"]
        for key in ("CANDIDATE_SHA", "PACKAGE_VERSION", "RUN_ATTEMPT")
    )
    assert "$GITHUB_STEP_SUMMARY" in summary["run"]


def test_release_tag_check_is_release_only_and_smoke_uses_frozen_version() -> None:
    prepare = load_workflow("pypi.yml")["jobs"]["prepare"]
    assert prepare["outputs"]["version"] == "${{ steps.version.outputs.version }}"

    version = step_named(prepare, "Read package version")
    assert "GITHUB_OUTPUT" in version["run"]
    tag_check = step_named(prepare, "Check release tag version")
    assert tag_check["if"] == "github.event_name == 'release'"
    assert tag_check["env"]["PACKAGE_VERSION"] == "${{ steps.version.outputs.version }}"

    smoke = step_named(
        load_workflow("pypi.yml")["jobs"]["build"], "Smoke test built wheel"
    )
    assert smoke["env"]["PACKAGE_VERSION"] == "${{ needs.prepare.outputs.version }}"
    assert 'removeprefix("v")' not in smoke["run"]


def test_release_checkouts_and_reusable_workflows_use_the_frozen_commit() -> None:
    workflow = load_workflow("pypi.yml")
    release = workflow["jobs"]
    prepare = release["prepare"]

    assert "lint_base" not in prepare.get("outputs", {})
    assert not any("baseline" in str(step).lower() for step in prepare["steps"])

    for checkout in checkout_steps(release["prepare"]):
        assert checkout["with"]["ref"] == "${{ github.sha }}"
    for job_name in ("build", "publish"):
        assert checkout_steps(release[job_name])[0]["with"]["ref"] == FROZEN_SHA

    for job_name in ("cpu", "gpu", "tpu"):
        assert release[job_name]["with"]["commit"] == FROZEN_SHA
    assert "lint-base" not in release["cpu"].get("with", {})

    for workflow_name, binding_name in (
        ("release-gpu.yml", "gpu-release-binding.json"),
        ("release-tpu.yml", "tpu-release-binding.json"),
    ):
        child = load_workflow(workflow_name)["jobs"]
        provider_job_name = "gpu" if "gpu" in child else "tpu"
        assert_hard_gates_fail_closed(child, ("prepare", provider_job_name))
        provider_job = child[provider_job_name]
        if provider_job_name == "gpu":
            assert provider_job["environment"] == "modal-gpu"
            controller = step_named(
                provider_job, "Confirm the release candidate on a Modal T4"
            )
            assert controller["env"] == {
                "MODAL_TOKEN_ID": "${{ secrets.MODAL_TOKEN_ID }}",
                "MODAL_TOKEN_SECRET": "${{ secrets.MODAL_TOKEN_SECRET }}",
            }
        else:
            assert provider_job["environment"] == "kaggle-tpu"
            controller = step_named(
                provider_job, "Confirm the release candidate on a Kaggle TPU"
            )
            assert controller["env"] == {
                "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
                "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
            }
        prepare = child["prepare"]
        freeze = step_named(
            prepare, "Freeze the release candidate and workflow attempt"
        )
        assert freeze["env"]["RELEASE_COMMIT"] == "${{ inputs.commit }}"
        provider_checkouts = checkout_steps(child[provider_job_name])
        assert provider_checkouts[0]["with"]["ref"] == "${{ github.sha }}"
        assert binding_name in str(child)


def test_release_tpu_wait_budgets_fit_the_hosted_job_and_keep_failure_reports() -> None:
    workflow = load_workflow("release-tpu.yml")
    jobs = workflow["jobs"]
    tpu = jobs["tpu"]
    provider = step_named(tpu, "Confirm the release candidate on a Kaggle TPU")

    assert jobs["prepare"]["timeout-minutes"] == "5"
    assert provider["timeout-minutes"] == "330"
    assert tpu["timeout-minutes"] == "350"
    assert 350 < 360  # GitHub-hosted job maximum.
    provider_seconds = int(provider["timeout-minutes"]) * 60
    job_seconds = int(tpu["timeout-minutes"]) * 60
    phase_seconds = (
        kaggle_ci.QUEUE_TIMEOUT_SECONDS + kaggle_ci.EXECUTION_TIMEOUT_SECONDS
    )
    # Leave time for bounded CLI calls, output collection and local validation.
    assert provider_seconds - phase_seconds >= 45 * 60
    # The job also needs time after the provider step for finish and artefact upload.
    assert job_seconds - provider_seconds >= 20 * 60
    assert ".venv/bin/python -u -m tools.kaggle_ci" in provider["run"]

    finish = step_named(tpu, "Validate the bound TPU result")
    upload = step_named(tpu, "Upload TPU release confirmation artefacts")
    assert finish["if"] == "always()"
    assert finish["run"].startswith("python -m tools.release_tpu_gate finish")
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")


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
        assert refs and all(ref.startswith("${{ inputs.commit || ") for ref in refs)

    lint = checks["lint"]
    assert len(checkout_steps(lint)) == 1
    assert (
        "lint-base" not in load_workflow("checks.yml")["on"]["workflow_call"]["inputs"]
    )
    assert step_named(lint, "Check Ruff lint")["run"].startswith(
        "uv run --no-sync --python 3.14 ruff check --no-respect-gitignore "
    )

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
