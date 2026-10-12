"""Contract tests for the release publication workflow graph."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
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


def step_with_id(job: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("id") == step_id)


def checkout_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]


def assert_checkout_precedes_run_steps(job: dict[str, Any]) -> None:
    steps = job["steps"]
    checkout_indexes = [
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    run_indexes = [index for index, step in enumerate(steps) if "run" in step]
    assert checkout_indexes and run_indexes
    assert min(checkout_indexes) < min(run_indexes)


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
        or "UV_PUBLISH_TOKEN" in step.get("env", {})
        for step in rehearsal["steps"]
    )

    freeze = step_with_id(jobs["prepare"], "freeze")
    assert freeze["env"]["RELEASE_COMMIT"] == "${{ github.sha }}"

    steps = rehearsal["steps"]
    binding_recheck = next(
        index
        for index, step in enumerate(steps)
        if step.get("env", {}).get("RELEASE_COMMIT")
        and "GH_TOKEN" in step.get("env", {})
    )
    downloads = [
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(downloads) == 2 and binding_recheck < downloads[1]
    assert all(
        "${{ github.run_attempt }}" in steps[index]["with"]["name"]
        for index in downloads
    )


def test_release_tag_check_is_release_only_and_smoke_uses_frozen_version(
    tmp_path: Path,
) -> None:
    prepare = load_workflow("pypi.yml")["jobs"]["prepare"]
    assert prepare["outputs"]["version"] == "${{ steps.version.outputs.version }}"

    version = step_with_id(prepare, "version")
    assert "GITHUB_OUTPUT" in version["run"]
    tag_check = next(
        step for step in prepare["steps"] if "RELEASE_TAG" in step.get("env", {})
    )
    assert tag_check["if"] == "github.event_name == 'release'"
    assert tag_check["env"]["PACKAGE_VERSION"] == "${{ steps.version.outputs.version }}"

    smoke = next(
        step
        for step in load_workflow("pypi.yml")["jobs"]["build"]["steps"]
        if "PACKAGE_VERSION" in step.get("env", {})
    )
    assert smoke["env"]["PACKAGE_VERSION"] == "${{ needs.prepare.outputs.version }}"

    smoke_lines = smoke["run"].splitlines()
    heredoc_index = next(
        index for index, line in enumerate(smoke_lines) if "<<" in line
    )
    delimiter = smoke_lines[heredoc_index].partition("<<")[2].strip().strip("'\"")
    body_start = heredoc_index + 1
    body_end = next(
        index
        for index in range(body_start, len(smoke_lines))
        if smoke_lines[index] == delimiter
    )
    smoke_script = "\n".join(smoke_lines[body_start:body_end])
    site_packages = tmp_path / "site-packages"
    renderer = site_packages / "renderer"
    renderer.mkdir(parents=True)
    (renderer / "__init__.py").write_text("", encoding="utf-8")
    distribution = site_packages / "jaxrenderer-1.2.3.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: jaxrenderer\nVersion: 1.2.3\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    environment["RELEASE_TAG"] = ""
    for expected, returncode in (("1.2.3", 0), ("9.9.9", 1)):
        environment["PACKAGE_VERSION"] = expected
        result = subprocess.run(
            [sys.executable, "-c", smoke_script],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert result.returncode == returncode, result.stderr
        if returncode:
            assert "AssertionError" in result.stderr


def test_release_checkouts_and_reusable_workflows_use_the_frozen_commit() -> None:
    workflow = load_workflow("pypi.yml")
    release = workflow["jobs"]
    prepare = release["prepare"]
    checks = load_workflow("checks.yml")

    assert release["cpu"]["uses"] == "./.github/workflows/checks.yml"
    assert checks["on"]["workflow_call"]["inputs"]["commit"]["required"] == "true"

    prepare_checkouts = checkout_steps(prepare)
    assert prepare_checkouts
    assert_checkout_precedes_run_steps(prepare)
    for checkout in prepare_checkouts:
        assert checkout["with"]["ref"] == "${{ github.sha }}"
    for job_name in ("build", "publish"):
        assert checkout_steps(release[job_name])[0]["with"]["ref"] == FROZEN_SHA

    for job_name in ("cpu", "gpu", "tpu"):
        assert release[job_name]["with"]["commit"] == FROZEN_SHA

    for workflow_name in ("release-gpu.yml", "release-tpu.yml"):
        child = load_workflow(workflow_name)["jobs"]
        provider_job_name = "gpu" if "gpu" in child else "tpu"
        assert_hard_gates_fail_closed(child, ("prepare", provider_job_name))
        provider_job = child[provider_job_name]
        if provider_job_name == "gpu":
            assert provider_job["environment"] == "modal-gpu"
            controller = step_with_id(provider_job, "provider")
            assert controller["env"] == {
                "MODAL_TOKEN_ID": "${{ secrets.MODAL_TOKEN_ID }}",
                "MODAL_TOKEN_SECRET": "${{ secrets.MODAL_TOKEN_SECRET }}",
            }
        else:
            assert provider_job["environment"] == "kaggle-tpu"
            controller = step_with_id(provider_job, "provider")
            assert controller["env"] == {
                "KAGGLE_API_TOKEN": "${{ secrets.KAGGLE_API_TOKEN }}",
                "KAGGLE_USERNAME": "${{ vars.KAGGLE_USERNAME }}",
            }
        prepare = child["prepare"]
        freeze = step_with_id(prepare, "freeze")
        assert freeze["env"]["RELEASE_COMMIT"] == "${{ inputs.commit }}"
        provider_checkouts = checkout_steps(child[provider_job_name])
        assert provider_checkouts[0]["with"]["ref"] == "${{ github.sha }}"
        uploaded_bindings = [
            step
            for step in prepare["steps"]
            if step.get("uses", "").startswith("actions/upload-artifact@")
        ]
        downloaded_bindings = [
            step
            for step in provider_job["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert len(uploaded_bindings) == len(downloaded_bindings) == 1
        binding_name = uploaded_bindings[0]["with"]["name"]
        assert "${{ github.run_attempt }}" in binding_name
        assert downloaded_bindings[0]["with"]["name"] == binding_name


def test_release_tpu_wait_budgets_fit_the_hosted_job_and_keep_failure_reports() -> None:
    workflow = load_workflow("release-tpu.yml")
    jobs = workflow["jobs"]
    tpu = jobs["tpu"]
    provider = step_with_id(tpu, "provider")

    provider_seconds = int(provider["timeout-minutes"]) * 60
    job_seconds = int(tpu["timeout-minutes"]) * 60
    phase_seconds = (
        kaggle_ci.QUEUE_TIMEOUT_SECONDS + kaggle_ci.EXECUTION_TIMEOUT_SECONDS
    )
    # Leave time for bounded CLI calls, output collection and local validation.
    assert provider_seconds - phase_seconds >= 45 * 60
    # The job also needs time after the provider step for finish and artefact upload.
    assert job_seconds - provider_seconds >= 20 * 60
    assert int(tpu["timeout-minutes"]) < 360  # GitHub-hosted job maximum.
    finish = next(
        step
        for step in tpu["steps"]
        if step.get("if") == "always()" and "PROVIDER_SUCCESS" in step.get("env", {})
    )
    upload = next(
        step
        for step in tpu["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
        and "release-artifacts/tpu/" in step.get("with", {}).get("path", "")
    )
    assert finish["if"] == "always()"
    assert "continue-on-error" not in finish
    assert upload["if"] == "always()"
    assert upload["uses"].startswith("actions/upload-artifact@")


def test_cpu_reuse_requires_frozen_candidate_and_all_cpu_gates() -> None:
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


def test_build_is_published_once_from_the_same_attempt_artifact() -> None:
    jobs = load_workflow("pypi.yml")["jobs"]
    build = jobs["build"]
    publish = jobs["publish"]

    retained = next(
        step
        for step in build["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
        and step.get("with", {}).get("path") == "dist/"
    )
    artifact = retained["with"]["name"]
    assert "${{ github.run_attempt }}" in artifact

    downloaded = next(
        step
        for step in publish["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
        and step.get("with", {}).get("name") == artifact
    )
    assert downloaded["with"]["path"] == "dist/"
    steps = publish["steps"]
    recheck_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("env", {}).get("RELEASE_COMMIT")
        and "GH_TOKEN" in step.get("env", {})
    )
    distribution_download_index = steps.index(downloaded)
    publication = next(
        step for step in steps if "UV_PUBLISH_TOKEN" in step.get("env", {})
    )
    publication_index = steps.index(publication)
    assert recheck_index < distribution_download_index < publication_index


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
    publication = next(
        step for step in publish["steps"] if "UV_PUBLISH_TOKEN" in step.get("env", {})
    )
    assert token_steps == [publication]
    assert publication["env"] == {"UV_PUBLISH_TOKEN": "${{ secrets.PYPI_API_TOKEN }}"}
