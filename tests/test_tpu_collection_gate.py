"""Validate original TPU evidence independently of a new collection attempt."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, cast
import zipfile

import pytest
from tools import tpu_collection_gate as gate

HEAD = "a" * 40
COLLECTOR_HEAD = "b" * 40
SOURCE_RUN = 789
SOURCE_ATTEMPT = 3


def binding() -> dict[str, object]:
    return {
        "kind": "release",
        "head_sha": HEAD,
        "head_repository": gate.REPOSITORY,
        "run_id": "789.3",
    }


def state() -> dict[str, object]:
    value: dict[str, object] = {
        field: None for field in gate.kaggle_ci.CONTROLLER_STATE_FIELDS
    }
    value.update(
        binding=binding(),
        kernel_id="joeyteng/jaxr-original-kernel",
        submitted_version=1,
        status="queued",
        outcome="queue_timeout",
    )
    return value


def legacy_state(version: object = 1) -> dict[str, object]:
    identity = json.dumps(binding(), sort_keys=True, separators=(",", ":"))
    run_tag = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return {
        "binding": binding(),
        "kernel_id": f"joeyteng/jaxr-{run_tag}-0123456789",
        "submitted_version": version,
    }


def result() -> dict[str, object]:
    return {
        **binding(),
        "backend": "tpu",
        "device_backend": "tpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
        "versions": {"jax": "0.11.2", "libtpu": "0.0.48"},
    }


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for key, value in {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/master",
        "GITHUB_REPOSITORY": gate.REPOSITORY,
        "GITHUB_SHA": COLLECTOR_HEAD,
        "GITHUB_RUN_ID": "990",
        "GITHUB_RUN_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(key, value)
    run = {
        "id": SOURCE_RUN,
        "run_attempt": SOURCE_ATTEMPT,
        "head_sha": HEAD,
        "path": gate.SOURCE_WORKFLOW,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "failure",
        "repository": {"full_name": gate.REPOSITORY},
        "head_repository": {"full_name": gate.REPOSITORY},
    }
    jobs = [
        {
            "name": name,
            "run_id": SOURCE_RUN,
            "status": "completed",
            "conclusion": "success",
        }
        for name in sorted(gate.REQUIRED_JOBS)
    ] + [
        {
            "name": "tpu / tpu",
            "run_id": SOURCE_RUN,
            "status": "completed",
            "conclusion": "failure",
        },
        {
            "name": "publish",
            "run_id": SOURCE_RUN,
            "status": "completed",
            "conclusion": "skipped",
        },
    ]
    artifacts = [
        {
            "id": index,
            "name": name,
            "expired": False,
            "size_in_bytes": 1024,
            "workflow_run": {"id": SOURCE_RUN, "head_sha": HEAD},
        }
        for index, name in enumerate(
            ("tpu-release-binding-3", f"release-tpu-{HEAD}-3"), 1
        )
    ]
    data: dict[str, Any] = {
        "run": run,
        "jobs": {"total_count": len(jobs), "jobs": jobs},
        "artifacts": {"total_count": 2, "artifacts": artifacts},
        "controller_state": state(),
    }

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if route.endswith("/attempts/3"):
            return data["run"]
        if route.endswith("/attempts/3/jobs?per_page=100"):
            return data["jobs"]
        if route.endswith("/artifacts?per_page=100"):
            return data["artifacts"]
        if route.endswith(f"/compare/{HEAD}...master"):
            return {"status": "ahead", "base_commit": {"sha": HEAD}}
        raise AssertionError(route)

    def download(artifact_id: int, member: str, path: Path) -> None:
        assert artifact_id in {1, 2}
        value = (
            binding()
            if member == "tpu-release-binding.json"
            else data["controller_state"]
        )
        path.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(gate, "_download_member", download)
    return data


def prepare(tmp_path: Path) -> Path:
    path = tmp_path / "source"
    gate.prepare("789", "3", HEAD, path, tmp_path / "github-output")
    return path


def test_schedule_context_requires_the_exact_canonical_master_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        f"{gate.REPOSITORY}/.github/workflows/collect-tpu-periodic.yml@refs/heads/master",
    )
    monkeypatch.setenv("GITHUB_SHA", COLLECTOR_HEAD)
    monkeypatch.setenv("GITHUB_RUN_ID", "990")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")

    assert gate._context() == "990.1"

    for name, value in (
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_REPOSITORY", "someone/jaxrenderer"),
        (
            "GITHUB_WORKFLOW_REF",
            f"{gate.REPOSITORY}/.github/workflows/release-tpu.yml@refs/heads/master",
        ),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(gate.CollectionGateError, match="scheduled collection"):
            gate._context()
        monkeypatch.setenv(
            name,
            value
            if name == "GITHUB_WORKFLOW_REF"
            else (
                f"{gate.REPOSITORY}/.github/workflows/collect-tpu-periodic.yml@refs/heads/master"
                if name == "GITHUB_WORKFLOW_REF"
                else "refs/heads/master"
                if name == "GITHUB_REF"
                else gate.REPOSITORY
            ),
        )


def test_manual_context_remains_supported_without_workflow_ref(
    source: dict[str, Any],
) -> None:
    assert gate._context() == "990.1"


def collected(tmp_path: Path) -> Path:
    output = tmp_path / "collected"
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result()), encoding="utf-8")
    (output / "kaggle-collection-report.json").write_text(
        json.dumps(
            {
                "binding": binding(),
                "kernel_id": state()["kernel_id"],
                "submitted_version": 1,
                "requested_version": 1,
                "success": True,
                "outcome": "success",
                "remote_status": "complete",
                "version_verified": True,
            }
        ),
        encoding="utf-8",
    )
    return output


def test_prepare_retains_original_identity(
    source: dict[str, Any], tmp_path: Path
) -> None:
    path = prepare(tmp_path)
    saved_state = json.loads((path / "kaggle-controller-state.json").read_text())
    saved = json.loads((path / "collection-source.json").read_text())
    assert saved["collector_run_id"] == "990.1"
    assert saved["collector_sha"] == COLLECTOR_HEAD
    assert saved["source_run_id"] == SOURCE_RUN
    assert json.loads((path / "binding.json").read_text()) == binding()
    assert saved_state["outcome"] == "queue_timeout"
    assert saved_state["status"] == "queued"
    assert (tmp_path / "github-output").read_text().splitlines() == [
        f"head_sha={HEAD}",
        "source_run_id=789",
        "source_attempt=3",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", 790),
        ("run_attempt", 4),
        ("head_sha", COLLECTOR_HEAD),
        ("path", ".github/workflows/release-tpu.yml"),
        ("event", "pull_request"),
        ("status", "in_progress"),
        ("conclusion", "success"),
        ("repository", {"full_name": "someone/jaxrenderer"}),
        ("head_repository", {"full_name": "someone/jaxrenderer"}),
    ],
)
def test_prepare_rejects_wrong_run(
    source: dict[str, Any], tmp_path: Path, field: str, value: object
) -> None:
    source["run"][field] = value
    with pytest.raises(gate.CollectionGateError, match="exact completed"):
        prepare(tmp_path)


@pytest.mark.parametrize(
    "name", ["build", "cpu / check (3.14)", "gpu / gpu", "tpu / prepare"]
)
def test_prepare_rejects_other_gate_failure(
    source: dict[str, Any], tmp_path: Path, name: str
) -> None:
    next(job for job in source["jobs"]["jobs"] if job["name"] == name)["conclusion"] = (
        "failure"
    )
    with pytest.raises(gate.CollectionGateError, match="required job"):
        prepare(tmp_path)


def test_prepare_rejects_incomplete_job_inventory(
    source: dict[str, Any], tmp_path: Path
) -> None:
    source["jobs"]["total_count"] += 1
    with pytest.raises(gate.CollectionGateError, match="inventory"):
        prepare(tmp_path)


def test_prepare_rejects_duplicate_jobs(source: dict[str, Any], tmp_path: Path) -> None:
    source["jobs"]["jobs"].append(copy.deepcopy(source["jobs"]["jobs"][0]))
    source["jobs"]["total_count"] += 1
    with pytest.raises(gate.CollectionGateError, match="ambiguous"):
        prepare(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expired", True),
        ("size_in_bytes", gate.MAX_ARCHIVE_BYTES + 1),
        ("workflow_run", {"id": SOURCE_RUN, "head_sha": COLLECTOR_HEAD}),
    ],
)
def test_prepare_rejects_invalid_artifact(
    source: dict[str, Any], tmp_path: Path, field: str, value: object
) -> None:
    source["artifacts"]["artifacts"][0][field] = value
    with pytest.raises(gate.CollectionGateError, match="artefact identity"):
        prepare(tmp_path)


def test_prepare_rejects_duplicate_artifacts(
    source: dict[str, Any], tmp_path: Path
) -> None:
    source["artifacts"]["artifacts"].append(
        copy.deepcopy(source["artifacts"]["artifacts"][0])
    )
    source["artifacts"]["total_count"] += 1
    with pytest.raises(gate.CollectionGateError, match="exactly one"):
        prepare(tmp_path)


@pytest.mark.parametrize(
    "outcome",
    [
        "preflight_error",
        "submission_error_unknown",
        "submission_version_unknown",
        "cli_error",
        "remote_terminal_failure",
        "result_validation_error",
    ],
)
def test_prepare_classifies_known_non_timeout_controller_results_as_ineligible(
    source: dict[str, Any], tmp_path: Path, outcome: str
) -> None:
    source["controller_state"]["outcome"] = outcome
    with pytest.raises(gate.IneligibleSource, match="not a timeout"):
        prepare(tmp_path)


def test_preflight_without_submitted_version_is_ineligible(
    source: dict[str, Any], tmp_path: Path
) -> None:
    source["controller_state"].update(outcome="preflight_error", submitted_version=None)

    with pytest.raises(gate.IneligibleSource, match="not a timeout"):
        prepare(tmp_path)


@pytest.mark.parametrize("version", [None, 1])
def test_prepare_skips_valid_legacy_state_without_timeout_evidence(
    source: dict[str, Any], tmp_path: Path, version: object
) -> None:
    source["controller_state"] = legacy_state(version)

    with pytest.raises(
        gate.IneligibleSource,
        match=r"source 789\.3 legacy controller state has no timeout evidence",
    ):
        prepare(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "binding",
            {**binding(), "run_id": "789.4"},
            "source 789.3 legacy controller binding is invalid",
        ),
        (
            "kernel_id",
            "joeyteng/jaxr-wrong-0123456789",
            "source 789.3 legacy controller identity is invalid",
        ),
        (
            "submitted_version",
            True,
            "source 789.3 legacy controller version is invalid",
        ),
        ("submitted_version", 0, "source 789.3 legacy controller version is invalid"),
        ("submitted_version", 2, "source 789.3 legacy controller version is invalid"),
        (
            "submitted_version",
            "1",
            "source 789.3 legacy controller version is invalid",
        ),
    ],
)
def test_prepare_rejects_legacy_state_with_wrong_identity(
    source: dict[str, Any],
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    legacy = legacy_state()
    legacy[field] = value
    source["controller_state"] = legacy

    with pytest.raises(gate.CollectionGateError, match=message):
        prepare(tmp_path)


@pytest.mark.parametrize("owner", ["", "bad user", "u" * 51, "joé"])
def test_prepare_rejects_malformed_legacy_kernel_owner(
    source: dict[str, Any], tmp_path: Path, owner: str
) -> None:
    legacy = legacy_state()
    valid_slug = str(legacy["kernel_id"]).split("/", maxsplit=1)[1]
    legacy["kernel_id"] = f"{owner}/{valid_slug}"
    source["controller_state"] = legacy

    with pytest.raises(
        gate.CollectionGateError,
        match="source 789.3 legacy controller identity is invalid",
    ) as error:
        prepare(tmp_path)

    assert not isinstance(error.value, gate.IneligibleSource)


def test_prepare_rejects_unknown_controller_schema_with_source_identity(
    source: dict[str, Any], tmp_path: Path
) -> None:
    source["controller_state"]["unrecognised"] = "value"

    with pytest.raises(
        gate.CollectionGateError,
        match=r"source 789\.3 controller state schema is unsupported",
    ):
        prepare(tmp_path)


def test_finish_separates_collection_from_release_credit(
    source: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "step-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    original, output = prepare(tmp_path), collected(tmp_path)
    receipt = gate.finish(original, output, True)
    assert receipt["binding"] == binding()
    assert receipt["collector_run_id"] == "990.1"
    assert receipt["validation_only"] is True
    assert receipt["release_gate_credit"] is False
    assert receipt["source_conclusion"] == "failure"
    assert f"Original commit: `{HEAD}`" in summary.read_text(encoding="utf-8")
    assert (
        gate.release_tpu_gate.finish(original / "binding.json", output, True)["state"]
        == "failure"
    )


@pytest.mark.parametrize("timeout_outcome", ["queue_timeout", "execution_timeout"])
@pytest.mark.parametrize("remote_status", sorted(gate.kaggle_ci.TERMINAL_SUCCESS))
def test_finish_accepts_terminal_success_after_timeout_as_validation_only(
    source: dict[str, Any],
    tmp_path: Path,
    timeout_outcome: str,
    remote_status: str,
) -> None:
    source["controller_state"].update(
        outcome=timeout_outcome,
        status=remote_status,
    )
    original, output = prepare(tmp_path), collected(tmp_path)

    receipt = gate.finish(original, output, True)

    assert receipt["success"] is True
    assert receipt["validation_only"] is True
    assert receipt["release_gate_credit"] is False
    with pytest.raises(gate.CollectionGateError, match="provider did not collect"):
        gate.finish(original, collected(tmp_path / "provider-failed"), False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("success", False),
        ("binding", {**binding(), "run_id": "990.1"}),
        ("kernel_id", "someone/another-kernel"),
        ("submitted_version", 2),
        ("requested_version", True),
        ("version_verified", False),
        ("remote_status", "queued"),
        ("outcome", "pending"),
    ],
)
def test_finish_rejects_unproven_result(
    source: dict[str, Any], tmp_path: Path, field: str, value: object
) -> None:
    original, output = prepare(tmp_path), collected(tmp_path)
    report = output / "kaggle-collection-report.json"
    value_dict = json.loads(report.read_text())
    value_dict[field] = value
    report.write_text(json.dumps(value_dict), encoding="utf-8")
    with pytest.raises(gate.CollectionGateError, match="original successful"):
        gate.finish(original, output, True)
    assert not (output / "collection-receipt.json").exists()


def test_finish_rechecks_original_artifact_ids(
    source: dict[str, Any], tmp_path: Path
) -> None:
    original, output = prepare(tmp_path), collected(tmp_path)
    source["artifacts"]["artifacts"][0]["id"] = 101
    with pytest.raises(gate.CollectionGateError, match="identity changed"):
        gate.finish(original, output, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submitted_version", True),
        ("outcome", "success"),
        ("status", "failed"),
        ("binding", {**binding(), "run_id": "789.4"}),
    ],
)
def test_finish_rejects_invalid_controller_state(
    source: dict[str, Any], tmp_path: Path, field: str, value: object
) -> None:
    original, output = prepare(tmp_path), collected(tmp_path)
    path = original / "kaggle-controller-state.json"
    value_dict = json.loads(path.read_text())
    value_dict[field] = value
    path.write_text(json.dumps(value_dict), encoding="utf-8")
    with pytest.raises(gate.CollectionGateError):
        gate.finish(original, output, True)


def test_download_selects_only_bounded_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GH_TOKEN", "fixture")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert command[-1].endswith("/actions/artifacts/1/zip")
        with zipfile.ZipFile(kwargs["stdout"], "w") as archive:
            archive.writestr("binding.json", "{}")
            archive.writestr("../unrelated", "not extracted")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gate.subprocess, "run", run)
    path = tmp_path / "binding.json"
    download_member = cast(
        Callable[[int, str, Path], None], getattr(gate, "_download_member")
    )
    download_member(1, "binding.json", path)
    assert path.read_text() == "{}"
    assert not path.with_suffix(".zip").exists()
    assert list(tmp_path.iterdir()) == [path]
