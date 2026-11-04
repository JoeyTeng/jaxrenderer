"""Behavioural tests for completing a release from a saved TPU result."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, cast
import zipfile

import pytest
from tools import accelerator_gate, release_candidate, release_tpu_gate
from tools import release_completion as completion
from tools import tpu_collection_gate as collection_gate
from tools import tpu_collection_periodic as periodic

SOURCE_SHA = "a" * 40
COLLECTOR_SHA = "b" * 40
COMPLETION_SHA = "c" * 40
TAG_SHA = "d" * 40
SOURCE_RUN = "789"
SOURCE_ATTEMPT = "3"
COLLECTOR_RUN = "990"
COLLECTOR_ATTEMPT = "2"
COMPLETION_RUN = "1001"
VERSION = "1.2.3"
TAG = f"v{VERSION}"
SOURCE_IDENTITY = f"{SOURCE_RUN}.{SOURCE_ATTEMPT}"
COLLECTOR_IDENTITY = f"{COLLECTOR_RUN}.{COLLECTOR_ATTEMPT}"


def _candidate(*, mode: str = "release", version: str = VERSION) -> dict[str, object]:
    recovery = mode in {"release", "recovery"}
    return {
        "mode": mode,
        "source_commit": SOURCE_SHA,
        "head_sha": SOURCE_SHA,
        "version": version,
        "tag": TAG if recovery else None,
        "tag_ref": f"refs/tags/{TAG}" if recovery else None,
        "tag_object_sha": TAG_SHA if recovery else None,
        "release_id": 42 if recovery else None,
        "controller_sha": SOURCE_SHA,
        "repository": release_candidate.REPOSITORY,
        "run_id": SOURCE_IDENTITY,
    }


def _write_distribution_archive(
    path: Path,
    *,
    version: str = VERSION,
    wheel_version: str | None = None,
    sdist_version: str | None = None,
) -> dict[str, bytes]:
    wheel_metadata_version = wheel_version or version
    sdist_metadata_version = sdist_version or version
    wheel_name = f"jaxrenderer-{version}-py3-none-any.whl"
    sdist_name = f"jaxrenderer-{version}.tar.gz"
    wheel_buffer = io.BytesIO()
    with zipfile.ZipFile(wheel_buffer, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr(
            f"jaxrenderer-{wheel_metadata_version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: jaxrenderer\nVersion: {wheel_metadata_version}\n\n",
        )
    sdist_buffer = io.BytesIO()
    with tarfile.open(fileobj=sdist_buffer, mode="w:gz") as sdist:
        metadata = (
            f"Metadata-Version: 2.1\nName: jaxrenderer\nVersion: {sdist_metadata_version}\n\n"
        ).encode()
        info = tarfile.TarInfo(f"jaxrenderer-{version}/PKG-INFO")
        info.size = len(metadata)
        sdist.addfile(info, io.BytesIO(metadata))
    distributions = {
        wheel_name: wheel_buffer.getvalue(),
        sdist_name: sdist_buffer.getvalue(),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in distributions.items():
            archive.writestr(name, content)
    return distributions


def _terminal_record(
    binding: dict[str, object], state: dict[str, object], *, outcome: str = "success"
) -> dict[str, object]:
    return {
        "terminal": True,
        "outcome": outcome,
        "source_run_id": SOURCE_RUN,
        "source_attempt": SOURCE_ATTEMPT,
        "head_sha": SOURCE_SHA,
        "collector_run_id": COLLECTOR_IDENTITY,
        "collector_attempt": COLLECTOR_ATTEMPT,
        "collector_sha": COLLECTOR_SHA,
        "binding": binding,
        "kernel_id": state["kernel_id"],
        "submitted_version": 1,
        "requested_version": 1,
        "version_verified": True,
        "remote_status": "complete",
    }


@pytest.fixture
def completion_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    candidate = _candidate()
    candidate_bytes = json.dumps(candidate, sort_keys=True).encode() + b"\n"
    source: dict[str, object] = {
        "source_run_id": int(SOURCE_RUN),
        "source_attempt": int(SOURCE_ATTEMPT),
        "head_sha": SOURCE_SHA,
        "collector_run_id": COLLECTOR_IDENTITY,
        "collector_sha": COLLECTOR_SHA,
        "binding_artifact_id": 101,
        "controller_artifact_id": 102,
        "candidate_artifact_id": 103,
        "candidate_artifact_name": f"candidate-binding-{SOURCE_ATTEMPT}",
        "marker_artifact_id": 104,
        "marker_artifact_name": f"release-pending-{SOURCE_ATTEMPT}",
        "_source_conclusion": "success",
    }
    tpu_binding: dict[str, object] = {
        "head_sha": SOURCE_SHA,
        "run_id": SOURCE_IDENTITY,
    }
    state: dict[str, object] = {"kernel_id": "joey/jaxrenderer-test-kernel"}

    collection = tmp_path / "collection"
    source_dir = collection / "source"
    source_dir.mkdir(parents=True)
    _ = (source_dir / "candidate-binding.json").write_bytes(candidate_bytes)
    _ = (source_dir / "binding.json").write_text('{"kind":"release"}\n')
    _ = (source_dir / "kaggle-controller-state.json").write_text(
        json.dumps(state) + "\n"
    )
    _ = (source_dir / "collection-source.json").write_text(
        json.dumps(
            {
                "source_run_id": int(SOURCE_RUN),
                "source_attempt": int(SOURCE_ATTEMPT),
                "head_sha": SOURCE_SHA,
                "collector_run_id": COLLECTOR_IDENTITY,
                "collector_sha": COLLECTOR_SHA,
            }
        )
        + "\n"
    )
    report = {"terminal": True, "outcome": "success"}
    _ = (collection / "periodic-collection.json").write_text(json.dumps(report) + "\n")
    terminal_record = _terminal_record(tpu_binding, state)
    _ = (collection / "periodic-terminal.json").write_text(
        json.dumps(terminal_record) + "\n"
    )
    _ = (collection / "collected").mkdir()

    dist_archive = tmp_path / "release-dist.zip"
    distributions = _write_distribution_archive(dist_archive)
    archive_digest = "sha256:" + hashlib.sha256(dist_archive.read_bytes()).hexdigest()
    collection_name = (
        f"tpu-periodic-{SOURCE_SHA}-{SOURCE_RUN}-{SOURCE_ATTEMPT}-"
        f"{COLLECTOR_RUN}-{COLLECTOR_ATTEMPT}"
    )
    terminal_name = f"tpu-periodic-terminal-{SOURCE_RUN}-{SOURCE_ATTEMPT}-{SOURCE_SHA}"
    terminal_artifact: dict[str, object] = {
        "id": 107,
        "name": terminal_name,
        "digest": "sha256:" + "7" * 64,
        "size_in_bytes": 256,
        "expired": False,
        "workflow_run": {
            "id": int(COLLECTOR_RUN),
            "head_sha": COLLECTOR_SHA,
        },
    }
    terminal_state = {"artifact_present": True, "upload_proven": True}
    names = {
        f"candidate-binding-{SOURCE_ATTEMPT}": 103,
        f"release-pending-{SOURCE_ATTEMPT}": 104,
        collection_name: 105,
        f"release-dist-{SOURCE_ATTEMPT}": 106,
    }
    metadata = {
        name: {
            "id": artifact_id,
            "name": name,
            "digest": archive_digest
            if name == f"release-dist-{SOURCE_ATTEMPT}"
            else "sha256:" + f"{artifact_id:064x}",
            "size_in_bytes": dist_archive.stat().st_size
            if name == f"release-dist-{SOURCE_ATTEMPT}"
            else 123,
            "expired": False,
        }
        for name, artifact_id in names.items()
    }

    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_REPOSITORY", periodic.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", periodic.WORKFLOW_REF)
    monkeypatch.setenv("GITHUB_RUN_ID", COLLECTOR_RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", COLLECTOR_ATTEMPT)
    monkeypatch.setenv("GITHUB_SHA", COLLECTOR_SHA)
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setattr(periodic, "_require_context", lambda: COLLECTOR_IDENTITY)

    def no_master_history_check(_sha: str) -> None:
        return None

    def revalidate_source(
        *_args: object, **_kwargs: object
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        return source, tpu_binding, state

    def finish_collection(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "binding": tpu_binding,
            "collector_run_id": COLLECTOR_IDENTITY,
            "collector_sha": COLLECTOR_SHA,
            "kernel_id": state["kernel_id"],
            "submitted_version": 1,
            "success": True,
            "validation_only": True,
            "release_gate_credit": False,
            "source_conclusion": source["_source_conclusion"],
        }

    def artifact_metadata(
        _run_id: int, name: str, _commit: str, **_kwargs: object
    ) -> dict[str, object]:
        return cast(dict[str, object], metadata[name])

    monkeypatch.setattr(
        release_tpu_gate, "verify_master_history", no_master_history_check
    )
    monkeypatch.setattr(collection_gate, "revalidate", revalidate_source)
    monkeypatch.setattr(collection_gate, "finish", finish_collection)
    monkeypatch.setattr(collection_gate, "_artifact_metadata", artifact_metadata)

    def verified_terminal_artifacts(
        source_run_id: str,
        source_attempt: str,
        commit: str,
        binding: dict[str, object],
        kernel_id: str,
    ) -> list[tuple[dict[str, object], dict[str, object]]]:
        assert source_run_id == SOURCE_RUN
        assert source_attempt == SOURCE_ATTEMPT
        assert commit == SOURCE_SHA
        assert binding == tpu_binding
        assert kernel_id == state["kernel_id"]
        if terminal_state["artifact_present"] and terminal_state["upload_proven"]:
            return [(terminal_artifact, terminal_record)]
        return []

    monkeypatch.setattr(
        periodic, "verified_terminal_artifacts", verified_terminal_artifacts
    )

    def download_member(_artifact_id: int, member: str, destination: Path) -> None:
        assert member == "candidate-binding.json"
        _ = destination.write_bytes(candidate_bytes)

    monkeypatch.setattr(collection_gate, "download_member", download_member)

    def verify_frozen(_path: Path) -> release_candidate.CandidateBinding:
        return release_candidate.validate_binding(candidate)

    monkeypatch.setattr(release_candidate, "verify_frozen", verify_frozen)

    def download_archive(_artifact_id: int, destination: Path, _limit: int) -> None:
        _ = destination.write_bytes(dist_archive.read_bytes())

    monkeypatch.setattr(completion, "_download_archive", download_archive)

    return {
        "candidate": candidate,
        "candidate_bytes": candidate_bytes,
        "source": source,
        "binding": tpu_binding,
        "state": state,
        "collection": collection,
        "dist_archive": dist_archive,
        "distributions": distributions,
        "metadata": metadata,
        "output": tmp_path / "completion",
        "github_output": tmp_path / "github-output",
        "source_sha": SOURCE_SHA,
        "collection_name": collection_name,
        "terminal_artifact": terminal_artifact,
        "terminal_name": terminal_name,
        "terminal_state": terminal_state,
    }


def _prepare(case: dict[str, Any]) -> dict[str, object]:
    return completion.prepare(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        case["collection"],
        case["output"],
        case["github_output"],
    )


def _write_original_distributions(case: dict[str, Any], destination: Path) -> None:
    destination.mkdir()
    for name, content in case["distributions"].items():
        _ = (destination / name).write_bytes(content)


def test_prepare_binds_terminal_success_and_original_distributions(
    completion_case: dict[str, Any],
) -> None:
    case = completion_case
    binding = _prepare(case)

    assert binding["publish"] is True
    assert binding["source_run_id"] == int(SOURCE_RUN)
    assert binding["collector_run_id"] == int(COLLECTOR_RUN)
    assert binding["release_dist_artifact_id"] == 106
    assert binding["terminal_marker_artifact_id"] == 107
    assert binding["terminal_marker_artifact_name"] == case["terminal_name"]
    assert (
        binding["terminal_marker_artifact_digest"]
        == case["terminal_artifact"]["digest"]
    )
    assert binding["terminal_marker_artifact_size"] == 256
    assert (case["output"] / "completion-binding.json").is_file()
    assert (case["output"] / "candidate-binding.json").read_bytes() == case[
        "candidate_bytes"
    ]
    assert (case["output"] / "collection" / "periodic-terminal.json").is_file()
    assert not (case["output"] / "dist").exists()
    assert case["github_output"].read_text().splitlines()[:2] == [
        "ready=true",
        "publish=true",
    ]

    dist = case["output"].parent / "dist"
    _write_original_distributions(case, dist)
    assert completion.verify(case["output"], dist) == binding


@pytest.mark.parametrize("upload_state", ["missing", "failed"])
def test_prepare_rejects_unconfirmed_native_terminal_marker_upload(
    completion_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    upload_state: str,
) -> None:
    case = completion_case
    if upload_state == "missing":
        case["terminal_state"]["artifact_present"] = False
    else:
        case["terminal_state"]["upload_proven"] = False

    def unexpected_distribution_download(*_args: object, **_kwargs: object) -> None:
        pytest.fail("unconfirmed terminal markers must not retrieve distributions")

    monkeypatch.setattr(
        completion, "_download_archive", unexpected_distribution_download
    )

    result = _prepare(case)

    assert result == {
        "ready": False,
        "publish": False,
        "outcome": "terminal_marker_unconfirmed",
    }
    assert case["github_output"].read_text().splitlines()[0] == "ready=false"
    assert not case["output"].exists()


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("id", 777),
        ("digest", "sha256:" + "8" * 64),
        ("size_in_bytes", 257),
    ],
)
def test_verify_rejects_terminal_marker_identity_drift_after_approval(
    completion_case: dict[str, Any],
    field: str,
    changed: object,
) -> None:
    case = completion_case
    _ = _prepare(case)
    dist = case["output"].parent / "dist"
    _write_original_distributions(case, dist)
    case["terminal_artifact"][field] = changed

    with pytest.raises(
        completion.CompletionError, match="terminal marker artefact identity changed"
    ):
        _ = completion.verify(case["output"], dist)


@pytest.mark.parametrize("outcome", ["pending", "remote_failure"])
def test_pending_or_remote_failure_is_not_ready_and_never_downloads_distributions(
    completion_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    case = completion_case
    _ = (case["collection"] / "periodic-collection.json").write_text(
        json.dumps({"terminal": outcome == "remote_failure", "outcome": outcome}) + "\n"
    )

    def unexpected_download(*_args: object) -> None:
        pytest.fail("pending or failed TPU results must not retrieve distributions")

    monkeypatch.setattr(completion, "_download_archive", unexpected_download)
    result = _prepare(case)

    assert result == {"ready": False, "publish": False, "outcome": outcome}
    assert case["github_output"].read_text().splitlines()[0] == "ready=false"
    assert not case["output"].exists()


def test_legacy_validation_only_success_is_not_ready_or_publishable(
    completion_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completion_case
    source = case["source"]
    for field in collection_gate.SOURCE_CANDIDATE_FIELDS:
        source.pop(field)
    source["_source_conclusion"] = "failure"
    (case["collection"] / "source" / "candidate-binding.json").unlink()

    artifact_lookups: list[str] = []

    def collection_artifact_lookup(
        run_id: int,
        name: str,
        commit: str,
        **kwargs: object,
    ) -> dict[str, object]:
        if name != case["collection_name"]:
            pytest.fail(
                "legacy validation-only source must not look up release artefacts"
            )
        assert run_id == int(COLLECTOR_RUN)
        assert commit == COLLECTOR_SHA
        assert kwargs == {"expected_attempt": int(COLLECTOR_ATTEMPT)}
        artifact_lookups.append(name)
        return cast(dict[str, object], case["metadata"][name])

    def unexpected_distribution_download(*_args: object, **_kwargs: object) -> None:
        pytest.fail("legacy validation-only source must not retrieve distributions")

    monkeypatch.setattr(completion, "_artifact_metadata", collection_artifact_lookup)
    monkeypatch.setattr(
        completion, "_download_archive", unexpected_distribution_download
    )

    result = _prepare(case)

    assert result == {
        "ready": False,
        "publish": False,
        "outcome": "validation_only",
    }
    assert case["github_output"].read_text().splitlines() == [
        "ready=false",
        "publish=false",
        "outcome=validation_only",
        "version=",
        "head_sha=",
        "mode=",
    ]
    assert artifact_lookups == [case["collection_name"]]
    assert not case["output"].exists()


def test_marker_bound_source_rejects_missing_candidate_evidence(
    completion_case: dict[str, Any],
) -> None:
    case = completion_case
    (case["collection"] / "source" / "candidate-binding.json").unlink()

    with pytest.raises(completion.CompletionError, match="not a bounded regular file"):
        _ = _prepare(case)

    assert not case["github_output"].exists()
    assert not case["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [("source_run_id", 999), ("source_attempt", 9), ("head_sha", "f" * 40)],
)
def test_prepare_rejects_wrong_source_receipt(
    completion_case: dict[str, Any], field: str, value: object
) -> None:
    case = completion_case
    receipt_path = case["collection"] / "source" / "collection-source.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    _ = receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(completion.CompletionError):
        _ = _prepare(case)


def test_prepare_rejects_terminal_record_for_another_source(
    completion_case: dict[str, Any],
) -> None:
    case = completion_case
    terminal_path = case["collection"] / "periodic-terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["source_run_id"] = "999"
    _ = terminal_path.write_text(json.dumps(terminal))

    with pytest.raises(completion.CompletionError, match="terminal evidence"):
        _ = _prepare(case)


@pytest.mark.parametrize(
    "wheel_version,sdist_version", [("9.9.9", VERSION), (VERSION, "9.9.9")]
)
def test_prepare_rejects_wheel_or_sdist_with_wrong_package_version(
    completion_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wheel_version: str,
    sdist_version: str,
) -> None:
    case = completion_case
    invalid_archive = tmp_path / "wrong-version.zip"
    _ = _write_distribution_archive(
        invalid_archive,
        wheel_version=wheel_version,
        sdist_version=sdist_version,
    )

    def download_invalid_archive(
        _artifact_id: int, destination: Path, _limit: int
    ) -> None:
        _ = destination.write_bytes(invalid_archive.read_bytes())

    monkeypatch.setattr(completion, "_download_archive", download_invalid_archive)
    dist_metadata = case["metadata"][f"release-dist-{SOURCE_ATTEMPT}"]
    dist_metadata["digest"] = (
        "sha256:" + hashlib.sha256(invalid_archive.read_bytes()).hexdigest()
    )
    dist_metadata["size_in_bytes"] = invalid_archive.stat().st_size

    with pytest.raises(completion.CompletionError, match="metadata|version"):
        _ = _prepare(case)


def test_verify_rejects_distribution_bytes_changed_after_completion_approval(
    completion_case: dict[str, Any],
) -> None:
    case = completion_case
    _ = _prepare(case)
    dist = case["output"].parent / "dist"
    _write_original_distributions(case, dist)
    changed = next(dist.iterdir())
    _ = changed.write_bytes(changed.read_bytes() + b"changed")

    with pytest.raises(completion.CompletionError, match="does not match its manifest"):
        _ = completion.verify(case["output"], dist)


def test_verify_rejects_original_artifact_id_or_digest_drift(
    completion_case: dict[str, Any],
) -> None:
    case = completion_case
    _ = _prepare(case)
    dist = case["output"].parent / "dist"
    _write_original_distributions(case, dist)
    original = dict(case["metadata"][f"release-dist-{SOURCE_ATTEMPT}"])

    for field, changed in (("id", 777), ("digest", "sha256:" + "f" * 64)):
        case["metadata"][f"release-dist-{SOURCE_ATTEMPT}"] = {
            **original,
            field: changed,
        }
        with pytest.raises(
            completion.CompletionError, match="distribution artefact identity"
        ):
            _ = completion.verify(case["output"], dist)
    case["metadata"][f"release-dist-{SOURCE_ATTEMPT}"] = original


def test_manual_replay_accepts_successful_collection_job_when_run_needs_approval(
    completion_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = completion_case
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", completion.MANUAL_WORKFLOW_REF)
    monkeypatch.setenv("GITHUB_RUN_ID", COMPLETION_RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", COMPLETION_SHA)

    def gh_api(route: str, _body: dict[str, object] | None = None) -> object:
        if route.endswith("/attempts/2"):
            return {
                "id": int(COLLECTOR_RUN),
                "run_attempt": int(COLLECTOR_ATTEMPT),
                "path": periodic.WORKFLOW_PATH,
                "event": "schedule",
                "head_branch": "master",
                "head_sha": COLLECTOR_SHA,
                "status": "completed",
                "conclusion": "cancelled",
                "repository": {"full_name": periodic.REPOSITORY},
                "head_repository": {"full_name": periodic.REPOSITORY},
            }
        if route.endswith("/jobs?per_page=100"):
            return {
                "total_count": 1,
                "jobs": [
                    {
                        "name": f"collect-{SOURCE_RUN}-{SOURCE_ATTEMPT}",
                        "run_id": int(COLLECTOR_RUN),
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            }
        raise AssertionError(f"unexpected GitHub API route: {route}")

    monkeypatch.setattr(accelerator_gate, "gh_api", gh_api)
    binding = completion.prepare(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        case["collection"],
        case["output"],
        case["github_output"],
        collector_run=COLLECTOR_RUN,
        collector_attempt=COLLECTOR_ATTEMPT,
    )

    assert binding["completion_kind"] == "manual"
    assert binding["publish"] is True
    dist = case["output"].parent / "manual-dist"
    _write_original_distributions(case, dist)
    assert completion.verify(case["output"], dist) == binding


def test_manual_replay_requires_paired_collector_identity(
    completion_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = completion_case
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_WORKFLOW_REF", completion.MANUAL_WORKFLOW_REF)
    monkeypatch.setenv("GITHUB_RUN_ID", COMPLETION_RUN)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", COMPLETION_SHA)

    with pytest.raises(completion.CompletionError, match="supplied together"):
        _ = completion.prepare(
            SOURCE_RUN,
            SOURCE_ATTEMPT,
            SOURCE_SHA,
            case["collection"],
            case["output"],
            case["github_output"],
            collector_run=COLLECTOR_RUN,
        )
