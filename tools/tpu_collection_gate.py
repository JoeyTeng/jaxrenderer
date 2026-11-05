"""Recover delayed TPU evidence without granting a release gate credit."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
from typing import cast
import zipfile

from tools import (
    accelerator_gate,
    kaggle_ci,
    kaggle_collect,
    release_candidate,
    release_tpu_gate,
)

REPOSITORY = release_tpu_gate.REPOSITORY
SOURCE_WORKFLOW = ".github/workflows/pypi.yml"
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024
REQUIRED_JOBS = {
    "prepare",
    "build",
    "cpu / lint",
    "cpu / check (3.12)",
    "cpu / check (3.13)",
    "cpu / check (3.14)",
    "cpu / numpy-minimum",
    "cpu / linux-render-regression",
    "cpu / macos-render-regression",
    "gpu / prepare",
    "gpu / gpu",
    "tpu / prepare",
}
TPU_SEGMENT_JOBS = (
    "tpu / tpu-segment-1",
    "tpu / tpu-segment-2",
    "tpu / tpu-segment-3",
)
SOURCE_BASE_FIELDS = {
    "source_run_id",
    "source_attempt",
    "head_sha",
    "collector_run_id",
    "collector_sha",
    "binding_artifact_id",
    "controller_artifact_id",
}
SOURCE_CANDIDATE_FIELDS = {
    "candidate_artifact_id",
    "candidate_artifact_name",
    "marker_artifact_id",
    "marker_artifact_name",
}
SOURCE_FIELDS = SOURCE_BASE_FIELDS
PENDING_JOB_NAMES = ("pending",)


class CollectionGateError(RuntimeError):
    """A source or collection receipt failed validation."""


class IneligibleSource(CollectionGateError):
    """A complete source inventory proves this run is not a TPU-only timeout."""


def positive_id(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,19}", value):
        raise CollectionGateError("run ID and attempt must be positive integers")
    return int(value)


def read_json(path: Path) -> dict[str, object]:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_JSON_BYTES
        ):
            raise CollectionGateError("evidence must be a small regular JSON file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CollectionGateError("could not read collection evidence") from error
    if not isinstance(value, dict):
        raise CollectionGateError("collection evidence must be a JSON object")
    return cast(dict[str, object], value)


def _write_json(path: Path, value: dict[str, object]) -> None:
    accelerator_gate.atomic_write_json(path, value)


def _context() -> str:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "workflow_dispatch":
        accelerator_gate.require_dispatch_context()
    elif event == "schedule":
        workflow_ref = (
            f"{REPOSITORY}/.github/workflows/collect-tpu-periodic.yml@refs/heads/master"
        )
        if (
            os.environ.get("GITHUB_REF") != "refs/heads/master"
            or os.environ.get("GITHUB_REPOSITORY", "").casefold()
            != REPOSITORY.casefold()
            or os.environ.get("GITHUB_WORKFLOW_REF") != workflow_ref
        ):
            raise CollectionGateError(
                "scheduled collection is restricted to the canonical master workflow"
            )
    else:
        raise CollectionGateError("collection requires workflow_dispatch or schedule")
    sha = os.environ.get("GITHUB_SHA", "")
    if not accelerator_gate.SHA_RE.fullmatch(sha):
        raise CollectionGateError("collector GITHUB_SHA must be a full commit SHA")
    return accelerator_gate.run_id_from_env()


def _source_snapshot(run_id: int, attempt: int, commit: str) -> dict[str, object]:
    """Require a pending async submission or a legacy TPU-only timeout."""
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise CollectionGateError("source commit must be a full lowercase SHA")
    route = f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}"
    run = accelerator_gate.gh_api(route)
    if not isinstance(run, dict):
        raise CollectionGateError(
            "source is not the exact completed publishing attempt"
        )
    run = cast(dict[str, object], run)
    repository = run.get("repository")
    head_repository = run.get("head_repository")
    repository = (
        cast(dict[str, object], repository) if isinstance(repository, dict) else {}
    )
    head_repository = (
        cast(dict[str, object], head_repository)
        if isinstance(head_repository, dict)
        else {}
    )
    if (
        type(run.get("id")) is not int
        or run["id"] != run_id
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] != attempt
        or run.get("head_sha") != commit
        or run.get("path") != SOURCE_WORKFLOW
        or not isinstance(run.get("event"), str)
        or run.get("event") not in {"workflow_dispatch", "release"}
        or run.get("status") != "completed"
        or not isinstance(run.get("conclusion"), str)
        or run.get("conclusion") not in {"failure", "success"}
        or not isinstance(run.get("repository"), dict)
        or str(repository.get("full_name", "")).casefold() != REPOSITORY.casefold()
        or not isinstance(run.get("head_repository"), dict)
        or str(head_repository.get("full_name", "")).casefold() != REPOSITORY.casefold()
    ):
        raise CollectionGateError(
            "source is not the exact completed publishing attempt"
        )
    release_tpu_gate.verify_master_history(commit)
    page = accelerator_gate.gh_api(f"{route}/jobs?per_page=100")
    if not isinstance(page, dict):
        raise CollectionGateError("source job inventory is incomplete or too large")
    page = cast(dict[str, object], page)
    jobs_value = page.get("jobs")
    jobs_list = cast(list[object], jobs_value) if isinstance(jobs_value, list) else []
    total = page.get("total_count")
    if (
        type(total) is not int
        or not isinstance(jobs_value, list)
        or total != len(jobs_list)
        or total > 100
    ):
        raise CollectionGateError("source job inventory is incomplete or too large")
    jobs: dict[str, dict[str, object]] = {}
    for job_value in jobs_list:
        if not isinstance(job_value, dict):
            raise CollectionGateError("source job metadata is malformed")
        job = cast(dict[str, object], job_value)
        if not isinstance(job.get("name"), str):
            raise CollectionGateError("source job metadata is malformed")
        name = cast(str, job["name"])
        if name in jobs or job.get("run_id") != run_id:
            raise CollectionGateError("source job identity is ambiguous")
        jobs[name] = job
    for name in REQUIRED_JOBS:
        if name not in jobs:
            raise CollectionGateError(f"source required job is missing: {name}")
        job = jobs[name]
        if job.get("status") != "completed":
            raise CollectionGateError(f"source required job is incomplete: {name}")
        conclusion = job.get("conclusion")
        if not isinstance(conclusion, str):
            raise CollectionGateError(
                f"source required job conclusion is malformed: {name}"
            )
        if conclusion in {"failure", "cancelled", "skipped"}:
            raise IneligibleSource(f"source required job did not pass: {name}")
        if conclusion != "success":
            raise CollectionGateError(f"source required job did not pass: {name}")
    if run.get("conclusion") == "success":
        pending_jobs = [name for name in PENDING_JOB_NAMES if name in jobs]
        if len(pending_jobs) != 1:
            raise IneligibleSource("source has no unique pending TPU completion job")
        pending = jobs[pending_jobs[0]]
        if (
            pending.get("status") != "completed"
            or pending.get("conclusion") != "success"
        ):
            raise IneligibleSource("source pending TPU completion job did not pass")
        if "tpu / tpu" in jobs or any(name in jobs for name in TPU_SEGMENT_JOBS):
            raise IneligibleSource(
                "pending source unexpectedly contains a TPU wait job"
            )
        collection_kind = "pending"
    else:
        tpu = jobs.get("tpu / tpu", {})
        if not tpu:
            raise IneligibleSource("failed source has no legacy TPU wait job")
        if tpu.get("status") != "completed":
            raise CollectionGateError("source TPU job metadata is incomplete")
        tpu_conclusion = tpu.get("conclusion")
        if not isinstance(tpu_conclusion, str) or tpu_conclusion not in {
            "failure",
            "success",
            "skipped",
            "cancelled",
        }:
            raise CollectionGateError("source TPU job conclusion is unrecognised")

        segments: list[dict[str, object] | None] = []
        failed_segments: list[int] = []
        for index, name in enumerate(TPU_SEGMENT_JOBS):
            job = jobs.get(name)
            segments.append(job)
            if job is None:
                continue
            if job.get("status") != "completed":
                raise CollectionGateError(
                    "source TPU segment job metadata is incomplete"
                )
            conclusion = job.get("conclusion")
            if not isinstance(conclusion, str):
                raise CollectionGateError(
                    "source TPU segment conclusion is unrecognised"
                )
            if conclusion == "failure":
                failed_segments.append(index)
            elif conclusion == "cancelled":
                raise IneligibleSource("source TPU segment was cancelled")
            elif conclusion not in {"success", "skipped"}:
                raise CollectionGateError(
                    "source TPU segment conclusion is unrecognised"
                )

        if failed_segments:
            if len(failed_segments) != 1 or tpu_conclusion != "skipped":
                raise IneligibleSource(
                    "source TPU stages have an invalid failure sequence"
                )
            failed_index = failed_segments[0]
            for index, job in enumerate(segments):
                if job is None or index == failed_index:
                    continue
                expected = "success" if index < failed_index else "skipped"
                if job.get("conclusion") != expected:
                    raise IneligibleSource(
                        "source TPU stages have an invalid failure sequence"
                    )
        elif tpu_conclusion != "failure":
            raise IneligibleSource("source TPU job did not fail")
        collection_kind = "legacy"

    for name, job in jobs.items():
        if name == "tpu / tpu" or name in TPU_SEGMENT_JOBS:
            continue
        if job.get("status") != "completed":
            raise CollectionGateError("source contains incomplete job metadata")
        conclusion = job.get("conclusion")
        if not isinstance(conclusion, str):
            raise CollectionGateError("source job conclusion is malformed")
        if conclusion in {"failure", "cancelled"}:
            raise IneligibleSource("source has another failed job")
        if conclusion not in {"success", "skipped"}:
            raise CollectionGateError("source has an unrecognised job conclusion")
    run["_collection_kind"] = collection_kind
    return run


def _artifact_inventory(run_id: int) -> list[object]:
    page = accelerator_gate.gh_api(
        f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"
    )
    if not isinstance(page, dict):
        raise CollectionGateError(
            "source artefact inventory is incomplete or too large"
        )
    page = cast(dict[str, object], page)
    artifacts_value = page.get("artifacts")
    artifact_list = (
        cast(list[object], artifacts_value) if isinstance(artifacts_value, list) else []
    )
    total = page.get("total_count")
    if (
        type(total) is not int
        or not isinstance(artifacts_value, list)
        or total != len(artifact_list)
        or total > 100
    ):
        raise CollectionGateError(
            "source artefact inventory is incomplete or too large"
        )
    return artifact_list


def _find_artifact_metadata(
    run_id: int,
    name: str,
    commit: str,
    *,
    max_size_bytes: int = MAX_ARCHIVE_BYTES,
    inventory: list[object] | None = None,
) -> dict[str, object] | None:
    if (
        type(max_size_bytes) is not int
        or not 0 < max_size_bytes <= MAX_SOURCE_ARTIFACT_BYTES
    ):
        raise CollectionGateError("source artefact size limit is invalid")
    artifact_list = _artifact_inventory(run_id) if inventory is None else inventory
    matches = [
        cast(dict[str, object], item)
        for item in artifact_list
        if isinstance(item, dict) and cast(dict[str, object], item).get("name") == name
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise CollectionGateError("source must have exactly one named artefact")
    item = matches[0]
    artifact_id = item.get("id")
    size = item.get("size_in_bytes")
    owner = item.get("workflow_run")
    if (
        type(artifact_id) is not int
        or artifact_id <= 0
        or type(size) is not int
        or not 0 < size <= max_size_bytes
        or item.get("expired") is not False
        or not isinstance(owner, dict)
        or cast(dict[str, object], owner).get("id") != run_id
        or cast(dict[str, object], owner).get("head_sha") != commit
    ):
        raise CollectionGateError(
            "source artefact identity, size or availability is invalid"
        )
    return item


def _artifact_metadata(
    run_id: int,
    name: str,
    commit: str,
    *,
    max_size_bytes: int = MAX_ARCHIVE_BYTES,
    inventory: list[object] | None = None,
) -> dict[str, object]:
    item = _find_artifact_metadata(
        run_id, name, commit, max_size_bytes=max_size_bytes, inventory=inventory
    )
    if item is None:
        raise CollectionGateError("source must have exactly one named artefact")
    return item


def artifact_metadata(
    run_id: int,
    name: str,
    commit: str,
    *,
    max_size_bytes: int = MAX_ARCHIVE_BYTES,
) -> dict[str, object]:
    """Validate one named source artefact against its workflow run owner."""
    return _artifact_metadata(run_id, name, commit, max_size_bytes=max_size_bytes)


def _artifact_id(metadata: dict[str, object]) -> int:
    value = metadata.get("id")
    if type(value) is not int or value <= 0:
        raise CollectionGateError("source artefact identity is invalid")
    return value


def download_member(artifact_id: int, member_name: str, destination: Path) -> None:
    """Let gh handle authenticated redirects, with a capped temporary archive."""
    archive_path = destination.with_suffix(".zip")
    env = os.environ.copy()
    token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    if not token:
        raise CollectionGateError("GitHub token is required for artefact retrieval")
    for key in (
        "GH_CONFIG_DIR",
        "GH_HOST",
        "GH_PATH",
        "GH_REPO",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
    ):
        _ = env.pop(key, None)
    env.update(GH_TOKEN=token, GH_HOST="github.com", GH_PROMPT_DISABLED="1")

    def limit_file() -> None:
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (MAX_ARCHIVE_BYTES, MAX_ARCHIVE_BYTES)
        )

    try:
        with archive_path.open("xb") as stream:
            completed = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
                ],
                stdout=stream,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=60,
                check=False,
                preexec_fn=limit_file,
            )
        if completed.returncode:
            raise CollectionGateError("GitHub artefact download failed")
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > 30 or len({item.filename for item in members}) != len(
                members
            ):
                raise CollectionGateError(
                    "source archive has too many or duplicate members"
                )
            matches = [item for item in members if item.filename == member_name]
            if len(matches) != 1 or matches[0].file_size > MAX_JSON_BYTES:
                raise CollectionGateError(
                    "source archive lacks one bounded evidence member"
                )
            data = archive.read(matches[0])
            if len(data) > MAX_JSON_BYTES:
                raise CollectionGateError("source evidence exceeds its byte limit")
            _ = destination.write_bytes(data)
    except (
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        subprocess.TimeoutExpired,
    ) as error:
        raise CollectionGateError(
            "could not retrieve the original source evidence"
        ) from error
    finally:
        archive_path.unlink(missing_ok=True)


def _validate_source_files(
    source_dir: Path,
    source: dict[str, object],
    *,
    source_conclusion: str | None = None,
) -> dict[str, object]:
    source_identity = f"{source['source_run_id']}.{source['source_attempt']}"
    attempt = positive_id(str(source["source_attempt"]))
    source_fields = set(source)
    has_candidate = source_fields == SOURCE_BASE_FIELDS | SOURCE_CANDIDATE_FIELDS
    if frozenset(source_fields) not in {
        frozenset(SOURCE_BASE_FIELDS),
        frozenset(SOURCE_BASE_FIELDS | SOURCE_CANDIDATE_FIELDS),
    }:
        raise CollectionGateError(
            f"source {source_identity} manifest has an unsupported schema"
        )
    for field in ("binding_artifact_id", "controller_artifact_id"):
        value = source.get(field)
        if type(value) is not int or value <= 0:
            raise CollectionGateError(
                f"source {source_identity} artefact identity is invalid"
            )

    candidate: dict[str, object] | None = None
    marker_name: str | None = None
    if has_candidate:
        candidate_name = f"candidate-binding-{attempt}"
        candidate_marker_name = source.get("candidate_artifact_name")
        marker_name_value = source.get("marker_artifact_name")
        if candidate_marker_name != candidate_name:
            raise CollectionGateError(
                f"source {source_identity} candidate artefact name is invalid"
            )
        if not isinstance(marker_name_value, str) or marker_name_value not in {
            f"release-pending-{attempt}",
            f"recovery-source-binding-{attempt}",
        }:
            raise CollectionGateError(
                f"source {source_identity} marker artefact name is invalid"
            )
        for field in ("candidate_artifact_id", "marker_artifact_id"):
            value = source.get(field)
            if type(value) is not int or value <= 0:
                raise CollectionGateError(
                    f"source {source_identity} artefact identity is invalid"
                )
        candidate_path = source_dir / "candidate-binding.json"
        marker_path = source_dir / "marker-binding.json"
        candidate_bytes = _read_bounded_bytes(candidate_path)
        marker_bytes = _read_bounded_bytes(marker_path)
        if candidate_bytes != marker_bytes:
            raise CollectionGateError(
                f"source {source_identity} marker does not match its candidate"
            )
        try:
            candidate = dict(
                release_candidate.validate_binding(read_json(candidate_path))
            )
        except release_candidate.CandidateError as error:
            raise CollectionGateError(
                f"source {source_identity} candidate binding is invalid"
            ) from error
        marker_name = marker_name_value
        if (
            candidate.get("controller_sha") != source.get("head_sha")
            or candidate.get("run_id") != source_identity
        ):
            raise CollectionGateError(
                f"source {source_identity} candidate does not match its controller"
            )
        mode = candidate.get("mode")
        if marker_name == f"recovery-source-binding-{attempt}":
            if mode != "recovery":
                raise CollectionGateError(
                    f"source {source_identity} recovery marker is not a recovery candidate"
                )
        elif mode == "release" and candidate.get("head_sha") != source.get("head_sha"):
            raise CollectionGateError(
                f"source {source_identity} release candidate does not match its controller"
            )

    binding = release_tpu_gate.validate_binding(read_json(source_dir / "binding.json"))
    expected = {
        "kind": "release",
        "head_repository": REPOSITORY,
        "head_sha": candidate["head_sha"]
        if candidate is not None
        else source["head_sha"],
        "run_id": f"{source['source_run_id']}.{source['source_attempt']}",
    }
    if binding != expected:
        raise CollectionGateError(
            f"source {source_identity} binding does not match the original attempt"
        )
    state_path = source_dir / "kaggle-controller-state.json"
    state = read_json(state_path)
    state_fields = set(state)
    if state_fields == kaggle_ci.LEGACY_CONTROLLER_STATE_FIELDS:
        if state.get("binding") != binding:
            raise CollectionGateError(
                f"source {source_identity} legacy controller binding is invalid"
            )
        submitted_version = state.get("submitted_version")
        if submitted_version is not None and (
            type(submitted_version) is not int or submitted_version != 1
        ):
            raise CollectionGateError(
                f"source {source_identity} legacy controller version is invalid"
            )
        kernel_id = state.get("kernel_id")
        try:
            _ = kaggle_collect.kernel_identity(dict(binding), kernel_id)
        except kaggle_collect.CollectionError as error:
            raise CollectionGateError(
                f"source {source_identity} legacy controller identity is invalid"
            ) from error
        raise IneligibleSource(
            f"source {source_identity} legacy controller state has no timeout evidence"
        )
    if state_fields != kaggle_ci.CONTROLLER_STATE_FIELDS:
        raise CollectionGateError(
            f"source {source_identity} controller state schema is unsupported"
        )
    if state.get("binding") != binding:
        raise CollectionGateError(
            f"source {source_identity} controller binding is invalid"
        )
    outcome = state.get("outcome")
    if not isinstance(outcome, str):
        raise CollectionGateError(
            f"source {source_identity} controller outcome is unknown"
        )
    if outcome in {"queue_timeout", "execution_timeout"}:
        if (
            type(state.get("submitted_version")) is not int
            or state["submitted_version"] != 1
        ):
            raise CollectionGateError(
                f"source {source_identity} timed-out controller did not submit version 1"
            )
        status = state.get("status")
        if not isinstance(status, str) or status not in (
            kaggle_ci.IN_PROGRESS | kaggle_ci.TERMINAL_SUCCESS
        ):
            raise CollectionGateError(
                f"source {source_identity} timed-out controller status is not eligible for collection"
            )
    elif outcome == "handover":
        if (
            not has_candidate
            or marker_name != f"release-pending-{attempt}"
            or source_conclusion != "success"
        ):
            raise IneligibleSource(
                f"source {source_identity} handover is not a timeout or successful pending source"
            )
        _validate_handover_state(state, dict(binding), source_identity)
    elif outcome in {
        "preflight_error",
        "submission_error_unknown",
        "submission_version_unknown",
        "cli_error",
        "unknown_api_status",
        "remote_terminal_success",
        "remote_terminal_failure",
        "output_retrieval_error",
        "result_validation_error",
        "version_verification_error",
        "pending",
        "success",
        "status_query_error",
    }:
        raise IneligibleSource(
            f"source {source_identity} controller outcome is not a timeout"
        )
    else:
        raise CollectionGateError(
            f"source {source_identity} controller outcome is unknown"
        )
    return dict(binding)


def _read_bounded_bytes(path: Path) -> bytes:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_JSON_BYTES
        ):
            raise CollectionGateError("evidence must be a small regular JSON file")
        return path.read_bytes()
    except OSError as error:
        raise CollectionGateError("could not read collection evidence") from error


def _validate_kernel_identity(
    state: dict[str, object], binding: dict[str, object], source_identity: str
) -> None:
    try:
        _ = kaggle_collect.kernel_identity(binding, state.get("kernel_id"))
    except kaggle_collect.CollectionError as error:
        raise CollectionGateError(
            f"source {source_identity} controller identity is invalid"
        ) from error


def _validate_handover_state(
    state: dict[str, object], binding: dict[str, object], source_identity: str
) -> None:
    status = state.get("status")
    phase = state.get("phase")
    if (
        type(state.get("submitted_version")) is not int
        or state["submitted_version"] != 1
        or not isinstance(status, str)
        or status not in kaggle_ci.IN_PROGRESS | {"submitted"}
        or not isinstance(phase, str)
        or phase not in {"queue", "execution"}
        or state.get("last_error") is not None
    ):
        raise CollectionGateError(
            f"source {source_identity} handover state is malformed"
        )
    timestamps: dict[str, datetime] = {}
    for field in ("submitted_at", "updated_at", "phase_started_at"):
        value = state.get(field)
        if not isinstance(value, str):
            raise CollectionGateError(
                f"source {source_identity} handover timestamp is malformed"
            )
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise CollectionGateError(
                f"source {source_identity} handover timestamp is malformed"
            ) from error
        if parsed.tzinfo is None:
            raise CollectionGateError(
                f"source {source_identity} handover timestamp has no timezone"
            )
        timestamps[field] = parsed
    if (
        timestamps["submitted_at"] > timestamps["updated_at"]
        or timestamps["phase_started_at"] > timestamps["updated_at"]
    ):
        raise CollectionGateError(
            f"source {source_identity} handover timestamps are inconsistent"
        )
    execution_started = state.get("execution_started_at")
    if execution_started is not None:
        if not isinstance(execution_started, str):
            raise CollectionGateError(
                f"source {source_identity} handover execution timestamp is malformed"
            )
        try:
            parsed_execution = datetime.fromisoformat(
                execution_started.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise CollectionGateError(
                f"source {source_identity} handover execution timestamp is malformed"
            ) from error
        if (
            parsed_execution.tzinfo is None
            or parsed_execution > timestamps["updated_at"]
        ):
            raise CollectionGateError(
                f"source {source_identity} handover execution timestamp is inconsistent"
            )
    expected_phase = "execution" if execution_started is not None else "queue"
    if phase != expected_phase:
        raise CollectionGateError(
            f"source {source_identity} handover phase is inconsistent"
        )
    for field in ("queue_elapsed_seconds", "execution_elapsed_seconds"):
        value = state.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise CollectionGateError(
                f"source {source_identity} handover elapsed time is malformed"
            )
    _validate_kernel_identity(state, binding, source_identity)


def prepare(
    source_run: str,
    source_attempt: str,
    commit: str,
    output_dir: Path,
    github_output: Path,
) -> dict[str, object]:
    collector = _context()
    run_id, attempt = positive_id(source_run), positive_id(source_attempt)
    if collector == f"{run_id}.{attempt}":
        raise CollectionGateError("collection must use a separate workflow attempt")
    snapshot = _source_snapshot(run_id, attempt, commit)
    collection_kind = snapshot["_collection_kind"]
    inventory = _artifact_inventory(run_id)
    binding_artifact = _artifact_metadata(
        run_id, f"tpu-release-binding-{attempt}", commit, inventory=inventory
    )
    pending_marker_name = f"release-pending-{attempt}"
    recovery_marker_name = f"recovery-source-binding-{attempt}"
    if collection_kind == "pending":
        marker_name = pending_marker_name
        marker_artifact = _find_artifact_metadata(
            run_id, marker_name, commit, inventory=inventory
        )
        if marker_artifact is None:
            raise IneligibleSource("source has no pending TPU marker artefact")
    else:
        pending_marker = _find_artifact_metadata(
            run_id, pending_marker_name, commit, inventory=inventory
        )
        if pending_marker is not None:
            raise IneligibleSource(
                "failed legacy source unexpectedly has a pending TPU marker"
            )
        marker_name = recovery_marker_name
        marker_artifact = _find_artifact_metadata(
            run_id, marker_name, commit, inventory=inventory
        )

    candidate_artifact: dict[str, object] | None = None
    if marker_artifact is not None:
        candidate_name = f"candidate-binding-{attempt}"
        candidate_artifact = _artifact_metadata(
            run_id, candidate_name, commit, inventory=inventory
        )
    else:
        candidate_name = ""

    output_dir.mkdir(parents=True, exist_ok=False)
    source: dict[str, object] = {
        "source_run_id": run_id,
        "source_attempt": attempt,
        "head_sha": commit,
        "collector_run_id": collector,
        "collector_sha": os.environ["GITHUB_SHA"],
        "binding_artifact_id": _artifact_id(binding_artifact),
    }
    candidate_sha = commit
    if candidate_artifact is not None and marker_artifact is not None:
        source.update(
            candidate_artifact_id=_artifact_id(candidate_artifact),
            candidate_artifact_name=candidate_name,
            marker_artifact_id=_artifact_id(marker_artifact),
            marker_artifact_name=marker_name,
        )
        download_member(
            _artifact_id(candidate_artifact),
            "candidate-binding.json",
            output_dir / "candidate-binding.json",
        )
        download_member(
            _artifact_id(marker_artifact),
            "candidate-binding.json",
            output_dir / "marker-binding.json",
        )
        candidate_bytes = _read_bounded_bytes(output_dir / "candidate-binding.json")
        marker_bytes = _read_bounded_bytes(output_dir / "marker-binding.json")
        if candidate_bytes != marker_bytes:
            raise CollectionGateError(
                "source candidate marker does not match candidate"
            )
        try:
            candidate = release_candidate.validate_binding(
                read_json(output_dir / "candidate-binding.json")
            )
        except release_candidate.CandidateError as error:
            raise CollectionGateError("source candidate binding is invalid") from error
        if (
            candidate["controller_sha"] != commit
            or candidate["run_id"] != f"{run_id}.{attempt}"
        ):
            raise CollectionGateError("source candidate does not match its controller")
        if marker_name == recovery_marker_name and candidate["mode"] != "recovery":
            raise CollectionGateError(
                "source recovery marker is not a recovery candidate"
            )
        if candidate["mode"] == "release" and candidate["head_sha"] != commit:
            raise CollectionGateError(
                "source release candidate does not match its controller"
            )
        candidate_sha = candidate["head_sha"]

    controller_artifact_name = f"release-tpu-{candidate_sha}-{attempt}"
    controller_artifact = _artifact_metadata(
        run_id, controller_artifact_name, commit, inventory=inventory
    )
    source["controller_artifact_id"] = _artifact_id(controller_artifact)
    download_member(
        _artifact_id(binding_artifact),
        "tpu-release-binding.json",
        output_dir / "binding.json",
    )
    download_member(
        _artifact_id(controller_artifact),
        "kaggle-controller-state.json",
        output_dir / "kaggle-controller-state.json",
    )
    _ = _validate_source_files(
        output_dir, source, source_conclusion=cast(str, snapshot["conclusion"])
    )
    _write_json(output_dir / "collection-source.json", source)
    with github_output.open("a", encoding="utf-8") as stream:
        _ = stream.write(
            f"head_sha={commit}\nsource_run_id={run_id}\nsource_attempt={attempt}\n"
        )
    return source


def revalidate(
    source_dir: Path,
    *,
    expected_collector_run_id: str | None = None,
    expected_collector_sha: str | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Recheck original source evidence under the current collector context.

    Explicit collector overrides support replay only after the caller has
    authenticated that historic collector attempt and its source artefact.
    """
    current_collector = _context()
    if (expected_collector_run_id is None) != (expected_collector_sha is None):
        raise CollectionGateError("expected collector identity must be complete")
    if expected_collector_run_id is None:
        expected_collector = current_collector
        expected_sha = os.environ["GITHUB_SHA"]
    else:
        expected_collector = expected_collector_run_id
        expected_sha = cast(str, expected_collector_sha)
        if not release_tpu_gate.RUN_ID_RE.fullmatch(expected_collector) or not (
            accelerator_gate.SHA_RE.fullmatch(expected_sha)
        ):
            raise CollectionGateError("expected collector identity is malformed")
    source = read_json(source_dir / "collection-source.json")
    if (
        frozenset(source)
        not in {
            frozenset(SOURCE_BASE_FIELDS),
            frozenset(SOURCE_BASE_FIELDS | SOURCE_CANDIDATE_FIELDS),
        }
        or source.get("collector_run_id") != expected_collector
        or source.get("collector_sha") != expected_sha
    ):
        raise CollectionGateError(
            "collection source receipt belongs to another collector"
        )
    run_id = positive_id(str(source["source_run_id"]))
    attempt = positive_id(str(source["source_attempt"]))
    commit = source["head_sha"]
    if not isinstance(commit, str):
        raise CollectionGateError("source SHA is malformed")
    snapshot = _source_snapshot(run_id, attempt, commit)
    inventory = _artifact_inventory(run_id)
    source_fields = set(source)
    has_candidate = source_fields == SOURCE_BASE_FIELDS | SOURCE_CANDIDATE_FIELDS
    collection_kind = snapshot["_collection_kind"]
    pending_marker_name = f"release-pending-{attempt}"
    recovery_marker_name = f"recovery-source-binding-{attempt}"
    marker_name: str | None = None
    if collection_kind == "pending":
        marker_name = pending_marker_name
        if not has_candidate:
            raise CollectionGateError("pending source has no frozen candidate marker")
    else:
        pending_marker = _find_artifact_metadata(
            run_id, pending_marker_name, commit, inventory=inventory
        )
        if pending_marker is not None:
            raise CollectionGateError("failed source unexpectedly has a pending marker")
        recovery_marker = _find_artifact_metadata(
            run_id, recovery_marker_name, commit, inventory=inventory
        )
        if recovery_marker is None:
            if has_candidate:
                raise CollectionGateError("legacy source candidate marker disappeared")
        else:
            marker_name = recovery_marker_name
            if not has_candidate:
                raise CollectionGateError(
                    "legacy recovery marker has no frozen candidate"
                )
    if has_candidate and source.get("marker_artifact_name") != marker_name:
        raise CollectionGateError("source marker artefact name changed")
    if not has_candidate and marker_name is not None:
        raise CollectionGateError("source marker requires a frozen candidate")

    binding = _validate_source_files(
        source_dir, source, source_conclusion=cast(str, snapshot["conclusion"])
    )
    artifact_names = [
        ("binding_artifact_id", f"tpu-release-binding-{attempt}"),
        (
            "controller_artifact_id",
            f"release-tpu-{binding['head_sha']}-{attempt}",
        ),
    ]
    if has_candidate:
        artifact_names.extend(
            (
                ("candidate_artifact_id", f"candidate-binding-{attempt}"),
                ("marker_artifact_id", cast(str, marker_name)),
            )
        )
    for field, name in artifact_names:
        if (
            _artifact_metadata(run_id, name, commit, inventory=inventory)["id"]
            != source[field]
        ):
            raise CollectionGateError("original source artefact identity changed")
    state = read_json(source_dir / "kaggle-controller-state.json")
    source["_source_conclusion"] = snapshot["conclusion"]
    return source, binding, state


def finish(
    source_dir: Path,
    output_dir: Path,
    provider_success: bool,
    *,
    expected_collector_run_id: str | None = None,
    expected_collector_sha: str | None = None,
) -> dict[str, object]:
    source, binding, state = revalidate(
        source_dir,
        expected_collector_run_id=expected_collector_run_id,
        expected_collector_sha=expected_collector_sha,
    )
    collector = str(source["collector_run_id"])
    report = read_json(output_dir / "kaggle-collection-report.json")
    if (
        not provider_success
        or report.get("success") is not True
        or report.get("binding") != binding
        or report.get("kernel_id") != state["kernel_id"]
        or type(report.get("submitted_version")) is not int
        or report["submitted_version"] != 1
        or type(report.get("requested_version")) is not int
        or report["requested_version"] != 1
        or report.get("version_verified") is not True
        or report.get("remote_status") not in kaggle_ci.TERMINAL_SUCCESS
        or report.get("outcome") != "success"
    ):
        raise CollectionGateError(
            "provider did not collect the original successful TPU result"
        )
    release_tpu_gate.read_result(
        output_dir / "result.json",
        cast(release_tpu_gate.Binding, binding),
        "tpu",
    )
    receipt = {
        "binding": binding,
        "collector_run_id": collector,
        "collector_sha": source["collector_sha"],
        "kernel_id": state["kernel_id"],
        "submitted_version": 1,
        "success": True,
        "validation_only": True,
        "release_gate_credit": False,
        "source_conclusion": source["_source_conclusion"],
    }
    _write_json(output_dir / "collection-receipt.json", receipt)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        commit = str(source["head_sha"])
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            _ = stream.write(
                "## Delayed TPU evidence collected\n\n"
                f"- Original commit: `{commit}`\n- Original attempt: `{binding['run_id']}`\n"
                f"- Collection attempt: `{collector}`\n"
                "- Original TPU result validated; logs and numerical report retained.\n"
                "- Validation only: the original attempt stays failed and no release gate credit is granted.\n"
            )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("prepare")
    _ = first.add_argument("--source-run", required=True)
    _ = first.add_argument("--source-attempt", required=True)
    _ = first.add_argument("--commit", required=True)
    _ = first.add_argument("--output-dir", type=Path, required=True)
    _ = first.add_argument("--github-output", type=Path, required=True)
    last = commands.add_parser("finish")
    _ = last.add_argument("--source-dir", type=Path, required=True)
    _ = last.add_argument("--output-dir", type=Path, required=True)
    _ = last.add_argument(
        "--provider-success", choices=("true", "false"), required=True
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            receipt = prepare(
                args.source_run,
                args.source_attempt,
                args.commit,
                args.output_dir,
                args.github_output,
            )
        else:
            receipt = finish(
                args.source_dir, args.output_dir, args.provider_success == "true"
            )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (
        CollectionGateError,
        accelerator_gate.GateError,
        release_tpu_gate.ReleaseGateError,
        OSError,
    ) as error:
        message = str(error)
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "KAGGLE_API_TOKEN"):
            token = os.environ.get(name, "")
            if token:
                message = message.replace(token, "[redacted]")
        print(f"TPU collection gate: {message[:400]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
