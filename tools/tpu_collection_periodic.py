"""Discover and collect delayed TPU results without changing release outcomes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Generator, cast
from urllib.parse import urlencode

from tools import accelerator_gate, kaggle_ci, kaggle_collect, release_tpu_gate
from tools import tpu_collection_gate as collection_gate

REPOSITORY = collection_gate.REPOSITORY
WORKFLOW_PATH = ".github/workflows/collect-tpu-periodic.yml"
WORKFLOW_REF = f"{REPOSITORY}/{WORKFLOW_PATH}@refs/heads/master"
SOURCE_WORKFLOW = collection_gate.SOURCE_WORKFLOW
MAX_AGE = timedelta(days=7)
MAX_RUNS = 100
MAX_CANDIDATES = 8
MAX_RUNTIME_SECONDS = 10 * 60
MAX_REPORT_BYTES = collection_gate.MAX_JSON_BYTES
TERMINAL_FIELDS = {
    "terminal",
    "outcome",
    "source_run_id",
    "source_attempt",
    "head_sha",
    "collector_run_id",
    "collector_attempt",
    "collector_sha",
    "binding",
    "kernel_id",
    "submitted_version",
    "requested_version",
    "version_verified",
    "remote_status",
}
TERMINAL_PROOF_FIELDS = {
    "terminal",
    "outcome",
    "source_run_id",
    "source_attempt",
    "head_sha",
    "binding",
    "kernel_id",
    "submitted_version",
    "requested_version",
    "version_verified",
    "remote_status",
}


class PeriodicError(RuntimeError):
    """A discovery or collection run could not be verified safely."""


class _Budget:
    def __init__(self) -> None:
        super().__init__()
        self.deadline = time.monotonic() + MAX_RUNTIME_SECONDS

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise PeriodicError("periodic collection exceeded its runtime budget")


@contextmanager
def _bounded_github_calls(budget: _Budget) -> Generator[None, None, None]:
    """Check the aggregate deadline before each already-bounded GitHub operation."""
    original_api = accelerator_gate.gh_api
    original_download = collection_gate.download_member

    def checked_api(route: str, body: dict[str, object] | None = None) -> object:
        budget.check()
        return original_api(route, body)

    def checked_download(artifact_id: int, member: str, destination: Path) -> None:
        budget.check()
        original_download(artifact_id, member, destination)
        budget.check()

    accelerator_gate.gh_api = checked_api
    collection_gate.download_member = checked_download
    try:
        yield
    finally:
        accelerator_gate.gh_api = original_api
        collection_gate.download_member = original_download


def _require_context() -> str:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event not in {"schedule", "workflow_dispatch"}:
        raise PeriodicError("periodic collector requires schedule or workflow_dispatch")
    if (
        os.environ.get("GITHUB_REPOSITORY", "").casefold() != REPOSITORY.casefold()
        or os.environ.get("GITHUB_REF") != "refs/heads/master"
        or os.environ.get("GITHUB_WORKFLOW_REF") != WORKFLOW_REF
    ):
        raise PeriodicError(
            "periodic collector is restricted to its canonical master workflow"
        )
    commit = os.environ.get("GITHUB_SHA", "")
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise PeriodicError("collector GITHUB_SHA must be a full lowercase commit SHA")
    return accelerator_gate.run_id_from_env()


def require_context() -> str:
    """Authenticate the canonical master collector workflow context."""
    return _require_context()


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise PeriodicError("GitHub run creation time is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PeriodicError("GitHub run creation time is malformed") from error
    if parsed.tzinfo is None:
        raise PeriodicError("GitHub run creation time has no timezone")
    return parsed.astimezone(timezone.utc)


def _identity_run(run: object, *, source: bool) -> dict[str, object]:
    if not isinstance(run, dict):
        raise PeriodicError("GitHub workflow run metadata is malformed")
    # The GitHub transport decodes JSON objects; identity checks remain below.
    run = cast(dict[str, object], run)
    repo = run.get("repository")
    head_repo = run.get("head_repository")
    run_id = run.get("id")
    attempt = run.get("run_attempt")
    commit = run.get("head_sha")
    if (
        type(run_id) is not int
        or run_id <= 0
        or type(attempt) is not int
        or attempt <= 0
        or not isinstance(commit, str)
        or not accelerator_gate.SHA_RE.fullmatch(commit)
        or run.get("path") != (SOURCE_WORKFLOW if source else WORKFLOW_PATH)
        or not isinstance(repo, dict)
        or str(cast(dict[str, object], repo).get("full_name", "")).casefold()
        != REPOSITORY.casefold()
        or not isinstance(head_repo, dict)
        or str(cast(dict[str, object], head_repo).get("full_name", "")).casefold()
        != REPOSITORY.casefold()
    ):
        raise PeriodicError("GitHub workflow run identity is not canonical")
    if source:
        event = run.get("event")
        if not isinstance(event, str) or event not in {
            "release",
            "workflow_dispatch",
        }:
            raise PeriodicError("source workflow event is not eligible for collection")
    if not source and (
        run.get("event") not in {"schedule", "workflow_dispatch"}
        or run.get("head_branch") != "master"
    ):
        raise PeriodicError("terminal artefact owner is not a master collector run")
    return run


def _latest_source_attempt(run: dict[str, object], budget: _Budget) -> None:
    run_id = run["id"]
    budget.check()
    latest = accelerator_gate.gh_api(f"repos/{REPOSITORY}/actions/runs/{run_id}")
    latest = _identity_run(latest, source=True)
    for field in (
        "id",
        "run_attempt",
        "head_sha",
        "path",
        "event",
        "status",
        "conclusion",
    ):
        if latest.get(field) != run.get(field):
            raise PeriodicError("source run changed after inventory discovery")


def _source_expired(
    source_run_id: str, source_attempt: str, commit: str, budget: _Budget
) -> bool:
    """Check the exact source attempt against the seven-day collection window."""
    route = f"repos/{REPOSITORY}/actions/runs/{source_run_id}/attempts/{source_attempt}"
    budget.check()
    run = _identity_run(accelerator_gate.gh_api(route), source=True)
    if (
        str(run["id"]) != source_run_id
        or str(run["run_attempt"]) != source_attempt
        or run.get("head_sha") != commit
        or run.get("status") != "completed"
        or not isinstance(run.get("conclusion"), str)
        or run.get("conclusion") not in {"failure", "success"}
    ):
        raise PeriodicError("source attempt changed after discovery")
    _latest_source_attempt(run, budget)
    return (
        _parse_timestamp(run.get("created_at")) < datetime.now(timezone.utc) - MAX_AGE
    )


def _list_recent_runs(budget: _Budget) -> list[dict[str, object]]:
    cutoff = datetime.now(timezone.utc) - MAX_AGE
    query = urlencode(
        {
            "created": f">={cutoff.isoformat(timespec='seconds')}",
            "per_page": str(MAX_RUNS),
            "page": "1",
        }
    )
    budget.check()
    page = accelerator_gate.gh_api(
        f"repos/{REPOSITORY}/actions/workflows/pypi.yml/runs?{query}"
    )
    if not isinstance(page, dict):
        raise PeriodicError("recent source run inventory is incomplete or too large")
    page = cast(dict[str, object], page)
    total = page.get("total_count")
    items = page.get("workflow_runs")
    if (
        type(total) is not int
        or total < 0
        or total > MAX_RUNS
        or not isinstance(items, list)
        or total != len(cast(list[object], items))
    ):
        raise PeriodicError("recent source run inventory is incomplete or too large")
    runs: list[dict[str, object]] = []
    for item in cast(list[object], items):
        run = _identity_run(item, source=True)
        created_at = _parse_timestamp(run.get("created_at"))
        if created_at > cutoff:
            runs.append(run)
    return runs


def _terminal_artifacts(
    source_run: int, artifact_name: str, budget: _Budget
) -> list[dict[str, object]]:
    query = urlencode({"name": artifact_name, "per_page": str(MAX_RUNS), "page": "1"})
    budget.check()
    page = accelerator_gate.gh_api(f"repos/{REPOSITORY}/actions/artifacts?{query}")
    if not isinstance(page, dict):
        raise PeriodicError("terminal artefact inventory is incomplete or too large")
    page = cast(dict[str, object], page)
    total = page.get("total_count")
    items = page.get("artifacts")
    if (
        type(total) is not int
        or total < 0
        or total > MAX_RUNS
        or not isinstance(items, list)
        or total != len(cast(list[object], items))
    ):
        raise PeriodicError("terminal artefact inventory is incomplete or too large")
    matches = [
        cast(dict[str, object], item)
        for item in cast(list[object], items)
        if isinstance(item, dict)
        and cast(dict[str, object], item).get("name") == artifact_name
    ]
    for item in matches:
        owner = item.get("workflow_run")
        artifact_id = item.get("id")
        size = item.get("size_in_bytes")
        owner_map = cast(dict[str, object], owner) if isinstance(owner, dict) else None
        owner_id = owner_map.get("id") if owner_map is not None else None
        owner_sha = owner_map.get("head_sha") if owner_map is not None else None
        if (
            type(artifact_id) is not int
            or artifact_id <= 0
            or type(size) is not int
            or not 0 < size <= collection_gate.MAX_ARCHIVE_BYTES
            or item.get("expired") is not False
            or not isinstance(owner, dict)
            or type(owner_id) is not int
            or owner_id <= 0
            or not isinstance(owner_sha, str)
            or not accelerator_gate.SHA_RE.fullmatch(owner_sha)
        ):
            raise PeriodicError(
                "terminal artefact identity, size or availability is invalid"
            )
    return matches


def _collection_artifact(
    *,
    source_run_id: str,
    source_attempt: str,
    commit: str,
    collector_id: int,
    collector_attempt: str,
    collector_sha: str,
    budget: _Budget,
) -> dict[str, object] | None:
    """Find the full collection artefact paired with a terminal marker."""
    artifact_name = (
        f"tpu-periodic-{commit}-{source_run_id}-{source_attempt}-"
        f"{collector_id}-{collector_attempt}"
    )
    query = urlencode({"name": artifact_name, "per_page": str(MAX_RUNS), "page": "1"})
    budget.check()
    page = accelerator_gate.gh_api(f"repos/{REPOSITORY}/actions/artifacts?{query}")
    if not isinstance(page, dict):
        raise PeriodicError("collection artefact inventory is incomplete or too large")
    page = cast(dict[str, object], page)
    total = page.get("total_count")
    items = page.get("artifacts")
    if (
        type(total) is not int
        or total < 0
        or total > MAX_RUNS
        or not isinstance(items, list)
        or total != len(cast(list[object], items))
    ):
        raise PeriodicError("collection artefact inventory is incomplete or too large")
    matches = [
        cast(dict[str, object], item)
        for item in cast(list[object], items)
        if isinstance(item, dict)
        and cast(dict[str, object], item).get("name") == artifact_name
    ]
    if len(matches) > 1:
        raise PeriodicError("multiple collection artefacts have the same identity")
    if not matches:
        # Uploads can briefly be absent from the API after the marker appears.
        # An orphan marker is retryable evidence, not proof of completion.
        return None
    item = matches[0]
    artifact_id = item.get("id")
    size = item.get("size_in_bytes")
    owner = item.get("workflow_run")
    owner_map = cast(dict[str, object], owner) if isinstance(owner, dict) else None
    owner_id = owner_map.get("id") if owner_map is not None else None
    owner_sha = owner_map.get("head_sha") if owner_map is not None else None
    if (
        type(artifact_id) is not int
        or artifact_id <= 0
        or type(size) is not int
        or not 0 < size <= collection_gate.MAX_SOURCE_ARTIFACT_BYTES
        or item.get("expired") is not False
        or not isinstance(owner, dict)
        or type(owner_id) is not int
        or owner_id != collector_id
        or not isinstance(owner_sha, str)
        or not accelerator_gate.SHA_RE.fullmatch(owner_sha)
        or owner_sha != collector_sha
    ):
        raise PeriodicError(
            "collection artefact identity, size or availability is invalid"
        )
    return item


def _coalesce_verified_terminals(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """Select a stable representative only when all verified proofs agree."""
    if not records:
        return None
    proof = {field: records[0][field] for field in TERMINAL_PROOF_FIELDS}
    if any(
        {field: record[field] for field in TERMINAL_PROOF_FIELDS} != proof
        for record in records[1:]
    ):
        raise PeriodicError(
            "conflicting verified terminal artefacts for the same source"
        )

    def collector_key(record: dict[str, object]) -> tuple[int, int, str]:
        collector_run = record.get("collector_run_id")
        collector_sha = record.get("collector_sha")
        match = (
            re.fullmatch(r"([1-9][0-9]*)\.([1-9][0-9]*)", collector_run)
            if isinstance(collector_run, str)
            else None
        )
        if match is None or not isinstance(collector_sha, str):
            raise PeriodicError("verified terminal collector identity is malformed")
        return (int(match.group(1)), int(match.group(2)), collector_sha)

    return min(records, key=collector_key)


_COLLECT_JOB_EVIDENCE_STEPS = {
    "Collect and validate the delayed TPU result",
    "Upload periodic collection evidence",
    "Record terminal collection marker",
}
_ACTIVE_RUN_STATUSES = {"queued", "requested", "waiting", "pending", "in_progress"}


def _collector_job_completed(
    collector_id: int,
    collector_sha: str,
    collector_attempt: str,
    source_run_id: str,
    source_attempt: str,
    outcome: str,
    budget: _Budget,
) -> bool:
    """Prove the exact collection job finished before an active run is skipped."""
    route = (
        f"repos/{REPOSITORY}/actions/runs/{collector_id}/attempts/"
        f"{collector_attempt}/jobs?per_page=100"
    )
    budget.check()
    page = accelerator_gate.gh_api(route)
    if not isinstance(page, dict):
        return False
    page = cast(dict[str, object], page)
    total = page.get("total_count")
    jobs = page.get("jobs")
    if (
        type(total) is not int
        or total < 0
        or total > MAX_RUNS
        or not isinstance(jobs, list)
        or total != len(cast(list[object], jobs))
    ):
        return False
    expected_name = f"collect-{source_run_id}-{source_attempt}"
    matches = [
        cast(dict[str, object], job)
        for job in cast(list[object], jobs)
        if isinstance(job, dict)
        and cast(dict[str, object], job).get("name") == expected_name
    ]
    if len(matches) != 1:
        return False
    job = matches[0]
    job_id = job.get("id")
    job_run_id = job.get("run_id")
    if outcome not in {"success", "remote_failure"}:
        return False
    expected_job_conclusion = "success" if outcome == "success" else "failure"
    if (
        type(job_id) is not int
        or job_id <= 0
        or type(job_run_id) is not int
        or job_run_id != collector_id
        or job.get("head_sha") != collector_sha
        or job.get("status") != "completed"
        or job.get("conclusion") != expected_job_conclusion
    ):
        return False
    try:
        completed_at = _parse_timestamp(job.get("completed_at"))
        started_at = _parse_timestamp(job.get("started_at"))
    except PeriodicError:
        return False
    if completed_at < started_at:
        return False
    steps = job.get("steps")
    if not isinstance(steps, list):
        return False
    found: set[str] = set()
    for value in cast(list[object], steps):
        if not isinstance(value, dict):
            return False
        step = cast(dict[str, object], value)
        name = step.get("name")
        if not isinstance(name, str) or step.get("status") != "completed":
            return False
        if name in _COLLECT_JOB_EVIDENCE_STEPS:
            expected_step_conclusion = (
                "failure"
                if outcome == "remote_failure"
                and name == "Collect and validate the delayed TPU result"
                else "success"
            )
            if step.get("conclusion") != expected_step_conclusion or name in found:
                return False
            found.add(name)
        elif step.get("conclusion") not in {"success", "skipped"}:
            return False
    return found == _COLLECT_JOB_EVIDENCE_STEPS


def _read_terminal(
    artifact: dict[str, object],
    *,
    source_run_id: str,
    source_attempt: str,
    commit: str,
    binding: Mapping[str, object],
    kernel_id: str,
    budget: _Budget,
    require_job_proof: bool = False,
) -> dict[str, object] | None:
    owner = artifact["workflow_run"]
    assert isinstance(owner, dict)
    owner = cast(dict[str, object], owner)
    collector_id = cast(int, owner["id"])
    collector_sha = cast(str, owner["head_sha"])
    name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{commit}"
    with tempfile.TemporaryDirectory(prefix="jaxrenderer-periodic-terminal-") as temp:
        path = Path(temp) / "periodic-terminal.json"
        collection_gate.download_member(
            cast(int, artifact["id"]), "periodic-terminal.json", path
        )
        record = collection_gate.read_json(path)
    if set(record) != TERMINAL_FIELDS:
        raise PeriodicError("terminal artefact has an unsupported schema")
    collector_attempt = record.get("collector_attempt")
    if (
        not isinstance(collector_attempt, str)
        or re.fullmatch(r"[1-9][0-9]*", collector_attempt) is None
        or record.get("collector_run_id") != f"{collector_id}.{collector_attempt}"
        or record.get("collector_sha") != collector_sha
    ):
        raise PeriodicError(
            "terminal artefact collector identity does not match its owner"
        )
    route = (
        f"repos/{REPOSITORY}/actions/runs/{collector_id}/attempts/{collector_attempt}"
    )
    budget.check()
    run = _identity_run(accelerator_gate.gh_api(route), source=False)
    if (
        run["id"] != collector_id
        or run["head_sha"] != collector_sha
        or run.get("run_attempt") != int(collector_attempt)
    ):
        raise PeriodicError("terminal artefact owner attempt is not exact")
    run_status = run.get("status")
    if run_status == "completed":
        if run.get("conclusion") not in {
            "success",
            "failure",
            "cancelled",
            "timed_out",
        }:
            raise PeriodicError(
                "terminal artefact owner attempt has no final conclusion"
            )
    elif run_status in _ACTIVE_RUN_STATUSES:
        if run.get("conclusion") is not None:
            raise PeriodicError("active terminal artefact owner has a conclusion")
    else:
        raise PeriodicError("terminal artefact owner attempt is not active or complete")
    release_tpu_gate.verify_master_history(str(collector_sha))
    if (
        record.get("terminal") is not True
        or record.get("source_run_id") != source_run_id
        or record.get("source_attempt") != source_attempt
        or record.get("head_sha") != commit
        or record.get("collector_sha") != collector_sha
        or record.get("binding") != binding
        or record.get("kernel_id") != kernel_id
        or type(record.get("submitted_version")) is not int
        or record["submitted_version"] != 1
        or type(record.get("requested_version")) is not int
        or record["requested_version"] != 1
        or record.get("version_verified") is not True
        or artifact.get("name") != name
    ):
        raise PeriodicError(
            "terminal artefact does not match its exact source and collector"
        )
    outcome = record.get("outcome")
    remote_status = record.get("remote_status")
    if outcome == "success":
        if remote_status not in kaggle_ci.TERMINAL_SUCCESS:
            raise PeriodicError(
                "success terminal record has no successful remote status"
            )
    elif outcome == "remote_failure":
        if remote_status not in kaggle_ci.TERMINAL_FAILURE:
            raise PeriodicError("failure terminal record has no failed remote status")
    else:
        raise PeriodicError("terminal artefact outcome is not terminal")
    if run_status in _ACTIVE_RUN_STATUSES or require_job_proof:
        if not _collector_job_completed(
            collector_id,
            collector_sha,
            collector_attempt,
            source_run_id,
            source_attempt,
            cast(str, outcome),
            budget,
        ):
            # A marker can become visible before its upload step and job finish.
            # Marker-bound publication candidates also require native collect-job
            # evidence after the owner run completes; otherwise a failed marker
            # upload could leave an orphan proof that suppresses future retries.
            return None
    return record


def _has_terminal(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    source_dir: Path,
    budget: _Budget,
) -> dict[str, object] | None:
    source = collection_gate.read_json(source_dir / "collection-source.json")
    binding = release_tpu_gate.validate_binding(
        collection_gate.read_json(source_dir / "binding.json")
    )
    state = collection_gate.read_json(source_dir / "kaggle-controller-state.json")
    name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{commit}"
    artifacts = _terminal_artifacts(int(source_run_id), name, budget)
    if not artifacts:
        return None
    if source.get("head_sha") != commit:
        raise PeriodicError("prepared source SHA does not match the terminal lookup")
    publication_capable = set(source) == (
        collection_gate.SOURCE_BASE_FIELDS | collection_gate.SOURCE_CANDIDATE_FIELDS
    )
    kernel_id = state.get("kernel_id")
    if not isinstance(kernel_id, str):
        raise PeriodicError("original controller kernel identity is malformed")
    verified: list[dict[str, object]] = []
    for artifact in artifacts:
        record = _read_terminal(
            artifact,
            source_run_id=source_run_id,
            source_attempt=source_attempt,
            commit=commit,
            binding=binding,
            kernel_id=kernel_id,
            budget=budget,
            require_job_proof=publication_capable,
        )
        if record is None:
            continue
        collector_run = record.get("collector_run_id")
        collector_attempt = record.get("collector_attempt")
        collector_sha = record.get("collector_sha")
        match = (
            re.fullmatch(r"([1-9][0-9]*)\.([1-9][0-9]*)", collector_run)
            if isinstance(collector_run, str)
            else None
        )
        if (
            match is None
            or not isinstance(collector_attempt, str)
            or not isinstance(collector_sha, str)
        ):
            raise PeriodicError("terminal artefact collector identity is malformed")
        collector_id = int(match.group(1))
        if match.group(2) != collector_attempt:
            raise PeriodicError("terminal artefact collector attempt is inconsistent")
        collection_artifact = _collection_artifact(
            source_run_id=source_run_id,
            source_attempt=source_attempt,
            commit=commit,
            collector_id=collector_id,
            collector_attempt=collector_attempt,
            collector_sha=collector_sha,
            budget=budget,
        )
        if collection_artifact is not None:
            verified.append(record)
    return _coalesce_verified_terminals(verified)


def verified_terminal_artifacts(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    binding: Mapping[str, object],
    kernel_id: str,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Return exact terminal marker artefacts with native collect-job proof.

    Completion callers may use the returned metadata to pin one marker to a
    previously validated collection receipt. Equivalent markers are returned
    together; conflicting terminal results fail closed.
    """
    if (
        re.fullmatch(r"[1-9][0-9]{0,19}", source_run_id) is None
        or re.fullmatch(r"[1-9][0-9]{0,19}", source_attempt) is None
        or not accelerator_gate.SHA_RE.fullmatch(commit)
        or not kernel_id
    ):
        raise PeriodicError("terminal lookup source identity is malformed")
    try:
        verified_binding = release_tpu_gate.validate_binding(dict(binding))
    except release_tpu_gate.ReleaseGateError as error:
        raise PeriodicError("terminal lookup binding is invalid") from error

    budget = _Budget()
    name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{commit}"
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    records: list[dict[str, object]] = []
    with _bounded_github_calls(budget):
        artifacts = _terminal_artifacts(int(source_run_id), name, budget)
        for artifact in artifacts:
            if artifact.get("name") != name:
                raise PeriodicError("terminal marker artefact name changed")
            record = _read_terminal(
                artifact,
                source_run_id=source_run_id,
                source_attempt=source_attempt,
                commit=commit,
                binding=verified_binding,
                kernel_id=kernel_id,
                budget=budget,
                require_job_proof=True,
            )
            if record is None:
                continue
            records.append(record)
            pairs.append((artifact, record))

    _ = _coalesce_verified_terminals(records)
    return pairs


def _candidate_batch(
    candidates: list[dict[str, str]], collector_run_number: int
) -> list[dict[str, str]]:
    """Choose a deterministic rotating slice so capped backlogs still drain."""
    if len(candidates) <= MAX_CANDIDATES:
        return candidates
    if type(collector_run_number) is not int or collector_run_number <= 0:
        raise PeriodicError("collector run number is malformed")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            int(candidate["source_run_id"]),
            int(candidate["source_attempt"]),
            candidate["commit"],
        ),
    )
    rotation = (collector_run_number - 1) % len(ordered)
    return [
        ordered[(rotation + index) % len(ordered)] for index in range(MAX_CANDIDATES)
    ]


def _write_json(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise PeriodicError("periodic output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(data) > MAX_REPORT_BYTES:
        raise PeriodicError("periodic report exceeded its size limit")
    _ = path.write_bytes(data)


def _write_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            if "\n" in name or "\n" in value:
                raise PeriodicError("GitHub output value contains a newline")
            _ = stream.write(f"{name}={value}\n")


def _write_summary(outcome: str, detail: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            _ = stream.write(f"## Periodic TPU collection: {outcome}\n\n{detail}\n")


def discover(output_dir: Path, github_output: Path) -> dict[str, object]:
    _ = _require_context()
    run_number = os.environ.get("GITHUB_RUN_NUMBER", "")
    if re.fullmatch(r"[1-9][0-9]*", run_number) is None:
        raise PeriodicError("GITHUB_RUN_NUMBER must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=False)
    budget = _Budget()
    candidates: list[dict[str, str]] = []
    scanned = 0
    with _bounded_github_calls(budget):
        runs = _list_recent_runs(budget)
        for run in runs:
            budget.check()
            conclusion = run.get("conclusion")
            if (
                run.get("status") != "completed"
                or not isinstance(conclusion, str)
                or conclusion not in {"failure", "success"}
            ):
                continue
            _latest_source_attempt(run, budget)
            if run.get("event") == "workflow_dispatch":
                branch = run.get("head_branch")
                if not isinstance(branch, str):
                    raise PeriodicError("manual source run branch metadata is missing")
                if branch != "master":
                    continue
            run_id, attempt = str(run["id"]), str(run["run_attempt"])
            commit = str(run["head_sha"])
            try:
                with tempfile.TemporaryDirectory(
                    prefix="jaxrenderer-periodic-source-"
                ) as temp:
                    source_dir = Path(temp) / "source"
                    github_out = Path(temp) / "github-output"
                    _ = collection_gate.prepare(
                        run_id, attempt, commit, source_dir, github_out
                    )
                    scanned += 1
                    terminal = _has_terminal(
                        run_id, attempt, commit, source_dir, budget
                    )
                    if terminal is None:
                        candidates.append(
                            {
                                "source_run_id": run_id,
                                "source_attempt": attempt,
                                "commit": commit,
                            }
                        )
            except collection_gate.IneligibleSource:
                continue
    eligible_count = len(candidates)
    candidates = _candidate_batch(candidates, int(run_number))
    matrix: dict[str, object] = {"include": candidates}
    report: dict[str, object] = {
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cutoff": (datetime.now(timezone.utc) - MAX_AGE)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "inventory_count": len(runs),
        "verified_source_count": scanned,
        "has_sources": bool(candidates),
        "matrix": matrix,
    }
    _write_json(output_dir / "discovery.json", report)
    _write_outputs(
        github_output,
        {
            "has_sources": "true" if candidates else "false",
            "matrix": json.dumps(matrix, separators=(",", ":")),
        },
    )
    _write_summary(
        "ready" if candidates else "no pending sources",
        f"Verified {scanned} source run(s); {len(candidates)} of {eligible_count} "
        f"need collection in this batch (maximum {MAX_CANDIDATES}).",
    )
    return report


def _terminal_record(
    source_dir: Path,
    report: dict[str, object],
    outcome: str,
    remote_status: str,
) -> dict[str, object]:
    source = collection_gate.read_json(source_dir / "collection-source.json")
    binding = release_tpu_gate.validate_binding(
        collection_gate.read_json(source_dir / "binding.json")
    )
    state = collection_gate.read_json(source_dir / "kaggle-controller-state.json")
    collector_run = str(source["collector_run_id"])
    match = re.fullmatch(r"([1-9][0-9]*)\.([1-9][0-9]*)", collector_run)
    if match is None:
        raise PeriodicError("collector run identity is malformed")
    record: dict[str, object] = {
        "terminal": True,
        "outcome": outcome,
        "source_run_id": str(source["source_run_id"]),
        "source_attempt": str(source["source_attempt"]),
        "head_sha": str(source["head_sha"]),
        "collector_run_id": collector_run,
        "collector_attempt": match.group(2),
        "collector_sha": str(source["collector_sha"]),
        "binding": binding,
        "kernel_id": state["kernel_id"],
        "submitted_version": 1,
        "requested_version": 1,
        "version_verified": True,
        "remote_status": remote_status,
    }
    if set(record) != TERMINAL_FIELDS:
        raise PeriodicError("terminal record did not match its frozen schema")
    if (
        report.get("binding") != binding
        or report.get("kernel_id") != state["kernel_id"]
    ):
        raise PeriodicError("provider report identity differs from the original source")
    return record


def collect_source(
    source_run: str,
    source_attempt: str,
    commit: str,
    output_dir: Path,
    github_output: Path,
) -> tuple[dict[str, object], int]:
    _ = _require_context()
    report: dict[str, object]
    run_id = collection_gate.positive_id(source_run)
    attempt = collection_gate.positive_id(source_attempt)
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise PeriodicError("source commit must be a full lowercase SHA")
    output_dir.mkdir(parents=True, exist_ok=False)
    budget = _Budget()
    with _bounded_github_calls(budget):
        if _source_expired(source_run, source_attempt, commit, budget):
            report = {"terminal": False, "outcome": "expired"}
            _write_json(output_dir / "periodic-collection.json", report)
            _write_outputs(github_output, {"terminal": "false", "outcome": "expired"})
            _write_summary(
                "expired",
                "The original source attempt is older than seven days; no provider query was made.",
            )
            return report, 0
        source_dir = output_dir / "source"
        with tempfile.TemporaryDirectory(prefix="jaxrenderer-periodic-output-") as temp:
            _ = collection_gate.prepare(
                str(run_id),
                str(attempt),
                commit,
                source_dir,
                Path(temp) / "github-output",
            )
        existing = _has_terminal(str(run_id), str(attempt), commit, source_dir, budget)
        if existing is not None:
            report = {"terminal": False, "outcome": "already-terminal"}
            _write_json(output_dir / "periodic-collection.json", report)
            _write_outputs(
                github_output, {"terminal": "false", "outcome": "already-terminal"}
            )
            _write_summary(
                "already terminal",
                "The original source already has a verified terminal record.",
            )
            return report, 0

        if _source_expired(source_run, source_attempt, commit, budget):
            report = {"terminal": False, "outcome": "expired"}
            _write_json(output_dir / "periodic-collection.json", report)
            _write_outputs(github_output, {"terminal": "false", "outcome": "expired"})
            _write_summary(
                "expired",
                "The original source attempt aged out before collection; no provider query was made.",
            )
            return report, 0

        binding = release_tpu_gate.validate_binding(
            collection_gate.read_json(source_dir / "binding.json")
        )
        provider_dir = output_dir / "collected"
        provider_report = kaggle_collect.collect(
            binding, source_dir / "kaggle-controller-state.json", provider_dir
        )
        outcome = provider_report.get("outcome")
        remote_status = provider_report.get("remote_status")
        if outcome == "pending":
            report = {"terminal": False, "outcome": "pending"}
            _write_json(output_dir / "periodic-collection.json", report)
            _write_outputs(github_output, {"terminal": "false", "outcome": "pending"})
            _write_summary(
                "pending",
                "The original TPU kernel is still queued or active; a later schedule may check it again.",
            )
            return report, 0
        if outcome == "success":
            _ = collection_gate.finish(source_dir, provider_dir, True)
            if not isinstance(remote_status, str):
                raise PeriodicError("successful provider report has no terminal status")
            terminal = _terminal_record(
                source_dir, provider_report, "success", remote_status
            )
            exit_code = 0
        elif outcome == "remote_failure":
            _, original_binding, state = collection_gate.revalidate(source_dir)
            if (
                provider_report.get("success") is not False
                or provider_report.get("binding") != original_binding
                or provider_report.get("version_verified") is not True
                or type(provider_report.get("submitted_version")) is not int
                or provider_report["submitted_version"] != 1
                or type(provider_report.get("requested_version")) is not int
                or provider_report["requested_version"] != 1
                or provider_report.get("kernel_id") != state.get("kernel_id")
                or not isinstance(remote_status, str)
                or remote_status not in kaggle_ci.TERMINAL_FAILURE
            ):
                raise PeriodicError(
                    "remote failure was not fully version and identity verified"
                )
            terminal = _terminal_record(
                source_dir, provider_report, "remote_failure", remote_status
            )
            exit_code = 1
        else:
            raise PeriodicError(
                "provider collection did not produce a verified terminal result"
            )
        _write_json(output_dir / "periodic-terminal.json", terminal)
        report = {"terminal": True, "outcome": str(outcome)}
        _write_json(output_dir / "periodic-collection.json", report)
        _write_outputs(
            github_output,
            {"terminal": "true", "outcome": str(outcome)},
        )
        _write_summary(
            str(outcome),
            f"Source `{run_id}.{attempt}` at `{commit}` reached verified remote status `{remote_status}`.",
        )
        return report, exit_code


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    discover_parser = commands.add_parser("discover")
    _ = discover_parser.add_argument("--output-dir", type=Path, required=True)
    _ = discover_parser.add_argument("--github-output", type=Path, required=True)
    collect_parser = commands.add_parser("collect")
    _ = collect_parser.add_argument("--source-run", required=True)
    _ = collect_parser.add_argument("--source-attempt", required=True)
    _ = collect_parser.add_argument("--commit", required=True)
    _ = collect_parser.add_argument("--output-dir", type=Path, required=True)
    _ = collect_parser.add_argument("--github-output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "discover":
            _ = discover(args.output_dir, args.github_output)
            return 0
        _, exit_code = collect_source(
            args.source_run,
            args.source_attempt,
            args.commit,
            args.output_dir,
            args.github_output,
        )
        return exit_code
    except (
        PeriodicError,
        collection_gate.CollectionGateError,
        accelerator_gate.GateError,
        release_tpu_gate.ReleaseGateError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        message = str(error)
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "KAGGLE_API_TOKEN"):
            token = os.environ.get(name, "")
            if token:
                message = message.replace(token, "[redacted]")
        print(f"Periodic TPU collection: {message[:400]}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
