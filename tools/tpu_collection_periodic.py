"""Discover and collect delayed TPU results without changing release outcomes."""

from __future__ import annotations

import argparse
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
from typing import Any, Iterator
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


class PeriodicError(RuntimeError):
    """A discovery or collection run could not be verified safely."""


class _Budget:
    def __init__(self) -> None:
        self.deadline = time.monotonic() + MAX_RUNTIME_SECONDS

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise PeriodicError("periodic collection exceeded its runtime budget")


@contextmanager
def _bounded_github_calls(budget: _Budget) -> Iterator[None]:
    """Check the aggregate deadline before each already-bounded GitHub operation."""
    original_api = accelerator_gate._gh_api
    original_download = collection_gate._download_member

    def checked_api(route: str, body: dict[str, object] | None = None) -> Any:
        budget.check()
        return original_api(route, body)

    def checked_download(artifact_id: int, member: str, destination: Path) -> None:
        budget.check()
        original_download(artifact_id, member, destination)
        budget.check()

    accelerator_gate._gh_api = checked_api
    collection_gate._download_member = checked_download
    try:
        yield
    finally:
        accelerator_gate._gh_api = original_api
        collection_gate._download_member = original_download


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
    return accelerator_gate._run_id_from_env()


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
    repo = run.get("repository")
    head_repo = run.get("head_repository")
    if (
        type(run.get("id")) is not int
        or run["id"] <= 0
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] <= 0
        or not isinstance(run.get("head_sha"), str)
        or not accelerator_gate.SHA_RE.fullmatch(run["head_sha"])
        or run.get("path") != (SOURCE_WORKFLOW if source else WORKFLOW_PATH)
        or not isinstance(repo, dict)
        or str(repo.get("full_name", "")).casefold() != REPOSITORY.casefold()
        or not isinstance(head_repo, dict)
        or str(head_repo.get("full_name", "")).casefold() != REPOSITORY.casefold()
    ):
        raise PeriodicError("GitHub workflow run identity is not canonical")
    if source and run.get("event") not in {"release", "workflow_dispatch"}:
        raise PeriodicError("source workflow event is not eligible for collection")
    if not source and (
        run.get("event") not in {"schedule", "workflow_dispatch"}
        or run.get("head_branch") != "master"
    ):
        raise PeriodicError("terminal artifact owner is not a master collector run")
    return run


def _latest_source_attempt(run: dict[str, object], budget: _Budget) -> None:
    run_id = run["id"]
    budget.check()
    latest = accelerator_gate._gh_api(f"repos/{REPOSITORY}/actions/runs/{run_id}")
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
    run = _identity_run(accelerator_gate._gh_api(route), source=True)
    if (
        str(run["id"]) != source_run_id
        or str(run["run_attempt"]) != source_attempt
        or run.get("head_sha") != commit
        or run.get("status") != "completed"
        or run.get("conclusion") != "failure"
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
    page = accelerator_gate._gh_api(
        f"repos/{REPOSITORY}/actions/workflows/pypi.yml/runs?{query}"
    )
    if (
        not isinstance(page, dict)
        or type(page.get("total_count")) is not int
        or page["total_count"] < 0
        or page["total_count"] > MAX_RUNS
        or not isinstance(page.get("workflow_runs"), list)
        or page["total_count"] != len(page["workflow_runs"])
    ):
        raise PeriodicError("recent source run inventory is incomplete or too large")
    runs: list[dict[str, object]] = []
    for item in page["workflow_runs"]:
        run = _identity_run(item, source=True)
        created_at = _parse_timestamp(run.get("created_at"))
        if created_at > cutoff:
            runs.append(run)
    return runs


def _terminal_artifact(
    source_run: int, artifact_name: str, budget: _Budget
) -> dict[str, object] | None:
    query = urlencode({"name": artifact_name, "per_page": str(MAX_RUNS), "page": "1"})
    budget.check()
    page = accelerator_gate._gh_api(f"repos/{REPOSITORY}/actions/artifacts?{query}")
    if (
        not isinstance(page, dict)
        or type(page.get("total_count")) is not int
        or page["total_count"] < 0
        or page["total_count"] > MAX_RUNS
        or not isinstance(page.get("artifacts"), list)
        or page["total_count"] != len(page["artifacts"])
    ):
        raise PeriodicError("terminal artifact inventory is incomplete or too large")
    matches = [
        item
        for item in page["artifacts"]
        if isinstance(item, dict) and item.get("name") == artifact_name
    ]
    if len(matches) > 1:
        raise PeriodicError("multiple terminal artifacts have the same source identity")
    if not matches:
        return None
    item = matches[0]
    owner = item.get("workflow_run")
    if (
        type(item.get("id")) is not int
        or item["id"] <= 0
        or type(item.get("size_in_bytes")) is not int
        or not 0 < item["size_in_bytes"] <= collection_gate.MAX_ARCHIVE_BYTES
        or item.get("expired") is not False
        or not isinstance(owner, dict)
        or type(owner.get("id")) is not int
        or owner["id"] <= 0
        or not isinstance(owner.get("head_sha"), str)
        or not accelerator_gate.SHA_RE.fullmatch(owner["head_sha"])
    ):
        raise PeriodicError(
            "terminal artifact identity, size or availability is invalid"
        )
    return item


def _read_terminal(
    artifact: dict[str, object],
    *,
    source_run_id: str,
    source_attempt: str,
    commit: str,
    binding: dict[str, object],
    kernel_id: str,
    budget: _Budget,
) -> dict[str, object]:
    owner = artifact["workflow_run"]
    assert isinstance(owner, dict)
    collector_id = owner["id"]
    collector_sha = owner["head_sha"]
    name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{commit}"
    with tempfile.TemporaryDirectory(prefix="jaxrenderer-periodic-terminal-") as temp:
        path = Path(temp) / "periodic-terminal.json"
        collection_gate._download_member(artifact["id"], "periodic-terminal.json", path)
        record = collection_gate._read_json(path)
    if set(record) != TERMINAL_FIELDS:
        raise PeriodicError("terminal artifact has an unsupported schema")
    collector_attempt = record.get("collector_attempt")
    if (
        not isinstance(collector_attempt, str)
        or re.fullmatch(r"[1-9][0-9]*", collector_attempt) is None
        or record.get("collector_run_id") != f"{collector_id}.{collector_attempt}"
        or record.get("collector_sha") != collector_sha
    ):
        raise PeriodicError(
            "terminal artifact collector identity does not match its owner"
        )
    route = (
        f"repos/{REPOSITORY}/actions/runs/{collector_id}/attempts/{collector_attempt}"
    )
    budget.check()
    run = _identity_run(accelerator_gate._gh_api(route), source=False)
    if (
        run["id"] != collector_id
        or run["head_sha"] != collector_sha
        or run.get("run_attempt") != int(collector_attempt)
        or run.get("status") != "completed"
        or run.get("conclusion") not in {"success", "failure"}
    ):
        raise PeriodicError("terminal artifact owner attempt is not complete and exact")
    release_tpu_gate._verify_master_history(str(collector_sha))
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
            "terminal artifact does not match its exact source and collector"
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
        raise PeriodicError("terminal artifact outcome is not terminal")
    return record


def _has_terminal(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    source_dir: Path,
    budget: _Budget,
) -> dict[str, object] | None:
    source = collection_gate._read_json(source_dir / "collection-source.json")
    binding = release_tpu_gate._validate_binding(
        collection_gate._read_json(source_dir / "binding.json")
    )
    state = collection_gate._read_json(source_dir / "kaggle-controller-state.json")
    name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{commit}"
    artifact = _terminal_artifact(int(source_run_id), name, budget)
    if artifact is None:
        return None
    if source.get("head_sha") != commit:
        raise PeriodicError("prepared source SHA does not match the terminal lookup")
    kernel_id = state.get("kernel_id")
    if not isinstance(kernel_id, str):
        raise PeriodicError("original controller kernel identity is malformed")
    return _read_terminal(
        artifact,
        source_run_id=source_run_id,
        source_attempt=source_attempt,
        commit=commit,
        binding=binding,
        kernel_id=kernel_id,
        budget=budget,
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise PeriodicError("periodic output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if len(data) > MAX_REPORT_BYTES:
        raise PeriodicError("periodic report exceeded its size limit")
    path.write_bytes(data)


def _write_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            if "\n" in name or "\n" in value:
                raise PeriodicError("GitHub output value contains a newline")
            stream.write(f"{name}={value}\n")


def _write_summary(outcome: str, detail: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(f"## Periodic TPU collection: {outcome}\n\n{detail}\n")


def discover(output_dir: Path, github_output: Path) -> dict[str, object]:
    _require_context()
    output_dir.mkdir(parents=True, exist_ok=False)
    budget = _Budget()
    candidates: list[dict[str, str]] = []
    scanned = 0
    with _bounded_github_calls(budget):
        runs = _list_recent_runs(budget)
        for run in runs:
            budget.check()
            if run.get("status") != "completed" or run.get("conclusion") != "failure":
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
                    collection_gate.prepare(
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
                        if len(candidates) > MAX_CANDIDATES:
                            raise PeriodicError(
                                "eligible TPU collection inventory exceeds its candidate limit"
                            )
            except collection_gate.IneligibleSource:
                continue
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
        f"Verified {scanned} source run(s); {len(candidates)} need collection.",
    )
    return report


def _terminal_record(
    source_dir: Path,
    report: dict[str, object],
    outcome: str,
    remote_status: str,
) -> dict[str, object]:
    source = collection_gate._read_json(source_dir / "collection-source.json")
    binding = release_tpu_gate._validate_binding(
        collection_gate._read_json(source_dir / "binding.json")
    )
    state = collection_gate._read_json(source_dir / "kaggle-controller-state.json")
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
    _require_context()
    run_id = collection_gate._positive_id(source_run)
    attempt = collection_gate._positive_id(source_attempt)
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
            collection_gate.prepare(
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

        binding = release_tpu_gate._validate_binding(
            collection_gate._read_json(source_dir / "binding.json")
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
            collection_gate.finish(source_dir, provider_dir, True)
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
    discover_parser.add_argument("--output-dir", type=Path, required=True)
    discover_parser.add_argument("--github-output", type=Path, required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--source-run", required=True)
    collect_parser.add_argument("--source-attempt", required=True)
    collect_parser.add_argument("--commit", required=True)
    collect_parser.add_argument("--output-dir", type=Path, required=True)
    collect_parser.add_argument("--github-output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "discover":
            discover(args.output_dir, args.github_output)
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
