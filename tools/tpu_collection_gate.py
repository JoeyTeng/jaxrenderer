"""Recover delayed TPU evidence without granting a release gate credit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import zipfile

from tools import accelerator_gate, kaggle_ci, kaggle_collect, release_tpu_gate

REPOSITORY = release_tpu_gate.REPOSITORY
SOURCE_WORKFLOW = ".github/workflows/pypi.yml"
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
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
SOURCE_FIELDS = {
    "source_run_id",
    "source_attempt",
    "head_sha",
    "collector_run_id",
    "collector_sha",
    "binding_artifact_id",
    "controller_artifact_id",
}


class CollectionGateError(RuntimeError):
    """A source or collection receipt failed validation."""


class IneligibleSource(CollectionGateError):
    """A complete source inventory proves this run is not a TPU-only timeout."""


def _positive_id(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]{0,19}", value):
        raise CollectionGateError("run ID and attempt must be positive integers")
    return int(value)


def _read_json(path: Path) -> dict[str, object]:
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
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    accelerator_gate._atomic_write_json(path, value)


def _context() -> str:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event == "workflow_dispatch":
        accelerator_gate._require_dispatch_context()
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
    return accelerator_gate._run_id_from_env()


def _source_snapshot(run_id: int, attempt: int, commit: str) -> dict[str, object]:
    """Require one failed publishing attempt whose only gate failure was TPU."""
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise CollectionGateError("source commit must be a full lowercase SHA")
    route = f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}"
    run = accelerator_gate._gh_api(route)
    if (
        not isinstance(run, dict)
        or type(run.get("id")) is not int
        or run["id"] != run_id
        or type(run.get("run_attempt")) is not int
        or run["run_attempt"] != attempt
        or run.get("head_sha") != commit
        or run.get("path") != SOURCE_WORKFLOW
        or run.get("event") not in {"workflow_dispatch", "release"}
        or run.get("status") != "completed"
        or run.get("conclusion") != "failure"
        or not isinstance(run.get("repository"), dict)
        or str(run["repository"].get("full_name", "")).casefold()
        != REPOSITORY.casefold()
        or not isinstance(run.get("head_repository"), dict)
        or str(run["head_repository"].get("full_name", "")).casefold()
        != REPOSITORY.casefold()
    ):
        raise CollectionGateError(
            "source is not the exact completed publishing attempt"
        )
    release_tpu_gate._verify_master_history(commit)
    page = accelerator_gate._gh_api(f"{route}/jobs?per_page=100")
    if (
        not isinstance(page, dict)
        or type(page.get("total_count")) is not int
        or not isinstance(page.get("jobs"), list)
        or page["total_count"] != len(page["jobs"])
        or page["total_count"] > 100
    ):
        raise CollectionGateError("source job inventory is incomplete or too large")
    jobs: dict[str, dict[str, object]] = {}
    for job in page["jobs"]:
        if not isinstance(job, dict) or not isinstance(job.get("name"), str):
            raise CollectionGateError("source job metadata is malformed")
        name = job["name"]
        if name in jobs or job.get("run_id") != run_id:
            raise CollectionGateError("source job identity is ambiguous")
        jobs[name] = job
    for name in REQUIRED_JOBS:
        if name not in jobs:
            raise CollectionGateError(f"source required job is missing: {name}")
        job = jobs[name]
        if job.get("status") != "completed":
            raise CollectionGateError(f"source required job is incomplete: {name}")
        if job.get("conclusion") in {"failure", "cancelled", "skipped"}:
            raise IneligibleSource(f"source required job did not pass: {name}")
        if job.get("conclusion") != "success":
            raise CollectionGateError(f"source required job did not pass: {name}")
    tpu = jobs.get("tpu / tpu", {})
    if not tpu:
        raise CollectionGateError("source TPU job metadata is missing")
    if tpu.get("status") != "completed":
        raise CollectionGateError("source TPU job metadata is incomplete")
    if tpu.get("conclusion") != "failure":
        if tpu.get("conclusion") in {"success", "skipped", "cancelled"}:
            raise IneligibleSource("source TPU job did not fail")
        raise CollectionGateError("source TPU job conclusion is unrecognised")
    for name, job in jobs.items():
        if name == "tpu / tpu":
            continue
        if job.get("status") != "completed":
            raise CollectionGateError("source contains incomplete job metadata")
        if job.get("conclusion") in {"failure", "cancelled"}:
            raise IneligibleSource("source has another failed job")
        if job.get("conclusion") not in {"success", "skipped"}:
            raise CollectionGateError("source has an unrecognised job conclusion")
    return run


def _artifact_metadata(run_id: int, name: str, commit: str) -> dict[str, object]:
    page = accelerator_gate._gh_api(
        f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100"
    )
    if (
        not isinstance(page, dict)
        or type(page.get("total_count")) is not int
        or not isinstance(page.get("artifacts"), list)
        or page["total_count"] != len(page["artifacts"])
        or page["total_count"] > 100
    ):
        raise CollectionGateError(
            "source artefact inventory is incomplete or too large"
        )
    matches = [
        item
        for item in page["artifacts"]
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(matches) != 1:
        raise CollectionGateError("source must have exactly one named artefact")
    item = matches[0]
    owner = item.get("workflow_run")
    if (
        type(item.get("id")) is not int
        or item["id"] <= 0
        or type(item.get("size_in_bytes")) is not int
        or not 0 < item["size_in_bytes"] <= MAX_ARCHIVE_BYTES
        or item.get("expired") is not False
        or not isinstance(owner, dict)
        or owner.get("id") != run_id
        or owner.get("head_sha") != commit
    ):
        raise CollectionGateError(
            "source artefact identity, size or availability is invalid"
        )
    return item


def _download_member(artifact_id: int, member_name: str, destination: Path) -> None:
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
        env.pop(key, None)
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
            destination.write_bytes(data)
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
    source_dir: Path, source: dict[str, object]
) -> dict[str, object]:
    source_identity = f"{source['source_run_id']}.{source['source_attempt']}"
    binding = release_tpu_gate._validate_binding(
        _read_json(source_dir / "binding.json")
    )
    expected = {
        "kind": "release",
        "head_repository": REPOSITORY,
        "head_sha": source["head_sha"],
        "run_id": f"{source['source_run_id']}.{source['source_attempt']}",
    }
    if binding != expected:
        raise CollectionGateError(
            f"source {source_identity} binding does not match the original attempt"
        )
    state_path = source_dir / "kaggle-controller-state.json"
    state = _read_json(state_path)
    state_fields = set(state)
    if state_fields == kaggle_ci.LEGACY_CONTROLLER_STATE_FIELDS:
        if state.get("binding") != binding:
            raise CollectionGateError(
                f"source {source_identity} legacy controller binding is invalid"
            )
        kernel_id = state.get("kernel_id")
        if not isinstance(kernel_id, str):
            raise CollectionGateError(
                f"source {source_identity} legacy controller identity is invalid"
            )
        kernel_match = kaggle_collect.KERNEL_ID_RE.fullmatch(kernel_id)
        if kernel_match is None:
            raise CollectionGateError(
                f"source {source_identity} legacy controller identity is invalid"
            )
        username = kernel_match.group("username")
        try:
            kaggle_collect._controller_identity(binding, state_path, username)
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
    if outcome in {"queue_timeout", "execution_timeout"}:
        if (
            type(state.get("submitted_version")) is not int
            or state["submitted_version"] != 1
        ):
            raise CollectionGateError(
                f"source {source_identity} timed-out controller did not submit version 1"
            )
        if (
            state.get("status")
            not in kaggle_ci.IN_PROGRESS | kaggle_ci.TERMINAL_SUCCESS
        ):
            raise CollectionGateError(
                f"source {source_identity} timed-out controller status is not eligible for collection"
            )
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
    return binding


def prepare(
    source_run: str,
    source_attempt: str,
    commit: str,
    output_dir: Path,
    github_output: Path,
) -> dict[str, object]:
    collector = _context()
    run_id, attempt = _positive_id(source_run), _positive_id(source_attempt)
    if collector == f"{run_id}.{attempt}":
        raise CollectionGateError("collection must use a separate workflow attempt")
    _source_snapshot(run_id, attempt, commit)
    binding_artifact = _artifact_metadata(
        run_id, f"tpu-release-binding-{attempt}", commit
    )
    controller_artifact = _artifact_metadata(
        run_id, f"release-tpu-{commit}-{attempt}", commit
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    source: dict[str, object] = {
        "source_run_id": run_id,
        "source_attempt": attempt,
        "head_sha": commit,
        "collector_run_id": collector,
        "collector_sha": os.environ["GITHUB_SHA"],
        "binding_artifact_id": binding_artifact["id"],
        "controller_artifact_id": controller_artifact["id"],
    }
    _download_member(
        binding_artifact["id"], "tpu-release-binding.json", output_dir / "binding.json"
    )
    _download_member(
        controller_artifact["id"],
        "kaggle-controller-state.json",
        output_dir / "kaggle-controller-state.json",
    )
    _validate_source_files(output_dir, source)
    _write_json(output_dir / "collection-source.json", source)
    with github_output.open("a", encoding="utf-8") as stream:
        stream.write(
            f"head_sha={commit}\nsource_run_id={run_id}\nsource_attempt={attempt}\n"
        )
    return source


def revalidate(
    source_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Recheck the exact prepared source attempt and its original artefact IDs."""
    collector = _context()
    source = _read_json(source_dir / "collection-source.json")
    if (
        set(source) != SOURCE_FIELDS
        or source.get("collector_run_id") != collector
        or source.get("collector_sha") != os.environ["GITHUB_SHA"]
    ):
        raise CollectionGateError(
            "collection source receipt belongs to another collector"
        )
    run_id = _positive_id(str(source["source_run_id"]))
    attempt = _positive_id(str(source["source_attempt"]))
    commit = source["head_sha"]
    if not isinstance(commit, str):
        raise CollectionGateError("source SHA is malformed")
    _source_snapshot(run_id, attempt, commit)
    for field, name in (
        ("binding_artifact_id", f"tpu-release-binding-{attempt}"),
        ("controller_artifact_id", f"release-tpu-{commit}-{attempt}"),
    ):
        if _artifact_metadata(run_id, name, commit)["id"] != source[field]:
            raise CollectionGateError("original source artefact identity changed")
    binding = _validate_source_files(source_dir, source)
    state = _read_json(source_dir / "kaggle-controller-state.json")
    return source, binding, state


def finish(
    source_dir: Path, output_dir: Path, provider_success: bool
) -> dict[str, object]:
    source, binding, state = revalidate(source_dir)
    collector = str(source["collector_run_id"])
    report = _read_json(output_dir / "kaggle-collection-report.json")
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
    release_tpu_gate._read_result(output_dir / "result.json", binding, "tpu")
    receipt = {
        "binding": binding,
        "collector_run_id": collector,
        "collector_sha": source["collector_sha"],
        "kernel_id": state["kernel_id"],
        "submitted_version": 1,
        "success": True,
        "validation_only": True,
        "release_gate_credit": False,
        "source_conclusion": "failure",
    }
    _write_json(output_dir / "collection-receipt.json", receipt)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        commit = str(source["head_sha"])
        with Path(summary_path).open("a", encoding="utf-8") as stream:
            stream.write(
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
    first.add_argument("--source-run", required=True)
    first.add_argument("--source-attempt", required=True)
    first.add_argument("--commit", required=True)
    first.add_argument("--output-dir", type=Path, required=True)
    first.add_argument("--github-output", type=Path, required=True)
    last = commands.add_parser("finish")
    last.add_argument("--source-dir", type=Path, required=True)
    last.add_argument("--output-dir", type=Path, required=True)
    last.add_argument("--provider-success", choices=("true", "false"), required=True)
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
