"""Complete a frozen release after a delayed periodic TPU result succeeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import cast
import zipfile

from tools import (
    accelerator_gate,
    release_candidate,
    release_tpu_gate,
)
from tools import (
    tpu_collection_gate as collection_gate,
)
from tools import (
    tpu_collection_periodic as periodic,
)

REPOSITORY = release_tpu_gate.REPOSITORY
WORKFLOW_PATH = periodic.WORKFLOW_PATH
MANUAL_WORKFLOW_REF = (
    f"{REPOSITORY}/.github/workflows/complete-release.yml@refs/heads/master"
)
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_COLLECTION_BYTES = 100 * 1024 * 1024
MAX_DIST_MEMBER_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_MEMBERS = 10_000
MAX_METADATA_BYTES = 1024 * 1024
DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPLETION_FIELDS = {
    "schema_version",
    "repository",
    "source_run_id",
    "source_attempt",
    "source_head_sha",
    "collector_run_id",
    "collector_attempt",
    "collector_sha",
    "collector_artifact_id",
    "collector_artifact_name",
    "collector_artifact_digest",
    "collector_artifact_size",
    "binding_artifact_id",
    "controller_artifact_id",
    "candidate_artifact_id",
    "candidate_artifact_name",
    "candidate_artifact_digest",
    "marker_artifact_id",
    "marker_artifact_name",
    "marker_artifact_digest",
    "release_dist_artifact_id",
    "release_dist_artifact_name",
    "release_dist_artifact_digest",
    "release_dist_artifact_size",
    "candidate_mode",
    "candidate_sha256",
    "collection_sha256",
    "version",
    "head_sha",
    "publish",
    "completion_kind",
    "completion_run_id",
    "completion_sha",
}


class CompletionError(RuntimeError):
    """A completion receipt or original distribution failed validation."""


def _object(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompletionError(message)
    return cast(dict[str, object], value)


def _is_repository(value: object) -> bool:
    return isinstance(value, str) and value.casefold() == REPOSITORY.casefold()


def _read_json(
    path: Path, *, limit: int = collection_gate.MAX_JSON_BYTES
) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise CompletionError("completion evidence is not a bounded regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except CompletionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompletionError("could not read completion evidence") from error
    return _object(value, "completion evidence must be a JSON object")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_pair(run_id: object, attempt: object) -> str:
    if (
        type(run_id) is not int
        or run_id <= 0
        or type(attempt) is not int
        or attempt <= 0
    ):
        raise CompletionError("collector run identity is malformed")
    return f"{run_id}.{attempt}"


def _parse_run_pair(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]{0,19})\.([1-9][0-9]{0,19})", value)
    if match is None:
        raise CompletionError("collector run identity is malformed")
    return int(match.group(1)), int(match.group(2))


def _current_run_identity() -> str:
    try:
        value = accelerator_gate.run_id_from_env()
    except accelerator_gate.GateError as error:
        raise CompletionError("completion workflow run identity is invalid") from error
    return value


def _manual_context() -> str:
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_REF") != "refs/heads/master"
        or os.environ.get("GITHUB_REPOSITORY", "").casefold() != REPOSITORY.casefold()
        or os.environ.get("GITHUB_WORKFLOW_REF") != MANUAL_WORKFLOW_REF
    ):
        raise CompletionError(
            "manual completion is restricted to the canonical master workflow"
        )
    current_sha = os.environ.get("GITHUB_SHA", "")
    if not accelerator_gate.SHA_RE.fullmatch(current_sha):
        raise CompletionError("manual completion SHA is malformed")
    try:
        release_tpu_gate.verify_master_history(current_sha)
    except release_tpu_gate.ReleaseGateError as error:
        raise CompletionError(
            "manual completion is not on current master history"
        ) from error
    return _current_run_identity()


def _collector_context(
    collector_run: str | None, collector_attempt: str | None
) -> tuple[str, str, str, str]:
    if (collector_run is None) != (collector_attempt is None):
        raise CompletionError("collector run and attempt must be supplied together")
    if collector_run is None:
        try:
            collector_identity = periodic.require_context()
        except periodic.PeriodicError as error:
            raise CompletionError(
                "completion requires the canonical master collector"
            ) from error
        collector_id, collector_try = _parse_run_pair(collector_identity)
        collector_sha = os.environ.get("GITHUB_SHA", "")
        completion_kind = "automatic"
        completion_identity = _current_run_identity()
        if completion_identity != collector_identity:
            raise CompletionError(
                "current workflow is not the original collector attempt"
            )
    else:
        _ = _manual_context()
        if (
            not collector_run.isdigit()
            or int(collector_run) <= 0
            or collector_attempt is None
            or not collector_attempt.isdigit()
            or int(collector_attempt) <= 0
        ):
            raise CompletionError("historical collector identity is malformed")
        collector_id, collector_try = int(collector_run), int(collector_attempt)
        collector_identity = f"{collector_id}.{collector_try}"
        collector_sha = ""
        completion_kind = "manual"
        completion_identity = _current_run_identity()
    if not accelerator_gate.SHA_RE.fullmatch(collector_sha) and collector_sha:
        raise CompletionError("collector SHA is malformed")
    return collector_identity, collector_sha, completion_kind, completion_identity


def _artifact_metadata(
    run_id: int,
    name: str,
    commit: str,
    *,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    expected_attempt: int | None = None,
) -> dict[str, object]:
    """Use the collection gate's exact-run metadata checks with a larger dist cap."""
    try:
        value = collection_gate.artifact_metadata(
            run_id, name, commit, max_size_bytes=max_archive_bytes
        )
    except (collection_gate.CollectionGateError, TypeError) as error:
        raise CompletionError("original GitHub artefact metadata is invalid") from error
    metadata = _object(value, "original GitHub artefact metadata is malformed")
    digest = metadata.get("digest")
    size = metadata.get("size_in_bytes")
    owner = metadata.get("workflow_run")
    owner_attempt = (
        cast(dict[str, object], owner).get("run_attempt")
        if isinstance(owner, dict)
        else None
    )
    if (
        not isinstance(digest, str)
        or DIGEST_RE.fullmatch(digest) is None
        or type(size) is not int
        or not 0 < size <= max_archive_bytes
        or (owner_attempt is not None and owner_attempt != expected_attempt)
    ):
        raise CompletionError("original artefact digest or size is unavailable")
    return metadata


def _artifact_id(metadata: dict[str, object]) -> int:
    value = metadata.get("id")
    if type(value) is not int or value <= 0:
        raise CompletionError("original artefact identity is malformed")
    return value


def _artifact_digest(metadata: dict[str, object]) -> str:
    value = metadata.get("digest")
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise CompletionError("original artefact digest is malformed")
    return value


def _collector_run_identity(
    run_id: int,
    attempt: int,
    commit: str,
    *,
    require_completed_job: bool,
    source_run: int,
    source_attempt: int,
) -> None:
    route = f"repos/{REPOSITORY}/actions/runs/{run_id}"
    attempt_route = f"{route}/attempts/{attempt}"
    exact = _object(
        accelerator_gate.gh_api(attempt_route),
        "collector attempt metadata is malformed",
    )
    repository = exact.get("repository")
    head_repository = exact.get("head_repository")
    if (
        exact.get("id") != run_id
        or exact.get("run_attempt") != attempt
        or exact.get("path") != WORKFLOW_PATH
        or exact.get("event") not in {"schedule", "workflow_dispatch"}
        or exact.get("head_branch") != "master"
        or exact.get("head_sha") != commit
        or exact.get("status") != "completed"
        or not isinstance(repository, dict)
        or not _is_repository(cast(dict[str, object], repository).get("full_name"))
        or not isinstance(head_repository, dict)
        or not _is_repository(cast(dict[str, object], head_repository).get("full_name"))
    ):
        raise CompletionError("historical collector attempt is not canonical")
    try:
        release_tpu_gate.verify_master_history(commit)
    except release_tpu_gate.ReleaseGateError as error:
        raise CompletionError(
            "collector controller is not trusted master history"
        ) from error
    if not require_completed_job:
        return
    jobs = _object(
        accelerator_gate.gh_api(f"{attempt_route}/jobs?per_page=100"),
        "collector job inventory is malformed",
    )
    values = jobs.get("jobs")
    total = jobs.get("total_count")
    if not isinstance(values, list) or type(total) is not int or total > 100:
        raise CompletionError("collector job inventory is incomplete")
    expected_name = f"collect-{source_run}-{source_attempt}"
    job_values = cast(list[object], values)
    if total != len(job_values):
        raise CompletionError("collector job inventory is incomplete")
    matches: list[dict[str, object]] = []
    for job_value in job_values:
        if isinstance(job_value, dict):
            job = cast(dict[str, object], job_value)
            if job.get("name") == expected_name:
                matches.append(job)
    if (
        len(matches) != 1
        or matches[0].get("run_id") != run_id
        or matches[0].get("status") != "completed"
        or matches[0].get("conclusion") != "success"
    ):
        raise CompletionError("original collection job did not complete successfully")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise CompletionError("could not hash completion file") from error
    return digest.hexdigest()


def _collection_tree_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise CompletionError("periodic collection evidence is missing")
    entries: list[tuple[str, int, str]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CompletionError("periodic collection evidence contains a symlink")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            continue
        if not path.is_file():
            raise CompletionError("periodic collection contains a non-regular file")
        size = path.stat().st_size
        total += size
        if size > MAX_COLLECTION_BYTES or total > MAX_COLLECTION_BYTES:
            raise CompletionError("periodic collection exceeds its size limit")
        entries.append((relative, size, _sha256(path)))
        if len(entries) > 1000:
            raise CompletionError("periodic collection contains too many files")
    digest = hashlib.sha256()
    for relative, size, file_digest in entries:
        digest.update(f"{relative}\0{size}\0{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    _ = _collection_tree_digest(source)
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = shutil.copyfile(path, target, follow_symlinks=False)


def _download_archive(artifact_id: int, destination: Path, max_bytes: int) -> None:
    env = os.environ.copy()
    token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    if not token:
        raise CompletionError("GitHub token is required for artefact retrieval")
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
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))

    try:
        with destination.open("xb") as stream:
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip",
                ],
                stdout=stream,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=180,
                check=False,
                preexec_fn=limit_file,
            )
        if result.returncode:
            raise CompletionError("GitHub artefact download failed")
        if destination.stat().st_size > max_bytes:
            raise CompletionError("GitHub artefact exceeds its byte limit")
    except CompletionError:
        raise
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CompletionError(
            "could not retrieve the original GitHub artefact"
        ) from error


def _validate_zip_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CompletionError("distribution archive contains an unsafe path")


def _copy_dist_archive(
    archive_path: Path,
    destination: Path,
    *,
    version: str,
) -> list[dict[str, object]]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 2 or len({item.filename for item in members}) != 2:
                raise CompletionError(
                    "release artefact must contain one wheel and one sdist"
                )
            names: set[str] = set()
            records: list[dict[str, object]] = []
            total = 0
            for item in members:
                _validate_zip_name(item.filename)
                if "/" in item.filename or item.flag_bits & 0x1:
                    raise CompletionError(
                        "release artefact member is not a regular file"
                    )
                mode = item.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG} or item.is_dir():
                    raise CompletionError(
                        "release artefact member is not a regular file"
                    )
                if item.file_size <= 0 or item.file_size > MAX_DIST_MEMBER_BYTES:
                    raise CompletionError("release distribution exceeds its size limit")
                total += item.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise CompletionError(
                        "release distributions exceed their total size limit"
                    )
                name = item.filename
                if name.endswith(".whl"):
                    _validate_wheel_filename(name, version)
                elif name.endswith(".tar.gz"):
                    _validate_sdist_filename(name, version)
                else:
                    raise CompletionError(
                        "release artefact contains an unsupported file"
                    )
                names.add(name)
                target = destination / name
                digest = hashlib.sha256()
                copied = 0
                with archive.open(item) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > item.file_size or copied > MAX_DIST_MEMBER_BYTES:
                            raise CompletionError(
                                "release distribution exceeded its declared size"
                            )
                        digest.update(chunk)
                        _ = output.write(chunk)
                if copied != item.file_size:
                    raise CompletionError(
                        "release artefact member size is inconsistent"
                    )
                records.append(
                    {"name": name, "size": copied, "sha256": digest.hexdigest()}
                )
            if (
                len(names) != 2
                or not any(name.endswith(".whl") for name in names)
                or not any(name.endswith(".tar.gz") for name in names)
            ):
                raise CompletionError(
                    "release artefact lacks its wheel or source archive"
                )
            for record in records:
                path = destination / cast(str, record["name"])
                if path.suffix == ".whl":
                    _validate_wheel(path, version)
                else:
                    _validate_sdist(path, version)
            return sorted(records, key=lambda item: cast(str, item["name"]))
    except CompletionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, EOFError) as error:
        raise CompletionError("release artefact ZIP is invalid") from error


def _canonical(value: str) -> str:
    return re.sub(r"[-_.]+", ".", value).casefold()


def _validate_wheel_filename(name: str, version: str) -> None:
    if not name.endswith(".whl"):
        raise CompletionError("release wheel filename is invalid")
    parts = name[:-4].split("-")
    if (
        len(parts) not in {5, 6}
        or _canonical(parts[0]) != "jaxrenderer"
        or _canonical(parts[1]) != _canonical(version)
        or any(not part for part in parts[2:])
    ):
        raise CompletionError("release wheel does not match the frozen version")


def _validate_sdist_filename(name: str, version: str) -> None:
    if not name.endswith(".tar.gz"):
        raise CompletionError("release source archive filename is invalid")
    parts = name[: -len(".tar.gz")].split("-")
    if (
        len(parts) != 2
        or _canonical(parts[0]) != "jaxrenderer"
        or _canonical(parts[1]) != _canonical(version)
    ):
        raise CompletionError(
            "release source archive does not match the frozen version"
        )


def _metadata_headers(raw: bytes) -> dict[str, str]:
    if len(raw) > MAX_METADATA_BYTES:
        raise CompletionError("distribution metadata exceeds its size limit")
    headers: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line:
            break
        if line[:1].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.casefold()
        if normalized in {"name", "version"}:
            if normalized in headers:
                raise CompletionError("distribution metadata has duplicate identity")
            headers[normalized] = value.strip()
    return headers


def _check_distribution_metadata(raw: bytes, version: str) -> None:
    headers = _metadata_headers(raw)
    if _canonical(headers.get("name", "")) != "jaxrenderer" or _canonical(
        headers.get("version", "")
    ) != _canonical(version):
        raise CompletionError("distribution metadata does not match the frozen package")


def _validate_wheel(path: Path, version: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_PACKAGE_MEMBERS:
                raise CompletionError("wheel contains too many package members")
            metadata_members = [
                item
                for item in members
                if item.filename.endswith(".dist-info/METADATA")
            ]
            if (
                len(metadata_members) != 1
                or metadata_members[0].file_size > MAX_METADATA_BYTES
            ):
                raise CompletionError("wheel package identity metadata is missing")
            _check_distribution_metadata(archive.read(metadata_members[0]), version)
    except CompletionError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, UnicodeError) as error:
        raise CompletionError("release wheel is invalid") from error


def _validate_sdist(path: Path, version: str) -> None:
    expected_root = f"jaxrenderer-{version}/"
    try:
        with tarfile.open(path, mode="r|gz") as archive:
            metadata: bytes | None = None
            total = 0
            count = 0
            for item in archive:
                count += 1
                if count > MAX_PACKAGE_MEMBERS:
                    raise CompletionError("source archive contains too many members")
                total += item.size
                if total > MAX_DIST_MEMBER_BYTES:
                    raise CompletionError(
                        "source archive expands beyond its size limit"
                    )
                if item.name == f"{expected_root}PKG-INFO":
                    if metadata is not None or not item.isreg():
                        raise CompletionError(
                            "source package identity metadata is ambiguous"
                        )
                    if item.size > MAX_METADATA_BYTES:
                        raise CompletionError(
                            "source package metadata exceeds its size limit"
                        )
                    stream = archive.extractfile(item)
                    if stream is None:
                        raise CompletionError("source package metadata is unreadable")
                    metadata = stream.read(MAX_METADATA_BYTES + 1)
            if metadata is None:
                raise CompletionError(
                    "source archive package identity metadata is missing"
                )
            _check_distribution_metadata(metadata, version)
    except CompletionError:
        raise
    except (OSError, RuntimeError, tarfile.TarError, EOFError, UnicodeError) as error:
        raise CompletionError("release source archive is invalid") from error


def _distribution_manifest(
    artifact_metadata: dict[str, object], files: list[dict[str, object]], version: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": version,
        "artifact_id": _artifact_id(artifact_metadata),
        "artifact_name": artifact_metadata.get("name"),
        "archive_digest": _artifact_digest(artifact_metadata),
        "archive_size": artifact_metadata.get("size_in_bytes"),
        "files": files,
    }


def _validate_terminal(
    collection_dir: Path,
    source: dict[str, object],
    binding: dict[str, object],
    state: dict[str, object],
    collector_run_id: str,
    collector_sha: str,
) -> dict[str, object]:
    report = _read_json(collection_dir / "periodic-collection.json")
    record = _read_json(collection_dir / "periodic-terminal.json")
    if set(record) != periodic.TERMINAL_FIELDS:
        raise CompletionError("periodic terminal record has an unsupported schema")
    run_id, attempt = _parse_run_pair(collector_run_id)
    if (
        report.get("terminal") is not True
        or report.get("outcome") != "success"
        or record.get("terminal") is not True
        or record.get("outcome") != "success"
        or record.get("source_run_id") != str(source["source_run_id"])
        or record.get("source_attempt") != str(source["source_attempt"])
        or record.get("head_sha") != source["head_sha"]
        or record.get("collector_run_id") != collector_run_id
        or record.get("collector_attempt") != str(attempt)
        or record.get("collector_sha") != collector_sha
        or record.get("binding") != binding
        or record.get("kernel_id") != state.get("kernel_id")
        or type(record.get("submitted_version")) is not int
        or record.get("submitted_version") != 1
        or type(record.get("requested_version")) is not int
        or record.get("requested_version") != 1
        or record.get("version_verified") is not True
        or record.get("remote_status")
        not in {
            "complete",
            "completed",
            "success",
            "succeeded",
        }
    ):
        raise CompletionError("periodic terminal evidence does not match this source")
    if run_id <= 0:
        raise CompletionError("collector identity is malformed")
    return record


def _source_candidate(
    source_dir: Path, source: dict[str, object], tpu_binding: dict[str, object]
) -> tuple[release_candidate.CandidateBinding, bytes]:
    candidate_path = source_dir / "candidate-binding.json"
    candidate = release_candidate.validate_binding(
        _read_json(candidate_path, limit=16_384)
    )
    raw = candidate_path.read_bytes()
    expected_run = f"{source['source_run_id']}.{source['source_attempt']}"
    if (
        candidate["run_id"] != expected_run
        or candidate["controller_sha"] != source["head_sha"]
        or candidate["head_sha"] != tpu_binding.get("head_sha")
        or candidate["mode"] not in {"rehearsal", "release", "recovery"}
        or candidate["repository"] != REPOSITORY
    ):
        raise CompletionError("frozen candidate does not match the original TPU source")
    if candidate["mode"] == "release" and candidate["head_sha"] != source["head_sha"]:
        raise CompletionError("release candidate head does not match its controller")
    try:
        _ = release_candidate.verify_frozen(candidate_path)
    except release_candidate.CandidateError as error:
        raise CompletionError(
            "frozen release candidate failed live revalidation"
        ) from error
    return candidate, raw


def _source_artifact_identity(
    source: dict[str, object], candidate: release_candidate.CandidateBinding
) -> tuple[dict[str, object], dict[str, object]]:
    source_run = int(cast(int, source["source_run_id"]))
    source_attempt = int(cast(int, source["source_attempt"]))
    controller_sha = cast(str, source["head_sha"])
    candidate_name = f"candidate-binding-{source_attempt}"
    candidate_artifact = _artifact_metadata(
        source_run, candidate_name, controller_sha, expected_attempt=source_attempt
    )
    recorded_candidate_id = source.get("candidate_artifact_id")
    if (
        recorded_candidate_id is not None
        and recorded_candidate_id != candidate_artifact["id"]
    ):
        raise CompletionError("candidate artefact identity changed")
    if candidate_artifact.get("name") != candidate_name:
        raise CompletionError("candidate artefact name changed")
    marker_name = source.get("marker_artifact_name")
    if not isinstance(marker_name, str):
        marker_name = (
            f"release-pending-{source_attempt}"
            if candidate["mode"] != "recovery"
            else f"recovery-source-binding-{source_attempt}"
        )
    if marker_name not in {
        f"release-pending-{source_attempt}",
        f"recovery-source-binding-{source_attempt}",
    }:
        raise CompletionError("source opt-in marker name is invalid")
    marker_artifact = _artifact_metadata(
        source_run, marker_name, controller_sha, expected_attempt=source_attempt
    )
    recorded_marker_id = source.get("marker_artifact_id")
    if recorded_marker_id is not None and recorded_marker_id != marker_artifact["id"]:
        raise CompletionError("source opt-in marker identity changed")
    candidate_id = _artifact_id(candidate_artifact)
    marker_id = _artifact_id(marker_artifact)
    if (
        source.get("candidate_artifact_name") not in {None, candidate_name}
        or candidate_artifact.get("expired") is not False
        or marker_artifact.get("expired") is not False
    ):
        raise CompletionError("source candidate or opt-in marker is unavailable")
    return (
        {**candidate_artifact, "id": candidate_id},
        {**marker_artifact, "id": marker_id},
    )


def _validate_candidate_files(
    source_dir: Path,
    source: dict[str, object],
    candidate_artifact: dict[str, object],
    marker_artifact: dict[str, object],
    candidate_raw: bytes,
    temp_dir: Path,
) -> None:
    candidate_tmp = temp_dir / "candidate-binding-original.json"
    marker_tmp = temp_dir / "marker-binding-original.json"
    try:
        collection_gate.download_member(
            _artifact_id(candidate_artifact), "candidate-binding.json", candidate_tmp
        )
        collection_gate.download_member(
            _artifact_id(marker_artifact), "candidate-binding.json", marker_tmp
        )
    except collection_gate.CollectionGateError as error:
        raise CompletionError(
            "could not recheck original candidate artefacts"
        ) from error
    if candidate_tmp.read_bytes() != candidate_raw:
        raise CompletionError("original candidate artefact bytes changed")
    marker_copy = source_dir / "marker-binding.json"
    if marker_copy.is_file() and not marker_copy.is_symlink():
        if marker_tmp.read_bytes() != marker_copy.read_bytes():
            raise CompletionError("source opt-in marker bytes changed")
    elif marker_tmp.read_bytes() != candidate_raw:
        raise CompletionError("source opt-in marker does not contain the candidate")


def _distribution_files_match(dist_dir: Path, manifest: dict[str, object]) -> None:
    values = manifest.get("files")
    if not isinstance(values, list):
        raise CompletionError(
            "distribution manifest must contain one wheel and one sdist"
        )
    file_values = cast(list[object], values)
    if len(file_values) != 2:
        raise CompletionError(
            "distribution manifest must contain one wheel and one sdist"
        )
    expected: dict[str, tuple[int, str]] = {}
    for value in file_values:
        entry = _object(value, "distribution manifest entry is malformed")
        name, size, digest = entry.get("name"), entry.get("size"), entry.get("sha256")
        if (
            not isinstance(name, str)
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or name in expected
        ):
            raise CompletionError("distribution manifest entry is invalid")
        _validate_zip_name(name)
        expected[name] = (size, digest)
    if dist_dir.is_symlink() or not dist_dir.is_dir():
        raise CompletionError("original distributions were not downloaded")
    actual: set[str] = set()
    for path in dist_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise CompletionError("downloaded distribution directory is unsafe")
        actual.add(path.name)
        expected_file = expected.get(path.name)
        if expected_file is None:
            raise CompletionError("downloaded distribution set changed")
        size, digest = expected_file
        if path.stat().st_size != size or _sha256(path) != digest:
            raise CompletionError("downloaded distribution does not match its manifest")
    if actual != set(expected):
        raise CompletionError("downloaded distribution set is incomplete")


def _write_outputs(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            _ = stream.write(f"{key}={value}\n")


def _not_ready(output_path: Path, outcome: str) -> dict[str, object]:
    _write_outputs(
        output_path,
        {
            "ready": "false",
            "publish": "false",
            "outcome": outcome,
            "version": "",
            "head_sha": "",
            "mode": "",
        },
    )
    return {"ready": False, "publish": False, "outcome": outcome}


def prepare(
    source_run: str,
    source_attempt: str,
    commit: str,
    collection_dir: Path,
    output_dir: Path,
    github_output: Path,
    *,
    collector_run: str | None = None,
    collector_attempt: str | None = None,
) -> dict[str, object]:
    source_id, attempt = (
        collection_gate.positive_id(source_run),
        collection_gate.positive_id(source_attempt),
    )
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise CompletionError("source commit must be a full lowercase SHA")
    if output_dir.exists() or output_dir.is_symlink():
        raise CompletionError("completion output directory already exists")
    collector_identity, collector_sha, completion_kind, completion_identity = (
        _collector_context(collector_run, collector_attempt)
    )
    report = _read_json(collection_dir / "periodic-collection.json")
    outcome = report.get("outcome")
    if outcome in {"pending", "already-terminal", "expired", "remote_failure"}:
        return _not_ready(github_output, cast(str, outcome))
    if outcome != "success" or report.get("terminal") is not True:
        raise CompletionError("periodic collection did not produce verified success")
    _ = _collection_tree_digest(collection_dir)
    collector_id, collector_try = _parse_run_pair(collector_identity)
    source_dir = collection_dir / "source"
    receipt = _read_json(source_dir / "collection-source.json")
    if (
        receipt.get("source_run_id") != source_id
        or receipt.get("source_attempt") != attempt
        or receipt.get("head_sha") != commit
        or receipt.get("collector_run_id") != collector_identity
    ):
        raise CompletionError(
            "periodic source receipt does not match the requested run"
        )
    source_sha = receipt.get("collector_sha")
    if not isinstance(source_sha, str) or not accelerator_gate.SHA_RE.fullmatch(
        source_sha
    ):
        raise CompletionError("periodic collector SHA is malformed")
    if collector_sha and collector_sha != source_sha:
        raise CompletionError("current collector SHA differs from its receipt")
    collector_sha = source_sha
    try:
        release_tpu_gate.verify_master_history(collector_sha)
    except release_tpu_gate.ReleaseGateError as error:
        raise CompletionError("collector SHA is not trusted master history") from error
    if completion_kind == "manual":
        _collector_run_identity(
            collector_id,
            collector_try,
            collector_sha,
            require_completed_job=True,
            source_run=source_id,
            source_attempt=attempt,
        )
    try:
        source, tpu_binding, state = collection_gate.revalidate(
            source_dir,
            expected_collector_run_id=collector_identity,
            expected_collector_sha=collector_sha,
        )
    except collection_gate.CollectionGateError as error:
        raise CompletionError("original source attempt failed revalidation") from error
    if (
        source.get("source_run_id") != source_id
        or source.get("source_attempt") != attempt
        or source.get("head_sha") != commit
    ):
        raise CompletionError("original source identity changed")
    terminal = _validate_terminal(
        collection_dir, source, tpu_binding, state, collector_identity, collector_sha
    )
    try:
        collection_receipt = collection_gate.finish(
            source_dir,
            collection_dir / "collected",
            True,
            expected_collector_run_id=collector_identity,
            expected_collector_sha=collector_sha,
        )
    except collection_gate.CollectionGateError as error:
        raise CompletionError("original TPU result failed full validation") from error
    source_fields = set(source)
    source_fields.discard("_source_conclusion")
    source_fields = frozenset(source_fields)
    if source_fields not in {
        frozenset(collection_gate.SOURCE_BASE_FIELDS),
        frozenset(
            collection_gate.SOURCE_BASE_FIELDS | collection_gate.SOURCE_CANDIDATE_FIELDS
        ),
    }:
        raise CompletionError("original source manifest has an unsupported schema")
    collection_artifact_name = (
        f"tpu-periodic-{commit}-{source_id}-{attempt}-{collector_id}-{collector_try}"
    )
    collection_artifact = _artifact_metadata(
        collector_id,
        collection_artifact_name,
        collector_sha,
        expected_attempt=collector_try,
    )
    if collection_artifact.get("name") != collection_artifact_name:
        raise CompletionError("periodic collection artefact name changed")
    if source_fields == frozenset(collection_gate.SOURCE_BASE_FIELDS):
        if (
            source.get("_source_conclusion") != "failure"
            or collection_receipt.get("binding") != tpu_binding
            or collection_receipt.get("collector_run_id") != collector_identity
            or collection_receipt.get("collector_sha") != collector_sha
            or collection_receipt.get("kernel_id") != state.get("kernel_id")
            or type(collection_receipt.get("submitted_version")) is not int
            or collection_receipt.get("submitted_version") != 1
            or collection_receipt.get("success") is not True
            or collection_receipt.get("validation_only") is not True
            or collection_receipt.get("release_gate_credit") is not False
            or collection_receipt.get("source_conclusion")
            != source.get("_source_conclusion")
        ):
            raise CompletionError(
                "legacy collection receipt does not prove validation-only success"
            )
        return _not_ready(github_output, "validation_only")
    candidate, candidate_raw = _source_candidate(source_dir, source, tpu_binding)
    candidate_artifact, marker_artifact = _source_artifact_identity(source, candidate)
    dist_artifact_name = f"release-dist-{attempt}"
    dist_artifact = _artifact_metadata(
        source_id,
        dist_artifact_name,
        commit,
        max_archive_bytes=MAX_ARCHIVE_BYTES,
        expected_attempt=attempt,
    )
    if dist_artifact.get("name") != dist_artifact_name:
        raise CompletionError("original release distribution artefact name changed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="jaxrenderer-release-completion-", dir=output_dir.parent
    ) as stage_name:
        stage = Path(stage_name) / "completion"
        stage.mkdir()
        collection_copy = stage / "collection"
        _copy_tree(collection_dir, collection_copy)
        _ = (stage / "candidate-binding.json").write_bytes(candidate_raw)
        with tempfile.TemporaryDirectory(
            prefix="jaxrenderer-release-candidate-"
        ) as candidate_temp:
            _validate_candidate_files(
                source_dir,
                source,
                candidate_artifact,
                marker_artifact,
                candidate_raw,
                Path(candidate_temp),
            )
        with tempfile.TemporaryDirectory(
            prefix="jaxrenderer-release-dist-"
        ) as dist_temp:
            archive_path = Path(dist_temp) / "release-dist.zip"
            _download_archive(
                _artifact_id(dist_artifact), archive_path, MAX_ARCHIVE_BYTES
            )
            archive_digest = _sha256(archive_path)
            if _artifact_digest(dist_artifact) != f"sha256:{archive_digest}":
                raise CompletionError("original distribution artefact digest changed")
            if archive_path.stat().st_size != dist_artifact.get("size_in_bytes"):
                raise CompletionError("original distribution artefact size changed")
            dist_tmp = Path(dist_temp) / "dist"
            dist_tmp.mkdir()
            files = _copy_dist_archive(
                archive_path, dist_tmp, version=candidate["version"]
            )
            dist_manifest = _distribution_manifest(
                dist_artifact, files, candidate["version"]
            )
        _write_json(stage / "dist-manifest.json", dist_manifest)
        collection_digest = _collection_tree_digest(collection_copy)
        candidate_digest = hashlib.sha256(candidate_raw).hexdigest()
        publish = candidate["mode"] in {"release", "recovery"}
        completion: dict[str, object] = {
            "schema_version": 1,
            "repository": REPOSITORY,
            "source_run_id": source_id,
            "source_attempt": attempt,
            "source_head_sha": commit,
            "collector_run_id": collector_id,
            "collector_attempt": collector_try,
            "collector_sha": collector_sha,
            "collector_artifact_id": _artifact_id(collection_artifact),
            "collector_artifact_name": collection_artifact_name,
            "collector_artifact_digest": _artifact_digest(collection_artifact),
            "collector_artifact_size": collection_artifact["size_in_bytes"],
            "binding_artifact_id": source["binding_artifact_id"],
            "controller_artifact_id": source["controller_artifact_id"],
            "candidate_artifact_id": _artifact_id(candidate_artifact),
            "candidate_artifact_name": candidate_artifact["name"],
            "candidate_artifact_digest": _artifact_digest(candidate_artifact),
            "marker_artifact_id": _artifact_id(marker_artifact),
            "marker_artifact_name": marker_artifact["name"],
            "marker_artifact_digest": _artifact_digest(marker_artifact),
            "release_dist_artifact_id": _artifact_id(dist_artifact),
            "release_dist_artifact_name": dist_artifact_name,
            "release_dist_artifact_digest": _artifact_digest(dist_artifact),
            "release_dist_artifact_size": dist_artifact["size_in_bytes"],
            "candidate_mode": candidate["mode"],
            "candidate_sha256": candidate_digest,
            "collection_sha256": collection_digest,
            "version": candidate["version"],
            "head_sha": candidate["head_sha"],
            "publish": publish,
            "completion_kind": completion_kind,
            "completion_run_id": completion_identity,
            "completion_sha": os.environ.get("GITHUB_SHA", ""),
        }
        _write_json(stage / "completion-binding.json", completion)
        if not stage.is_dir():
            raise CompletionError("completion bundle was not prepared")
        _ = stage.rename(output_dir)
    with github_output.open("a", encoding="utf-8") as stream:
        _ = stream.write(
            "ready=true\n"
            f"publish={str(publish).lower()}\n"
            f"version={candidate['version']}\n"
            f"head_sha={candidate['head_sha']}\n"
            f"mode={candidate['mode']}\n"
            f"outcome={terminal['outcome']}\n"
        )
    return completion


def _validate_completion_binding(value: object) -> dict[str, object]:
    binding = _object(value, "completion binding is malformed")
    if set(binding) != COMPLETION_FIELDS or binding.get("schema_version") != 1:
        raise CompletionError("completion binding has missing or unexpected fields")
    if binding.get("repository") != REPOSITORY:
        raise CompletionError("completion binding repository is invalid")
    for field in (
        "source_run_id",
        "source_attempt",
        "collector_run_id",
        "collector_attempt",
    ):
        if type(binding.get(field)) is not int or cast(int, binding[field]) <= 0:
            raise CompletionError("completion binding run identity is invalid")
    for field in (
        "collector_artifact_id",
        "collector_artifact_size",
        "binding_artifact_id",
        "controller_artifact_id",
        "candidate_artifact_id",
        "marker_artifact_id",
        "release_dist_artifact_id",
        "release_dist_artifact_size",
    ):
        if type(binding.get(field)) is not int or cast(int, binding[field]) <= 0:
            raise CompletionError("completion binding artefact identity is invalid")
    for field in ("source_head_sha", "collector_sha", "head_sha", "completion_sha"):
        value_sha = binding.get(field)
        if not isinstance(value_sha, str) or not accelerator_gate.SHA_RE.fullmatch(
            value_sha
        ):
            raise CompletionError(f"completion binding {field} is invalid")
    for field in (
        "collector_artifact_digest",
        "candidate_artifact_digest",
        "marker_artifact_digest",
        "release_dist_artifact_digest",
    ):
        value_digest = binding.get(field)
        if (
            not isinstance(value_digest, str)
            or DIGEST_RE.fullmatch(value_digest) is None
        ):
            raise CompletionError(f"completion binding {field} is invalid")
    if binding.get("candidate_mode") not in {"rehearsal", "release", "recovery"}:
        raise CompletionError("completion candidate mode is invalid")
    if type(binding.get("publish")) is not bool:
        raise CompletionError("completion publish flag is invalid")
    if binding.get("publish") != (
        binding.get("candidate_mode") in {"release", "recovery"}
    ):
        raise CompletionError(
            "completion publish flag does not match its candidate mode"
        )
    if (
        not isinstance(binding.get("version"), str)
        or not isinstance(binding.get("candidate_artifact_name"), str)
        or not isinstance(binding.get("marker_artifact_name"), str)
    ):
        raise CompletionError("completion candidate identity is malformed")
    if not isinstance(binding.get("collector_artifact_name"), str) or not isinstance(
        binding.get("release_dist_artifact_name"), str
    ):
        raise CompletionError("completion artefact names are malformed")
    for field in ("candidate_sha256", "collection_sha256"):
        value_digest = binding.get(field)
        if (
            not isinstance(value_digest, str)
            or SHA256_RE.fullmatch(value_digest) is None
        ):
            raise CompletionError(f"completion binding {field} is invalid")
    return binding


def verify(bundle_dir: Path, dist_dir: Path) -> dict[str, object]:
    if bundle_dir.is_symlink() or not bundle_dir.is_dir():
        raise CompletionError("completion bundle directory is unsafe")
    binding = _validate_completion_binding(
        _read_json(bundle_dir / "completion-binding.json", limit=256 * 1024)
    )
    if binding["completion_kind"] == "automatic":
        try:
            current = periodic.require_context()
        except periodic.PeriodicError as error:
            raise CompletionError(
                "publish requires the original periodic collector context"
            ) from error
    elif binding["completion_kind"] == "manual":
        _ = _manual_context()
        current = _current_run_identity()
    else:
        raise CompletionError("completion workflow kind is invalid")
    if (
        current != binding["completion_run_id"]
        or os.environ.get("GITHUB_SHA") != binding["completion_sha"]
    ):
        raise CompletionError("completion bundle belongs to another workflow attempt")
    source_id = cast(int, binding["source_run_id"])
    source_attempt = cast(int, binding["source_attempt"])
    collector_id = cast(int, binding["collector_run_id"])
    collector_attempt = cast(int, binding["collector_attempt"])
    collector_identity = _run_pair(collector_id, collector_attempt)
    source_head = cast(str, binding["source_head_sha"])
    collector_sha = cast(str, binding["collector_sha"])
    try:
        release_tpu_gate.verify_master_history(collector_sha)
    except release_tpu_gate.ReleaseGateError as error:
        raise CompletionError("collector SHA is not trusted master history") from error
    if (
        _collection_tree_digest(bundle_dir / "collection")
        != binding["collection_sha256"]
    ):
        raise CompletionError("frozen collection evidence changed")
    if binding["completion_kind"] == "manual":
        _collector_run_identity(
            collector_id,
            collector_attempt,
            collector_sha,
            require_completed_job=True,
            source_run=source_id,
            source_attempt=source_attempt,
        )
    collection_artifact = _artifact_metadata(
        collector_id,
        cast(str, binding["collector_artifact_name"]),
        collector_sha,
        expected_attempt=collector_attempt,
    )
    if (
        _artifact_id(collection_artifact) != binding["collector_artifact_id"]
        or _artifact_digest(collection_artifact) != binding["collector_artifact_digest"]
        or collection_artifact.get("size_in_bytes")
        != binding["collector_artifact_size"]
    ):
        raise CompletionError("periodic collection artefact identity changed")
    source_dir = bundle_dir / "collection" / "source"
    try:
        source, tpu_binding, state = collection_gate.revalidate(
            source_dir,
            expected_collector_run_id=collector_identity,
            expected_collector_sha=collector_sha,
        )
    except collection_gate.CollectionGateError as error:
        raise CompletionError(
            "original source attempt failed fresh revalidation"
        ) from error
    if (
        source.get("source_run_id") != source_id
        or source.get("source_attempt") != source_attempt
        or source.get("head_sha") != source_head
    ):
        raise CompletionError("original source attempt identity changed")
    _ = _validate_terminal(
        bundle_dir / "collection",
        source,
        tpu_binding,
        state,
        collector_identity,
        collector_sha,
    )
    try:
        _ = collection_gate.finish(
            source_dir,
            bundle_dir / "collection" / "collected",
            True,
            expected_collector_run_id=collector_identity,
            expected_collector_sha=collector_sha,
        )
    except collection_gate.CollectionGateError as error:
        raise CompletionError("original TPU result failed fresh validation") from error
    candidate, candidate_raw = _source_candidate(source_dir, source, tpu_binding)
    if (
        candidate["mode"] != binding["candidate_mode"]
        or candidate["version"] != binding["version"]
        or candidate["head_sha"] != binding["head_sha"]
        or hashlib.sha256(candidate_raw).hexdigest() != binding["candidate_sha256"]
    ):
        raise CompletionError("frozen completion evidence changed")
    candidate_artifact, marker_artifact = _source_artifact_identity(source, candidate)
    if (
        _artifact_id(candidate_artifact) != binding["candidate_artifact_id"]
        or candidate_artifact.get("name") != binding["candidate_artifact_name"]
        or _artifact_digest(candidate_artifact) != binding["candidate_artifact_digest"]
        or _artifact_id(marker_artifact) != binding["marker_artifact_id"]
        or marker_artifact.get("name") != binding["marker_artifact_name"]
        or _artifact_digest(marker_artifact) != binding["marker_artifact_digest"]
        or source.get("binding_artifact_id") != binding["binding_artifact_id"]
        or source.get("controller_artifact_id") != binding["controller_artifact_id"]
    ):
        raise CompletionError("original source artefact identities changed")
    candidate_copy = bundle_dir / "candidate-binding.json"
    if candidate_copy.is_symlink() or not candidate_copy.is_file():
        raise CompletionError("frozen candidate copy is unsafe")
    if candidate_copy.read_bytes() != candidate_raw:
        raise CompletionError("frozen candidate copy changed")
    with tempfile.TemporaryDirectory(
        prefix="jaxrenderer-release-candidate-"
    ) as candidate_temp:
        _validate_candidate_files(
            source_dir,
            source,
            candidate_artifact,
            marker_artifact,
            candidate_raw,
            Path(candidate_temp),
        )
    dist_name = cast(str, binding["release_dist_artifact_name"])
    dist_artifact = _artifact_metadata(
        source_id,
        dist_name,
        source_head,
        max_archive_bytes=MAX_ARCHIVE_BYTES,
        expected_attempt=source_attempt,
    )
    if (
        dist_name != f"release-dist-{source_attempt}"
        or _artifact_id(dist_artifact) != binding["release_dist_artifact_id"]
        or _artifact_digest(dist_artifact) != binding["release_dist_artifact_digest"]
        or dist_artifact.get("size_in_bytes") != binding["release_dist_artifact_size"]
    ):
        raise CompletionError("original distribution artefact identity changed")
    manifest = _read_json(bundle_dir / "dist-manifest.json", limit=256 * 1024)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("version") != binding["version"]
        or manifest.get("artifact_id") != binding["release_dist_artifact_id"]
        or manifest.get("artifact_name") != dist_name
        or manifest.get("archive_digest") != binding["release_dist_artifact_digest"]
        or manifest.get("archive_size") != binding["release_dist_artifact_size"]
    ):
        raise CompletionError("original distribution manifest changed")
    _distribution_files_match(dist_dir, manifest)
    return binding


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _ = prepare_parser.add_argument("--source-run", required=True)
    _ = prepare_parser.add_argument("--source-attempt", required=True)
    _ = prepare_parser.add_argument("--commit", required=True)
    _ = prepare_parser.add_argument("--collector-run")
    _ = prepare_parser.add_argument("--collector-attempt")
    _ = prepare_parser.add_argument("--collection-dir", type=Path, required=True)
    _ = prepare_parser.add_argument("--output-dir", type=Path, required=True)
    _ = prepare_parser.add_argument("--github-output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    _ = verify_parser.add_argument("--bundle-dir", type=Path, required=True)
    _ = verify_parser.add_argument("--dist-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            _ = prepare(
                args.source_run,
                args.source_attempt,
                args.commit,
                args.collection_dir,
                args.output_dir,
                args.github_output,
                collector_run=args.collector_run,
                collector_attempt=args.collector_attempt,
            )
        else:
            _ = verify(args.bundle_dir, args.dist_dir)
    except (
        CompletionError,
        accelerator_gate.GateError,
        collection_gate.CollectionGateError,
        periodic.PeriodicError,
        release_candidate.CandidateError,
        release_tpu_gate.ReleaseGateError,
        OSError,
    ) as error:
        print(f"release completion: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
