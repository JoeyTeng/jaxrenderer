"""Select and bind a rehearsal, published release, or recovery candidate."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from typing import TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from tools import accelerator_gate, release_tpu_gate

REPOSITORY = "JoeyTeng/jaxrenderer"
MAX_TAG_PEEL_DEPTH = release_tpu_gate.MAX_TAG_PEEL_DEPTH
MAX_CONTENT_BYTES = 65_536
MAX_PYPI_BYTES = 65_536
MAX_EVENT_BYTES = 1_048_576
TAG_RE = re.compile(r"^v?[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$")
FIELDS = {
    "mode",
    "source_commit",
    "head_sha",
    "version",
    "tag",
    "tag_ref",
    "tag_object_sha",
    "release_id",
    "controller_sha",
    "repository",
    "run_id",
}


class CandidateBinding(TypedDict):
    mode: str
    source_commit: str
    head_sha: str
    version: str
    tag: str | None
    tag_ref: str | None
    tag_object_sha: str | None
    release_id: int | None
    controller_sha: str
    repository: str
    run_id: str


class CandidateError(RuntimeError):
    """A candidate validation error that must fail closed."""


def _json_object(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


def _context_identity() -> tuple[str, str]:
    try:
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")
        if event_name == "workflow_dispatch":
            accelerator_gate.require_dispatch_context()
        elif event_name == "release":
            if (
                os.environ.get("GITHUB_REPOSITORY", "").casefold()
                != REPOSITORY.casefold()
            ):
                raise CandidateError(f"release event is restricted to {REPOSITORY}")
        else:
            raise CandidateError(
                "release candidate requires workflow_dispatch or release event"
            )
        run_id = accelerator_gate.run_id_from_env()
    except accelerator_gate.GateError as error:
        raise CandidateError(str(error)) from error
    controller_sha = os.environ.get("GITHUB_SHA", "")
    if not accelerator_gate.SHA_RE.fullmatch(controller_sha):
        raise CandidateError("GITHUB_SHA must be a full lowercase controller SHA")
    return controller_sha, run_id


def _validate_tag(tag: str) -> None:
    if len(tag) > 128 or not TAG_RE.fullmatch(tag):
        raise CandidateError("recovery tag is malformed")


def _release_for_tag(tag: str) -> tuple[int, str, str]:
    route = f"repos/{REPOSITORY}/releases/tags/{quote(tag, safe='')}"
    try:
        release_value = accelerator_gate.gh_api(route)
    except accelerator_gate.GateError as error:
        raise CandidateError("GitHub could not confirm a published release") from error
    release = _json_object(release_value)
    if release is None:
        raise CandidateError("GitHub returned malformed release metadata")
    release_tag = release.get("tag_name")
    if release_tag != tag:
        raise CandidateError("published release tag does not match the requested tag")
    if release.get("draft") is not False:
        raise CandidateError("candidate requires an already published release")
    published_at = release.get("published_at")
    if not isinstance(published_at, str) or not published_at:
        raise CandidateError("candidate requires an already published release")
    release_id = release.get("id")
    if type(release_id) is not int or release_id <= 0:
        raise CandidateError("GitHub returned malformed release identity")
    return release_id, tag, f"refs/tags/{tag}"


def _release_event_identity() -> tuple[str, int]:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        raise CandidateError("GITHUB_EVENT_PATH is required for release events")
    path = Path(event_path)
    try:
        if path.stat().st_size > MAX_EVENT_BYTES:
            raise CandidateError(
                "release event payload exceeds the supported size limit"
            )
        event_value = json.loads(path.read_text(encoding="utf-8"))
    except CandidateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateError("could not read release event payload") from error
    event = _json_object(event_value)
    repository = _json_object(event.get("repository")) if event is not None else None
    release = _json_object(event.get("release")) if event is not None else None
    repository_name = repository.get("full_name") if repository is not None else None
    if event is None or event.get("action") != "published":
        raise CandidateError("release event action must be published")
    if (
        not isinstance(repository_name, str)
        or repository_name.casefold() != REPOSITORY.casefold()
    ):
        raise CandidateError(
            "release event repository does not match the target repository"
        )
    if release is None or release.get("draft") is not False:
        raise CandidateError("release event must describe an already published release")
    tag = release.get("tag_name")
    if not isinstance(tag, str):
        raise CandidateError("release event tag is malformed")
    _validate_tag(tag)
    release_id = release.get("id")
    if type(release_id) is not int or release_id <= 0:
        raise CandidateError("release event identity is malformed")
    return tag, release_id


def _resolve_tag(tag: str) -> tuple[str, str]:
    route = f"repos/{REPOSITORY}/git/ref/tags/{quote(tag, safe='')}"
    try:
        reference_value = accelerator_gate.gh_api(route)
    except accelerator_gate.GateError as error:
        raise CandidateError(
            "GitHub could not resolve the published release tag"
        ) from error
    reference = _json_object(reference_value)
    tag_ref = f"refs/tags/{tag}"
    if reference is None or reference.get("ref") != tag_ref:
        raise CandidateError("GitHub did not return the exact published release tag")
    target: object = reference.get("object")
    first_object_sha: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_TAG_PEEL_DEPTH):
        target_object = _json_object(target)
        if target_object is None:
            raise CandidateError("published release tag target is malformed")
        object_type = target_object.get("type")
        object_sha = target_object.get("sha")
        if not isinstance(object_sha, str) or not accelerator_gate.SHA_RE.fullmatch(
            object_sha
        ):
            raise CandidateError("published release tag target SHA is malformed")
        if first_object_sha is None:
            first_object_sha = object_sha
        if object_type == "commit":
            return first_object_sha, object_sha
        if object_type != "tag" or object_sha in seen:
            raise CandidateError("published release tag has an unsupported target")
        seen.add(object_sha)
        try:
            tag_value = accelerator_gate.gh_api(
                f"repos/{REPOSITORY}/git/tags/{object_sha}"
            )
        except accelerator_gate.GateError as error:
            raise CandidateError(
                "GitHub could not peel the published release tag"
            ) from error
        tag_object = _json_object(tag_value)
        if tag_object is None or tag_object.get("sha") != object_sha:
            raise CandidateError("GitHub did not confirm the annotated tag object")
        target = tag_object.get("object")
    raise CandidateError("published release tag exceeds the annotated-tag depth limit")


def _version_at_commit(commit: str) -> str:
    route = f"repos/{REPOSITORY}/contents/pyproject.toml?ref={commit}"
    try:
        content_value = accelerator_gate.gh_api(route)
    except accelerator_gate.GateError as error:
        raise CandidateError(
            "GitHub could not read project version at the candidate commit"
        ) from error
    content = _json_object(content_value)
    if content is None or content.get("encoding") != "base64":
        raise CandidateError("GitHub returned malformed pyproject.toml content")
    size = content.get("size")
    encoded = content.get("content")
    if type(size) is not int or size < 1 or size > MAX_CONTENT_BYTES:
        raise CandidateError("pyproject.toml exceeds the supported size limit")
    if not isinstance(encoded, str) or len(encoded) > MAX_CONTENT_BYTES * 2:
        raise CandidateError("GitHub returned malformed pyproject.toml content")
    try:
        raw = base64.b64decode("".join(encoded.split()), validate=True)
        if len(raw) != size or len(raw) > MAX_CONTENT_BYTES:
            raise CandidateError("GitHub returned inconsistent pyproject.toml content")
        project = tomllib.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise CandidateError("candidate pyproject.toml is invalid") from error
    project_table = _json_object(project.get("project"))
    version = project_table.get("version") if project_table is not None else None
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise CandidateError("candidate pyproject.toml has no valid project version")
    return version


def _require_pypi_version_absent(version: str) -> None:
    url = f"https://pypi.org/pypi/jaxrenderer/{quote(version, safe='')}/json"
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise CandidateError("PyPI returned an unexpected response")
            body = response.read(MAX_PYPI_BYTES + 1)
    except HTTPError as error:
        if error.code == 404:
            return
        raise CandidateError(
            "PyPI could not confirm the target version is absent"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise CandidateError(
            "PyPI could not confirm the target version is absent"
        ) from error
    if len(body) > MAX_PYPI_BYTES:
        raise CandidateError("PyPI response exceeds the supported size limit")
    try:
        metadata = json.loads(body)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CandidateError("PyPI returned malformed version metadata") from error
    metadata_object = _json_object(metadata)
    info = _json_object(metadata_object.get("info")) if metadata_object else None
    if info is None or info.get("version") != version:
        raise CandidateError("PyPI returned mismatched version metadata")
    raise CandidateError("target version already exists on PyPI")


def select_candidate(recovery_tag: str | None) -> CandidateBinding:
    controller_sha, run_id = _context_identity()
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name == "release":
        if recovery_tag not in {None, ""}:
            raise CandidateError(
                "release events select their tag from the event payload"
            )
        tag, event_release_id = _release_event_identity()
        release_id, tag, tag_ref = _release_for_tag(tag)
        if release_id != event_release_id:
            raise CandidateError(
                "release event identity does not match the current GitHub release"
            )
        tag_object_sha, commit = _resolve_tag(tag)
        if os.environ.get("GITHUB_REF") != tag_ref:
            raise CandidateError("GITHUB_REF does not match the published release tag")
        if controller_sha != commit:
            raise CandidateError(
                "release event GITHUB_SHA does not match the published tag"
            )
        try:
            release_tpu_gate.verify_master_history(commit)
        except release_tpu_gate.ReleaseGateError as error:
            raise CandidateError(
                "release commit is not on the master history"
            ) from error
        version = _version_at_commit(commit)
        if version != tag.removeprefix("v"):
            raise CandidateError(
                "release tag does not match project.version at its commit"
            )
        _require_pypi_version_absent(version)
        mode = "release"
    elif recovery_tag is None or recovery_tag == "":
        commit = controller_sha
        version = _version_at_commit(commit)
        mode = "rehearsal"
        tag: str | None = None
        tag_ref: str | None = None
        tag_object_sha: str | None = None
        release_id: int | None = None
    else:
        _validate_tag(recovery_tag)
        release_id, tag, tag_ref = _release_for_tag(recovery_tag)
        tag_object_sha, commit = _resolve_tag(recovery_tag)
        try:
            release_tpu_gate.verify_master_history(commit)
        except release_tpu_gate.ReleaseGateError as error:
            raise CandidateError(
                "recovery commit is not on the master history"
            ) from error
        version = _version_at_commit(commit)
        if version != recovery_tag.removeprefix("v"):
            raise CandidateError(
                "release tag does not match project.version at its commit"
            )
        _require_pypi_version_absent(version)
        mode = "recovery"
    return validate_binding(
        {
            "mode": mode,
            "source_commit": commit,
            "head_sha": commit,
            "version": version,
            "tag": tag,
            "tag_ref": tag_ref,
            "tag_object_sha": tag_object_sha,
            "release_id": release_id,
            "controller_sha": controller_sha,
            "repository": REPOSITORY,
            "run_id": run_id,
        }
    )


def validate_binding(value: object) -> CandidateBinding:
    binding = _json_object(value)
    if binding is None or set(binding) != FIELDS:
        raise CandidateError("candidate binding has missing or unexpected fields")
    mode = binding.get("mode")
    if not isinstance(mode, str) or mode not in {"rehearsal", "release", "recovery"}:
        raise CandidateError("candidate binding mode is invalid")
    for field in ("source_commit", "head_sha", "controller_sha"):
        sha = binding.get(field)
        if not isinstance(sha, str) or not accelerator_gate.SHA_RE.fullmatch(sha):
            raise CandidateError(f"candidate binding {field} is invalid")
    if binding.get("head_sha") != binding.get("source_commit"):
        raise CandidateError("candidate binding head_sha does not match source_commit")
    if binding.get("repository") != REPOSITORY:
        raise CandidateError("candidate binding repository is invalid")
    version = binding.get("version")
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise CandidateError("candidate binding version is invalid")
    run_id = binding.get("run_id")
    if not isinstance(run_id, str) or not release_tpu_gate.RUN_ID_RE.fullmatch(run_id):
        raise CandidateError("candidate binding run_id is invalid")
    if mode == "rehearsal":
        if any(
            binding.get(field) is not None
            for field in ("tag", "tag_ref", "tag_object_sha", "release_id")
        ) or binding.get("source_commit") != binding.get("controller_sha"):
            raise CandidateError("candidate rehearsal binding is inconsistent")
    else:
        tag = binding.get("tag")
        if not isinstance(tag, str):
            raise CandidateError("candidate recovery tag is invalid")
        _validate_tag(tag)
        if binding.get("tag_ref") != f"refs/tags/{tag}":
            raise CandidateError("candidate recovery tag reference is invalid")
        tag_object_sha = binding.get("tag_object_sha")
        if not isinstance(tag_object_sha, str) or not accelerator_gate.SHA_RE.fullmatch(
            tag_object_sha
        ):
            raise CandidateError("candidate recovery tag object is invalid")
        release_id = binding.get("release_id")
        if type(release_id) is not int or release_id <= 0:
            raise CandidateError("candidate recovery release identity is invalid")
        if version != tag.removeprefix("v"):
            raise CandidateError("candidate recovery version does not match its tag")
    return cast(CandidateBinding, binding)


def _read_binding(path: Path) -> CandidateBinding:
    try:
        if path.stat().st_size > 16_384:
            raise CandidateError("candidate binding exceeds the supported size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except CandidateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateError("could not read candidate binding") from error
    return validate_binding(value)


def verify(binding_path: Path, recovery_tag: str | None) -> CandidateBinding:
    existing = _read_binding(binding_path)
    current = select_candidate(recovery_tag)
    if current != existing:
        raise CandidateError("candidate changed after the binding was prepared")
    return current


def verify_frozen(binding_path: Path) -> CandidateBinding:
    """Recheck an authenticated original binding without selecting a new run."""
    binding = _read_binding(binding_path)
    commit = binding["source_commit"]
    version = binding["version"]
    if binding["mode"] == "rehearsal":
        if _version_at_commit(commit) != version:
            raise CandidateError(
                "candidate version changed after the binding was prepared"
            )
        return binding

    tag = cast(str, binding["tag"])
    release_id, live_tag, live_ref = _release_for_tag(tag)
    if (
        release_id != binding["release_id"]
        or live_tag != tag
        or live_ref != binding["tag_ref"]
    ):
        raise CandidateError(
            "published release identity changed after the binding was prepared"
        )
    tag_object_sha, live_commit = _resolve_tag(tag)
    if tag_object_sha != binding["tag_object_sha"] or live_commit != commit:
        raise CandidateError(
            "published release tag changed after the binding was prepared"
        )
    try:
        release_tpu_gate.verify_master_history(commit)
    except release_tpu_gate.ReleaseGateError as error:
        raise CandidateError("release commit is no longer on master history") from error
    if _version_at_commit(commit) != version or version != tag.removeprefix("v"):
        raise CandidateError("release version changed after the binding was prepared")
    if binding["mode"] == "release" and binding["controller_sha"] != commit:
        raise CandidateError("release controller does not match its published tag")
    _require_pypi_version_absent(version)
    return binding


def prepare(
    recovery_tag: str | None, output: Path, github_output: Path
) -> CandidateBinding:
    if output.resolve() == github_output.resolve():
        raise CandidateError("binding output and GITHUB_OUTPUT must be different files")
    binding = select_candidate(recovery_tag)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
    publish = binding["mode"] in {"release", "recovery"}
    with github_output.open("a", encoding="utf-8") as stream:
        _ = stream.write(
            f"head_sha={binding['head_sha']}\n"
            f"version={binding['version']}\n"
            f"mode={binding['mode']}\n"
            f"publish={str(publish).lower()}\n"
        )
    return binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    _ = prepare_parser.add_argument("--recovery-tag", default="")
    _ = prepare_parser.add_argument("--output", type=Path, required=True)
    _ = prepare_parser.add_argument("--github-output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    _ = verify_parser.add_argument("--binding", type=Path, required=True)
    _ = verify_parser.add_argument("--recovery-tag", default="")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            _ = prepare(
                arguments.recovery_tag, arguments.output, arguments.github_output
            )
        else:
            _ = verify(arguments.binding, arguments.recovery_tag)
    except CandidateError as error:
        print(f"release candidate: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
