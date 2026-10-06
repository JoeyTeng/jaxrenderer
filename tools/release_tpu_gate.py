"""Freeze and validate a release accelerator regression."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import quote

from tools import accelerator_gate

REPOSITORY = "JoeyTeng/jaxrenderer"
FIELDS = {"kind", "head_sha", "head_repository", "run_id"}
RUN_ID_RE = re.compile(r"^[1-9][0-9]*\.[1-9][0-9]*$")
MAX_TAG_PEEL_DEPTH = 5
RESULT_FIELDS = {
    "kind",
    "head_sha",
    "head_repository",
    "run_id",
    "backend",
    "device_backend",
    "device_count",
    "success",
    "devices",
    "versions",
}


class ReleaseGateError(RuntimeError):
    """A release accelerator gate error that must fail closed."""


def _validate_binding(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ReleaseGateError("binding has missing or unexpected fields")
    if value["kind"] != "release":
        raise ReleaseGateError("binding kind is not release")
    if not isinstance(value["head_sha"], str) or not accelerator_gate.SHA_RE.fullmatch(
        value["head_sha"]
    ):
        raise ReleaseGateError("binding head_sha must be a full lowercase commit SHA")
    if value["head_repository"] != REPOSITORY:
        raise ReleaseGateError("binding head_repository is not the release repository")
    if not isinstance(value["run_id"], str) or not RUN_ID_RE.fullmatch(value["run_id"]):
        raise ReleaseGateError("binding run_id is invalid")
    return value


def _verify_master_history(commit: str) -> None:
    try:
        comparison = accelerator_gate._gh_api(
            f"repos/{REPOSITORY}/compare/{commit}...master"
        )
    except accelerator_gate.GateError as error:
        raise ReleaseGateError(
            "GitHub could not validate release master history"
        ) from error
    if not isinstance(comparison, dict) or comparison.get("status") not in {
        "ahead",
        "identical",
    }:
        raise ReleaseGateError("release commit is not on the master history")
    base_commit = comparison.get("base_commit")
    if not isinstance(base_commit, dict) or base_commit.get("sha") != commit:
        raise ReleaseGateError("GitHub did not confirm the requested release commit")


def _verify_release_event(commit: str) -> None:
    if os.environ.get("GITHUB_REPOSITORY", "").casefold() != REPOSITORY.casefold():
        raise ReleaseGateError(f"release event is restricted to {REPOSITORY}")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        raise ReleaseGateError("GITHUB_EVENT_PATH is required for release events")
    try:
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseGateError("could not read release event payload") from error
    if not isinstance(payload, dict) or payload.get("action") != "published":
        raise ReleaseGateError("release event action must be published")
    repository = payload.get("repository")
    repository_name = (
        repository.get("full_name") if isinstance(repository, dict) else None
    )
    if (
        not isinstance(repository_name, str)
        or repository_name.casefold() != REPOSITORY.casefold()
    ):
        raise ReleaseGateError(
            "release payload repository does not match this repository"
        )
    release = payload.get("release")
    if not isinstance(release, dict) or release.get("draft") is not False:
        raise ReleaseGateError("release payload must describe a published release")
    tag_name = release.get("tag_name")
    if (
        not isinstance(tag_name, str)
        or not tag_name
        or len(tag_name) > 255
        or any(ord(character) < 32 for character in tag_name)
    ):
        raise ReleaseGateError("release payload tag name is malformed")
    if os.environ.get("GITHUB_REF") != f"refs/tags/{tag_name}":
        raise ReleaseGateError("GITHUB_REF does not match the published release tag")
    github_sha = os.environ.get("GITHUB_SHA", "")
    if not accelerator_gate.SHA_RE.fullmatch(github_sha) or github_sha != commit:
        raise ReleaseGateError(
            "requested commit must match the release event GITHUB_SHA"
        )
    encoded_tag_name = quote(tag_name, safe="")
    try:
        reference = accelerator_gate._gh_api(
            f"repos/{REPOSITORY}/git/ref/tags/{encoded_tag_name}"
        )
    except accelerator_gate.GateError as error:
        raise ReleaseGateError("could not resolve the published release tag") from error
    if (
        not isinstance(reference, dict)
        or reference.get("ref") != f"refs/tags/{tag_name}"
    ):
        raise ReleaseGateError("GitHub did not return the exact published release tag")
    target = reference.get("object")
    seen: set[str] = set()
    for _ in range(MAX_TAG_PEEL_DEPTH):
        if not isinstance(target, dict):
            raise ReleaseGateError("published release tag target is malformed")
        object_type = target.get("type")
        object_sha = target.get("sha")
        if not isinstance(object_sha, str) or not accelerator_gate.SHA_RE.fullmatch(
            object_sha
        ):
            raise ReleaseGateError("published release tag target SHA is malformed")
        if object_type == "commit":
            if object_sha != commit:
                raise ReleaseGateError(
                    "published release tag no longer resolves to the requested commit"
                )
            return
        if object_type != "tag" or object_sha in seen:
            raise ReleaseGateError("published release tag has an unsupported target")
        seen.add(object_sha)
        try:
            tag_object = accelerator_gate._gh_api(
                f"repos/{REPOSITORY}/git/tags/{object_sha}"
            )
        except accelerator_gate.GateError as error:
            raise ReleaseGateError(
                "could not peel the published release tag"
            ) from error
        if not isinstance(tag_object, dict) or tag_object.get("sha") != object_sha:
            raise ReleaseGateError("GitHub did not confirm the annotated tag object")
        target = tag_object.get("object")
    else:
        raise ReleaseGateError(
            "published release tag exceeds the annotated-tag depth limit"
        )


def _verify_context(commit: str) -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name == "workflow_dispatch":
        try:
            accelerator_gate._require_dispatch_context()
        except accelerator_gate.GateError as error:
            raise ReleaseGateError(str(error)) from error
    elif event_name == "release":
        _verify_release_event(commit)
    else:
        raise ReleaseGateError(
            "release gate requires workflow_dispatch or release event"
        )
    try:
        _verify_master_history(commit)
    except accelerator_gate.GateError as error:
        raise ReleaseGateError(str(error)) from error


def prepare(commit: str, output: Path, github_output: Path) -> dict[str, object]:
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise ReleaseGateError("commit must be a full lowercase SHA")
    if output.resolve() == github_output.resolve():
        raise ReleaseGateError(
            "binding output and GITHUB_OUTPUT must be different files"
        )
    _verify_context(commit)
    run_id = accelerator_gate._run_id_from_env()
    binding = _validate_binding(
        {
            "kind": "release",
            "head_sha": commit,
            "head_repository": REPOSITORY,
            "run_id": run_id,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
    with github_output.open("a", encoding="utf-8") as stream:
        stream.write(f"head_sha={commit}\nrun_id={run_id}\n")
    return binding


def _read_result(path: Path, binding: dict[str, object], backend: str) -> None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseGateError(f"could not read result.json: {error}") from error
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        raise ReleaseGateError("result.json has missing or unexpected fields")
    for field in ("kind", "head_sha", "head_repository", "run_id"):
        if result[field] != binding[field] or type(result[field]) is not type(
            binding[field]
        ):
            raise ReleaseGateError(
                f"result.json {field} does not match the frozen binding"
            )
    if result["backend"] != backend or result["device_backend"] != backend:
        raise ReleaseGateError(
            f"result.json does not prove a {backend.upper()} device backend"
        )
    if type(result["success"]) is not bool or result["success"] is not True:
        raise ReleaseGateError("result.json does not report success")
    devices = result["devices"]
    if not isinstance(devices, list) or not devices:
        raise ReleaseGateError(f"result.json has no {backend.upper()} devices")
    if type(result["device_count"]) is not int or result["device_count"] != len(
        devices
    ):
        raise ReleaseGateError(
            "result.json device_count does not match its device list"
        )
    for device in devices:
        if (
            not isinstance(device, dict)
            or set(device) != {"platform", "device_kind", "id"}
            or device["platform"] != backend
            or not all(isinstance(device[key], str) and device[key] for key in device)
        ):
            raise ReleaseGateError(
                f"result.json contains invalid {backend.upper()} device metadata"
            )
    versions = result["versions"]
    if (
        not isinstance(versions, dict)
        or not versions
        or not all(
            isinstance(name, str) and name and isinstance(version, str) and version
            for name, version in versions.items()
        )
    ):
        raise ReleaseGateError("result.json versions are invalid")


def finish(
    binding_path: Path,
    output_dir: Path,
    provider_success: bool,
    backend: str = "tpu",
) -> dict[str, object]:
    if backend not in {"gpu", "tpu"}:
        raise ReleaseGateError("backend must be gpu or tpu")
    try:
        binding = _validate_binding(
            json.loads(binding_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseGateError(f"could not read binding file: {error}") from error
    current_run = accelerator_gate._run_id_from_env()
    if binding["run_id"] != current_run:
        return {
            "state": "failure",
            "reason": "binding belongs to another workflow run or attempt",
        }
    try:
        _verify_context(str(binding["head_sha"]))
    except ReleaseGateError as error:
        return {"state": "failure", "reason": str(error), "backend": backend}
    if not provider_success:
        return {"state": "failure", "reason": "provider command did not succeed"}
    try:
        _read_result(output_dir / "result.json", binding, backend)
    except ReleaseGateError as error:
        return {"state": "failure", "reason": str(error), "backend": backend}
    return {
        "state": "success",
        "reason": None,
        "backend": backend,
        "head_sha": binding["head_sha"],
        "run_id": current_run,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--commit", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--github-output", type=Path, required=True)
    finish_parser = commands.add_parser("finish")
    finish_parser.add_argument("--binding", type=Path, required=True)
    finish_parser.add_argument("--output-dir", type=Path, required=True)
    finish_parser.add_argument("--backend", choices=("gpu", "tpu"), default="tpu")
    finish_parser.add_argument(
        "--provider-success", choices=("true", "false"), required=True
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            value: Any = prepare(args.commit, args.output, args.github_output)
            code = 0
        else:
            value = finish(
                args.binding,
                args.output_dir,
                args.provider_success == "true",
                args.backend,
            )
            code = 0 if value["state"] == "success" else 1
        print(json.dumps(value, sort_keys=True))
        return code
    except (ReleaseGateError, accelerator_gate.GateError) as error:
        print(f"release accelerator gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
