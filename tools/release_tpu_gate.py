"""Freeze and validate a release accelerator regression."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import TypedDict, cast
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


class Binding(TypedDict):
    kind: str
    head_sha: str
    head_repository: str
    run_id: str


def _json_object(value: object) -> dict[str, object] | None:
    """Narrow a decoded JSON object at the validation boundary."""
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


class ReleaseGateError(RuntimeError):
    """A release accelerator gate error that must fail closed."""


def validate_binding(value: object) -> Binding:
    binding = _json_object(value)
    if binding is None or set(binding) != FIELDS:
        raise ReleaseGateError("binding has missing or unexpected fields")
    if binding["kind"] != "release":
        raise ReleaseGateError("binding kind is not release")
    if not isinstance(
        binding["head_sha"], str
    ) or not accelerator_gate.SHA_RE.fullmatch(binding["head_sha"]):
        raise ReleaseGateError("binding head_sha must be a full lowercase commit SHA")
    if binding["head_repository"] != REPOSITORY:
        raise ReleaseGateError("binding head_repository is not the release repository")
    if not isinstance(binding["run_id"], str) or not RUN_ID_RE.fullmatch(
        binding["run_id"]
    ):
        raise ReleaseGateError("binding run_id is invalid")
    return cast(Binding, binding)


def verify_master_history(commit: str) -> None:
    try:
        comparison = accelerator_gate.gh_api(
            f"repos/{REPOSITORY}/compare/{commit}...master"
        )
    except accelerator_gate.GateError as error:
        raise ReleaseGateError(
            "GitHub could not validate release master history"
        ) from error
    comparison_object = _json_object(comparison)
    if comparison_object is None or comparison_object.get("status") not in {
        "ahead",
        "identical",
    }:
        raise ReleaseGateError("release commit is not on the master history")
    base_commit = _json_object(comparison_object.get("base_commit"))
    if base_commit is None or base_commit.get("sha") != commit:
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
    event = _json_object(payload)
    if event is None or event.get("action") != "published":
        raise ReleaseGateError("release event action must be published")
    repository = _json_object(event.get("repository"))
    repository_name = repository.get("full_name") if repository is not None else None
    if (
        not isinstance(repository_name, str)
        or repository_name.casefold() != REPOSITORY.casefold()
    ):
        raise ReleaseGateError(
            "release payload repository does not match this repository"
        )
    release = _json_object(event.get("release"))
    if release is None or release.get("draft") is not False:
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
        reference = accelerator_gate.gh_api(
            f"repos/{REPOSITORY}/git/ref/tags/{encoded_tag_name}"
        )
    except accelerator_gate.GateError as error:
        raise ReleaseGateError("could not resolve the published release tag") from error
    reference_object = _json_object(reference)
    if (
        reference_object is None
        or reference_object.get("ref") != f"refs/tags/{tag_name}"
    ):
        raise ReleaseGateError("GitHub did not return the exact published release tag")
    target: object = reference_object.get("object")
    seen: set[str] = set()
    for _ in range(MAX_TAG_PEEL_DEPTH):
        tag_target = _json_object(target)
        if tag_target is None:
            raise ReleaseGateError("published release tag target is malformed")
        object_type = tag_target.get("type")
        object_sha = tag_target.get("sha")
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
            tag_object = accelerator_gate.gh_api(
                f"repos/{REPOSITORY}/git/tags/{object_sha}"
            )
        except accelerator_gate.GateError as error:
            raise ReleaseGateError(
                "could not peel the published release tag"
            ) from error
        tag_object_data = _json_object(tag_object)
        if tag_object_data is None or tag_object_data.get("sha") != object_sha:
            raise ReleaseGateError("GitHub did not confirm the annotated tag object")
        target = tag_object_data.get("object")
    else:
        raise ReleaseGateError(
            "published release tag exceeds the annotated-tag depth limit"
        )


def _verify_context(commit: str) -> None:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name == "workflow_dispatch":
        try:
            accelerator_gate.require_dispatch_context()
        except accelerator_gate.GateError as error:
            raise ReleaseGateError(str(error)) from error
    elif event_name == "release":
        _verify_release_event(commit)
    else:
        raise ReleaseGateError(
            "release gate requires workflow_dispatch or release event"
        )
    try:
        verify_master_history(commit)
    except accelerator_gate.GateError as error:
        raise ReleaseGateError(str(error)) from error


def prepare(commit: str, output: Path, github_output: Path) -> Binding:
    if not accelerator_gate.SHA_RE.fullmatch(commit):
        raise ReleaseGateError("commit must be a full lowercase SHA")
    if output.resolve() == github_output.resolve():
        raise ReleaseGateError(
            "binding output and GITHUB_OUTPUT must be different files"
        )
    _verify_context(commit)
    run_id = accelerator_gate.run_id_from_env()
    binding = validate_binding(
        {
            "kind": "release",
            "head_sha": commit,
            "head_repository": REPOSITORY,
            "run_id": run_id,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = output.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
    with github_output.open("a", encoding="utf-8") as stream:
        _ = stream.write(f"head_sha={commit}\nrun_id={run_id}\n")
    return binding


def read_result(path: Path, binding: Binding, backend: str) -> None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseGateError(f"could not read result.json: {error}") from error
    result_object = _json_object(result)
    if result_object is None or set(result_object) != RESULT_FIELDS:
        raise ReleaseGateError("result.json has missing or unexpected fields")
    result = result_object
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
    devices_value = result["devices"]
    if not isinstance(devices_value, list) or not devices_value:
        raise ReleaseGateError(f"result.json has no {backend.upper()} devices")
    devices = cast(list[object], devices_value)
    if type(result["device_count"]) is not int or result["device_count"] != len(
        devices
    ):
        raise ReleaseGateError(
            "result.json device_count does not match its device list"
        )
    for device_value in devices:
        device = _json_object(device_value)
        if device is None or set(device) != {"platform", "device_kind", "id"}:
            raise ReleaseGateError(
                f"result.json contains invalid {backend.upper()} device metadata"
            )
        if device["platform"] != backend or not all(
            isinstance(device[key], str) and device[key]
            for key in ("platform", "device_kind", "id")
        ):
            raise ReleaseGateError(
                f"result.json contains invalid {backend.upper()} device metadata"
            )
    versions_value = result["versions"]
    versions = _json_object(versions_value)
    if (
        versions is None
        or not versions
        or not all(
            name and isinstance(version, str) and version
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
        binding = validate_binding(json.loads(binding_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseGateError(f"could not read binding file: {error}") from error
    current_run = accelerator_gate.run_id_from_env()
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
        read_result(output_dir / "result.json", binding, backend)
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
    _ = prepare_parser.add_argument("--commit", required=True)
    _ = prepare_parser.add_argument("--output", type=Path, required=True)
    _ = prepare_parser.add_argument("--github-output", type=Path, required=True)
    finish_parser = commands.add_parser("finish")
    _ = finish_parser.add_argument("--binding", type=Path, required=True)
    _ = finish_parser.add_argument("--output-dir", type=Path, required=True)
    _ = finish_parser.add_argument("--backend", choices=("gpu", "tpu"), default="tpu")
    _ = finish_parser.add_argument(
        "--provider-success", choices=("true", "false"), required=True
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            value = dict(prepare(args.commit, args.output, args.github_output))
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
