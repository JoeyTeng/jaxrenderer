"""Bind manually dispatched accelerator checks to an exact public PR head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import cast

REPOSITORY = "JoeyTeng/jaxrenderer"
BASE_BRANCH = "master"
CONTEXTS = {"gpu": "accelerator/gpu"}
BINDING_FIELDS = {
    "pr",
    "head_sha",
    "base_sha",
    "base_ref",
    "head_repository",
    "run_id",
}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^([1-9][0-9]*)\.([1-9][0-9]*)$")
REPOSITORY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GateError(RuntimeError):
    """A validation or GitHub API error that must fail the gate closed."""


def run_id_from_env() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run_id.isdigit() or int(run_id) <= 0:
        raise GateError("GITHUB_RUN_ID must be a positive integer")
    if not attempt.isdigit() or int(attempt) <= 0:
        raise GateError("GITHUB_RUN_ATTEMPT must be a positive integer")
    return f"{int(run_id)}.{int(attempt)}"


def gh_api(route: str, body: dict[str, object] | None = None) -> object:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GateError("GH_TOKEN or GITHUB_TOKEN is required for GitHub API access")
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    env["GH_PROMPT_DISABLED"] = "1"
    command = ["gh", "api", route]
    command.extend(["--header", "Cache-Control: no-cache"])
    encoded_body = None
    if body is not None:
        command.extend(["--method", "POST", "--input", "-"])
        encoded_body = json.dumps(body)
    try:
        completed = subprocess.run(
            command,
            input=encoded_body,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=45,
        )
    except subprocess.TimeoutExpired as error:
        raise GateError(f"gh api timed out for {route}") from error
    if completed.returncode:
        detail = completed.stderr.strip()[:400]
        raise GateError(f"gh api failed for {route}: {detail or 'unknown error'}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GateError(f"gh api returned invalid JSON for {route}") from error


def require_dispatch_context() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise GateError("prepare must run from workflow_dispatch")
    if os.environ.get("GITHUB_REF") != "refs/heads/master":
        raise GateError("prepare must run from the default master branch")
    if os.environ.get("GITHUB_REPOSITORY", "").casefold() != REPOSITORY.casefold():
        raise GateError(f"prepare is restricted to {REPOSITORY}")


def _read_pr(pr_number: int) -> dict[str, object]:
    value = gh_api(f"repos/{REPOSITORY}/pulls/{pr_number}")
    if not isinstance(value, dict):
        raise GateError("GitHub returned a malformed pull request")
    return cast(dict[str, object], value)


def _is_repository(value: object, expected: str) -> bool:
    return isinstance(value, str) and value.casefold() == expected.casefold()


def _master_sha() -> str:
    value = gh_api(f"repos/{REPOSITORY}/branches/{BASE_BRANCH}")
    if not isinstance(value, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    commit = cast(dict[str, object], value).get("commit")
    if not isinstance(commit, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    sha = cast(dict[str, object], commit).get("sha")
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise GateError("GitHub returned malformed master SHA")
    return sha


def _require_head_contains(base_sha: str, head_sha: str, description: str) -> None:
    comparison = gh_api(f"repos/{REPOSITORY}/compare/{base_sha}...{head_sha}")
    if not isinstance(comparison, dict) or cast(dict[str, object], comparison).get(
        "status"
    ) not in {"ahead", "identical"}:
        raise GateError(f"pull request head is not up to date with {description}")


def _binding_from_pr(
    pr_number: int, run_id: str, pr: dict[str, object] | None = None
) -> dict[str, object]:
    if pr is None:
        pr = _read_pr(pr_number)
    base = pr.get("base")
    head = pr.get("head")
    if not isinstance(base, dict) or not isinstance(head, dict):
        raise GateError("pull request is missing base or head metadata")
    base = cast(dict[str, object], base)
    head = cast(dict[str, object], head)
    base_repo = base.get("repo")
    head_repo = head.get("repo")
    if not isinstance(base_repo, dict) or not isinstance(head_repo, dict):
        raise GateError("pull request base or head repository is unavailable")
    base_repo = cast(dict[str, object], base_repo)
    head_repo = cast(dict[str, object], head_repo)
    base_ref = base.get("ref")
    base_sha = base.get("sha")
    head_sha = head.get("sha")
    head_repository = head_repo.get("full_name")
    if pr.get("state") != "open":
        raise GateError("pull request is not open")
    if not isinstance(base_ref, str) or not base_ref:
        raise GateError("pull request base reference is malformed")
    if not _is_repository(base_repo.get("full_name"), REPOSITORY):
        raise GateError(
            "pull request base repository does not match the gate repository"
        )
    if head_repo.get("private") is not False:
        raise GateError("pull request head repository must be public")
    if not isinstance(head_repository, str) or not REPOSITORY_NAME_RE.fullmatch(
        head_repository
    ):
        raise GateError("pull request head repository is unavailable")
    if not isinstance(base_sha, str) or not SHA_RE.fullmatch(base_sha):
        raise GateError("pull request base SHA is malformed")
    if not isinstance(head_sha, str) or not SHA_RE.fullmatch(head_sha):
        raise GateError("pull request head SHA is malformed")

    stack_value = pr.get("stack")
    if base_ref != BASE_BRANCH and not isinstance(stack_value, dict):
        raise GateError("non-master base requires native stack metadata")
    if stack_value is not None:
        if not isinstance(stack_value, dict):
            raise GateError("pull request has malformed native stack metadata")
        stack = cast(dict[str, object], stack_value)
        stack_base = stack.get("base")
        if not isinstance(stack_base, dict):
            raise GateError("pull request has malformed native stack base metadata")
        stack_base = cast(dict[str, object], stack_base)
        coordinates = ("id", "number", "position", "size")
        if (
            any(
                type(stack.get(key)) is not int or cast(int, stack[key]) <= 0
                for key in coordinates
            )
            or cast(int, stack["size"]) < cast(int, stack["position"])
            or stack_base.get("ref") != BASE_BRANCH
            or not isinstance(stack_base.get("sha"), str)
            or not SHA_RE.fullmatch(cast(str, stack_base["sha"]))
        ):
            raise GateError("pull request has malformed native stack metadata")
        if base_ref != BASE_BRANCH and not _is_repository(
            head_repo.get("full_name"), REPOSITORY
        ):
            raise GateError("native stack layers must use the target repository")
        if base_ref == BASE_BRANCH and stack_base["sha"] != base_sha:
            raise GateError("native stack base does not match the pull request base")
    live_master_sha = _master_sha()
    if (
        stack_value is not None
        and cast(dict[str, object], cast(dict[str, object], stack_value)["base"])["sha"]
        != live_master_sha
    ):
        raise GateError("native stack base does not match the live master SHA")
    _require_head_contains(base_sha, head_sha, "its base")
    _require_head_contains(live_master_sha, head_sha, "the live master")
    return {
        "pr": pr_number,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "base_ref": base_ref,
        "head_repository": head_repository,
        "run_id": run_id,
    }


def validate_binding(
    value: object, *, allow_non_master: bool = False
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GateError("binding.json has missing or unexpected fields")
    value = cast(dict[str, object], value)
    if set(value) != BINDING_FIELDS:
        raise GateError("binding.json has missing or unexpected fields")
    pr = value["pr"]
    if type(pr) is not int or pr <= 0:
        raise GateError("binding PR number is invalid")
    for name in ("head_sha", "base_sha"):
        sha = value[name]
        if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
            raise GateError(f"binding {name} is invalid")
    if (
        not isinstance(value["base_ref"], str)
        or not value["base_ref"]
        or (not allow_non_master and value["base_ref"] != BASE_BRANCH)
    ):
        raise GateError("binding base_ref is not master")
    head_repository = value["head_repository"]
    if not isinstance(head_repository, str) or not REPOSITORY_NAME_RE.fullmatch(
        head_repository
    ):
        raise GateError("binding head_repository is invalid")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise GateError("binding run_id is invalid")
    return value


def _run_url_parts(run_id: str) -> tuple[str, str]:
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None:
        raise GateError("run ID is malformed")
    return cast(tuple[str, str], match.groups())


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            _ = json.dump(value, temporary, sort_keys=True)
            _ = temporary.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_github_output(path: Path, values: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise GateError(f"unsafe multiline GitHub output: {name}")
            _ = output.write(f"{name}={rendered}\n")


def prepare(
    pr_number: int, backend_selection: str, output: Path, github_output: Path
) -> dict[str, object]:
    require_dispatch_context()
    if backend_selection != "gpu":
        raise GateError("PR accelerator gate only supports the GPU backend")
    if output.resolve() == github_output.resolve():
        raise GateError("binding output and GITHUB_OUTPUT must be different files")
    run_id = run_id_from_env()
    binding = _binding_from_pr(pr_number, run_id)
    atomic_write_json(output, binding)
    backends = (backend_selection,)
    run_number, attempt = _run_url_parts(run_id)
    run_url = (
        f"https://github.com/{REPOSITORY}/actions/runs/{run_number}/attempts/{attempt}"
    )
    for backend in backends:
        _ = gh_api(
            f"repos/{REPOSITORY}/statuses/{binding['head_sha']}",
            {
                "state": "pending",
                "context": CONTEXTS[backend],
                "description": f"{backend.upper()} accelerator run {run_id} pending",
                "target_url": run_url,
            },
        )
    _append_github_output(
        github_output,
        {
            "pr": pr_number,
            "head_sha": binding["head_sha"],
            "base_sha": binding["base_sha"],
            "run_id": run_id,
        },
    )
    return binding


def _load_binding(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"could not read binding file: {error}") from error
    return validate_binding(value, allow_non_master=True)


def _load_result(path: Path, binding: dict[str, object], backend: str) -> None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"could not read result.json: {error}") from error
    fields = {
        "pr",
        "head_sha",
        "base_sha",
        "run_id",
        "backend",
        "device_backend",
        "device_count",
        "success",
        "devices",
        "versions",
    }
    if not isinstance(result, dict):
        raise GateError("result.json has missing or unexpected fields")
    result = cast(dict[str, object], result)
    if set(result) != fields:
        raise GateError("result.json has missing or unexpected fields")
    for name in ("pr", "head_sha", "base_sha", "run_id"):
        if result[name] != binding[name] or type(result[name]) is not type(
            binding[name]
        ):
            raise GateError(f"result.json {name} does not match the frozen binding")
    if result["backend"] != backend:
        raise GateError("result.json backend does not match the selected backend")
    if result["device_backend"] != backend:
        raise GateError("result.json reports a different accelerator backend")
    if type(result["success"]) is not bool or result["success"] is not True:
        raise GateError("result.json does not report success")
    devices = result["devices"]
    if not isinstance(devices, list) or not devices:
        raise GateError("result.json has no accelerator devices")
    devices = cast(list[object], devices)
    if type(result["device_count"]) is not int or result["device_count"] != len(
        devices
    ):
        raise GateError("result.json device_count does not match its device list")
    for raw_device in devices:
        if not isinstance(raw_device, dict):
            raise GateError("result.json contains malformed device metadata")
        device = cast(dict[str, object], raw_device)
        if set(device) != {
            "platform",
            "device_kind",
            "id",
        }:
            raise GateError("result.json contains malformed device metadata")
        if device["platform"] != backend:
            raise GateError("result.json includes a device from the wrong backend")
        if any(
            not isinstance(device[key], str) or not device[key]
            for key in ("device_kind", "id")
        ):
            raise GateError("result.json contains incomplete device metadata")
    versions = result["versions"]
    if not isinstance(versions, dict):
        raise GateError("result.json versions must map non-empty names to versions")
    versions = cast(dict[str, object], versions)
    if not all(isinstance(value, str) and value for value in versions.values()):
        raise GateError("result.json versions must map non-empty names to versions")


def _current_pr_matches(binding: dict[str, object]) -> None:
    current = _binding_from_pr(cast(int, binding["pr"]), cast(str, binding["run_id"]))
    if current != binding:
        raise GateError("pull request identity or frozen head/base has changed")


def _latest_status(
    binding: dict[str, object], backend: str
) -> dict[str, object] | None:
    statuses = gh_api(
        f"repos/{REPOSITORY}/commits/{binding['head_sha']}/statuses?per_page=100"
    )
    if not isinstance(statuses, list):
        raise GateError("GitHub returned malformed commit statuses")
    statuses = cast(list[object], statuses)
    for raw_status in statuses:
        if isinstance(raw_status, dict):
            status = cast(dict[str, object], raw_status)
            if status.get("context") == CONTEXTS[backend]:
                return status
    return None


def finish(
    binding_path: Path, backend: str, output_dir: Path, provider_success: bool
) -> dict[str, object]:
    if backend not in CONTEXTS:
        raise GateError("backend must be gpu or tpu")
    binding = _load_binding(binding_path)
    run_id = run_id_from_env()
    if binding["run_id"] != run_id:
        run_number, attempt = _run_url_parts(cast(str, binding["run_id"]))
        return {
            "backend": backend,
            "state": "failure",
            "run_url": (
                f"https://github.com/{REPOSITORY}/actions/runs/"
                f"{run_number}/attempts/{attempt}"
            ),
            "reason": "binding belongs to a different workflow run or attempt",
            "status_updated": False,
        }
    failure_reason = ""
    if not provider_success:
        failure_reason = failure_reason or "provider command did not succeed"
    try:
        _load_result(output_dir / "result.json", binding, backend)
    except GateError as error:
        failure_reason = failure_reason or str(error)
    try:
        _current_pr_matches(binding)
    except GateError as error:
        failure_reason = failure_reason or str(error)
    latest = _latest_status(binding, backend)
    expected_pending = f"{backend.upper()} accelerator run {binding['run_id']} pending"
    if latest is None:
        failure_reason = failure_reason or "no pending status exists for this backend"
    elif (
        latest.get("state") != "pending"
        or latest.get("description") != expected_pending
    ):
        # A later invocation may already own this context; do not clobber it.
        run_number, attempt = _run_url_parts(cast(str, binding["run_id"]))
        return {
            "backend": backend,
            "state": "failure",
            "run_url": (
                f"https://github.com/{REPOSITORY}/actions/runs/"
                f"{run_number}/attempts/{attempt}"
            ),
            "reason": "latest status belongs to another run or is already terminal",
            "status_updated": False,
        }

    run_number, attempt = _run_url_parts(cast(str, binding["run_id"]))
    run_url = (
        f"https://github.com/{REPOSITORY}/actions/runs/{run_number}/attempts/{attempt}"
    )
    state = "failure" if failure_reason else "success"
    _ = gh_api(
        f"repos/{REPOSITORY}/statuses/{binding['head_sha']}",
        {
            "state": state,
            "context": CONTEXTS[backend],
            "description": (
                f"{backend.upper()} accelerator run {binding['run_id']} {state}"
                if not failure_reason
                else f"{backend.upper()} accelerator run {binding['run_id']} failed"
            ),
            "target_url": run_url,
        },
    )
    return {
        "backend": backend,
        "state": state,
        "run_url": run_url,
        "reason": failure_reason or None,
        "status_updated": True,
    }


def _positive_pr(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError(
            "PR number must be a canonical positive decimal integer"
        )
    return int(value)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    _ = prepare_parser.add_argument("--pr", required=True, type=_positive_pr)
    _ = prepare_parser.add_argument("--backend", choices=("gpu",), required=True)
    _ = prepare_parser.add_argument("--output", type=Path, required=True)
    _ = prepare_parser.add_argument("--github-output", type=Path, required=True)
    finish_parser = commands.add_parser("finish")
    _ = finish_parser.add_argument("--binding", type=Path, required=True)
    _ = finish_parser.add_argument("--backend", choices=tuple(CONTEXTS), required=True)
    _ = finish_parser.add_argument("--output-dir", type=Path, required=True)
    _ = finish_parser.add_argument(
        "--provider-success", choices=("true", "false"), required=True
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.pr, args.backend, args.output, args.github_output)
            exit_code = 0
        else:
            result = finish(
                args.binding,
                args.backend,
                args.output_dir,
                args.provider_success == "true",
            )
            exit_code = 0 if result["state"] == "success" else 1
        print(json.dumps(result, sort_keys=True))
        return exit_code
    except GateError as error:
        print(f"accelerator gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
