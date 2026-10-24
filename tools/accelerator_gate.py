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
MANIFEST_FIELDS = {
    "version",
    "run_id",
    "repository",
    "requested_pr",
    "trunk_sha",
    "stack",
    "pull_requests",
}


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
    if base_ref != BASE_BRANCH:
        raise GateError(f"pull request must target {BASE_BRANCH}")
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

    comparison = gh_api(f"repos/{REPOSITORY}/compare/{base_sha}...{head_sha}")
    if not isinstance(comparison, dict) or cast(dict[str, object], comparison).get(
        "status"
    ) not in {
        "ahead",
        "identical",
    }:
        raise GateError("pull request head is not up to date with its base")
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
    if not isinstance(value["base_ref"], str) or (
        not allow_non_master and value["base_ref"] != BASE_BRANCH
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


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise GateError(f"{name} is malformed")
    return value


def _pr_number(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise GateError("stack contains a malformed pull request number")
    return value


def _positive_id(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise GateError(f"{name} is malformed")
    return value


def _pr_parts(pr: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    base, head = pr.get("base"), pr.get("head")
    if not isinstance(base, dict) or not isinstance(head, dict):
        raise GateError("stack pull request is missing base or head metadata")
    return cast(dict[str, object], base), cast(dict[str, object], head)


def _stack_snapshot(
    requested_pr: int, run_id: str, initial: dict[str, object]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    stack_value = initial.get("stack")
    if stack_value is None:
        binding = _binding_from_pr(requested_pr, run_id, initial)
        base, head = _pr_parts(initial)
        trunk = gh_api(f"repos/{REPOSITORY}/branches/{BASE_BRANCH}")
        if not isinstance(trunk, dict):
            raise GateError("GitHub returned malformed master branch metadata")
        trunk_commit = cast(dict[str, object], trunk).get("commit")
        if not isinstance(trunk_commit, dict) or cast(
            dict[str, object], trunk_commit
        ).get("sha") != base.get("sha"):
            raise GateError("pull request base does not match the live master SHA")
        return (
            {"id": None, "number": None},
            [
                {
                    "pr": requested_pr,
                    "id": _positive_id(initial.get("id"), "pull request ID"),
                    "head_sha": binding["head_sha"],
                    "base_sha": binding["base_sha"],
                    "base_ref": binding["base_ref"],
                    "head_ref": head.get("ref"),
                    "head_repository": binding["head_repository"],
                }
            ],
        )
    if not isinstance(stack_value, dict):
        raise GateError("pull request has malformed native stack metadata")
    stack_ref = cast(dict[str, object], stack_value)
    stack_id = stack_ref.get("id")
    stack_number = stack_ref.get("number")
    position = stack_ref.get("position")
    size = stack_ref.get("size")
    base_ref = stack_ref.get("base")
    if not isinstance(base_ref, dict):
        raise GateError("pull request has malformed native stack base metadata")
    pr_stack_base_sha = _sha(
        cast(dict[str, object], base_ref).get("sha"), "pull request stack base SHA"
    )
    if (
        type(stack_id) is not int
        or stack_id <= 0
        or type(stack_number) is not int
        or stack_number <= 0
        or type(position) is not int
        or position <= 0
        or type(size) is not int
        or size < position
        or cast(dict[str, object], base_ref).get("ref") != BASE_BRANCH
    ):
        raise GateError("pull request has malformed native stack coordinates")
    stack = gh_api(f"repos/{REPOSITORY}/stacks/{stack_number}")
    if not isinstance(stack, dict):
        raise GateError("GitHub returned malformed native stack metadata")
    stack = cast(dict[str, object], stack)
    if (
        type(stack.get("id")) is not int
        or type(stack.get("number")) is not int
        or stack.get("id") != stack_id
        or stack.get("number") != stack_number
        or stack.get("open") is not True
    ):
        raise GateError("native stack identity or open state does not match")
    members_value = stack.get("pull_requests")
    if (
        not isinstance(members_value, list)
        or len(cast(list[object], members_value)) != size
    ):
        raise GateError("native stack has malformed ordered pull requests")
    members = cast(list[object], members_value)
    if position > len(members):
        raise GateError(
            "native stack position does not match the selected pull request"
        )
    member_numbers: list[int] = []
    member_ids: dict[int, int] = {}
    for member in members[:position]:
        if not isinstance(member, dict):
            raise GateError("native stack has malformed ordered pull requests")
        member_dict = cast(dict[str, object], member)
        number = _pr_number(member_dict.get("number"))
        member_id = _positive_id(member_dict.get("id"), "stack pull request ID")
        if number in member_numbers:
            raise GateError("native stack contains duplicate pull requests")
        member_numbers.append(number)
        member_ids[number] = member_id
    listed_base = stack.get("base")
    if not isinstance(listed_base, dict):
        raise GateError("native stack base metadata is missing")
    listed_base = cast(dict[str, object], listed_base)
    if listed_base.get("ref") != BASE_BRANCH:
        raise GateError("native stack base does not match master")
    if member_numbers[-1] != requested_pr:
        raise GateError(
            "native stack position does not match the selected pull request"
        )

    # Read every layer through the selected PR so merged layers cannot be hidden
    # by a partial or stale stack listing.
    active: list[dict[str, object]] = []
    active_stack_base_shas: list[str] = []
    previous_open = False
    for index, number in enumerate(member_numbers[:position], start=1):
        pr = _read_pr(number)
        if (
            type(pr.get("number")) is not int
            or pr.get("number") != number
            or pr.get("id") != member_ids[number]
        ):
            raise GateError("stack member identity does not match its pull request")
        state = pr.get("state")
        merged = pr.get("merged_at") is not None
        member = cast(dict[str, object], members[index - 1])
        if (
            not isinstance(member.get("state"), str)
            or member.get("state") != state
            or "merged_at" not in member
            or member.get("merged_at") != pr.get("merged_at")
            or (
                member.get("merged_at") is not None
                and not isinstance(member.get("merged_at"), str)
            )
            or (
                pr.get("merged_at") is not None
                and not isinstance(pr.get("merged_at"), str)
            )
        ):
            raise GateError("stack member state changed between API snapshots")
        for field in ("base", "head"):
            listed = member.get(field)
            current = pr.get(field)
            if not isinstance(listed, dict) or not isinstance(current, dict):
                raise GateError("stack member branch metadata is malformed")
            listed_branch = cast(dict[str, object], listed)
            current_branch = cast(dict[str, object], current)
            if (
                not isinstance(listed_branch.get("ref"), str)
                or not listed_branch["ref"]
                or not isinstance(current_branch.get("ref"), str)
                or not current_branch["ref"]
            ):
                raise GateError("stack member branch reference is malformed")
            _ = _sha(listed_branch.get("sha"), "stack member listed SHA")
            _ = _sha(current_branch.get("sha"), "stack member current SHA")
            if any(
                listed_branch.get(key) != current_branch.get(key)
                for key in ("ref", "sha")
            ):
                raise GateError("stack member branch snapshot does not match")
        if state == "closed" and merged:
            if previous_open:
                raise GateError("native stack has an interleaved merged layer")
            continue
        if state != "open" or merged:
            raise GateError("native stack contains a closed, unmerged layer")
        previous_open = True
        pr_stack = pr.get("stack")
        if not isinstance(pr_stack, dict):
            raise GateError("stack member has missing native stack coordinates")
        pr_stack = cast(dict[str, object], pr_stack)
        if (
            pr_stack.get("id") != stack_id
            or pr_stack.get("number") != stack_number
            or pr_stack.get("position") != index
            or pr_stack.get("size") != size
        ):
            raise GateError("stack member coordinates do not match ordered layers")
        pr_stack_base = pr_stack.get("base")
        if not isinstance(pr_stack_base, dict):
            raise GateError("stack member native base metadata is missing")
        pr_stack_base = cast(dict[str, object], pr_stack_base)
        if pr_stack_base.get("ref") != BASE_BRANCH:
            raise GateError("stack member native base does not match master")
        active_stack_base_shas.append(
            _sha(pr_stack_base.get("sha"), "stack member native base SHA")
        )
        base, head = _pr_parts(pr)
        base_repo, head_repo = base.get("repo"), head.get("repo")
        if not isinstance(base_repo, dict) or not isinstance(head_repo, dict):
            raise GateError("stack member repository metadata is unavailable")
        base_repo = cast(dict[str, object], base_repo)
        head_repo = cast(dict[str, object], head_repo)
        if not _is_repository(base_repo.get("full_name"), REPOSITORY):
            raise GateError("stack member targets a different repository")
        if not _is_repository(head_repo.get("full_name"), REPOSITORY):
            raise GateError("native stack layers must use the target repository")
        base_ref = base.get("ref")
        head_ref = head.get("ref")
        if not isinstance(base_ref, str) or not isinstance(head_ref, str):
            raise GateError("stack member branch reference is malformed")
        head_repository = head_repo.get("full_name")
        if not isinstance(head_repository, str) or not REPOSITORY_NAME_RE.fullmatch(
            head_repository
        ):
            raise GateError("stack member head repository is unavailable")
        active.append(
            {
                "pr": number,
                "id": _positive_id(pr.get("id"), "stack pull request ID"),
                "head_sha": _sha(head.get("sha"), "stack head SHA"),
                "base_sha": _sha(base.get("sha"), "stack base SHA"),
                "base_ref": base_ref,
                "head_ref": head_ref,
                "head_repository": head_repository,
            }
        )
    if not active or active[-1]["pr"] != requested_pr:
        raise GateError("selected pull request is not an active stack layer")

    trunk = gh_api(f"repos/{REPOSITORY}/branches/{BASE_BRANCH}")
    if not isinstance(trunk, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    trunk = cast(dict[str, object], trunk)
    trunk_commit = trunk.get("commit")
    if not isinstance(trunk_commit, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    trunk_sha = _sha(cast(dict[str, object], trunk_commit).get("sha"), "master SHA")
    if pr_stack_base_sha != trunk_sha or any(
        base_sha != trunk_sha for base_sha in active_stack_base_shas
    ):
        raise GateError("native stack base does not match the live master SHA")
    previous_sha = trunk_sha
    previous_ref = BASE_BRANCH
    for layer in active:
        if layer["base_ref"] != previous_ref or layer["base_sha"] != previous_sha:
            raise GateError("native stack base/head references are not contiguous")
        comparison = gh_api(
            f"repos/{REPOSITORY}/compare/{layer['base_sha']}...{layer['head_sha']}"
        )
        if not isinstance(comparison, dict) or cast(dict[str, object], comparison).get(
            "status"
        ) not in {"ahead", "identical"}:
            raise GateError("stack pull request head is not up to date with its base")
        previous_ref = cast(str, layer["head_ref"])
        previous_sha = cast(str, layer["head_sha"])

    # Persist the original native coordinates, including position/size, so the
    # final read can detect reordered, resized, or replaced stacks.
    return {"id": stack_id, "number": stack_number}, active


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"could not read stack manifest: {error}") from error
    if not isinstance(value, dict):
        raise GateError("stack manifest has missing or unexpected fields")
    manifest = cast(dict[str, object], value)
    if set(manifest) != MANIFEST_FIELDS:
        raise GateError("stack manifest has missing or unexpected fields")
    if (
        type(manifest.get("version")) is not int
        or manifest.get("version") != 1
        or manifest.get("repository") != REPOSITORY
    ):
        raise GateError("stack manifest version or repository is invalid")
    requested_pr = manifest.get("requested_pr")
    if type(requested_pr) is not int or requested_pr <= 0:
        raise GateError("stack manifest selected pull request is invalid")
    if not isinstance(manifest.get("run_id"), str) or not RUN_ID_RE.fullmatch(
        cast(str, manifest["run_id"])
    ):
        raise GateError("stack manifest run ID is invalid")
    _ = _sha(manifest.get("trunk_sha"), "stack manifest trunk SHA")
    stack = manifest.get("stack")
    if not isinstance(stack, dict) or set(cast(dict[str, object], stack)) != {
        "id",
        "number",
    }:
        raise GateError("stack manifest stack identity is malformed")
    stack_identity = cast(dict[str, object], stack)
    if stack_identity["id"] is None or stack_identity["number"] is None:
        if stack_identity["id"] is not None or stack_identity["number"] is not None:
            raise GateError("stack manifest stack identity is malformed")
    else:
        _ = _positive_id(stack_identity["id"], "manifest stack ID")
        _ = _positive_id(stack_identity["number"], "manifest stack number")
    layers_value = manifest.get("pull_requests")
    if (
        not isinstance(layers_value, list)
        or not layers_value
        or len(cast(list[object], layers_value)) > 256
    ):
        raise GateError("stack manifest pull request scope is invalid")
    layers = cast(list[object], layers_value)
    seen: set[int] = set()
    for layer_value in layers:
        if not isinstance(layer_value, dict):
            raise GateError("stack manifest contains a malformed pull request")
        layer = cast(dict[str, object], layer_value)
        if set(layer) != {
            "pr",
            "id",
            "head_sha",
            "base_sha",
            "base_ref",
            "head_ref",
            "head_repository",
        }:
            raise GateError("stack manifest contains a malformed pull request")
        number = _pr_number(layer.get("pr"))
        _ = _positive_id(layer.get("id"), "manifest pull request ID")
        if number in seen:
            raise GateError("stack manifest contains duplicate pull requests")
        seen.add(number)
        _ = _sha(layer.get("head_sha"), "manifest head SHA")
        _ = _sha(layer.get("base_sha"), "manifest base SHA")
        for key in ("base_ref", "head_ref", "head_repository"):
            if not isinstance(layer.get(key), str) or not layer[key]:
                raise GateError("stack manifest contains malformed branch metadata")
        if not REPOSITORY_NAME_RE.fullmatch(cast(str, layer["head_repository"])):
            raise GateError("stack manifest contains malformed head repository")
    head_shas = [cast(dict[str, object], layer)["head_sha"] for layer in layers]
    if len(set(head_shas)) != len(head_shas):
        raise GateError("stack manifest contains duplicate active head SHAs")
    if manifest["requested_pr"] != cast(dict[str, object], layers[-1])["pr"]:
        raise GateError(
            "stack manifest scope does not end at the selected pull request"
        )
    return manifest


def prepare_stack(
    pr_number: int, backend: str, output_dir: Path, github_output: Path
) -> dict[str, object]:
    require_dispatch_context()
    if backend != "gpu":
        raise GateError("PR accelerator gate only supports the GPU backend")
    if output_dir.resolve() == github_output.resolve():
        raise GateError("binding output and GITHUB_OUTPUT must be different files")
    run_id = run_id_from_env()
    initial = _read_pr(pr_number)
    stack_identity, layers = _stack_snapshot(pr_number, run_id, initial)
    if len(layers) > 256:
        raise GateError("native stack exceeds the 256-entry matrix limit")
    head_shas = [layer["head_sha"] for layer in layers]
    if len(set(head_shas)) != len(head_shas):
        raise GateError("native stack contains duplicate active head SHAs")
    trunk_sha = cast(str, layers[0]["base_sha"])
    manifest = {
        "version": 1,
        "run_id": run_id,
        "repository": REPOSITORY,
        "requested_pr": pr_number,
        "trunk_sha": trunk_sha,
        "stack": stack_identity,
        "pull_requests": layers,
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_json(manifest_path, cast(dict[str, object], manifest))
    matrix: list[dict[str, object]] = []
    for layer in layers:
        binding = {
            "pr": layer["pr"],
            "head_sha": layer["head_sha"],
            "base_sha": layer["base_sha"],
            "base_ref": layer["base_ref"],
            "head_repository": layer["head_repository"],
            "run_id": run_id,
        }
        binding_path = output_dir / f"pr-{layer['pr']}.json"
        atomic_write_json(binding_path, cast(dict[str, object], binding))
        matrix.append(
            {
                "pr": layer["pr"],
                "head_sha": layer["head_sha"],
                "binding": binding_path.as_posix(),
            }
        )
    _append_github_output(
        github_output,
        {
            "matrix": json.dumps({"include": matrix}, separators=(",", ":")),
            "manifest": manifest_path.as_posix(),
            "run_id": run_id,
            "count": len(layers),
        },
    )

    # Re-read stack metadata and every frozen PR immediately before publishing
    # statuses, so a moving capture cannot initialise only part of the matrix.
    final_identity, final_layers = _stack_snapshot(
        pr_number, run_id, _read_pr(pr_number)
    )
    if final_identity != stack_identity or final_layers != layers:
        raise GateError("native stack changed while preparing the matrix")
    run_number, attempt = _run_url_parts(run_id)
    run_url = (
        f"https://github.com/{REPOSITORY}/actions/runs/{run_number}/attempts/{attempt}"
    )
    for layer in layers:
        _ = gh_api(
            f"repos/{REPOSITORY}/statuses/{layer['head_sha']}",
            {
                "state": "pending",
                "context": CONTEXTS[backend],
                "description": f"{backend.upper()} accelerator run {run_id} pending",
                "target_url": run_url,
            },
        )
    return {
        "matrix": {"include": matrix},
        "manifest": manifest_path.as_posix(),
        "run_id": run_id,
        "count": len(layers),
    }


def _load_binding(path: Path, *, allow_non_master: bool = False) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"could not read binding file: {error}") from error
    return validate_binding(value, allow_non_master=allow_non_master)


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
    pr = _read_pr(binding["pr"])  # type: ignore[arg-type]
    base = pr.get("base")
    head = pr.get("head")
    if not isinstance(base, dict) or not isinstance(head, dict):
        raise GateError("current pull request is missing base or head metadata")
    base = cast(dict[str, object], base)
    head = cast(dict[str, object], head)
    head_repo = head.get("repo")
    if not isinstance(head_repo, dict):
        raise GateError("current pull request head repository is unavailable")
    head_repo = cast(dict[str, object], head_repo)
    base_repo_value = base.get("repo")
    if not isinstance(base_repo_value, dict):
        raise GateError("current pull request base repository is unavailable")
    base_repo = cast(dict[str, object], base_repo_value)
    if (
        pr.get("state") != "open"
        or not _is_repository(base_repo.get("full_name"), REPOSITORY)
        or base.get("ref") != binding["base_ref"]
        or base.get("sha") != binding["base_sha"]
        or head.get("sha") != binding["head_sha"]
        or head_repo.get("full_name") != binding["head_repository"]
        or head_repo.get("private") is not False
    ):
        raise GateError("pull request identity or frozen head/base has changed")
    comparison = gh_api(
        f"repos/{REPOSITORY}/compare/{binding['base_sha']}...{binding['head_sha']}"
    )
    if not isinstance(comparison, dict) or cast(dict[str, object], comparison).get(
        "status"
    ) not in {
        "ahead",
        "identical",
    }:
        raise GateError("pull request head is no longer up to date with its base")


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


def _manifest_matches(path: Path, binding: dict[str, object]) -> dict[str, object]:
    manifest = _read_manifest(path)
    if manifest["run_id"] != binding["run_id"]:
        raise GateError("manifest belongs to a different workflow run or attempt")
    layers = cast(list[dict[str, object]], manifest["pull_requests"])
    layer = next((item for item in layers if item["pr"] == binding["pr"]), None)
    if layer is None:
        raise GateError("binding PR is not a member of the frozen stack manifest")
    expected = {
        "pr": layer["pr"],
        "head_sha": layer["head_sha"],
        "base_sha": layer["base_sha"],
        "base_ref": layer["base_ref"],
        "head_repository": layer["head_repository"],
        "run_id": manifest["run_id"],
    }
    if binding != expected:
        raise GateError("binding does not match its frozen stack manifest member")

    selected = _read_pr(cast(int, manifest["requested_pr"]))
    identity, current_layers = _stack_snapshot(
        cast(int, manifest["requested_pr"]), cast(str, manifest["run_id"]), selected
    )
    if identity != manifest["stack"] or current_layers != layers:
        raise GateError(
            "native stack topology or frozen pull request coordinates changed"
        )
    if current_layers[0]["base_sha"] != manifest["trunk_sha"]:
        raise GateError("frozen master SHA has changed")
    trunk = gh_api(f"repos/{REPOSITORY}/branches/{BASE_BRANCH}")
    if not isinstance(trunk, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    trunk = cast(dict[str, object], trunk)
    trunk_commit = trunk.get("commit")
    if not isinstance(trunk_commit, dict):
        raise GateError("GitHub returned malformed master branch metadata")
    live_trunk_sha = _sha(
        cast(dict[str, object], trunk_commit).get("sha"), "master SHA"
    )
    if live_trunk_sha != manifest["trunk_sha"]:
        raise GateError("frozen master SHA has changed")
    return manifest


def finish(
    binding_path: Path,
    backend: str,
    output_dir: Path,
    provider_success: bool,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    if backend not in CONTEXTS:
        raise GateError("backend must be gpu or tpu")
    binding = _load_binding(binding_path, allow_non_master=manifest_path is not None)
    run_id = run_id_from_env()
    if binding["run_id"] != run_id:
        return {
            "backend": backend,
            "state": "failure",
            "run_url": None,
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
    if manifest_path is not None:
        try:
            _ = _manifest_matches(manifest_path, binding)
        except GateError as error:
            failure_reason = failure_reason or str(error)
    else:
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


def close_pending(manifest_path: Path, backend: str) -> dict[str, object]:
    if backend not in CONTEXTS:
        raise GateError("backend must be gpu or tpu")
    manifest = _read_manifest(manifest_path)
    run_id = cast(str, manifest["run_id"])
    if run_id != run_id_from_env():
        raise GateError("manifest belongs to a different workflow run or attempt")
    run_number, attempt = _run_url_parts(run_id)
    run_url = (
        f"https://github.com/{REPOSITORY}/actions/runs/{run_number}/attempts/{attempt}"
    )
    closed: list[int] = []
    errors: list[str] = []
    for layer_value in cast(list[object], manifest["pull_requests"]):
        layer = cast(dict[str, object], layer_value)
        head_sha = cast(str, layer["head_sha"])
        try:
            latest = _latest_status({"head_sha": head_sha}, backend)
            expected = f"{backend.upper()} accelerator run {run_id} pending"
            if (
                latest is None
                or latest.get("state") != "pending"
                or latest.get("description") != expected
            ):
                continue
            _ = gh_api(
                f"repos/{REPOSITORY}/statuses/{head_sha}",
                {
                    "state": "failure",
                    "context": CONTEXTS[backend],
                    "description": f"{backend.upper()} accelerator run {run_id} failed",
                    "target_url": run_url,
                },
            )
            closed.append(cast(int, layer["pr"]))
        except GateError as error:
            errors.append(f"PR {layer['pr']}: {error}")
    return {
        "backend": backend,
        "state": "failure" if closed or errors else "success",
        "closed": closed,
        "errors": errors,
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
    _ = finish_parser.add_argument("--manifest", type=Path)
    _ = finish_parser.add_argument("--backend", choices=tuple(CONTEXTS), required=True)
    _ = finish_parser.add_argument("--output-dir", type=Path, required=True)
    _ = finish_parser.add_argument(
        "--provider-success", choices=("true", "false"), required=True
    )
    stack_parser = commands.add_parser("prepare-stack")
    _ = stack_parser.add_argument("--pr", required=True, type=_positive_pr)
    _ = stack_parser.add_argument("--backend", choices=("gpu",), required=True)
    _ = stack_parser.add_argument("--output-dir", type=Path, required=True)
    _ = stack_parser.add_argument("--github-output", type=Path, required=True)
    cleanup_parser = commands.add_parser("close-pending")
    _ = cleanup_parser.add_argument("--manifest", type=Path, required=True)
    _ = cleanup_parser.add_argument("--backend", choices=tuple(CONTEXTS), required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.pr, args.backend, args.output, args.github_output)
            exit_code = 0
        elif args.command == "prepare-stack":
            result = prepare_stack(
                args.pr, args.backend, args.output_dir, args.github_output
            )
            exit_code = 0
        elif args.command == "close-pending":
            result = close_pending(args.manifest, args.backend)
            exit_code = 0 if result["state"] == "success" else 1
        else:
            result = finish(
                args.binding,
                args.backend,
                args.output_dir,
                args.provider_success == "true",
                args.manifest,
            )
            exit_code = 0 if result["state"] == "success" else 1
        print(json.dumps(result, sort_keys=True))
        return exit_code
    except GateError as error:
        print(f"accelerator gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
