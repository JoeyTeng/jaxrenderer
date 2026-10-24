"""Tests for native-stack accelerator bindings."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import ContextManager, cast
from unittest.mock import MagicMock, patch

import pytest
from tools import accelerator_gate as gate

TRUNK = "0" * 40
SHA_ONE = "1" * 40
SHA_TWO = "2" * 40
SHA_THREE = "3" * 40
RUN_ID = "456.2"
HEADS = {1: SHA_ONE, 2: SHA_TWO, 3: SHA_THREE}


def stack_pr(
    number: int, state: str = "open", merged: bool = False
) -> dict[str, object]:
    position = number
    base_ref = "master" if number == 1 else f"topic-{number - 1}"
    base_sha = TRUNK if number == 1 else HEADS[number - 1]
    return {
        "id": 1000 + number,
        "number": number,
        "state": state,
        "merged_at": "2026-10-01T00:00:00Z" if merged else None,
        "base": {
            "ref": base_ref,
            "sha": base_sha,
            "repo": {"full_name": gate.REPOSITORY},
        },
        "head": {
            "ref": f"topic-{number}",
            "sha": HEADS[number],
            "repo": {"full_name": gate.REPOSITORY, "private": False},
        },
        "stack": {
            "id": 2557273,
            "number": 56,
            "position": position,
            "size": 3,
            "base": {"ref": "master", "sha": TRUNK},
        },
    }


class StackGitHub:
    def __init__(self) -> None:
        super().__init__()
        self.prs = {number: stack_pr(number) for number in HEADS}
        self.posts: list[dict[str, object]] = []
        self.statuses: dict[str, list[dict[str, object]]] = {
            sha: [] for sha in HEADS.values()
        }
        self.stack_open = True
        self.fail_post_number: int | None = None
        self.post_count = 0

    def __call__(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        route = command[2]
        if route.startswith(f"repos/{gate.REPOSITORY}/pulls/"):
            number = int(route.rsplit("/", 1)[1])
            value: object = self.prs[number]
        elif route == f"repos/{gate.REPOSITORY}/stacks/56":
            value = {
                "id": 2557273,
                "number": 56,
                "open": self.stack_open,
                "base": {"ref": "master"},
                "pull_requests": [
                    {
                        "id": 1000 + number,
                        "number": number,
                        "state": self.prs[number]["state"],
                        "merged_at": self.prs[number]["merged_at"],
                        "base": self.prs[number]["base"],
                        "head": self.prs[number]["head"],
                    }
                    for number in sorted(self.prs)
                ],
            }
        elif route == f"repos/{gate.REPOSITORY}/branches/master":
            value = {"commit": {"sha": TRUNK}}
        elif "/compare/" in route:
            value = {"status": "ahead"}
        elif "/statuses?per_page=100" in route:
            sha = route.split("/commits/", 1)[1].split("/statuses", 1)[0]
            value = self.statuses[sha]
        elif route.startswith(f"repos/{gate.REPOSITORY}/statuses/"):
            body = json.loads(str(kwargs["input"]))
            self.post_count += 1
            if self.post_count == self.fail_post_number:
                return subprocess.CompletedProcess(command, 1, "", "simulated failure")
            self.posts.append(body)
            sha = route.rsplit("/", 1)[1]
            self.statuses[sha].insert(
                0,
                {
                    "context": body["context"],
                    "state": body["state"],
                    "description": body["description"],
                },
            )
            value = body
        else:
            raise AssertionError(f"unexpected GitHub route: {route}")
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> StackGitHub:
    monkeypatch.setenv("GITHUB_TOKEN", "codex_synth_v1_access_a")
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    return StackGitHub()


def install_api(github: StackGitHub) -> ContextManager[MagicMock]:
    return patch.object(gate.subprocess, "run", side_effect=github)


def test_prepare_stack_freezes_only_prefix_through_selected_pr(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    github_output = tmp_path / "github-output"
    with install_api(github):
        result = gate.prepare_stack(2, "gpu", output_dir, github_output)

    assert result["count"] == 2
    matrix = result["matrix"]
    assert isinstance(matrix, dict)
    include = cast(list[dict[str, object]], matrix["include"])
    assert [item["pr"] for item in include] == [1, 2]
    assert not (output_dir / "pr-3.json").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["trunk_sha"] == TRUNK
    assert [layer["pr"] for layer in manifest["pull_requests"]] == [1, 2]
    assert [post["state"] for post in github.posts] == ["pending", "pending"]
    assert github_output.read_text(encoding="utf-8").splitlines()[1:] == [
        f"manifest={output_dir / 'manifest.json'}",
        f"run_id={RUN_ID}",
        "count=2",
    ]


def test_prepare_stack_keeps_ordinary_single_pr_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    pr = {
        "id": 1024,
        "number": 24,
        "state": "open",
        "base": {
            "ref": "master",
            "sha": TRUNK,
            "repo": {"full_name": gate.REPOSITORY},
        },
        "head": {
            "ref": "topic",
            "sha": SHA_ONE,
            "repo": {"full_name": "Contributor/jaxrenderer", "private": False},
        },
    }

    def api(route: str, body: dict[str, object] | None = None) -> object:
        if route == f"repos/{gate.REPOSITORY}/pulls/24":
            return pr
        if route == f"repos/{gate.REPOSITORY}/compare/{TRUNK}...{SHA_ONE}":
            return {"status": "ahead"}
        if route == f"repos/{gate.REPOSITORY}/branches/master":
            return {"commit": {"sha": TRUNK}}
        if route == f"repos/{gate.REPOSITORY}/statuses/{SHA_ONE}" and body is not None:
            return body
        raise AssertionError(f"unexpected route: {route} {body}")

    monkeypatch.setattr(gate, "gh_api", api)
    output_dir = tmp_path / "bindings"
    result = gate.prepare_stack(24, "gpu", output_dir, tmp_path / "out")
    assert result["count"] == 1
    binding = json.loads((output_dir / "pr-24.json").read_text(encoding="utf-8"))
    assert binding["pr"] == 24
    assert binding["base_ref"] == "master"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stack"] == {"id": None, "number": None}
    assert manifest["pull_requests"][0]["head_ref"] == "topic"


def test_prepare_stack_ignores_appended_layer_above_selected_prefix(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")

    for number in (1, 2, 3):
        stack = cast(dict[str, object], github.prs[number]["stack"])
        stack["size"] = 4
    fourth = stack_pr(3)
    fourth["number"] = 4
    fourth["id"] = 1004
    fourth["base"] = {
        "ref": "topic-3",
        "sha": SHA_THREE,
        "repo": {"full_name": gate.REPOSITORY},
    }
    fourth["head"] = {
        "ref": "topic-4",
        "sha": "4" * 40,
        "repo": {"full_name": gate.REPOSITORY},
    }
    fourth["stack"] = {
        "id": 2557273,
        "number": 56,
        "position": 4,
        "size": 4,
        "base": {"ref": "master", "sha": TRUNK},
    }
    github.prs[4] = fourth
    with install_api(github):
        result = gate.finish(
            output_dir / "pr-2.json",
            "gpu",
            tmp_path / "missing-result",
            provider_success=False,
            manifest_path=output_dir / "manifest.json",
        )
    assert result["status_updated"] is True
    assert "provider command did not succeed" in str(result["reason"])


def test_prepare_stack_skips_only_a_merged_lower_prefix(
    tmp_path: Path, github: StackGitHub
) -> None:
    github.prs[1] = stack_pr(1, state="closed", merged=True)
    # The surviving first active layer is rebased onto master.
    github.prs[2]["base"] = {
        "ref": "master",
        "sha": TRUNK,
        "repo": {"full_name": gate.REPOSITORY},
    }
    github.prs[2]["stack"] = {
        "id": 2557273,
        "number": 56,
        "position": 2,
        "size": 3,
        "base": {"ref": "master", "sha": TRUNK},
    }
    with install_api(github):
        result = gate.prepare_stack(2, "gpu", tmp_path / "bindings", tmp_path / "out")

    assert result["count"] == 1
    manifest = json.loads(
        (tmp_path / "bindings" / "manifest.json").read_text(encoding="utf-8")
    )
    assert [layer["pr"] for layer in manifest["pull_requests"]] == [2]


def test_prepare_stack_rejects_duplicate_active_head_status_ownership(
    tmp_path: Path, github: StackGitHub
) -> None:
    second_head = cast(dict[str, object], github.prs[2]["head"])
    second_head["sha"] = SHA_ONE
    with install_api(github):
        with pytest.raises(gate.GateError, match="duplicate active head SHAs"):
            _ = gate.prepare_stack(2, "gpu", tmp_path / "bindings", tmp_path / "out")
    assert github.posts == []


def test_stack_finish_requires_matching_layer_and_live_topology(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")
    binding_path = output_dir / "pr-2.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["head_sha"] == SHA_TWO
    with install_api(github):
        result = gate.finish(
            binding_path,
            "gpu",
            tmp_path / "missing-result",
            provider_success=False,
            manifest_path=output_dir / "manifest.json",
        )
    assert result["state"] == "failure"
    assert result["status_updated"] is True
    assert github.posts[-1]["state"] == "failure"


def test_each_stack_binding_requires_its_own_head_result(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")

    for number, head_sha in ((1, SHA_ONE), (2, SHA_TWO)):
        binding_path = output_dir / f"pr-{number}.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        result_dir = tmp_path / f"provider-{number}"
        result_dir.mkdir()
        _ = (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "pr": number,
                    "head_sha": head_sha,
                    "base_sha": binding["base_sha"],
                    "run_id": RUN_ID,
                    "backend": "gpu",
                    "device_backend": "gpu",
                    "device_count": 1,
                    "success": True,
                    "devices": [
                        {"platform": "gpu", "device_kind": "T4", "id": str(number)}
                    ],
                    "versions": {"jax": "1"},
                }
            ),
            encoding="utf-8",
        )
        with install_api(github):
            finished = gate.finish(
                binding_path,
                "gpu",
                result_dir,
                provider_success=True,
                manifest_path=output_dir / "manifest.json",
            )
        assert finished["state"] == "success"
        assert github.statuses[head_sha][0]["state"] == "success"


def test_prepare_partial_statuses_can_be_closed_from_manifest(
    tmp_path: Path, github: StackGitHub
) -> None:
    github.fail_post_number = 2
    output_dir = tmp_path / "bindings"
    with install_api(github):
        with pytest.raises(gate.GateError, match="gh api failed"):
            _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")
    assert (output_dir / "manifest.json").exists()
    assert len(github.posts) == 1

    github.fail_post_number = None
    with install_api(github):
        cleanup = gate.close_pending(output_dir / "manifest.json", "gpu")
    assert cleanup["closed"] == [1]
    assert github.statuses[SHA_ONE][0]["state"] == "failure"


def test_finish_and_cleanup_reject_a_different_attempt(
    tmp_path: Path, github: StackGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    with install_api(github):
        finished = gate.finish(
            output_dir / "pr-2.json",
            "gpu",
            tmp_path / "missing-result",
            provider_success=False,
            manifest_path=output_dir / "manifest.json",
        )
        with pytest.raises(gate.GateError, match="different workflow run or attempt"):
            _ = gate.close_pending(output_dir / "manifest.json", "gpu")
    assert finished["status_updated"] is False
    assert len(github.posts) == 2


def test_stack_finish_rejects_changed_parent_head(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")
    parent_head = cast(dict[str, object], github.prs[1]["head"])
    parent_head["sha"] = "a" * 40
    binding_path = output_dir / "pr-2.json"
    result_dir = tmp_path / "provider"
    result_dir.mkdir()
    _ = (result_dir / "result.json").write_text(
        json.dumps(
            {
                "pr": 2,
                "head_sha": SHA_TWO,
                "base_sha": SHA_ONE,
                "run_id": RUN_ID,
                "backend": "gpu",
                "device_backend": "gpu",
                "device_count": 1,
                "success": True,
                "devices": [{"platform": "gpu", "device_kind": "T4", "id": "0"}],
                "versions": {"jax": "1"},
            }
        ),
        encoding="utf-8",
    )
    with install_api(github):
        result = gate.finish(
            binding_path,
            "gpu",
            result_dir,
            provider_success=True,
            manifest_path=output_dir / "manifest.json",
        )
    assert result["state"] == "failure"
    assert result["reason"] is not None


def test_close_pending_only_closes_exact_run_owned_statuses(
    tmp_path: Path, github: StackGitHub
) -> None:
    output_dir = tmp_path / "bindings"
    with install_api(github):
        _ = gate.prepare_stack(2, "gpu", output_dir, tmp_path / "out")
    # A newer run owns PR 1; only this run's PR 2 status may be closed.
    github.statuses[SHA_ONE].insert(
        0,
        {
            "context": gate.CONTEXTS["gpu"],
            "state": "pending",
            "description": "GPU accelerator run 457.1 pending",
        },
    )
    with install_api(github):
        result = gate.close_pending(output_dir / "manifest.json", "gpu")

    assert result["closed"] == [2]
    assert (
        github.statuses[SHA_ONE][0]["description"]
        == "GPU accelerator run 457.1 pending"
    )
    assert github.posts[-1]["state"] == "failure"
