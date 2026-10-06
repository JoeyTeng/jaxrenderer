"""Tests for exact-commit accelerator status reporting."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import pytest
from tools import accelerator_gate as gate

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
RUN_ID = "456.2"
# Synthetic fixture catalogue: access-a and access-b (active), joey-private-v3.
SYNTHETIC_ACCESS_A = "codex_synth_v1_access_a"
SYNTHETIC_ACCESS_B = "codex_synth_v1_access_b"


def pull_request() -> dict[str, Any]:
    return {
        "state": "open",
        "base": {
            "ref": "master",
            "sha": BASE_SHA,
            "repo": {"full_name": gate.REPOSITORY},
        },
        "head": {
            "sha": HEAD_SHA,
            "repo": {"full_name": "Contributor/jaxrenderer", "private": False},
        },
    }


def binding() -> dict[str, object]:
    return {
        "pr": 24,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "base_ref": "master",
        "head_repository": "Contributor/jaxrenderer",
        "run_id": RUN_ID,
    }


def result(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "pr": 24,
        "head_sha": HEAD_SHA,
        "base_sha": BASE_SHA,
        "run_id": RUN_ID,
        "backend": "gpu",
        "device_backend": "gpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "gpu", "device_kind": "T4", "id": "0"}],
        "versions": {"jax": "0.11.2", "numpy": "2.1.3"},
    }
    value.update(overrides)
    return value


class FakeGitHub:
    def __init__(self) -> None:
        self.pr = pull_request()
        self.compare_status = "ahead"
        self.statuses: list[dict[str, object]] = []
        self.posts: list[tuple[str, dict[str, object]]] = []

    def __call__(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        route = command[2]
        if route == f"repos/{gate.REPOSITORY}/pulls/24":
            value: object = self.pr
        elif route == f"repos/{gate.REPOSITORY}/compare/{BASE_SHA}...{HEAD_SHA}":
            value = {"status": self.compare_status}
        elif (
            route == f"repos/{gate.REPOSITORY}/commits/{HEAD_SHA}/statuses?per_page=100"
        ):
            value = self.statuses
        elif route == f"repos/{gate.REPOSITORY}/statuses/{HEAD_SHA}":
            body = json.loads(str(kwargs["input"]))
            self.posts.append((route, body))
            self.statuses.insert(
                0,
                {
                    "context": body["context"],
                    "state": body["state"],
                    "description": body["description"],
                },
            )
            value = body
        else:
            raise AssertionError(f"unexpected GitHub API route: {route}")
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> FakeGitHub:
    monkeypatch.setenv("GITHUB_TOKEN", SYNTHETIC_ACCESS_A)
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    return FakeGitHub()


def install_fake_github(github: FakeGitHub) -> patch:
    return patch.object(gate.subprocess, "run", side_effect=github)


def write_binding(path: Path) -> Path:
    path.write_text(json.dumps(binding()), encoding="utf-8")
    return path


def write_result(directory: Path, data: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(data), encoding="utf-8")


def add_pending_status(github: FakeGitHub, backend: str = "gpu") -> None:
    github.statuses.append(
        {
            "context": gate.CONTEXTS[backend],
            "state": "pending",
            "description": f"{backend.upper()} accelerator run {RUN_ID} pending",
        }
    )


def test_prepare_freezes_pr_and_marks_only_selected_backend(
    tmp_path: Path, github: FakeGitHub
) -> None:
    binding_path = tmp_path / "binding.json"
    output_path = tmp_path / "github-output.txt"
    with install_fake_github(github):
        actual = gate.prepare(24, "gpu", binding_path, output_path)

    assert actual == binding()
    assert json.loads(binding_path.read_text(encoding="utf-8")) == binding()
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "pr=24",
        f"head_sha={HEAD_SHA}",
        f"base_sha={BASE_SHA}",
        f"run_id={RUN_ID}",
    ]
    assert [body["context"] for _, body in github.posts] == ["accelerator/gpu"]
    assert github.posts[0][1]["state"] == "pending"
    assert github.posts[0][1]["target_url"] == (
        "https://github.com/JoeyTeng/jaxrenderer/actions/runs/456/attempts/2"
    )


def test_prepare_rejects_non_default_dispatch_before_api(
    tmp_path: Path, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    with install_fake_github(github) as api:
        with pytest.raises(gate.GateError, match="default master"):
            gate.prepare(24, "gpu", tmp_path / "binding.json", tmp_path / "out")
    api.assert_not_called()


def test_prepare_rejects_head_that_is_behind_base(
    tmp_path: Path, github: FakeGitHub
) -> None:
    github.compare_status = "behind"
    with install_fake_github(github):
        with pytest.raises(gate.GateError, match="up to date"):
            gate.prepare(24, "gpu", tmp_path / "binding.json", tmp_path / "out")
    assert github.posts == []


@pytest.mark.parametrize("backend", ["tpu", "both"])
def test_prepare_rejects_non_gpu_backends(
    tmp_path: Path, github: FakeGitHub, backend: str
) -> None:
    with install_fake_github(github) as api:
        with pytest.raises(gate.GateError, match="only supports the GPU"):
            gate.prepare(24, backend, tmp_path / "binding.json", tmp_path / "out")
    api.assert_not_called()


@pytest.mark.parametrize("value", ["025", "+25", " 25", "25 ", "0", "-25", ""])
def test_pr_cli_value_requires_canonical_positive_decimal(value: str) -> None:
    with pytest.raises(SystemExit):
        gate.parser().parse_args(
            [
                "prepare",
                "--pr",
                value,
                "--backend",
                "gpu",
                "--output",
                "binding.json",
                "--github-output",
                "output.txt",
            ]
        )


def test_gh_api_prefers_gh_token_and_bounds_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", SYNTHETIC_ACCESS_A)
    monkeypatch.setenv("GH_TOKEN", SYNTHETIC_ACCESS_B)
    completed = subprocess.CompletedProcess(["gh"], 0, "{}", "")
    with patch.object(gate.subprocess, "run", return_value=completed) as run:
        assert gate._gh_api("repos/example/project") == {}

    assert run.call_args.kwargs["env"]["GH_TOKEN"] == SYNTHETIC_ACCESS_B
    assert run.call_args.kwargs["timeout"] == 45


def test_finish_posts_success_only_for_exact_accelerator_result(
    tmp_path: Path, github: FakeGitHub
) -> None:
    binding_path = write_binding(tmp_path / "binding.json")
    output_dir = tmp_path / "provider-output"
    write_result(output_dir, result())
    add_pending_status(github)

    with install_fake_github(github):
        actual = gate.finish(binding_path, "gpu", output_dir, provider_success=True)

    assert actual["state"] == "success"
    assert actual["run_url"].endswith("/actions/runs/456/attempts/2")
    assert github.posts[-1][1]["state"] == "success"
    assert github.posts[-1][1]["context"] == "accelerator/gpu"


@pytest.mark.parametrize(
    ("result_value", "provider_success", "change_base", "reason"),
    [
        (None, True, False, "could not read result.json"),
        (
            result(
                device_backend="cpu",
                devices=[{"platform": "cpu", "device_kind": "CPU", "id": "0"}],
            ),
            True,
            False,
            "different accelerator backend",
        ),
        (result(run_id="455.2"), True, False, "run_id does not match"),
        (result(), False, False, "provider command did not succeed"),
        (result(), True, True, "frozen head/base has changed"),
    ],
)
def test_finish_fails_closed_for_missing_cpu_stale_or_changed_run(
    tmp_path: Path,
    github: FakeGitHub,
    result_value: dict[str, object] | None,
    provider_success: bool,
    change_base: bool,
    reason: str,
) -> None:
    binding_path = write_binding(tmp_path / "binding.json")
    output_dir = tmp_path / "provider-output"
    if result_value is not None:
        write_result(output_dir, result_value)
    if change_base:
        github.pr["base"]["sha"] = "c" * 40  # type: ignore[index]
    add_pending_status(github)

    with install_fake_github(github):
        actual = gate.finish(
            binding_path, "gpu", output_dir, provider_success=provider_success
        )

    assert actual["state"] == "failure"
    assert reason in str(actual["reason"])
    assert github.posts[-1][1]["state"] == "failure"


def test_finish_rejects_result_from_another_pr_head(
    tmp_path: Path, github: FakeGitHub
) -> None:
    binding_path = write_binding(tmp_path / "binding.json")
    output_dir = tmp_path / "provider-output"
    write_result(output_dir, result(head_sha="c" * 40))
    add_pending_status(github)

    with install_fake_github(github):
        actual = gate.finish(binding_path, "gpu", output_dir, provider_success=True)

    assert actual["state"] == "failure"
    assert "head_sha does not match" in str(actual["reason"])


def test_finish_requires_pending_status_from_this_run(
    tmp_path: Path, github: FakeGitHub
) -> None:
    binding_path = write_binding(tmp_path / "binding.json")
    output_dir = tmp_path / "provider-output"
    write_result(output_dir, result())
    with install_fake_github(github):
        actual = gate.finish(binding_path, "gpu", output_dir, provider_success=True)

    assert actual["state"] == "failure"
    assert "no pending status" in str(actual["reason"])
    assert github.posts[-1][1]["state"] == "failure"


def test_finish_does_not_overwrite_a_newer_run_status(
    tmp_path: Path, github: FakeGitHub
) -> None:
    binding_path = write_binding(tmp_path / "binding.json")
    output_dir = tmp_path / "provider-output"
    write_result(output_dir, result())
    github.statuses = [
        {
            "context": gate.CONTEXTS["gpu"],
            "state": "pending",
            "description": "GPU accelerator run 457.1 pending",
        },
        {
            "context": gate.CONTEXTS["gpu"],
            "state": "pending",
            "description": f"GPU accelerator run {RUN_ID} pending",
        },
    ]

    with install_fake_github(github):
        actual = gate.finish(binding_path, "gpu", output_dir, provider_success=True)

    assert actual["state"] == "failure"
    assert actual["status_updated"] is False
    assert "another run" in str(actual["reason"])
    assert github.posts == []
