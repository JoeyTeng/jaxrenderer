"""Tests for the release-only TPU gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tools import release_tpu_gate as gate

HEAD = "a" * 40
RUN_ID = "789.3"


def binding(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": "release",
        "head_sha": HEAD,
        "head_repository": "JoeyTeng/jaxrenderer",
        "run_id": RUN_ID,
    }
    value.update(overrides)
    return value


def result(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        **binding(),
        "backend": "tpu",
        "device_backend": "tpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
        "versions": {"jax": "0.11.2", "libtpu": "0.0.48"},
    }
    value.update(overrides)
    return value


@pytest.fixture
def workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_REPOSITORY", "JoeyTeng/jaxrenderer")
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")


def test_prepare_freezes_master_ancestor_and_run_identity(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        assert body is None
        calls.append(route)
        return {"status": "ahead", "base_commit": {"sha": HEAD}}

    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", api)
    output = tmp_path / "binding.json"
    github_output = tmp_path / "github-output.txt"
    actual = gate.prepare(HEAD, output, github_output)

    assert actual == binding()
    assert json.loads(output.read_text()) == binding()
    assert github_output.read_text().splitlines() == [
        f"head_sha={HEAD}",
        f"run_id={RUN_ID}",
    ]
    assert calls == [f"repos/{gate.REPOSITORY}/compare/{HEAD}...master"]


@pytest.mark.parametrize("commit", ["A" * 40, "a" * 39, "not-a-sha"])
def test_prepare_rejects_noncanonical_commit_before_api(
    tmp_path: Path,
    workflow: None,
    monkeypatch: pytest.MonkeyPatch,
    commit: str,
) -> None:
    monkeypatch.setattr(
        gate.accelerator_gate,
        "_gh_api",
        lambda *args, **kwargs: pytest.fail("invalid commit reached GitHub"),
    )
    with pytest.raises(gate.ReleaseGateError, match="full lowercase"):
        gate.prepare(commit, tmp_path / "binding.json", tmp_path / "out")


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        ({"status": "behind", "base_commit": {"sha": HEAD}}, "not on the master"),
        ({"status": "ahead", "base_commit": {"sha": "b" * 40}}, "did not confirm"),
    ],
)
def test_prepare_rejects_nonmaster_or_mismatched_commit(
    tmp_path: Path,
    workflow: None,
    monkeypatch: pytest.MonkeyPatch,
    comparison: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", lambda *args: comparison)
    with pytest.raises(gate.ReleaseGateError, match=message):
        gate.prepare(HEAD, tmp_path / "binding.json", tmp_path / "out")


def _write_inputs(tmp_path: Path, data: dict[str, object]) -> tuple[Path, Path]:
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding()), encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "result.json").write_text(json.dumps(data), encoding="utf-8")
    return binding_path, output_dir


def test_finish_accepts_only_matching_release_tpu_result(
    tmp_path: Path, workflow: None
) -> None:
    binding_path, output_dir = _write_inputs(tmp_path, result())
    assert gate.finish(binding_path, output_dir, provider_success=True) == {
        "state": "success",
        "reason": None,
        "head_sha": HEAD,
        "run_id": RUN_ID,
    }


@pytest.mark.parametrize(
    ("manifest", "provider_success", "message"),
    [
        (result(head_sha="b" * 40), True, "head_sha does not match"),
        (result(run_id="789.2"), True, "run_id does not match"),
        (result(kind="pr", pr=25, base_sha="b" * 40), True, "unexpected fields"),
        (
            result(
                device_backend="cpu",
                devices=[{"platform": "cpu", "device_kind": "CPU", "id": "0"}],
            ),
            True,
            "does not prove a TPU",
        ),
        (result(device_count=0), True, "device_count does not match"),
        (
            result(devices=[{"platform": "gpu", "device_kind": "GPU", "id": "0"}]),
            True,
            "invalid TPU device",
        ),
        (result(success=False), True, "does not report success"),
        (result(), False, "provider command did not succeed"),
    ],
)
def test_finish_fails_closed_for_wrong_identity_provider_or_device(
    tmp_path: Path,
    workflow: None,
    manifest: dict[str, object],
    provider_success: bool,
    message: str,
) -> None:
    binding_path, output_dir = _write_inputs(tmp_path, manifest)
    outcome = gate.finish(binding_path, output_dir, provider_success)
    assert outcome["state"] == "failure"
    assert message in str(outcome["reason"])


def test_finish_rejects_binding_from_another_attempt(
    tmp_path: Path, workflow: None
) -> None:
    binding_path, output_dir = _write_inputs(tmp_path, result())
    other = binding(run_id="789.2")
    binding_path.write_text(json.dumps(other), encoding="utf-8")
    outcome = gate.finish(binding_path, output_dir, provider_success=True)
    assert outcome["state"] == "failure"
    assert "another workflow run or attempt" in str(outcome["reason"])


def test_finish_refuses_non_dispatch_context(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding_path, output_dir = _write_inputs(tmp_path, result())
    monkeypatch.setenv("GITHUB_REF", "refs/heads/feature")
    with pytest.raises(gate.ReleaseGateError, match="default master"):
        gate.finish(binding_path, output_dir, provider_success=True)
