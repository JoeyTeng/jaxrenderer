"""Tests for the release-only TPU gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    monkeypatch.setattr(
        gate.accelerator_gate,
        "_gh_api",
        lambda _route, body=None: {
            "status": "ahead",
            "base_commit": {"sha": HEAD},
        },
    )


def use_release_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    action: str = "published",
    tag: str = "v0.4.0",
    repository: str = "JoeyTeng/jaxrenderer",
    draft: bool = False,
    event_sha: str = HEAD,
    resolved_sha: str = HEAD,
    master_status: str = "ahead",
) -> list[str]:
    payload_path = tmp_path / "event.json"
    payload_path.write_text(
        json.dumps(
            {
                "action": action,
                "repository": {"full_name": repository},
                "release": {"tag_name": tag, "draft": draft},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(payload_path))
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/{tag}")
    monkeypatch.setenv("GITHUB_SHA", event_sha)
    calls: list[str] = []

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        assert body is None
        calls.append(route)
        if route.endswith("/git/ref/tags/" + quote(tag, safe="")):
            return {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "commit", "sha": resolved_sha},
            }
        return {"status": master_status, "base_commit": {"sha": HEAD}}

    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", api)
    return calls


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


def test_prepare_accepts_only_matching_published_release_event(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = use_release_event(tmp_path, monkeypatch, tag="release/v0.4.0")
    output = tmp_path / "binding.json"
    binding_value = gate.prepare(HEAD, output, tmp_path / "github-output.txt")
    assert binding_value == binding()
    assert calls == [
        f"repos/{gate.REPOSITORY}/git/ref/tags/release%2Fv0.4.0",
        f"repos/{gate.REPOSITORY}/compare/{HEAD}...master",
    ]


def test_prepare_peels_annotated_release_tag_to_commit(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag = "v0.4.0"
    calls: list[str] = []
    first_tag_sha = "b" * 40
    second_tag_sha = "c" * 40

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        assert body is None
        calls.append(route)
        if route.endswith("/git/ref/tags/v0.4.0"):
            return {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "tag", "sha": first_tag_sha},
            }
        if route.endswith(f"/git/tags/{first_tag_sha}"):
            return {
                "sha": first_tag_sha,
                "object": {"type": "tag", "sha": second_tag_sha},
            }
        if route.endswith(f"/git/tags/{second_tag_sha}"):
            return {
                "sha": second_tag_sha,
                "object": {"type": "commit", "sha": HEAD},
            }
        return {"status": "ahead", "base_commit": {"sha": HEAD}}

    use_release_event(tmp_path, monkeypatch, tag=tag)
    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", api)
    gate.prepare(HEAD, tmp_path / "binding.json", tmp_path / "out")
    assert calls == [
        f"repos/{gate.REPOSITORY}/git/ref/tags/v0.4.0",
        f"repos/{gate.REPOSITORY}/git/tags/{first_tag_sha}",
        f"repos/{gate.REPOSITORY}/git/tags/{second_tag_sha}",
        f"repos/{gate.REPOSITORY}/compare/{HEAD}...master",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tag": ""}, "tag name is malformed"),
        ({"action": "created"}, "action must be published"),
        ({"repository": "someone/fork"}, "repository does not match"),
        ({"draft": True}, "published release"),
        ({"tag": "v0.4.0", "event_sha": "b" * 40}, "GITHUB_SHA"),
        ({"tag": "v0.4.0", "resolved_sha": "b" * 40}, "no longer resolves"),
        ({"tag": "v0.4.0", "master_status": "behind"}, "master history"),
    ],
)
def test_prepare_rejects_malformed_or_unbound_release_event(
    tmp_path: Path,
    workflow: None,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    use_release_event(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(gate.ReleaseGateError, match=message):
        gate.prepare(HEAD, tmp_path / "binding.json", tmp_path / "out")


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


def test_prepare_does_not_echo_github_api_diagnostics(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gate.accelerator_gate,
        "_gh_api",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            gate.accelerator_gate.GateError("GH_TOKEN: private diagnostic")
        ),
    )
    with pytest.raises(gate.ReleaseGateError) as error:
        gate.prepare(HEAD, tmp_path / "binding.json", tmp_path / "out")
    assert "private diagnostic" not in str(error.value)


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
        "backend": "tpu",
        "head_sha": HEAD,
        "run_id": RUN_ID,
    }


def test_finish_accepts_gpu_result_for_published_release(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch)
    gpu_result = result(
        backend="gpu",
        device_backend="gpu",
        devices=[{"platform": "gpu", "device_kind": "T4", "id": "0"}],
    )
    binding_path, output_dir = _write_inputs(tmp_path, gpu_result)
    outcome = gate.finish(
        binding_path, output_dir, provider_success=True, backend="gpu"
    )
    assert outcome == {
        "state": "success",
        "reason": None,
        "backend": "gpu",
        "head_sha": HEAD,
        "run_id": RUN_ID,
    }


def test_finish_rejects_release_tag_moved_after_prepare(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch)
    binding_path, output_dir = _write_inputs(tmp_path, result())
    current_tag_sha = HEAD

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        assert body is None
        if "/git/ref/tags/" in route:
            return {
                "ref": "refs/tags/v0.4.0",
                "object": {"type": "commit", "sha": current_tag_sha},
            }
        return {"status": "ahead", "base_commit": {"sha": HEAD}}

    monkeypatch.setattr(gate.accelerator_gate, "_gh_api", api)
    current_tag_sha = "b" * 40
    outcome = gate.finish(binding_path, output_dir, provider_success=True)
    assert outcome["state"] == "failure"
    assert "no longer resolves" in str(outcome["reason"])


def test_finish_rejects_release_context_with_mismatched_tag_ref(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch)
    binding_path, output_dir = _write_inputs(tmp_path, result())
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.4.1")
    outcome = gate.finish(binding_path, output_dir, provider_success=True)
    assert outcome["state"] == "failure"
    assert "does not match the published release tag" in str(outcome["reason"])


def test_finish_rejects_unsupported_event_context(
    tmp_path: Path, workflow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding_path, output_dir = _write_inputs(tmp_path, result())
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_run")
    outcome = gate.finish(binding_path, output_dir, provider_success=True)
    assert outcome["state"] == "failure"
    assert "requires workflow_dispatch or release event" in str(outcome["reason"])


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
    outcome = gate.finish(binding_path, output_dir, provider_success=True)
    assert outcome["state"] == "failure"
    assert "default master" in str(outcome["reason"])
