"""Tests for published-release recovery candidate selection."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from email.message import Message
import json
from pathlib import Path
from typing import ClassVar
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from tools import release_candidate as gate

HEAD = "a" * 40
TAG_OBJECT = "b" * 40
CONTROLLER = "c" * 40
VERSION = "0.5.0"
TAG = "v0.5.0"
RUN_ID = "789.3"
PROJECT_FILE = b'[project]\nname = "jaxrenderer"\nversion = "0.5.0"\n'


@dataclass
class FakeResponse:
    status: ClassVar[int] = 200
    body: bytes

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.body[:limit]


@pytest.fixture
def dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "789")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    monkeypatch.setenv("GITHUB_SHA", CONTROLLER)


def _content() -> dict[str, object]:
    return {
        "encoding": "base64",
        "size": len(PROJECT_FILE),
        "content": base64.b64encode(PROJECT_FILE).decode("ascii"),
    }


def install_api(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> list[str]:
    options: dict[str, object] = {
        "release_id": 12,
        "draft": False,
        "published_at": "2026-11-01T12:00:00Z",
        "tag": TAG,
        "tag_object_sha": TAG_OBJECT,
        "peeled_commit": HEAD,
        "master_status": "ahead",
        "base_commit": HEAD,
        "version_content": _content(),
    }
    options.update(overrides)
    calls: list[str] = []

    def api(route: str, _body: dict[str, object] | None = None) -> object:
        calls.append(route)
        if "/releases/tags/" in route:
            return {
                "id": options["release_id"],
                "tag_name": options["tag"],
                "draft": options["draft"],
                "published_at": options["published_at"],
            }
        if "/git/ref/tags/" in route:
            return {
                "ref": f"refs/tags/{TAG}",
                "object": {"type": "tag", "sha": options["tag_object_sha"]},
            }
        if "/git/tags/" in route:
            return {
                "sha": options["tag_object_sha"],
                "object": {"type": "commit", "sha": options["peeled_commit"]},
            }
        if "/contents/pyproject.toml?ref=" in route:
            return options["version_content"]
        return {
            "status": options["master_status"],
            "base_commit": {"sha": options["base_commit"]},
        }

    monkeypatch.setattr(gate.accelerator_gate, "gh_api", api)
    return calls


def install_pypi_absent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def open_url(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == 15
        urls.append(request.full_url)
        raise HTTPError(request.full_url, 404, "not found", Message(), None)

    monkeypatch.setattr(gate, "urlopen", open_url)
    return urls


def use_release_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    release_id: int = 12,
    tag: str = TAG,
    repository: str = gate.REPOSITORY,
    action: str = "published",
    draft: bool = False,
) -> None:
    event_path = tmp_path / "event.json"
    _ = event_path.write_text(
        json.dumps(
            {
                "action": action,
                "repository": {"full_name": repository},
                "release": {
                    "id": release_id,
                    "tag_name": tag,
                    "draft": draft,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_NAME", "release")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", gate.REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/{tag}")
    monkeypatch.setenv("GITHUB_SHA", HEAD)


def test_recovery_binds_release_tag_commit_version_and_run(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_api(monkeypatch)
    urls = install_pypi_absent(monkeypatch)
    output = tmp_path / "candidate-binding.json"
    github_output = tmp_path / "github-output"

    binding = gate.prepare(TAG, output, github_output)

    assert binding == {
        "mode": "recovery",
        "source_commit": HEAD,
        "head_sha": HEAD,
        "version": VERSION,
        "tag": TAG,
        "tag_ref": f"refs/tags/{TAG}",
        "tag_object_sha": TAG_OBJECT,
        "release_id": 12,
        "controller_sha": CONTROLLER,
        "repository": gate.REPOSITORY,
        "run_id": RUN_ID,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == binding
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        f"head_sha={HEAD}",
        f"version={VERSION}",
        "mode=recovery",
        "publish=true",
    ]
    assert urls == [f"https://pypi.org/pypi/jaxrenderer/{VERSION}/json"]
    assert calls == [
        f"repos/{gate.REPOSITORY}/releases/tags/{TAG}",
        f"repos/{gate.REPOSITORY}/git/ref/tags/{TAG}",
        f"repos/{gate.REPOSITORY}/git/tags/{TAG_OBJECT}",
        f"repos/{gate.REPOSITORY}/compare/{HEAD}...master",
        f"repos/{gate.REPOSITORY}/contents/pyproject.toml?ref={HEAD}",
    ]


def test_published_release_event_binds_current_event_release_and_publishes(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch)
    calls = install_api(monkeypatch)
    urls = install_pypi_absent(monkeypatch)
    output = tmp_path / "candidate-binding.json"
    github_output = tmp_path / "github-output"

    binding = gate.prepare("", output, github_output)

    assert binding["mode"] == "release"
    assert binding["tag"] == TAG
    assert binding["tag_object_sha"] == TAG_OBJECT
    assert binding["source_commit"] == HEAD
    assert binding["controller_sha"] == HEAD
    assert binding["release_id"] == 12
    assert github_output.read_text(encoding="utf-8").splitlines()[-1] == "publish=true"
    assert urls == [f"https://pypi.org/pypi/jaxrenderer/{VERSION}/json"]
    assert calls.count(f"repos/{gate.REPOSITORY}/git/ref/tags/{TAG}") == 1
    assert f"repos/{gate.REPOSITORY}/compare/{HEAD}...master" in calls


def test_published_release_rejects_event_and_api_identity_disagreement(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch, release_id=13)
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)

    with pytest.raises(gate.CandidateError, match="identity does not match"):
        _ = gate.select_candidate("")


def test_published_release_verify_rejects_tampered_binding(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_release_event(tmp_path, monkeypatch)
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)
    tampered = gate.select_candidate("")
    tampered["release_id"] = 99
    binding_path = tmp_path / "binding.json"
    _ = binding_path.write_text(json.dumps(tampered), encoding="utf-8")
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)

    with pytest.raises(gate.CandidateError, match="changed after"):
        _ = gate.verify(binding_path, "")


def test_verify_reselects_and_rejects_changed_release_id(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)
    binding_path = tmp_path / "binding.json"
    _ = binding_path.write_text(
        json.dumps(gate.select_candidate(TAG)), encoding="utf-8"
    )
    _ = install_api(monkeypatch, release_id=13)
    _ = install_pypi_absent(monkeypatch)

    with pytest.raises(gate.CandidateError, match="changed after"):
        _ = gate.verify(binding_path, TAG)


@pytest.mark.parametrize(
    ("variable", "value"),
    [("GITHUB_RUN_ATTEMPT", "4"), ("GITHUB_SHA", "d" * 40)],
)
def test_verify_rejects_changed_attempt_or_controller_commit(
    tmp_path: Path,
    dispatch: None,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)
    binding_path = tmp_path / "binding.json"
    _ = binding_path.write_text(
        json.dumps(gate.select_candidate(TAG)), encoding="utf-8"
    )
    _ = install_api(monkeypatch)
    _ = install_pypi_absent(monkeypatch)
    monkeypatch.setenv(variable, value)

    with pytest.raises(gate.CandidateError, match="changed after"):
        _ = gate.verify(binding_path, TAG)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"draft": True}, "already published"),
        ({"published_at": None}, "already published"),
        ({"tag": "v0.5.1"}, "does not match"),
        ({"peeled_commit": "d" * 40}, "not on the master history"),
        ({"master_status": "behind"}, "not on the master history"),
        ({"base_commit": "d" * 40}, "not on the master history"),
        (
            {
                "version_content": {
                    "encoding": "base64",
                    "size": len(b'[project]\nversion = "0.5.1"\n'),
                    "content": base64.b64encode(
                        b'[project]\nversion = "0.5.1"\n'
                    ).decode("ascii"),
                }
            },
            "does not match project.version",
        ),
    ],
)
def test_recovery_rejects_unpublished_mismatched_or_untrusted_release(
    dispatch: None,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    _ = install_api(monkeypatch, **overrides)
    _ = install_pypi_absent(monkeypatch)
    with pytest.raises(gate.CandidateError, match=message):
        _ = gate.select_candidate(TAG)


def test_rehearsal_binds_controller_commit_and_never_checks_or_publishes_pypi(
    tmp_path: Path, dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = install_api(
        monkeypatch,
        version_content={
            "encoding": "base64",
            "size": len(b'[project]\nversion = "0.6.0.dev0"\n'),
            "content": base64.b64encode(b'[project]\nversion = "0.6.0.dev0"\n').decode(
                "ascii"
            ),
        },
    )

    def unexpected_urlopen(_request: Request, *, timeout: float) -> FakeResponse:
        _ = timeout
        pytest.fail("rehearsal contacted PyPI")

    monkeypatch.setattr(gate, "urlopen", unexpected_urlopen)
    binding = gate.prepare("", tmp_path / "binding.json", tmp_path / "out")
    assert binding["mode"] == "rehearsal"
    assert binding["source_commit"] == CONTROLLER
    assert binding["version"] == "0.6.0.dev0"
    assert binding["release_id"] is None
    assert (tmp_path / "out").read_text(encoding="utf-8").splitlines()[
        -1
    ] == "publish=false"
    assert calls == [
        f"repos/{gate.REPOSITORY}/contents/pyproject.toml?ref={CONTROLLER}"
    ]


@pytest.mark.parametrize("event", ["push", "workflow_run", "pull_request"])
def test_candidate_requires_manual_dispatch_on_master(
    dispatch: None, monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    _ = install_api(monkeypatch)
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    with pytest.raises(gate.CandidateError, match="workflow_dispatch"):
        _ = gate.select_candidate("")


def test_recovery_requires_pypi_404_and_rejects_existing_or_uncertain_versions(
    dispatch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_api(monkeypatch)

    def existing(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == 15
        body = json.dumps({"info": {"version": VERSION}}).encode()
        return FakeResponse(body)

    monkeypatch.setattr(gate, "urlopen", existing)
    with pytest.raises(gate.CandidateError, match="already exists"):
        _ = gate.select_candidate(TAG)

    def unavailable(request: Request, *, timeout: float) -> FakeResponse:
        _ = timeout
        raise HTTPError(request.full_url, 429, "rate limited", Message(), None)

    monkeypatch.setattr(gate, "urlopen", unavailable)
    with pytest.raises(gate.CandidateError, match="could not confirm"):
        _ = gate.select_candidate(TAG)

    def malformed(_request: Request, *, timeout: float) -> FakeResponse:
        _ = timeout
        return FakeResponse(b"not json")

    monkeypatch.setattr(gate, "urlopen", malformed)
    with pytest.raises(gate.CandidateError, match="malformed version metadata"):
        _ = gate.select_candidate(TAG)
