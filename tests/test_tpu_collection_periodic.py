# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from tools import accelerator_gate, kaggle_ci, kaggle_collect, release_tpu_gate
from tools import tpu_collection_gate as collection_gate
from tools import tpu_collection_periodic as periodic

SOURCE_SHA = "a" * 40
COLLECTOR_SHA = "b" * 40
SOURCE_RUN = "789"
SOURCE_ATTEMPT = "3"
COLLECTOR_RUN = "990"
COLLECTOR_ATTEMPT = "2"


def _binding(
    source_run: str = SOURCE_RUN, attempt: str = SOURCE_ATTEMPT
) -> dict[str, object]:
    return {
        "kind": "release",
        "head_sha": SOURCE_SHA,
        "head_repository": periodic.REPOSITORY,
        "run_id": f"{source_run}.{attempt}",
    }


def _controller_state(
    outcome: str = "queue_timeout", status: str = "queued"
) -> dict[str, object]:
    state: dict[str, object] = {
        field: None for field in kaggle_ci.CONTROLLER_STATE_FIELDS
    }
    state.update(
        binding=_binding(),
        kernel_id="joey/jaxr-frozen-kernel",
        submitted_version=1,
        outcome=outcome,
        status=status,
    )
    return state


def _source_record() -> dict[str, object]:
    return {
        "source_run_id": int(SOURCE_RUN),
        "source_attempt": int(SOURCE_ATTEMPT),
        "head_sha": SOURCE_SHA,
        "collector_run_id": f"{COLLECTOR_RUN}.1",
        "collector_sha": COLLECTOR_SHA,
        "binding_artifact_id": 101,
        "controller_artifact_id": 102,
    }


def _prepare_source(output: Path) -> None:
    output.mkdir(parents=True)
    (output / "binding.json").write_text(json.dumps(_binding()), encoding="utf-8")
    (output / "kaggle-controller-state.json").write_text(
        json.dumps(_controller_state()), encoding="utf-8"
    )
    (output / "collection-source.json").write_text(
        json.dumps(_source_record()), encoding="utf-8"
    )


def _run(
    run_id: int = int(SOURCE_RUN),
    attempt: int = int(SOURCE_ATTEMPT),
    *,
    created_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "id": run_id,
        "run_attempt": attempt,
        "head_sha": SOURCE_SHA,
        "path": periodic.SOURCE_WORKFLOW,
        "event": "workflow_dispatch",
        "head_branch": "master",
        "status": "completed",
        "conclusion": "failure",
        "created_at": (created_at or datetime.now(timezone.utc))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "repository": {"full_name": periodic.REPOSITORY},
        "head_repository": {"full_name": periodic.REPOSITORY},
    }


def _collector_run(
    *, attempt: int = int(COLLECTOR_ATTEMPT), path: str = periodic.WORKFLOW_PATH
) -> dict[str, object]:
    return {
        "id": int(COLLECTOR_RUN),
        "run_attempt": attempt,
        "head_sha": COLLECTOR_SHA,
        "head_branch": "master",
        "path": path,
        "event": "schedule",
        "status": "completed",
        "conclusion": "failure",
        "repository": {"full_name": periodic.REPOSITORY},
        "head_repository": {"full_name": periodic.REPOSITORY},
    }


def _set_periodic_context(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "GITHUB_EVENT_NAME": "schedule",
        "GITHUB_REF": "refs/heads/master",
        "GITHUB_REPOSITORY": periodic.REPOSITORY,
        "GITHUB_WORKFLOW_REF": periodic.WORKFLOW_REF,
        "GITHUB_SHA": COLLECTOR_SHA,
        "GITHUB_RUN_ID": COLLECTOR_RUN,
        "GITHUB_RUN_ATTEMPT": "1",
    }.items():
        monkeypatch.setenv(name, value)


def _patch_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(periodic, "_source_expired", lambda *args: False)

    def prepare(
        source_run: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        assert source_run == SOURCE_RUN
        assert source_attempt == SOURCE_ATTEMPT
        assert commit == SOURCE_SHA
        _prepare_source(output)
        github_output.write_text("", encoding="utf-8")
        return _source_record()

    monkeypatch.setattr(collection_gate, "prepare", prepare)


def _report(outcome: str, *, status: str = "complete") -> dict[str, object]:
    return {
        "binding": _binding(),
        "kernel_id": _controller_state()["kernel_id"],
        "submitted_version": 1,
        "requested_version": 1,
        "version_verified": True,
        "success": outcome == "success",
        "outcome": outcome,
        "remote_status": status,
    }


def _terminal_record(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "terminal": True,
        "outcome": "success",
        "source_run_id": SOURCE_RUN,
        "source_attempt": SOURCE_ATTEMPT,
        "head_sha": SOURCE_SHA,
        "collector_run_id": f"{COLLECTOR_RUN}.{COLLECTOR_ATTEMPT}",
        "collector_attempt": COLLECTOR_ATTEMPT,
        "collector_sha": COLLECTOR_SHA,
        "binding": _binding(),
        "kernel_id": _controller_state()["kernel_id"],
        "submitted_version": 1,
        "requested_version": 1,
        "version_verified": True,
        "remote_status": "complete",
    }
    value.update(changes)
    return value


def _terminal_artifact() -> dict[str, object]:
    return {
        "id": 500,
        "name": f"tpu-periodic-terminal-{SOURCE_RUN}-{SOURCE_ATTEMPT}-{SOURCE_SHA}",
        "workflow_run": {"id": int(COLLECTOR_RUN), "head_sha": COLLECTOR_SHA},
    }


def test_periodic_context_is_exact_and_manual_event_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_periodic_context(monkeypatch)
    assert periodic._require_context() == f"{COLLECTOR_RUN}.1"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    assert periodic._require_context() == f"{COLLECTOR_RUN}.1"

    for name, value in (
        ("GITHUB_REF", "refs/heads/feature"),
        ("GITHUB_REPOSITORY", "someone/jaxrenderer"),
        (
            "GITHUB_WORKFLOW_REF",
            f"{periodic.REPOSITORY}/.github/workflows/unrelated.yml@refs/heads/master",
        ),
    ):
        _set_periodic_context(monkeypatch)
        monkeypatch.setenv(name, value)
        with pytest.raises(periodic.PeriodicError, match="canonical master workflow"):
            periodic._require_context()


def test_discover_returns_exact_candidates_and_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    run = _run()

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 1, "workflow_runs": [run]}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return run
        if route.startswith(f"repos/{periodic.REPOSITORY}/actions/artifacts?"):
            return {"total_count": 0, "artifacts": []}
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    output = tmp_path / "discovery"
    github_output = tmp_path / "github-output"
    result = periodic.discover(output, github_output)

    assert result["has_sources"] is True
    assert result["matrix"] == {
        "include": [
            {
                "source_run_id": SOURCE_RUN,
                "source_attempt": SOURCE_ATTEMPT,
                "commit": SOURCE_SHA,
            }
        ]
    }
    lines = github_output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "has_sources=true"
    assert json.loads(lines[1].removeprefix("matrix=")) == result["matrix"]


def test_discover_fails_on_incomplete_run_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    monkeypatch.setattr(
        accelerator_gate,
        "_gh_api",
        lambda route, body=None: {"total_count": 2, "workflow_runs": [_run()]},
    )

    with pytest.raises(periodic.PeriodicError, match="inventory is incomplete"):
        periodic.discover(tmp_path / "discovery", tmp_path / "github-output")


def test_discover_ignores_sources_older_than_seven_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    old = _run(created_at=datetime.now(timezone.utc) - timedelta(days=8))

    def api(route: str, body: object = None) -> Any:
        assert "actions/workflows/pypi.yml/runs?" in route
        return {"total_count": 1, "workflow_runs": [old]}

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    result = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")

    assert result["has_sources"] is False
    assert result["inventory_count"] == 0


def test_discover_skips_non_master_manual_source_after_identity_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    run = _run()
    run["head_branch"] = "wip/feature"

    def api(route: str, body: object = None) -> Any:
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 1, "workflow_runs": [run]}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return run
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "prepare",
        lambda *args: pytest.fail("feature branch must be known ineligible"),
    )

    result = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")

    assert result["has_sources"] is False


def test_discover_skips_only_explicitly_ineligible_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    run = _run()

    def api(route: str, body: object = None) -> Any:
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 1, "workflow_runs": [run]}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return run
        raise AssertionError(route)

    def ineligible(*args: Any, **kwargs: Any) -> None:
        raise collection_gate.IneligibleSource("known CPU failure")

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", ineligible)
    result = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")
    assert result["has_sources"] is False

    monkeypatch.setattr(
        collection_gate,
        "prepare",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            collection_gate.CollectionGateError("GitHub artefact lookup failed")
        ),
    )
    with pytest.raises(collection_gate.CollectionGateError, match="lookup failed"):
        periodic.discover(
            tmp_path / "discovery-error", tmp_path / "github-output-error"
        )


def test_discover_skips_valid_legacy_source_and_keeps_timeout_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    legacy_run = _run(run_id=788, attempt=1)
    timeout_run = _run(run_id=int(SOURCE_RUN), attempt=int(SOURCE_ATTEMPT))
    runs = {788: legacy_run, int(SOURCE_RUN): timeout_run}

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 2, "workflow_runs": [legacy_run, timeout_run]}
        for run_id, run in runs.items():
            if route.endswith(f"/actions/runs/{run_id}"):
                return run
        if route.startswith(f"repos/{periodic.REPOSITORY}/actions/artifacts?"):
            return {"total_count": 0, "artifacts": []}
        raise AssertionError(route)

    def prepare(
        source_run: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        run_id = int(source_run)
        binding = _binding(source_run, source_attempt)
        state: dict[str, object] = (
            {
                "binding": binding,
                "kernel_id": legacy_kernel_id(binding),
                "submitted_version": 1,
            }
            if run_id == 788
            else {
                **_controller_state(),
                "binding": binding,
            }
        )
        source: dict[str, object] = {
            "source_run_id": run_id,
            "source_attempt": int(source_attempt),
            "head_sha": commit,
            "collector_run_id": f"{COLLECTOR_RUN}.1",
            "collector_sha": COLLECTOR_SHA,
            "binding_artifact_id": 101,
            "controller_artifact_id": 102,
        }
        output.mkdir(parents=True)
        (output / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
        state_path = output / "kaggle-controller-state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        collection_gate._validate_source_files(output, source)
        (output / "collection-source.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        github_output.write_text("", encoding="utf-8")
        return source

    def legacy_kernel_id(binding: dict[str, object]) -> str:
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        run_tag = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        return f"joey/jaxr-{run_tag}-0123456789"

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", prepare)
    monkeypatch.setattr(periodic, "_source_expired", lambda *args: False)

    report = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")

    assert report["matrix"] == {
        "include": [
            {
                "source_run_id": SOURCE_RUN,
                "source_attempt": SOURCE_ATTEMPT,
                "commit": SOURCE_SHA,
            }
        ]
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", {"full_name": "someone/jaxrenderer"}),
        ("head_repository", {"full_name": "someone/jaxrenderer"}),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_discover_rejects_foreign_or_wrong_workflow_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    _set_periodic_context(monkeypatch)
    bad_run = _run()
    bad_run[field] = value
    monkeypatch.setattr(
        accelerator_gate,
        "_gh_api",
        lambda route, body=None: {"total_count": 1, "workflow_runs": [bad_run]},
    )
    with pytest.raises(periodic.PeriodicError, match="identity is not canonical"):
        periodic.discover(tmp_path / "discovery", tmp_path / "github-output")


@pytest.mark.parametrize("owner_conclusion", ["cancelled", "timed_out"])
@pytest.mark.parametrize(
    ("outcome", "remote_status"),
    [("success", "complete"), ("remote_failure", "failed")],
)
def test_terminal_marker_accepts_post_upload_owner_cancellation_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    owner_conclusion: str,
    outcome: str,
    remote_status: str,
) -> None:
    record = _terminal_record(outcome=outcome, remote_status=remote_status)
    exact_attempt = _collector_run(attempt=int(COLLECTOR_ATTEMPT))
    exact_attempt["conclusion"] = owner_conclusion

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return exact_attempt
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(route)

    def download(artifact_id: int, member: str, destination: Path) -> None:
        assert artifact_id == 500
        assert member == "periodic-terminal.json"
        destination.write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(collection_gate, "_download_member", download)
    monkeypatch.setattr(release_tpu_gate, "_verify_master_history", lambda commit: None)

    loaded = periodic._read_terminal(
        _terminal_artifact(),
        source_run_id=SOURCE_RUN,
        source_attempt=SOURCE_ATTEMPT,
        commit=SOURCE_SHA,
        binding=_binding(),
        kernel_id=str(_controller_state()["kernel_id"]),
        budget=periodic._Budget(),
    )
    assert loaded["outcome"] == outcome


@pytest.mark.parametrize(
    "run_changes",
    [
        {"status": "in_progress", "conclusion": None},
        {"repository": {"full_name": "someone/else"}},
        {"run_attempt": int(COLLECTOR_ATTEMPT) + 1},
        {"head_sha": "c" * 40},
    ],
)
def test_terminal_marker_rejects_running_or_mismatched_owner_attempt(
    monkeypatch: pytest.MonkeyPatch,
    run_changes: dict[str, object],
) -> None:
    exact_attempt = _collector_run()
    exact_attempt.update(run_changes)

    def api(route: str, body: object = None) -> Any:
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return exact_attempt
        raise AssertionError(f"unexpected API route: {route}")

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "_download_member",
        lambda artifact_id, member, destination: destination.write_text(
            json.dumps(_terminal_record()), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(release_tpu_gate, "_verify_master_history", lambda commit: None)
    with pytest.raises(periodic.PeriodicError):
        periodic._read_terminal(
            _terminal_artifact(),
            source_run_id=SOURCE_RUN,
            source_attempt=SOURCE_ATTEMPT,
            commit=SOURCE_SHA,
            binding=_binding(),
            kernel_id=str(_controller_state()["kernel_id"]),
            budget=periodic._Budget(),
        )


def test_terminal_marker_uses_recorded_attempt_after_owner_was_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _terminal_record()
    exact_attempt = _collector_run(attempt=2)
    called: list[str] = []

    def api(route: str, body: object = None) -> Any:
        called.append(route)
        if route.endswith(f"/actions/runs/{COLLECTOR_RUN}/attempts/2"):
            return exact_attempt
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(f"unexpected or latest-run query: {route}")

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "_download_member",
        lambda artifact_id, member, destination: destination.write_text(
            json.dumps(record), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(release_tpu_gate, "_verify_master_history", lambda commit: None)

    periodic._read_terminal(
        _terminal_artifact(),
        source_run_id=SOURCE_RUN,
        source_attempt=SOURCE_ATTEMPT,
        commit=SOURCE_SHA,
        binding=_binding(),
        kernel_id=str(_controller_state()["kernel_id"]),
        budget=periodic._Budget(),
    )
    assert any(route.endswith("/attempts/2") for route in called)
    assert not any(route.endswith(f"/actions/runs/{COLLECTOR_RUN}") for route in called)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_attempt": "4"}, "exact source and collector"),
        ({"head_sha": "c" * 40}, "exact source and collector"),
    ],
)
def test_terminal_marker_rejects_attempt_or_sha_mismatch(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object], message: str
) -> None:
    record = _terminal_record(**changes)
    exact_attempt = _collector_run()

    def api(route: str, body: object = None) -> Any:
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return exact_attempt
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "_download_member",
        lambda artifact_id, member, destination: destination.write_text(
            json.dumps(record), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(release_tpu_gate, "_verify_master_history", lambda commit: None)
    with pytest.raises(periodic.PeriodicError, match=message):
        periodic._read_terminal(
            _terminal_artifact(),
            source_run_id=SOURCE_RUN,
            source_attempt=SOURCE_ATTEMPT,
            commit=SOURCE_SHA,
            binding=_binding(),
            kernel_id=str(_controller_state()["kernel_id"]),
            budget=periodic._Budget(),
        )


@pytest.mark.parametrize("outcome", ["pending", "success", "remote_failure"])
def test_collect_source_terminal_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    monkeypatch.setattr(periodic, "_has_terminal", lambda *args: None)
    finish_calls: list[bool] = []
    monkeypatch.setattr(
        collection_gate,
        "finish",
        lambda source, output, provider_success: finish_calls.append(provider_success),
    )
    monkeypatch.setattr(
        collection_gate,
        "revalidate",
        lambda source: (_source_record(), _binding(), _controller_state()),
    )
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        lambda binding, resume_state_path, output_dir: _report(
            outcome,
            status=(
                "queued"
                if outcome == "pending"
                else "failed"
                if outcome == "remote_failure"
                else "complete"
            ),
        ),
    )
    output = tmp_path / "artifacts"
    github_output = tmp_path / "github-output"

    report, exit_code = periodic.collect_source(
        SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, output, github_output
    )

    if outcome == "pending":
        assert exit_code == 0
        assert report == {"terminal": False, "outcome": "pending"}
        assert not (output / "periodic-terminal.json").exists()
        assert github_output.read_text().splitlines() == [
            "terminal=false",
            "outcome=pending",
        ]
    else:
        assert report == {"terminal": True, "outcome": outcome}
        record = json.loads((output / "periodic-terminal.json").read_text())
        assert record["binding"] == _binding()
        assert record["source_attempt"] == SOURCE_ATTEMPT
        assert record["requested_version"] == 1
        assert record["version_verified"] is True
        assert exit_code == (0 if outcome == "success" else 1)
    assert finish_calls == ([True] if outcome == "success" else [])


def test_collect_source_expires_before_provider_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    monkeypatch.setattr(periodic, "_source_expired", lambda *args: True)
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        lambda *args: pytest.fail("expired source must not query the provider"),
    )

    report, exit_code = periodic.collect_source(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        tmp_path / "expired",
        tmp_path / "github-output",
    )

    assert report == {"terminal": False, "outcome": "expired"}
    assert exit_code == 0
    assert not (tmp_path / "expired" / "periodic-terminal.json").exists()


def test_source_expiry_uses_exact_attempt_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _run(created_at=datetime.now(timezone.utc) - timedelta(days=8))
    routes: list[str] = []

    def api(route: str, body: object = None) -> Any:
        routes.append(route)
        if route.endswith(f"/actions/runs/{SOURCE_RUN}/attempts/{SOURCE_ATTEMPT}"):
            return old
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return old
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    budget = periodic._Budget()

    assert periodic._source_expired(SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, budget)
    assert routes[0].endswith(f"/actions/runs/{SOURCE_RUN}/attempts/{SOURCE_ATTEMPT}")


def test_collect_source_remote_failure_revalidates_original_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    monkeypatch.setattr(periodic, "_has_terminal", lambda *args: None)
    revalidate_calls: list[Path] = []
    monkeypatch.setattr(
        collection_gate,
        "revalidate",
        lambda source: (
            revalidate_calls.append(source),
            _source_record(),
            _binding(),
            _controller_state(),
        )[1:],
    )
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        lambda binding, resume_state_path, output_dir: _report(
            "remote_failure", status="failed"
        ),
    )

    _, exit_code = periodic.collect_source(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        tmp_path / "artifacts",
        tmp_path / "github-output",
    )

    assert exit_code == 1
    assert revalidate_calls == [tmp_path / "artifacts" / "source"]


def test_collect_source_collection_error_writes_no_terminal_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    monkeypatch.setattr(periodic, "_has_terminal", lambda *args: None)
    attempts = 0

    def collection_error(
        binding: dict[str, object], state_path: Path, output: Path
    ) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        return {**_report("collection_error"), "last_error": "temporary SDK error"}

    monkeypatch.setattr(kaggle_collect, "collect", collection_error)
    for suffix in ("first", "retry"):
        output = tmp_path / suffix
        with pytest.raises(periodic.PeriodicError, match="verified terminal result"):
            periodic.collect_source(
                SOURCE_RUN,
                SOURCE_ATTEMPT,
                SOURCE_SHA,
                output,
                tmp_path / f"github-output-{suffix}",
            )
        assert not (output / "periodic-terminal.json").exists()
    assert attempts == 2


@pytest.mark.parametrize(
    ("outcome", "remote_status"),
    [("success", "complete"), ("remote_failure", "failed")],
)
def test_collect_source_skips_provider_when_terminal_marker_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
    remote_status: str,
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    terminal = _terminal_record(outcome=outcome, remote_status=remote_status)
    owner_attempt = _collector_run()
    owner_attempt["conclusion"] = "cancelled"

    def api(route: str, body: object = None) -> Any:
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return owner_attempt
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(f"unexpected API route: {route}")

    monkeypatch.setattr(accelerator_gate, "_gh_api", api)
    monkeypatch.setattr(
        periodic,
        "_terminal_artifact",
        lambda source_run, name, budget: _terminal_artifact(),
    )
    monkeypatch.setattr(
        collection_gate,
        "_download_member",
        lambda artifact_id, member, destination: destination.write_text(
            json.dumps(terminal), encoding="utf-8"
        ),
    )
    monkeypatch.setattr(release_tpu_gate, "_verify_master_history", lambda commit: None)
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        lambda *args: pytest.fail("must not query provider after terminal evidence"),
    )

    report, exit_code = periodic.collect_source(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        tmp_path / "artifacts",
        tmp_path / "github-output",
    )

    assert report == {"terminal": False, "outcome": "already-terminal"}
    assert exit_code == 0
