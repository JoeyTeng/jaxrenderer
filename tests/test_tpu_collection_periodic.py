from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from tools import accelerator_gate, kaggle_ci, kaggle_collect, release_tpu_gate
from tools import tpu_collection_gate as collection_gate
from tools import tpu_collection_periodic as periodic

SOURCE_SHA = "a" * 40
COLLECTOR_SHA = "b" * 40
SOURCE_RUN_ID = 789
SOURCE_ATTEMPT_ID = 3
COLLECTOR_RUN_ID = 990
COLLECTOR_ATTEMPT_ID = 2
SOURCE_RUN = str(SOURCE_RUN_ID)
SOURCE_ATTEMPT = str(SOURCE_ATTEMPT_ID)
COLLECTOR_RUN = str(COLLECTOR_RUN_ID)
COLLECTOR_ATTEMPT = str(COLLECTOR_ATTEMPT_ID)


class _BudgetLike(Protocol):
    def check(self) -> None: ...


NoTerminal = Callable[[str, str, str, Path, _BudgetLike], dict[str, object] | None]
Revalidate = Callable[
    [Path], tuple[dict[str, object], dict[str, object], dict[str, object]]
]
CollectResult = Callable[[dict[str, object], Path, Path], dict[str, object]]


def _require_periodic_context() -> str:
    helper = cast(Callable[[], str], getattr(periodic, "_require_context"))
    return helper()


def _new_budget() -> _BudgetLike:
    factory = cast(Callable[[], _BudgetLike], getattr(periodic, "_Budget"))
    return factory()


def _read_periodic_terminal(
    artifact: dict[str, object],
    *,
    source_run_id: str,
    source_attempt: str,
    commit: str,
    binding: dict[str, object],
    kernel_id: str,
    budget: _BudgetLike,
) -> dict[str, object] | None:
    helper = cast(
        Callable[..., dict[str, object] | None], getattr(periodic, "_read_terminal")
    )
    return helper(
        artifact,
        source_run_id=source_run_id,
        source_attempt=source_attempt,
        commit=commit,
        binding=binding,
        kernel_id=kernel_id,
        budget=budget,
    )


def _has_periodic_terminal(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    source_dir: Path,
    budget: _BudgetLike,
) -> dict[str, object] | None:
    helper = cast(
        Callable[[str, str, str, Path, _BudgetLike], dict[str, object] | None],
        getattr(periodic, "_has_terminal"),
    )
    return helper(source_run_id, source_attempt, commit, source_dir, budget)


def _source_expired(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    budget: _BudgetLike,
) -> bool:
    helper = cast(
        Callable[[str, str, str, _BudgetLike], bool],
        getattr(periodic, "_source_expired"),
    )
    return helper(source_run_id, source_attempt, commit, budget)


def _validate_source_files(
    source_dir: Path, source: dict[str, object]
) -> dict[str, object]:
    helper = cast(
        Callable[[Path, dict[str, object]], dict[str, object]],
        getattr(collection_gate, "_validate_source_files"),
    )
    return helper(source_dir, source)


def _write_record(record: dict[str, object]) -> Callable[[int, str, Path], None]:
    def download(artifact_id: int, member: str, destination: Path) -> None:
        del artifact_id, member
        _ = destination.write_text(json.dumps(record), encoding="utf-8")

    return download


def _no_master_history(commit: str) -> None:
    del commit


def _not_expired(
    source_run: str, source_attempt: str, commit: str, budget: _BudgetLike
) -> bool:
    del source_run, source_attempt, commit, budget
    return False


def _has_expired(
    source_run: str, source_attempt: str, commit: str, budget: _BudgetLike
) -> bool:
    del source_run, source_attempt, commit, budget
    return True


def _raise_provider_not_called(
    binding: dict[str, object], resume_state_path: Path, output_dir: Path
) -> dict[str, object]:
    del binding, resume_state_path, output_dir
    pytest.fail("provider must not be queried")


def _raise_terminal_provider_not_called(
    binding: dict[str, object], resume_state_path: Path, output_dir: Path
) -> dict[str, object]:
    del binding, resume_state_path, output_dir
    pytest.fail("must not query provider after terminal evidence")


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
    _ = (output / "binding.json").write_text(json.dumps(_binding()), encoding="utf-8")
    _ = (output / "kaggle-controller-state.json").write_text(
        json.dumps(_controller_state()), encoding="utf-8"
    )
    _ = (output / "collection-source.json").write_text(
        json.dumps(_source_record()), encoding="utf-8"
    )


def _prepare_named_source(
    output: Path, source_run_id: str, source_attempt: str, commit: str
) -> dict[str, object]:
    output.mkdir(parents=True)
    binding = _binding(source_run_id, source_attempt)
    state = {**_controller_state(), "binding": binding}
    source = {
        **_source_record(),
        "source_run_id": int(source_run_id),
        "source_attempt": int(source_attempt),
        "head_sha": commit,
    }
    _ = (output / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
    _ = (output / "kaggle-controller-state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    _ = (output / "collection-source.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    return source


def _run(
    run_id: int = SOURCE_RUN_ID,
    attempt: int = SOURCE_ATTEMPT_ID,
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
    *, attempt: int = COLLECTOR_ATTEMPT_ID, path: str = periodic.WORKFLOW_PATH
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
        "GITHUB_RUN_NUMBER": "42",
    }.items():
        monkeypatch.setenv(name, value)


def _patch_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_expired(
        source_run: str, source_attempt: str, commit: str, budget: _BudgetLike
    ) -> bool:
        del source_run, source_attempt, commit, budget
        return False

    monkeypatch.setattr(periodic, "_source_expired", not_expired)

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
        _ = github_output.write_text("", encoding="utf-8")
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


def _terminal_record_for(
    source_run_id: str,
    source_attempt: str,
    commit: str,
    binding: dict[str, object],
    kernel_id: str,
    *,
    collector_id: int = COLLECTOR_RUN_ID,
    collector_attempt: int = COLLECTOR_ATTEMPT_ID,
    collector_sha: str = COLLECTOR_SHA,
    outcome: str = "success",
    remote_status: str = "complete",
) -> dict[str, object]:
    return _terminal_record(
        source_run_id=source_run_id,
        source_attempt=source_attempt,
        head_sha=commit,
        collector_run_id=f"{collector_id}.{collector_attempt}",
        collector_attempt=str(collector_attempt),
        collector_sha=collector_sha,
        binding=binding,
        kernel_id=kernel_id,
        outcome=outcome,
        remote_status=remote_status,
    )


def _terminal_artifact() -> dict[str, object]:
    return {
        "id": 500,
        "name": f"tpu-periodic-terminal-{SOURCE_RUN}-{SOURCE_ATTEMPT}-{SOURCE_SHA}",
        "workflow_run": {"id": int(COLLECTOR_RUN), "head_sha": COLLECTOR_SHA},
    }


def _artifact_metadata(
    artifact_id: int,
    name: str,
    collector_id: int,
    collector_sha: str,
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "name": name,
        "size_in_bytes": 128,
        "expired": False,
        "workflow_run": {"id": collector_id, "head_sha": collector_sha},
    }


def _complete_collect_job(
    collector_id: int,
    collector_sha: str,
    source_run_id: str,
    source_attempt: str,
    outcome: str = "success",
) -> dict[str, object]:
    collect_conclusion = "failure" if outcome == "remote_failure" else "success"
    return {
        "id": 701,
        "run_id": collector_id,
        "head_sha": collector_sha,
        "name": f"collect-{source_run_id}-{source_attempt}",
        "status": "completed",
        "conclusion": collect_conclusion,
        "started_at": "2026-11-04T11:00:00Z",
        "completed_at": "2026-11-04T11:01:00Z",
        "steps": [
            {
                "name": name,
                "status": "completed",
                "conclusion": (
                    collect_conclusion
                    if name == "Collect and validate the delayed TPU result"
                    else "success"
                ),
            }
            for name in (
                "Collect and validate the delayed TPU result",
                "Upload periodic collection evidence",
                "Record terminal collection marker",
            )
        ],
    }


def _terminal_source_dir(path: Path, *, publication_capable: bool) -> Path:
    path.mkdir()
    source = _source_record()
    if publication_capable:
        source.update(
            candidate_artifact_id=103,
            candidate_artifact_name=f"candidate-binding-{SOURCE_ATTEMPT}",
            marker_artifact_id=104,
            marker_artifact_name=f"release-pending-{SOURCE_ATTEMPT}",
        )
    _ = (path / "collection-source.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    _ = (path / "binding.json").write_text(json.dumps(_binding()), encoding="utf-8")
    _ = (path / "kaggle-controller-state.json").write_text(
        json.dumps(_controller_state()), encoding="utf-8"
    )
    return path


def _patch_completed_terminal_lookup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    job: dict[str, object] | None,
    record: dict[str, object] | None = None,
) -> list[str]:
    owner_attempt = _collector_run()
    owner_attempt["conclusion"] = "failure"
    routes: list[str] = []

    def api(route: str, body: object = None) -> Any:
        assert body is None
        routes.append(route)
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return owner_attempt
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}/jobs?per_page=100"
        ):
            return {
                "total_count": 1 if job is not None else 0,
                "jobs": [job] if job is not None else [],
            }
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(f"unexpected API route: {route}")

    def terminal_artifacts(
        source_run: int, name: str, budget: _BudgetLike
    ) -> list[dict[str, object]]:
        del source_run, name, budget
        return [_terminal_artifact()]

    def collection_artifact(
        *,
        source_run_id: str,
        source_attempt: str,
        commit: str,
        collector_id: int,
        collector_attempt: str,
        collector_sha: str,
        budget: _BudgetLike,
    ) -> dict[str, object]:
        del (
            source_run_id,
            source_attempt,
            commit,
            collector_id,
            collector_attempt,
            collector_sha,
            budget,
        )
        return {"id": 600, "name": "verified-collection"}

    terminal = _terminal_record() if record is None else record
    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(periodic, "_terminal_artifacts", terminal_artifacts)
    monkeypatch.setattr(periodic, "_collection_artifact", collection_artifact)
    monkeypatch.setattr(collection_gate, "download_member", _write_record(terminal))
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)
    return routes


def test_completed_publication_source_with_failed_marker_upload_remains_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = _complete_collect_job(
        int(COLLECTOR_RUN), COLLECTOR_SHA, SOURCE_RUN, SOURCE_ATTEMPT
    )
    job["conclusion"] = "failure"
    steps = cast(list[dict[str, object]], job["steps"])
    steps[-1]["conclusion"] = "failure"
    routes = _patch_completed_terminal_lookup(monkeypatch, job=job)
    source_dir = _terminal_source_dir(tmp_path / "source", publication_capable=True)

    result = _has_periodic_terminal(
        SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, source_dir, _new_budget()
    )

    assert result is None
    assert any(route.endswith("/jobs?per_page=100") for route in routes)


def test_completed_publication_source_with_valid_collect_proof_is_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    job = _complete_collect_job(
        int(COLLECTOR_RUN), COLLECTOR_SHA, SOURCE_RUN, SOURCE_ATTEMPT
    )
    _ = _patch_completed_terminal_lookup(monkeypatch, job=job)
    source_dir = _terminal_source_dir(tmp_path / "source", publication_capable=True)

    result = _has_periodic_terminal(
        SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, source_dir, _new_budget()
    )

    assert result == _terminal_record()


def test_completed_legacy_base_receipt_keeps_validation_only_terminal_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routes = _patch_completed_terminal_lookup(monkeypatch, job=None)
    source_dir = _terminal_source_dir(tmp_path / "source", publication_capable=False)

    result = _has_periodic_terminal(
        SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, source_dir, _new_budget()
    )

    assert result == _terminal_record()
    assert not any(route.endswith("/jobs?per_page=100") for route in routes)


def test_public_verified_terminal_artifacts_requires_job_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _complete_collect_job(
        int(COLLECTOR_RUN), COLLECTOR_SHA, SOURCE_RUN, SOURCE_ATTEMPT
    )
    routes = _patch_completed_terminal_lookup(monkeypatch, job=job)

    proofs = periodic.verified_terminal_artifacts(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        _binding(),
        str(_controller_state()["kernel_id"]),
    )

    assert proofs == [(_terminal_artifact(), _terminal_record())]
    assert any(route.endswith("/jobs?per_page=100") for route in routes)


def test_equivalent_verified_terminal_proofs_choose_stable_collector_identity() -> None:
    first = _terminal_record_for(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        _binding(),
        str(_controller_state()["kernel_id"]),
        collector_id=991,
        collector_attempt=1,
        collector_sha="c" * 40,
    )
    second = _terminal_record_for(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        _binding(),
        str(_controller_state()["kernel_id"]),
        collector_id=990,
        collector_attempt=2,
        collector_sha=COLLECTOR_SHA,
    )

    coalesce = cast(
        Callable[[list[dict[str, object]]], dict[str, object] | None],
        getattr(periodic, "_coalesce_verified_terminals"),
    )
    selected = coalesce([first, second])
    reversed_selected = coalesce([second, first])

    assert selected is not None
    assert reversed_selected == selected
    assert selected["collector_run_id"] == f"{COLLECTOR_RUN}.2"


def test_full_collection_artifact_uses_100_mib_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = (
        f"tpu-periodic-{SOURCE_SHA}-{SOURCE_RUN}-{SOURCE_ATTEMPT}-"
        f"{COLLECTOR_RUN}-{COLLECTOR_ATTEMPT}"
    )
    artifact = _artifact_metadata(601, name, COLLECTOR_RUN_ID, COLLECTOR_SHA)
    artifact["size_in_bytes"] = 3 * 1024 * 1024

    def api(route: str, body: object = None) -> Any:
        del body
        assert f"actions/artifacts?name={name}" in route
        return {"total_count": 1, "artifacts": [artifact]}

    monkeypatch.setattr(accelerator_gate, "gh_api", api)

    collection_artifact = cast(
        Callable[..., dict[str, object] | None],
        getattr(periodic, "_collection_artifact"),
    )
    loaded = collection_artifact(
        source_run_id=SOURCE_RUN,
        source_attempt=SOURCE_ATTEMPT,
        commit=SOURCE_SHA,
        collector_id=COLLECTOR_RUN_ID,
        collector_attempt=COLLECTOR_ATTEMPT,
        collector_sha=COLLECTOR_SHA,
        budget=_new_budget(),
    )

    assert loaded is not None
    assert loaded["size_in_bytes"] == 3 * 1024 * 1024


def test_periodic_context_is_exact_and_manual_event_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_periodic_context(monkeypatch)
    assert _require_periodic_context() == f"{COLLECTOR_RUN}.1"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    assert _require_periodic_context() == f"{COLLECTOR_RUN}.1"

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
            _ = _require_periodic_context()


@pytest.mark.parametrize("conclusion", ["failure", "success"])
def test_discover_returns_exact_candidates_and_matrix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, conclusion: str
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)
    run = _run()
    run["conclusion"] = conclusion

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 1, "workflow_runs": [run]}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}/artifacts?per_page=100&page=1"):
            return {"total_count": 0, "artifacts": []}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return run
        if route.startswith(f"repos/{periodic.REPOSITORY}/actions/artifacts?"):
            return {"total_count": 0, "artifacts": []}
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
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


def test_discover_caps_batch_and_rotates_over_all_eligible_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    source_ids = list(range(100, 110))
    runs = {source_id: _run(run_id=source_id, attempt=1) for source_id in source_ids}

    def api(route: str, body: object = None) -> Any:
        del body
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": len(runs), "workflow_runs": list(runs.values())}
        for source_id, run in runs.items():
            if route.endswith(f"/actions/runs/{source_id}"):
                return run
        if route.startswith(f"repos/{periodic.REPOSITORY}/actions/artifacts?"):
            return {"total_count": 0, "artifacts": []}
        raise AssertionError(f"unexpected API route: {route}")

    def prepare(
        source_run_id: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        source = _prepare_named_source(output, source_run_id, source_attempt, commit)
        _ = github_output.write_text("", encoding="utf-8")
        return source

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", prepare)
    selected: set[int] = set()
    for index, (run_number, run_id) in enumerate(
        ((7, "5123"), (8, "912000"), (9, "7000000"))
    ):
        monkeypatch.setenv("GITHUB_RUN_NUMBER", str(run_number))
        monkeypatch.setenv("GITHUB_RUN_ID", run_id)
        report = periodic.discover(
            tmp_path / f"discovery-{index}", tmp_path / f"github-output-{index}"
        )
        include = cast(dict[str, object], report["matrix"])["include"]
        assert isinstance(include, list)
        included = cast(list[object], include)
        assert len(included) == periodic.MAX_CANDIDATES
        assert report["inventory_count"] == len(source_ids)
        assert report["verified_source_count"] == len(source_ids)
        for value in included:
            assert isinstance(value, dict)
            candidate = cast(dict[str, object], value)
            source_id = candidate.get("source_run_id")
            assert isinstance(source_id, str)
            selected.add(int(source_id))

    assert selected == set(source_ids)


def test_discover_fails_on_incomplete_run_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        del body
        return {"total_count": 2, "workflow_runs": [_run()]}

    monkeypatch.setattr(
        accelerator_gate,
        "gh_api",
        api,
    )

    with pytest.raises(periodic.PeriodicError, match="inventory is incomplete"):
        _ = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")


def test_discover_ignores_sources_older_than_seven_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    old = _run(created_at=datetime.now(timezone.utc) - timedelta(days=8))

    def api(route: str, body: object = None) -> Any:
        assert "actions/workflows/pypi.yml/runs?" in route
        return {"total_count": 1, "workflow_runs": [old]}

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
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

    monkeypatch.setattr(accelerator_gate, "gh_api", api)

    def ineligible_prepare(
        source_run: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        del source_run, source_attempt, commit, output, github_output
        pytest.fail("feature branch must be known ineligible")

    monkeypatch.setattr(
        collection_gate,
        "prepare",
        ineligible_prepare,
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
        if route.endswith(f"/actions/runs/{SOURCE_RUN}/artifacts?per_page=100&page=1"):
            return {"total_count": 0, "artifacts": []}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return run
        raise AssertionError(route)

    def ineligible(*args: Any, **kwargs: Any) -> None:
        raise collection_gate.IneligibleSource("known CPU failure")

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", ineligible)
    result = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")
    assert result["has_sources"] is False

    def fail_prepare(
        source_run: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        del source_run, source_attempt, commit, output, github_output
        raise collection_gate.CollectionGateError("GitHub artefact lookup failed")

    monkeypatch.setattr(
        collection_gate,
        "prepare",
        fail_prepare,
    )
    with pytest.raises(collection_gate.CollectionGateError, match="lookup failed"):
        _ = periodic.discover(
            tmp_path / "discovery-error", tmp_path / "github-output-error"
        )


def test_discover_skips_known_non_timeout_sources_and_keeps_timeout_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    legacy_run = _run(run_id=788, attempt=1)
    version_error_run = _run(run_id=787, attempt=2)
    timeout_run = _run(run_id=int(SOURCE_RUN), attempt=int(SOURCE_ATTEMPT))
    runs = {788: legacy_run, 787: version_error_run, int(SOURCE_RUN): timeout_run}

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if "actions/workflows/pypi.yml/runs?" in route:
            return {
                "total_count": 3,
                "workflow_runs": [legacy_run, version_error_run, timeout_run],
            }
        if (
            route.endswith("/actions/runs/788/artifacts?per_page=100&page=1")
            or route.endswith("/actions/runs/787/artifacts?per_page=100&page=1")
            or route.endswith(
                f"/actions/runs/{SOURCE_RUN}/artifacts?per_page=100&page=1"
            )
        ):
            return {"total_count": 0, "artifacts": []}
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
                "submitted_version": None,
            }
            if run_id == 788
            else {
                **_controller_state(
                    outcome=(
                        "version_verification_error"
                        if run_id == 787
                        else "queue_timeout"
                    ),
                    status="running",
                ),
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
        _ = (output / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
        state_path = output / "kaggle-controller-state.json"
        _ = state_path.write_text(json.dumps(state), encoding="utf-8")
        _ = _validate_source_files(output, source)
        _ = (output / "collection-source.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        _ = github_output.write_text("", encoding="utf-8")
        return source

    def legacy_kernel_id(binding: dict[str, object]) -> str:
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        run_tag = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        return f"joey/jaxr-{run_tag}-0123456789"

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", prepare)
    monkeypatch.setattr(periodic, "_source_expired", _not_expired)

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


def test_discover_includes_validated_recovery_marker_and_ordinary_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    ordinary_run = _run(run_id=788, attempt=1)
    recovery_run = _run(run_id=790, attempt=2)
    recovery_sha = "c" * 40
    release_sha = "d" * 40
    recovery_run["head_sha"] = recovery_sha
    runs = {788: ordinary_run, 790: recovery_run}
    prepared: list[str] = []

    def api(route: str, body: object = None) -> Any:
        assert body is None
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 2, "workflow_runs": [ordinary_run, recovery_run]}
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
        prepared.append(source_run)
        output.mkdir(parents=True)
        binding = _binding(source_run, source_attempt)
        source: dict[str, object] = {
            "source_run_id": int(source_run),
            "source_attempt": int(source_attempt),
            "head_sha": commit,
            "collector_run_id": f"{COLLECTOR_RUN}.1",
            "collector_sha": COLLECTOR_SHA,
            "binding_artifact_id": 101,
            "controller_artifact_id": 102,
        }
        if source_run == "790":
            source.update(
                candidate_artifact_id=103,
                candidate_artifact_name="candidate-binding-2",
                marker_artifact_id=505,
                marker_artifact_name="recovery-source-binding-2",
            )
            binding = {
                "kind": "release",
                "head_sha": release_sha,
                "head_repository": periodic.REPOSITORY,
                "run_id": "790.2",
            }
        _ = (output / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
        state = _controller_state()
        _ = (output / "kaggle-controller-state.json").write_text(
            json.dumps({**state, "binding": binding}), encoding="utf-8"
        )
        if source_run == "790":
            candidate = {
                "mode": "recovery",
                "source_commit": release_sha,
                "head_sha": release_sha,
                "version": "1.2.3",
                "tag": "v1.2.3",
                "tag_ref": "refs/tags/v1.2.3",
                "tag_object_sha": "e" * 40,
                "release_id": 123,
                "controller_sha": recovery_sha,
                "repository": periodic.REPOSITORY,
                "run_id": "790.2",
            }
            candidate_bytes = json.dumps(candidate).encode()
            _ = (output / "candidate-binding.json").write_bytes(candidate_bytes)
            _ = (output / "marker-binding.json").write_bytes(candidate_bytes)
        _ = (output / "collection-source.json").write_text(
            json.dumps(source), encoding="utf-8"
        )
        _ = github_output.write_text("", encoding="utf-8")
        return source

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", prepare)
    monkeypatch.setattr(periodic, "_source_expired", _not_expired)

    report = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")

    assert prepared == ["788", "790"]
    assert report["matrix"] == {
        "include": [
            {"source_run_id": "788", "source_attempt": "1", "commit": SOURCE_SHA},
            {"source_run_id": "790", "source_attempt": "2", "commit": recovery_sha},
        ]
    }


@pytest.mark.parametrize(
    ("owner_status", "outcome"),
    [
        ("waiting", "success"),
        ("in_progress", "success"),
        ("waiting", "remote_failure"),
    ],
)
def test_discover_skips_completed_collect_job_while_owner_waits_for_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owner_status: str,
    outcome: str,
) -> None:
    _set_periodic_context(monkeypatch)
    waiting_source = _run(run_id=788, attempt=1)
    other_source = _run(run_id=787, attempt=2)
    source_runs = {788: waiting_source, 787: other_source}
    marker_name = f"tpu-periodic-terminal-788-1-{SOURCE_SHA}"
    full_name = f"tpu-periodic-{SOURCE_SHA}-788-1-{COLLECTOR_RUN}-2"
    marker_record = _terminal_record_for(
        "788",
        "1",
        SOURCE_SHA,
        _binding("788", "1"),
        str(_controller_state()["kernel_id"]),
        outcome=outcome,
        remote_status="failed" if outcome == "remote_failure" else "complete",
    )
    owner_attempt = _collector_run()
    owner_attempt.update(status=owner_status, conclusion=None)

    def api(route: str, body: object = None) -> Any:
        del body
        if "actions/workflows/pypi.yml/runs?" in route:
            return {"total_count": 2, "workflow_runs": list(source_runs.values())}
        for run_id, run in source_runs.items():
            if route.endswith(f"/actions/runs/{run_id}"):
                return run
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return owner_attempt
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}/jobs?per_page=100"
        ):
            job = _complete_collect_job(
                int(COLLECTOR_RUN), COLLECTOR_SHA, "788", "1", outcome
            )
            return {"total_count": 1, "jobs": [job]}
        if f"actions/artifacts?name={marker_name}" in route:
            return {
                "total_count": 1,
                "artifacts": [
                    _artifact_metadata(
                        500, marker_name, int(COLLECTOR_RUN), COLLECTOR_SHA
                    )
                ],
            }
        if f"actions/artifacts?name={full_name}" in route:
            return {
                "total_count": 1,
                "artifacts": [
                    _artifact_metadata(
                        600, full_name, int(COLLECTOR_RUN), COLLECTOR_SHA
                    )
                ],
            }
        if route.startswith(f"repos/{periodic.REPOSITORY}/actions/artifacts?"):
            return {"total_count": 0, "artifacts": []}
        raise AssertionError(f"unexpected API route: {route}")

    def prepare(
        source_run_id: str,
        source_attempt: str,
        commit: str,
        output: Path,
        github_output: Path,
    ) -> dict[str, object]:
        source = _prepare_named_source(output, source_run_id, source_attempt, commit)
        _ = github_output.write_text("", encoding="utf-8")
        return source

    def download(artifact_id: int, member: str, destination: Path) -> None:
        assert artifact_id == 500
        assert member == "periodic-terminal.json"
        _ = destination.write_text(json.dumps(marker_record), encoding="utf-8")

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "prepare", prepare)
    monkeypatch.setattr(periodic, "_source_expired", _not_expired)
    monkeypatch.setattr(collection_gate, "download_member", download)
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)

    report = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")

    assert report["matrix"] == {
        "include": [
            {"source_run_id": "787", "source_attempt": "2", "commit": SOURCE_SHA}
        ]
    }


@pytest.mark.parametrize(
    ("old_marker_becomes_valid", "old_outcome"),
    [(False, "success"), (True, "success"), (True, "remote_failure")],
)
def test_discover_retries_orphan_marker_then_uses_unique_verified_recollection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    old_marker_becomes_valid: bool,
    old_outcome: str,
) -> None:
    _set_periodic_context(monkeypatch)
    source = _run()
    source_run_id = SOURCE_RUN
    source_attempt = SOURCE_ATTEMPT
    source_binding = _binding(source_run_id, source_attempt)
    kernel_id = str(_controller_state()["kernel_id"])
    old_id, old_attempt, old_sha = 990, 2, COLLECTOR_SHA
    new_id, new_attempt, new_sha = 991, 1, "c" * 40
    marker_name = f"tpu-periodic-terminal-{source_run_id}-{source_attempt}-{SOURCE_SHA}"
    old_collection_name = (
        f"tpu-periodic-{SOURCE_SHA}-{source_run_id}-{source_attempt}-"
        f"{old_id}-{old_attempt}"
    )
    new_collection_name = (
        f"tpu-periodic-{SOURCE_SHA}-{source_run_id}-{source_attempt}-"
        f"{new_id}-{new_attempt}"
    )
    old_record = _terminal_record_for(
        source_run_id,
        source_attempt,
        SOURCE_SHA,
        source_binding,
        kernel_id,
        collector_id=old_id,
        collector_attempt=old_attempt,
        collector_sha=old_sha,
        outcome=old_outcome,
        remote_status="failed" if old_outcome == "remote_failure" else "complete",
    )
    new_record = _terminal_record_for(
        source_run_id,
        source_attempt,
        SOURCE_SHA,
        source_binding,
        kernel_id,
        collector_id=new_id,
        collector_attempt=new_attempt,
        collector_sha=new_sha,
    )
    records = {500: old_record, 501: new_record}
    old_owner = _collector_run(attempt=old_attempt)
    old_owner["conclusion"] = "cancelled"
    new_owner = {
        **_collector_run(attempt=new_attempt),
        "id": new_id,
        "head_sha": new_sha,
        "conclusion": "success",
    }
    round_number = 0

    def api(route: str, body: object = None) -> Any:
        nonlocal round_number
        del body
        if "actions/workflows/pypi.yml/runs?" in route:
            round_number += 1
            return {"total_count": 1, "workflow_runs": [source]}
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return source
        if route.endswith(f"/actions/runs/{old_id}/attempts/{old_attempt}"):
            return old_owner
        if route.endswith(f"/actions/runs/{new_id}/attempts/{new_attempt}"):
            return new_owner
        if f"actions/artifacts?name={marker_name}" in route:
            markers = [_artifact_metadata(500, marker_name, old_id, old_sha)]
            if round_number >= 2:
                markers.append(_artifact_metadata(501, marker_name, new_id, new_sha))
            return {"total_count": len(markers), "artifacts": markers}
        if f"actions/artifacts?name={old_collection_name}" in route:
            if old_marker_becomes_valid and round_number >= 2:
                old_item = _artifact_metadata(600, old_collection_name, old_id, old_sha)
                return {"total_count": 1, "artifacts": [old_item]}
            return {"total_count": 0, "artifacts": []}
        if f"actions/artifacts?name={new_collection_name}" in route:
            if round_number >= 2:
                new_item = _artifact_metadata(601, new_collection_name, new_id, new_sha)
                return {"total_count": 1, "artifacts": [new_item]}
            return {"total_count": 0, "artifacts": []}
        if route.endswith(f"/compare/{old_sha}...master"):
            return {"status": "ahead", "base_commit": {"sha": old_sha}}
        if route.endswith(f"/compare/{new_sha}...master"):
            return {"status": "ahead", "base_commit": {"sha": new_sha}}
        raise AssertionError(f"unexpected API route: {route}")

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    _patch_prepare(monkeypatch)
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)

    def download(artifact_id: int, member: str, destination: Path) -> None:
        del member
        _ = destination.write_text(json.dumps(records[artifact_id]), encoding="utf-8")

    monkeypatch.setattr(
        collection_gate,
        "download_member",
        download,
    )

    first = periodic.discover(
        tmp_path / "first-discovery", tmp_path / "first-github-output"
    )
    assert first["matrix"] == {
        "include": [
            {
                "source_run_id": source_run_id,
                "source_attempt": source_attempt,
                "commit": SOURCE_SHA,
            }
        ]
    }

    if old_marker_becomes_valid and old_outcome == "remote_failure":
        with pytest.raises(
            periodic.PeriodicError, match="conflicting verified terminal"
        ):
            _ = periodic.discover(
                tmp_path / "second-discovery", tmp_path / "second-github-output"
            )
    else:
        second = periodic.discover(
            tmp_path / "second-discovery", tmp_path / "second-github-output"
        )
        assert second["has_sources"] is False


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

    def api(route: str, body: dict[str, object] | None = None) -> Any:
        del body
        return {"total_count": 1, "workflow_runs": [bad_run]}

    monkeypatch.setattr(
        accelerator_gate,
        "gh_api",
        api,
    )
    with pytest.raises(periodic.PeriodicError, match="identity is not canonical"):
        _ = periodic.discover(tmp_path / "discovery", tmp_path / "github-output")


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
        _ = destination.write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(collection_gate, "download_member", download)
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)

    loaded = _read_periodic_terminal(
        _terminal_artifact(),
        source_run_id=SOURCE_RUN,
        source_attempt=SOURCE_ATTEMPT,
        commit=SOURCE_SHA,
        binding=_binding(),
        kernel_id=str(_controller_state()["kernel_id"]),
        budget=_new_budget(),
    )
    assert loaded is not None
    assert loaded["outcome"] == outcome


@pytest.mark.parametrize(
    "run_changes",
    [
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

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "download_member",
        _write_record(_terminal_record()),
    )
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)
    with pytest.raises(periodic.PeriodicError):
        _ = _read_periodic_terminal(
            _terminal_artifact(),
            source_run_id=SOURCE_RUN,
            source_attempt=SOURCE_ATTEMPT,
            commit=SOURCE_SHA,
            binding=_binding(),
            kernel_id=str(_controller_state()["kernel_id"]),
            budget=_new_budget(),
        )


def test_terminal_marker_waits_for_active_collection_job_to_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_attempt = _collector_run()
    active_attempt.update(status="in_progress", conclusion=None)
    job: dict[str, object] = {
        "id": 701,
        "run_id": int(COLLECTOR_RUN),
        "head_sha": COLLECTOR_SHA,
        "name": f"collect-{SOURCE_RUN}-{SOURCE_ATTEMPT}",
        "status": "in_progress",
        "conclusion": None,
        "started_at": "2026-11-04T11:00:00Z",
        "completed_at": None,
        "steps": [],
    }

    def api(route: str, body: object = None) -> Any:
        del body
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}"
        ):
            return active_attempt
        if route.endswith(
            f"/actions/runs/{COLLECTOR_RUN}/attempts/{COLLECTOR_ATTEMPT}/jobs?per_page=100"
        ):
            return {"total_count": 1, "jobs": [job]}
        if route.endswith(f"/compare/{COLLECTOR_SHA}...master"):
            return {"status": "ahead", "base_commit": {"sha": COLLECTOR_SHA}}
        raise AssertionError(f"unexpected API route: {route}")

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "download_member",
        _write_record(_terminal_record()),
    )
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)

    assert (
        _read_periodic_terminal(
            _terminal_artifact(),
            source_run_id=SOURCE_RUN,
            source_attempt=SOURCE_ATTEMPT,
            commit=SOURCE_SHA,
            binding=_binding(),
            kernel_id=str(_controller_state()["kernel_id"]),
            budget=_new_budget(),
        )
        is None
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

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "download_member",
        _write_record(record),
    )
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)

    _ = _read_periodic_terminal(
        _terminal_artifact(),
        source_run_id=SOURCE_RUN,
        source_attempt=SOURCE_ATTEMPT,
        commit=SOURCE_SHA,
        binding=_binding(),
        kernel_id=str(_controller_state()["kernel_id"]),
        budget=_new_budget(),
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

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    monkeypatch.setattr(
        collection_gate,
        "download_member",
        _write_record(record),
    )
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)
    with pytest.raises(periodic.PeriodicError, match=message):
        _ = _read_periodic_terminal(
            _terminal_artifact(),
            source_run_id=SOURCE_RUN,
            source_attempt=SOURCE_ATTEMPT,
            commit=SOURCE_SHA,
            binding=_binding(),
            kernel_id=str(_controller_state()["kernel_id"]),
            budget=_new_budget(),
        )


@pytest.mark.parametrize("outcome", ["pending", "success", "remote_failure"])
def test_collect_source_terminal_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)

    def no_terminal(
        source_run: str,
        attempt: str,
        commit: str,
        source: Path,
        budget: _BudgetLike,
    ) -> dict[str, object] | None:
        del source_run, attempt, commit, source, budget
        return None

    monkeypatch.setattr(periodic, "_has_terminal", no_terminal)
    finish_calls: list[bool] = []

    def finish(source: dict[str, object], output: Path, provider_success: bool) -> None:
        del source, output
        finish_calls.append(provider_success)

    monkeypatch.setattr(
        collection_gate,
        "finish",
        finish,
    )

    def revalidate(
        source: Path,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        del source
        return _source_record(), _binding(), _controller_state()

    monkeypatch.setattr(
        collection_gate,
        "revalidate",
        revalidate,
    )

    def collect_result(
        binding: dict[str, object], resume_state_path: Path, output_dir: Path
    ) -> dict[str, object]:
        del binding, resume_state_path, output_dir
        status = (
            "queued"
            if outcome == "pending"
            else "failed"
            if outcome == "remote_failure"
            else "complete"
        )
        return _report(outcome, status=status)

    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        collect_result,
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


def test_collect_source_can_move_from_pending_to_verified_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)

    def no_terminal(
        source_run: str,
        attempt: str,
        commit: str,
        source: Path,
        budget: _BudgetLike,
    ) -> dict[str, object] | None:
        del source_run, attempt, commit, source, budget
        return None

    monkeypatch.setattr(periodic, "_has_terminal", no_terminal)
    outcomes = iter(("pending", "success"))
    finish_calls: list[bool] = []

    def collect_result(
        binding: dict[str, object], resume_state_path: Path, output_dir: Path
    ) -> dict[str, object]:
        del binding, resume_state_path, output_dir
        outcome = next(outcomes)
        return _report(outcome, status="queued" if outcome == "pending" else "complete")

    def finish(source: dict[str, object], output: Path, provider_success: bool) -> None:
        del source, output
        finish_calls.append(provider_success)

    monkeypatch.setattr(kaggle_collect, "collect", collect_result)
    monkeypatch.setattr(collection_gate, "finish", finish)

    pending, pending_exit = periodic.collect_source(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        tmp_path / "pending",
        tmp_path / "pending-output",
    )
    success, success_exit = periodic.collect_source(
        SOURCE_RUN,
        SOURCE_ATTEMPT,
        SOURCE_SHA,
        tmp_path / "success",
        tmp_path / "success-output",
    )

    assert pending == {"terminal": False, "outcome": "pending"}
    assert pending_exit == 0
    assert success == {"terminal": True, "outcome": "success"}
    assert success_exit == 0
    assert (
        json.loads((tmp_path / "success" / "periodic-terminal.json").read_text())[
            "remote_status"
        ]
        == "complete"
    )
    assert finish_calls == [True]


def test_collect_source_expires_before_provider_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    monkeypatch.setattr(periodic, "_source_expired", _has_expired)
    fail_collect: CollectResult = _raise_provider_not_called
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        fail_collect,
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


@pytest.mark.parametrize("conclusion", ["failure", "success"])
def test_source_expiry_uses_exact_attempt_metadata(
    monkeypatch: pytest.MonkeyPatch, conclusion: str
) -> None:
    old = _run(created_at=datetime.now(timezone.utc) - timedelta(days=8))
    old["conclusion"] = conclusion
    routes: list[str] = []

    def api(route: str, body: object = None) -> Any:
        routes.append(route)
        if route.endswith(f"/actions/runs/{SOURCE_RUN}/attempts/{SOURCE_ATTEMPT}"):
            return old
        if route.endswith(f"/actions/runs/{SOURCE_RUN}"):
            return old
        raise AssertionError(route)

    monkeypatch.setattr(accelerator_gate, "gh_api", api)
    budget = _new_budget()

    assert _source_expired(SOURCE_RUN, SOURCE_ATTEMPT, SOURCE_SHA, budget)
    assert routes[0].endswith(f"/actions/runs/{SOURCE_RUN}/attempts/{SOURCE_ATTEMPT}")


def test_collect_source_remote_failure_revalidates_original_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_periodic_context(monkeypatch)
    _patch_prepare(monkeypatch)

    def no_terminal(
        source_run: str,
        attempt: str,
        commit: str,
        source: Path,
        budget: _BudgetLike,
    ) -> dict[str, object] | None:
        del source_run, attempt, commit, source, budget
        return None

    monkeypatch.setattr(periodic, "_has_terminal", no_terminal)
    revalidate_calls: list[Path] = []

    def revalidate(
        source: Path,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        revalidate_calls.append(source)
        return _source_record(), _binding(), _controller_state()

    monkeypatch.setattr(
        collection_gate,
        "revalidate",
        revalidate,
    )

    def collect_result(
        binding: dict[str, object], state_path: Path, output: Path
    ) -> dict[str, object]:
        del binding, state_path, output
        return _report("remote_failure", status="failed")

    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        collect_result,
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

    def no_terminal(
        source_run: str,
        attempt: str,
        commit: str,
        source: Path,
        budget: _BudgetLike,
    ) -> dict[str, object] | None:
        del source_run, attempt, commit, source, budget
        return None

    monkeypatch.setattr(periodic, "_has_terminal", no_terminal)
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
            _ = periodic.collect_source(
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

    monkeypatch.setattr(accelerator_gate, "gh_api", api)

    def terminal_artifacts(
        source_run: int, name: str, budget: _BudgetLike
    ) -> list[dict[str, object]]:
        del source_run, name, budget
        return [_terminal_artifact()]

    monkeypatch.setattr(
        periodic,
        "_terminal_artifacts",
        terminal_artifacts,
    )

    def collection_artifact(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {"id": 600, "name": "verified-collection"}

    monkeypatch.setattr(
        periodic,
        "_collection_artifact",
        collection_artifact,
    )
    monkeypatch.setattr(
        collection_gate,
        "download_member",
        _write_record(terminal),
    )
    monkeypatch.setattr(release_tpu_gate, "verify_master_history", _no_master_history)
    fail_collect: CollectResult = _raise_terminal_provider_not_called
    monkeypatch.setattr(
        kaggle_collect,
        "collect",
        fail_collect,
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
