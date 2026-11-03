from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest
from tools import kaggle_ci, kaggle_collect, kaggle_wait

BINDING: dict[str, object] = {
    "kind": "release",
    "head_sha": "a" * 40,
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "371234567.1",
}
SYNTHETIC_RESULT: dict[str, object] = {
    **BINDING,
    "backend": "tpu",
    "device_backend": "tpu",
    "device_count": 1,
    "success": True,
    "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
    "versions": {"jax": "0.11.2"},
}
SUBMITTED = datetime(2026, 11, 3, 4, 0, 0, tzinfo=timezone.utc)


def _kernel_id() -> str:
    run_tag = hashlib.sha256(
        json.dumps(BINDING, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"test-user/jaxr-{run_tag}-0123456789"


def _state() -> dict[str, object]:
    updated = SUBMITTED + timedelta(seconds=kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5)
    return {
        "binding": BINDING,
        "kernel_id": _kernel_id(),
        "submitted_version": 1,
        "status": "queued",
        "phase": "queue",
        "phase_started_at": SUBMITTED.isoformat(),
        "submitted_at": SUBMITTED.isoformat(),
        "execution_started_at": None,
        "updated_at": updated.isoformat(),
        "queue_elapsed_seconds": float(kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5),
        "execution_elapsed_seconds": 0.0,
        "outcome": "queue_timeout",
        "last_error": "Kaggle queue polling timed out",
    }


def _binding_file(tmp_path: Path) -> Path:
    path = tmp_path / "binding.json"
    _ = path.write_text(json.dumps(BINDING), encoding="utf-8")
    return path


def _resume_dir(tmp_path: Path, state: dict[str, object] | None = None) -> Path:
    path = tmp_path / "previous"
    _ = path.mkdir()
    _ = (path / "kaggle-controller-state.json").write_text(
        json.dumps(_state() if state is None else state), encoding="utf-8"
    )
    _ = (path / "kaggle-controller.log").write_text("{}\n", encoding="utf-8")
    return path


def _report(
    output_dir: Path,
    status: str,
    *,
    outcome: str = "pending",
    success: bool = False,
) -> dict[str, object]:
    _ = output_dir.mkdir()
    report: dict[str, object] = {
        "binding": BINDING,
        "kernel_id": _kernel_id(),
        "submitted_version": 1,
        "remote_status": status,
        "version_verified": True,
        "outcome": outcome,
        "success": success,
    }
    _ = (output_dir / "kaggle-collection-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    if success:
        _ = (output_dir / "result.json").write_text(
            json.dumps(SYNTHETIC_RESULT), encoding="utf-8"
        )
    return report


def test_submit_returns_handoff_ack_only_for_valid_first_queue_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    now = SUBMITTED + timedelta(seconds=kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5)
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    monkeypatch.setattr(kaggle_wait, "_utc_now", lambda: now)

    def timed_out(
        binding: dict[str, object],
        backend: str,
        output_dir: Path,
        *,
        queue_timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        assert binding == BINDING
        assert backend == "tpu"
        assert queue_timeout_seconds == 4 * 60 * 60
        _ = (output_dir / "kaggle-controller-state.json").write_text(
            json.dumps(_state()), encoding="utf-8"
        )
        raise kaggle_ci.KaggleError("Kaggle queue polling timed out")

    monkeypatch.setattr(kaggle_ci, "run", timed_out)
    assert kaggle_wait.submit(_binding_file(tmp_path), tmp_path / "output") == 0
    assert "validated continuation handoff" in capsys.readouterr().out


def test_submit_does_not_ack_other_errors_even_if_result_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    state = _state()
    state.update(
        status="failed", outcome="remote_terminal_failure", last_error="failed"
    )

    def failed(
        binding: dict[str, object], backend: str, output_dir: Path, **kwargs: object
    ) -> dict[str, object]:
        del binding, backend, kwargs
        _ = (output_dir / "kaggle-controller-state.json").write_text(json.dumps(state))
        _ = (output_dir / "result.json").write_text(json.dumps(SYNTHETIC_RESULT))
        raise kaggle_ci.KaggleError("remote run failed")

    monkeypatch.setattr(kaggle_ci, "run", failed)
    with pytest.raises(kaggle_ci.KaggleError, match="remote run failed"):
        _ = kaggle_wait.submit(_binding_file(tmp_path), output)


def test_submit_rejects_a_nonempty_output_directory_before_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    _ = output.mkdir()
    _ = (output / "kaggle-controller-state.json").write_text(json.dumps(_state()))

    def unexpected(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        pytest.fail("stale output must not reach Kaggle")

    monkeypatch.setattr(kaggle_ci, "run", unexpected)
    with pytest.raises(kaggle_ci.KaggleError, match="must be fresh"):
        _ = kaggle_wait.submit(_binding_file(tmp_path), output)


def test_continue_polls_same_kernel_and_collects_result_without_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    current = [SUBMITTED + timedelta(seconds=kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5)]
    monotonic = [0.0]
    monkeypatch.setattr(kaggle_wait, "_utc_now", lambda: current[0])
    monkeypatch.setattr(kaggle_wait, "_monotonic", lambda: monotonic[0])

    def sleep(seconds: float) -> None:
        current[0] += timedelta(seconds=seconds)
        monotonic[0] += seconds

    monkeypatch.setattr(kaggle_wait, "_sleep", sleep)
    statuses = iter(("queued", "running", "complete"))
    collector_dirs: list[Path] = []

    def collect(
        binding: dict[str, object], state_path: Path, output_dir: Path
    ) -> dict[str, object]:
        assert binding == BINDING
        assert state_path.parent.name == "previous"
        collector_dirs.append(output_dir)
        status = next(statuses)
        return _report(
            output_dir,
            status,
            outcome="success" if status == "complete" else "pending",
            success=status == "complete",
        )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("continuation must not submit or query through kaggle_ci")

    monkeypatch.setattr(kaggle_collect, "collect", collect)
    monkeypatch.setattr(kaggle_ci, "run", forbidden)
    monkeypatch.setattr(kaggle_ci, "_command_ok", forbidden)
    resume = _resume_dir(tmp_path)
    original_state = (resume / "kaggle-controller-state.json").read_text()

    assert (
        kaggle_wait.continue_wait(_binding_file(tmp_path), resume, tmp_path / "output")
        == 0
    )
    assert len(set(collector_dirs)) == 3
    assert (resume / "kaggle-controller-state.json").read_text() == original_state
    final_state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert final_state["kernel_id"] == _kernel_id()
    assert final_state["phase"] == "execution"
    assert final_state["execution_started_at"] is not None
    assert final_state["outcome"] == "success"
    assert (tmp_path / "output" / "result.json").is_file()


def test_continue_applies_eight_hour_total_from_original_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    now = SUBMITTED + timedelta(hours=8) - timedelta(seconds=10)
    current = [now]
    monotonic = [0.0]
    monkeypatch.setattr(kaggle_wait, "_utc_now", lambda: current[0])
    monkeypatch.setattr(kaggle_wait, "_monotonic", lambda: monotonic[0])
    calls = 0

    def collect(
        binding: dict[str, object], state_path: Path, output_dir: Path
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        current[0] += timedelta(seconds=15)
        monotonic[0] += 15
        return _report(output_dir, "queued")

    monkeypatch.setattr(kaggle_collect, "collect", collect)
    resume = _resume_dir(tmp_path)
    with pytest.raises(kaggle_ci.KaggleError, match="total queue wait timed out"):
        _ = kaggle_wait.continue_wait(
            _binding_file(tmp_path), resume, tmp_path / "output"
        )
    assert calls == 1
    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["outcome"] == "queue_timeout"
    assert state["queue_elapsed_seconds"] >= kaggle_ci.QUEUE_TIMEOUT_SECONDS


def test_execution_deadline_does_not_reset_when_status_returns_to_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    current = [SUBMITTED + timedelta(seconds=kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5)]
    monotonic = [0.0]
    monkeypatch.setattr(kaggle_wait, "_utc_now", lambda: current[0])
    monkeypatch.setattr(kaggle_wait, "_monotonic", lambda: monotonic[0])
    monkeypatch.setattr(kaggle_wait, "POLL_INTERVAL_SECONDS", 20)
    monkeypatch.setattr(kaggle_ci, "EXECUTION_TIMEOUT_SECONDS", 30)

    def sleep(seconds: float) -> None:
        current[0] += timedelta(seconds=seconds)
        monotonic[0] += seconds

    monkeypatch.setattr(kaggle_wait, "_sleep", sleep)
    statuses = iter(("running", "queued"))
    calls = 0

    def collect(
        binding: dict[str, object], state_path: Path, output_dir: Path
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _report(output_dir, next(statuses))

    monkeypatch.setattr(kaggle_collect, "collect", collect)
    with pytest.raises(kaggle_ci.KaggleError, match="execution wait timed out"):
        _ = kaggle_wait.continue_wait(
            _binding_file(tmp_path), _resume_dir(tmp_path), tmp_path / "output"
        )
    assert calls == 2
    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["phase"] == "execution"
    assert state["status"] == "queued"
    assert state["outcome"] == "execution_timeout"
    assert state["execution_elapsed_seconds"] == 30


@pytest.mark.parametrize("invalid", ["binding", "future", "version", "nonfinite"])
def test_continue_rejects_invalid_handoffs_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    current = SUBMITTED + timedelta(seconds=kaggle_wait.FIRST_QUEUE_WAIT_SECONDS + 5)
    monkeypatch.setattr(kaggle_wait, "_utc_now", lambda: current)
    state = _state()
    if invalid == "binding":
        state["binding"] = {**BINDING, "run_id": "371234567.2"}
    elif invalid == "future":
        state["updated_at"] = (current + timedelta(seconds=1)).isoformat()
    elif invalid == "version":
        state["submitted_version"] = 2
    else:
        state["queue_elapsed_seconds"] = float("nan")
    calls = 0

    def unexpected(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        del args, kwargs
        pytest.fail("invalid handoff reached the provider")

    monkeypatch.setattr(kaggle_collect, "collect", unexpected)
    with pytest.raises(kaggle_ci.KaggleError):
        _ = kaggle_wait.continue_wait(
            _binding_file(tmp_path),
            _resume_dir(tmp_path, state),
            tmp_path / "output",
        )
    assert calls == 0


def test_continue_skips_provider_for_a_valid_existing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = tmp_path / "previous"
    _ = resume.mkdir()
    _ = (resume / "result.json").write_text(json.dumps(SYNTHETIC_RESULT))

    def unexpected(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        pytest.fail("a validated completed result must skip provider polling")

    monkeypatch.setattr(kaggle_collect, "collect", unexpected)
    assert (
        kaggle_wait.continue_wait(_binding_file(tmp_path), resume, tmp_path / "output")
        == 0
    )
    assert (tmp_path / "output" / "result.json").read_text() == json.dumps(
        SYNTHETIC_RESULT
    )
