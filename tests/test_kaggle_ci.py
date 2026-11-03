from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, cast
from unittest.mock import patch

import pytest
from tools import kaggle_ci

# Synthetic fixture catalogue: api-key-a (active), joey-private-v3.
SYNTHETIC_API_KEY = "codex_synth_v1_api_key_a"

BINDING: dict[str, object] = {
    "kind": "release",
    "head_sha": "a" * 40,
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "371234567.1",
}


def _private_helper(name: str) -> Callable[..., Any]:
    return cast(Callable[..., Any], getattr(kaggle_ci, name))


def success_result(binding: dict[str, object] = BINDING) -> dict[str, object]:
    return {
        "kind": binding["kind"],
        "head_sha": binding["head_sha"],
        "head_repository": binding["head_repository"],
        "run_id": binding["run_id"],
        "backend": "tpu",
        "device_backend": "tpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
        "versions": {"jax": "0.11.2"},
    }


def completed(
    args: list[str], stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, "")


def _no_provider_access(*args: object, **kwargs: object) -> None:
    del args, kwargs
    pytest.fail("unexpected provider access")


def _record_version_verification(
    binding: dict[str, object], path: Path, calls: list[Path]
) -> None:
    del binding
    calls.append(path)


def _ignore_version_verification(binding: dict[str, object], path: Path) -> None:
    del binding, path


def authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", SYNTHETIC_API_KEY)
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")


def _relay_state(
    path: Path,
    *,
    outcome: str = "handover",
    status: str = "queued",
    queue_elapsed: float = 0.0,
    execution_elapsed: float = 0.0,
    execution_started_at: datetime | None = None,
    updated_at: datetime | None = None,
    binding: dict[str, object] = BINDING,
) -> dict[str, object]:
    now = datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)
    run_tag = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    last_update = updated_at or now
    submitted_at = last_update
    execution_started = execution_started_at
    state: dict[str, object] = {
        "binding": binding,
        "kernel_id": f"test-user/jaxr-{run_tag}-0123456789",
        "submitted_version": 1,
        "status": status,
        "phase": "execution" if execution_started is not None else "queue",
        "phase_started_at": submitted_at.isoformat(),
        "submitted_at": submitted_at.isoformat(),
        "execution_started_at": (
            execution_started.isoformat() if execution_started is not None else None
        ),
        "updated_at": last_update.isoformat(),
        "queue_elapsed_seconds": queue_elapsed,
        "execution_elapsed_seconds": execution_elapsed,
        "outcome": outcome,
        "last_error": None,
    }
    _ = path.write_text(json.dumps(state), encoding="utf-8")
    return state


def test_quota_requires_unambiguous_tpu_headroom() -> None:
    assert (
        _private_helper("_read_quota")(
            '[{"resource":"GPU","remaining":"10.00h"},'
            '{"resource":"TPU","remaining":"0.75h"}]'
        )
        == 0.75
    )
    for response in (
        "not json",
        "",
        "No quota information available",
        "[]",
        '[{"resource":"TPU","remaining":"unknown"}]',
        '[{"resource":"TPU","remaining":"0.74h"}]',
        '[{"resource":"TPU","remaining":"1h"},{"resource":"TPU","remaining":"2h"}]',
    ):
        with pytest.raises(kaggle_ci.KaggleError):
            _ = _private_helper("_read_quota")(response)


def test_run_cli_places_no_warn_before_the_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_popen = subprocess.Popen
    commands: list[list[str]] = []

    def harmless_popen(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        commands.append(command)
        return cast(
            Any,
            actual_popen(
                [sys.executable, "-c", "print('[]')"],
                **cast(Any, kwargs),
            ),
        )

    monkeypatch.setattr(kaggle_ci.subprocess, "Popen", harmless_popen)
    result = _private_helper("_run_cli")(
        ["kaggle", "quota", "--format", "json"], env={}
    )

    assert result.stdout == "[]\n"
    assert commands == [
        [
            sys.executable,
            "-m",
            "tools.kaggle_cli",
            "--no-warn",
            "quota",
            "--format",
            "json",
        ]
    ]


@pytest.mark.parametrize(
    ("quota_stdout", "expected_error"),
    [
        ("", "Kaggle quota response was empty"),
        ("No quota information available", "Kaggle quota information is unavailable"),
        (
            f"not JSON; token={SYNTHETIC_API_KEY}; https://example.invalid/signed?sig=secret",
            "Kaggle quota response was not valid JSON",
        ),
    ],
)
def test_quota_preflight_diagnostics_fail_closed_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    quota_stdout: str,
    expected_error: str,
) -> None:
    authenticated(monkeypatch)
    commands: list[list[str]] = []

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del env, timeout
        commands.append(args)
        return completed(args, quota_stdout)

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match=expected_error):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

    state_text = (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    log_text = (tmp_path / "output" / "kaggle-controller.log").read_text()
    captured = capsys.readouterr().out
    assert len(commands) == 1
    assert commands[0][1:3] == ["quota", "--format"]
    assert "kernels" not in " ".join(commands[0])
    assert expected_error in state_text + log_text + captured
    assert SYNTHETIC_API_KEY not in state_text + log_text + captured
    assert "signed?sig=secret" not in state_text + log_text + captured


def test_status_accepts_kaggle_worker_status_enum_output() -> None:
    assert (
        _private_helper("_status")('has status "KernelWorkerStatus.QUEUED"') == "queued"
    )
    assert (
        _private_helper("_status")('has status "KernelWorkerStatus.COMPLETE"')
        == "complete"
    )
    with pytest.raises(kaggle_ci.KaggleError, match="status type"):
        _ = _private_helper("_status")('has status "UntrustedStatus.QUEUED"')


def test_missing_credentials_fail_before_any_kaggle_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    def unexpected_command(*args: object, **kwargs: object) -> None:
        pytest.fail("Kaggle command was reached without credentials")

    monkeypatch.setattr(kaggle_ci, "_command_ok", unexpected_command)
    with pytest.raises(kaggle_ci.KaggleError, match="KAGGLE_API_TOKEN"):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path)
    assert "KAGGLE_API_TOKEN" in (tmp_path / "kaggle-controller.log").read_text()


def test_private_kernel_embeds_only_the_trusted_runner(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    folder = tmp_path / "kernel"
    _private_helper("_make_kernel")(
        folder, "test-user", "jaxr-0123456789abcdef-0123456789", BINDING, runner
    )

    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((folder / "bootstrap.ipynb").read_text(encoding="utf-8"))
    bootstrap = "".join(notebook["cells"][0]["source"])
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "TpuV5E8"
    assert metadata["kernel_type"] == "notebook"
    assert metadata["code_file"] == "bootstrap.ipynb"
    assert metadata["id"].endswith("/jaxr-0123456789abcdef-0123456789")
    assert "def run(binding, backend, output_dir):" in bootstrap
    assert "uv==0.12.20" in bootstrap
    assert json.dumps(BINDING, sort_keys=True) in bootstrap


@pytest.mark.parametrize("relay", [False, True])
def test_success_uses_a_fresh_private_slug_polls_and_downloads_bound_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relay: bool
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    if relay:
        monkeypatch.setattr(
            kaggle_ci, "_verify_version_one", _ignore_version_verification
        )
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])

    def advance_time(duration: float) -> None:
        fake_time[0] += duration

    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        advance_time,
    )
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 1900)
    commands: list[list[str]] = []
    status_responses = iter(
        (
            'has status "KernelWorkerStatus.QUEUED"',
            'has status "KernelWorkerStatus.RUNNING"',
            'has status "KernelWorkerStatus.COMPLETE"',
        )
    )

    def fake_command(
        args: list[str],
        *,
        env: dict[str, str],
        timeout: float = kaggle_ci.CLI_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        commands.append(args)
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"5.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            kernel_dir = Path(args[args.index("--path") + 1])
            metadata = json.loads(
                (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8")
            )
            controller_state = json.loads(
                (tmp_path / "output" / "kaggle-controller-state.json").read_text(
                    encoding="utf-8"
                )
            )
            assert metadata["is_private"] is True
            assert metadata["id"] == controller_state["kernel_id"]
            assert controller_state["binding"] == BINDING
            assert controller_state["kernel_id"].startswith("test-user/jaxr-")
            assert controller_state["submitted_version"] is None
            assert args[args.index("--accelerator") + 1] == "TpuV5E8"
            assert args[args.index("--timeout") + 1] == str(
                kaggle_ci.EXECUTION_TIMEOUT_SECONDS
            )
            return completed(args, "Kernel version 1 successfully pushed.\n")
        if args[1:3] == ["kernels", "status"]:
            return completed(args, next(status_responses))
        if args[1:3] == ["kernels", "output"]:
            assert args[3].endswith("/1")
            assert "render-artifacts/" in args[args.index("--file-pattern") + 1]
            output_dir = Path(args[args.index("--path") + 1])
            # Remote output must preserve the controller's submission identity.
            pattern = re.compile(args[args.index("--file-pattern") + 1])
            if pattern.fullmatch("kaggle-controller-state.json"):
                _ = (output_dir / "kaggle-controller-state.json").write_text("{}")
            output = output_dir / "result.json"
            _ = output.write_text(json.dumps(success_result()), encoding="utf-8")
            return completed(args)
        pytest.fail(f"unexpected Kaggle command: {args}")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    result = kaggle_ci.run(BINDING, "tpu", tmp_path / "output", relay=relay)

    assert result == success_result()
    assert fake_time[0] == 3800
    controller_state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert controller_state["submitted_version"] == 1
    assert controller_state["binding"] == BINDING
    assert controller_state["outcome"] == "success"
    assert controller_state["queue_elapsed_seconds"] == 1900
    pushes = [command for command in commands if command[1:3] == ["kernels", "push"]]
    assert len(pushes) == 1
    # The private slug includes a random suffix so output from an older run cannot match.
    status_commands = [
        command for command in commands if command[1:3] == ["kernels", "status"]
    ]
    assert len(status_commands) == 3
    assert status_commands[0][-1] == status_commands[1][-1]
    assert controller_state["kernel_id"] == status_commands[0][-1]
    assert re.fullmatch(
        r"test-user/jaxr-[0-9a-f]{16}-[0-9a-f]{10}", status_commands[0][-1]
    )


def test_queue_timeout_persists_submitted_identity_without_downloading_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "QUEUE_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 3)
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])

    def advance_time(duration: float) -> None:
        fake_time[0] += duration

    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        advance_time,
    )
    commands: list[list[str]] = []

    def fake_command(
        args: list[str],
        *,
        env: dict[str, str],
        timeout: float = kaggle_ci.CLI_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        commands.append(args)
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"5.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.\n")
        if args[1:3] == ["kernels", "status"]:
            return completed(args, 'has status "QUEUED"')
        pytest.fail("a timed-out kernel must not download results")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="queue polling timed out"):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
    assert sum(command[1:3] == ["kernels", "push"] for command in commands) == 1
    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["binding"] == BINDING
    assert state["kernel_id"].startswith("test-user/jaxr-")
    assert state["submitted_version"] == 1
    assert state["status"] == "queued"
    assert state["phase"] == "queue"
    assert state["queue_elapsed_seconds"] == 5
    assert state["outcome"] == "queue_timeout"


def test_status_returned_at_queue_deadline_times_out_before_result_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "QUEUE_TIMEOUT_SECONDS", 5)
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])
    commands: list[list[str]] = []

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del env, timeout
        commands.append(args)
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"1.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.")
        if args[1:3] == ["kernels", "status"]:
            fake_time[0] = 5
            return completed(args, 'has status "COMPLETE"')
        pytest.fail("a deadline-bound status response must not download output")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="queue polling timed out"):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert fake_time[0] == 5
    assert state["status"] == "complete"
    assert state["outcome"] == "queue_timeout"
    assert state["queue_elapsed_seconds"] == 5
    assert not any(command[1:3] == ["kernels", "output"] for command in commands)


def test_failed_remote_status_still_retrieves_scoped_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    commands: list[list[str]] = []

    def fake_command(
        args: list[str],
        *,
        env: dict[str, str],
        timeout: float = kaggle_ci.CLI_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        commands.append(args)
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"5.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.\n")
        if args[1:3] == ["kernels", "status"]:
            return completed(
                args,
                f'has status "KernelWorkerStatus.ERROR" {SYNTHETIC_API_KEY}',
            )
        if args[1:3] == ["kernels", "output"]:
            output = Path(args[args.index("--path") + 1]) / "diagnostics.log"
            _ = output.write_text("device backend was cpu", encoding="utf-8")
            return completed(args)
        pytest.fail(f"unexpected Kaggle command: {args}")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="status error") as raised:
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
    state_text = (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    log_text = (tmp_path / "output" / "kaggle-controller.log").read_text()
    assert SYNTHETIC_API_KEY not in state_text
    assert SYNTHETIC_API_KEY not in log_text
    assert SYNTHETIC_API_KEY not in str(raised.value)

    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["status"] == "error"
    assert state["outcome"] == "remote_terminal_failure"

    output_commands = [
        command for command in commands if command[1:3] == ["kernels", "output"]
    ]
    assert len(output_commands) == 1
    assert (
        tmp_path / "output" / "diagnostics.log"
    ).read_text() == "device backend was cpu"


def test_execution_deadline_survives_status_flapping_back_to_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "QUEUE_TIMEOUT_SECONDS", 1000)
    monkeypatch.setattr(kaggle_ci, "EXECUTION_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 3)
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])

    def advance_time(duration: float) -> None:
        fake_time[0] += duration

    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        advance_time,
    )
    commands: list[list[str]] = []

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del env, timeout
        commands.append(args)
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"1.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.")
        if args[1:3] == ["kernels", "status"]:
            status = (
                "RUNNING"
                if sum(c[1:3] == ["kernels", "status"] for c in commands) == 1
                else "QUEUED"
            )
            return completed(args, f'has status "{status}"')
        pytest.fail("execution timeout must not retrieve output")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="execution polling timed out"):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["phase"] == "execution"
    assert state["status"] == "queued"
    assert state["outcome"] == "execution_timeout"
    assert state["queue_elapsed_seconds"] == 0
    assert state["execution_elapsed_seconds"] == 10
    assert sum(c[1:3] == ["kernels", "push"] for c in commands) == 1


@pytest.mark.parametrize(
    ("status_stdout", "expected_outcome"),
    [
        ('has status "WAITING"', "unknown_api_status"),
        (f'has status "WAITING-{SYNTHETIC_API_KEY}"', "unknown_api_status"),
    ],
)
def test_unknown_kaggle_status_is_saved_as_unknown_api_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status_stdout: str,
    expected_outcome: str,
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del env, timeout
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"1.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.")
        if args[1:3] == ["kernels", "status"]:
            return completed(args, status_stdout)
        pytest.fail("unknown status must not retrieve output")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError) as raised:
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
    state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    )
    assert state["outcome"] == expected_outcome
    assert state["status"].startswith("waiting")
    if SYNTHETIC_API_KEY in status_stdout:
        state_text = (tmp_path / "output" / "kaggle-controller-state.json").read_text()
        log_text = (tmp_path / "output" / "kaggle-controller.log").read_text()
        stdout = capsys.readouterr().out
        assert SYNTHETIC_API_KEY not in (
            state_text + log_text + str(raised.value) + stdout
        )


def test_status_cli_error_is_distinct_and_redacted_in_state_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    _ = runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del env, timeout
        if args[1:3] == ["quota", "--format"]:
            return completed(args, '[{"resource":"TPU","remaining":"1.00h"}]')
        if args[1:3] == ["kernels", "push"]:
            return completed(args, "Kernel version 1 successfully pushed.")
        if args[1:3] == ["kernels", "status"]:
            raise kaggle_ci.KaggleError(f"CLI failed {SYNTHETIC_API_KEY}")
        pytest.fail("CLI status failure must not retrieve output")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError) as raised:
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
    state_text = (tmp_path / "output" / "kaggle-controller-state.json").read_text()
    log_text = (tmp_path / "output" / "kaggle-controller.log").read_text()
    state = json.loads(state_text)
    assert state["outcome"] == "cli_error"
    assert SYNTHETIC_API_KEY not in state_text + log_text + str(raised.value)


@pytest.mark.parametrize("current_schema", [False, True])
def test_resume_queries_status_once_without_mutating_input_or_collecting_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_schema: bool,
) -> None:
    authenticated(monkeypatch)
    run_tag = hashlib.sha256(
        json.dumps(BINDING, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    kernel_id = f"test-user/jaxr-{run_tag}-0123456789"
    stored: dict[str, object] = {
        "binding": BINDING,
        "kernel_id": kernel_id,
        "submitted_version": 1,
    }
    if current_schema:
        stored.update(
            status="queued",
            phase="queue",
            phase_started_at="2026-10-08T09:00:00+00:00",
            submitted_at="2026-10-08T09:00:00+00:00",
            execution_started_at=None,
            updated_at="2026-10-08T09:00:10+00:00",
            queue_elapsed_seconds=10.0,
            execution_elapsed_seconds=0.0,
            outcome="pending",
            last_error=None,
        )
    resume_state = tmp_path / "original-state.json"
    original_text = json.dumps(stored)
    _ = resume_state.write_text(original_text)
    commands: list[list[str]] = []

    def fake_command(args: list[str], *, env: dict[str, str], timeout: float = 90):
        del timeout
        commands.append(args)
        assert env["KAGGLE_CONFIG_DIR"]
        assert args[1:3] == ["kernels", "status"]
        assert args[3] == kernel_id
        return completed(args, 'has status "KernelWorkerStatus.RUNNING"')

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    collector = tmp_path / "collector"
    report = kaggle_ci.resume_status(BINDING, resume_state, collector)

    assert len(commands) == 1
    assert resume_state.read_text() == original_text
    assert sorted(path.name for path in collector.iterdir()) == [
        "kaggle-resume-report.json",
        "kaggle-resume.log",
    ]
    assert report == json.loads((collector / "kaggle-resume-report.json").read_text())
    assert report["binding"] == BINDING
    assert report["kernel_id"] == kernel_id
    assert report["submitted_version"] == 1
    assert report["remote_status"] == "running"
    assert report["version_verified"] is False
    assert report["success"] is False
    assert report["outcome"] == "status_only_version_unverified"


def test_resume_rejects_binding_mismatch_before_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    resume_state = tmp_path / "original-state.json"
    _ = resume_state.write_text(
        json.dumps(
            {
                "binding": {**BINDING, "run_id": "371234567.2"},
                "kernel_id": "test-user/jaxr-0000000000000000-0123456789",
                "submitted_version": 1,
            }
        )
    )

    def fail_command(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("mismatched binding reached Kaggle")

    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        fail_command,
    )
    with pytest.raises(kaggle_ci.KaggleError, match="binding does not exactly match"):
        _ = kaggle_ci.resume_status(BINDING, resume_state, tmp_path / "collector")


def test_resume_unknown_status_fails_closed_without_gate_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    run_tag = hashlib.sha256(
        json.dumps(BINDING, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    state_path = tmp_path / "state.json"
    _ = state_path.write_text(
        json.dumps(
            {
                "binding": BINDING,
                "kernel_id": f"test-user/jaxr-{run_tag}-0123456789",
                "submitted_version": 1,
            }
        )
    )

    def unknown_status(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        return completed(args, 'has status "WAITING"')

    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        unknown_status,
    )
    with pytest.raises(kaggle_ci.KaggleError, match="unrecognised kernel status"):
        _ = kaggle_ci.resume_status(BINDING, state_path, tmp_path / "collector")
    report = json.loads(
        (tmp_path / "collector" / "kaggle-resume-report.json").read_text()
    )
    assert report["remote_status"] == "unknown"
    assert report["outcome"] == "unknown_api_status"
    assert report["version_verified"] is False
    assert report["success"] is False


def _install_relay_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [0.0]
    fixed_now = datetime(2026, 10, 8, 10, 0, tzinfo=timezone.utc)

    def advance(delay: float) -> None:
        clock[0] += delay

    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(kaggle_ci.time, "sleep", advance)
    monkeypatch.setattr(kaggle_ci, "_utc_now", lambda: fixed_now)
    return clock


def _relay_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "queued",
    outcome: str = "handover",
    queue_elapsed: float = 0,
    execution_elapsed: float = 0,
    execution_started_at: datetime | None = None,
    updated_at: datetime | None = None,
    segment_timeout: int = 10,
    queue_timeout: int = 100,
) -> tuple[Path, list[float], list[list[str]], list[Path]]:
    authenticated(monkeypatch)
    clock = _install_relay_clock(monkeypatch)
    monkeypatch.setattr(kaggle_ci, "RELAY_QUEUE_TIMEOUT_SECONDS", queue_timeout)
    monkeypatch.setattr(kaggle_ci, "RELAY_SEGMENT_TIMEOUT_SECONDS", segment_timeout)
    monkeypatch.setattr(kaggle_ci, "EXECUTION_TIMEOUT_SECONDS", 100)
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 1)
    path = tmp_path / "output" / "kaggle-controller-state.json"
    path.parent.mkdir(exist_ok=True)
    _ = _relay_state(
        path,
        status=status,
        outcome=outcome,
        queue_elapsed=queue_elapsed,
        execution_elapsed=execution_elapsed,
        execution_started_at=execution_started_at,
        updated_at=updated_at,
    )
    calls: list[list[str]] = []
    verify_calls: list[Path] = []

    def record_verification(binding: dict[str, object], file: Path) -> None:
        _record_version_verification(binding, file, verify_calls)

    monkeypatch.setattr(kaggle_ci, "_verify_version_one", record_verification)
    return path, clock, calls, verify_calls


def _continue(path: Path, *, final_segment: bool = False) -> dict[str, object]:
    return kaggle_ci.run(
        BINDING,
        "tpu",
        path.parent,
        relay=True,
        continue_state_path=path,
        final_segment=final_segment,
    )


@pytest.mark.parametrize(
    ("prior", "statuses", "queue", "execution", "start", "expected"),
    [
        ("queued", ("RUNNING", "COMPLETE"), 20, 0, None, (23, 1)),
        (
            "running",
            ("QUEUED", "COMPLETE"),
            12,
            10,
            datetime(2026, 10, 8, 9, 59, 50, tzinfo=timezone.utc),
            (12, 14),
        ),
    ],
)
def test_relay_continuation_reuses_kernel_and_cumulative_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior: str,
    statuses: tuple[str, str],
    queue: float,
    execution: float,
    start: datetime | None,
    expected: tuple[float, float],
) -> None:
    path, clock, calls, verified = _relay_case(
        tmp_path,
        monkeypatch,
        status=prior,
        queue_elapsed=queue,
        execution_elapsed=execution,
        execution_started_at=start,
        updated_at=datetime(2026, 10, 8, 9, 59, 57, tzinfo=timezone.utc),
        segment_timeout=50,
    )
    status_iter = iter(statuses)

    def command(args: list[str], **kwargs: object):
        del kwargs
        calls.append(args)
        if args[1:3] == ["kernels", "status"]:
            return completed(args, f'has status "{next(status_iter)}"')
        if args[1:3] == ["kernels", "output"]:
            output = Path(args[args.index("--path") + 1]) / "result.json"
            _ = output.write_text(json.dumps(success_result()), encoding="utf-8")
            return completed(args)
        pytest.fail("continuation must not quota or submit another kernel")

    monkeypatch.setattr(kaggle_ci, "_command_ok", command)
    old = json.loads(path.read_text())
    assert _continue(path) == success_result()
    state = json.loads(path.read_text())
    assert (state["kernel_id"], state["submitted_at"]) == (
        old["kernel_id"],
        old["submitted_at"],
    )
    assert (
        state["queue_elapsed_seconds"],
        state["execution_elapsed_seconds"],
    ) == expected
    assert state["outcome"] == "success" and clock[0] == 1
    assert len(verified) == 3
    assert not any(
        c[1:3] in (["quota", "--format"], ["kernels", "push"]) for c in calls
    )


def test_relay_job_gap_and_cutoff_statuses_keep_queue_and_execution_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _, calls, _ = _relay_case(
        tmp_path,
        monkeypatch,
        queue_elapsed=8,
        updated_at=datetime(2026, 10, 8, 9, 59, 57, tzinfo=timezone.utc),
        queue_timeout=10,
    )
    monkeypatch.setattr(kaggle_ci, "_command_ok", _no_provider_access)
    with pytest.raises(kaggle_ci.KaggleError, match="queue polling timed out"):
        _ = _continue(path)
    state = json.loads(path.read_text())
    assert not calls and state["queue_elapsed_seconds"] == 11
    assert state["outcome"] == "queue_timeout"

    for status, outcome, phase in (
        ("RUNNING", "handover", "execution"),
        ("WAITING", "unknown_api_status", "queue"),
    ):
        path, clock, _, _ = _relay_case(tmp_path, monkeypatch, segment_timeout=3)

        def at_cutoff(args: list[str], **kwargs: object):
            del kwargs
            clock[0] = 3
            return completed(args, f'has status "{status}"')

        monkeypatch.setattr(kaggle_ci, "_command_ok", at_cutoff)
        if status == "WAITING":
            with pytest.raises(kaggle_ci.KaggleError, match="unknown status"):
                _ = _continue(path)
        else:
            assert _continue(path)["pending"] is True
        state = json.loads(path.read_text())
        assert (state["outcome"], state["phase"]) == (outcome, phase)
        if status == "RUNNING":
            assert state["execution_started_at"] is not None


def test_relay_terminal_status_can_finish_output_collection_after_soft_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, clock, _, verified = _relay_case(tmp_path, monkeypatch)

    def late_output(args: list[str], **kwargs: object):
        del kwargs
        if args[1:3] == ["kernels", "status"]:
            clock[0] = 9
            return completed(args, 'has status "COMPLETE"')
        if args[1:3] == ["kernels", "output"]:
            clock[0] = 12
            output = Path(args[args.index("--path") + 1]) / "result.json"
            _ = output.write_text(json.dumps(success_result()), encoding="utf-8")
            return completed(args)
        pytest.fail("continuation must not submit again")

    def record_verification(binding: dict[str, object], file: Path) -> None:
        _record_version_verification(binding, file, verified)

    monkeypatch.setattr(kaggle_ci, "_verify_version_one", record_verification)
    monkeypatch.setattr(kaggle_ci, "_command_ok", late_output)
    assert _continue(path) == success_result()
    assert clock[0] == 12 and len(verified) == 3
    assert json.loads(path.read_text())["outcome"] == "success"


@pytest.mark.parametrize(
    ("provider_error", "final", "expected"),
    [
        (True, False, "provider error"),
        (False, False, "pending"),
        (False, True, "final segment cannot continue"),
    ],
)
def test_relay_handover_only_accepts_pending_ack_from_nonfinal_provider_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_error: bool,
    final: bool,
    expected: str,
) -> None:
    path, clock, calls, _ = _relay_case(tmp_path, monkeypatch, segment_timeout=3)

    def command(args: list[str], **kwargs: object):
        del kwargs
        calls.append(args)
        clock[0] = 3
        if args[1:3] == ["kernels", "status"] and not provider_error:
            return completed(args, 'has status "QUEUED"')
        raise kaggle_ci.KaggleError("Kaggle CLI command failed: provider error")

    monkeypatch.setattr(kaggle_ci, "_command_ok", command)
    if provider_error or final:
        message = expected
        with pytest.raises(kaggle_ci.KaggleError, match=message):
            _ = _continue(path, final_segment=final)
    else:
        assert _continue(path)["pending"] is True
    state = json.loads(path.read_text())
    assert state["outcome"] == (
        "cli_error" if provider_error else "segment_timeout" if final else "handover"
    )
    assert len(calls) == 1
    assert not (path.parent / "result.json").exists()


@pytest.mark.parametrize(
    "case", ["success", "cross-binding", "schema", "error-outcome"]
)
def test_relay_success_passthrough_and_invalid_state_fail_before_provider_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    path, _, _, _ = _relay_case(
        tmp_path, monkeypatch, outcome="success", status="complete"
    )
    state = json.loads(path.read_text())
    state.update(phase="execution", execution_started_at=state["submitted_at"])
    if case == "cross-binding":
        state["binding"] = {**BINDING, "run_id": "371234567.2"}
    elif case == "schema":
        del state["updated_at"]
    elif case == "error-outcome":
        state["outcome"] = "cli_error"
    _ = path.write_text(json.dumps(state), encoding="utf-8")
    if case in {"success", "error-outcome"}:
        _ = (path.parent / "result.json").write_text(json.dumps(success_result()))
    monkeypatch.setattr(kaggle_ci, "_command_ok", _no_provider_access)
    monkeypatch.setattr(kaggle_ci, "_verify_version_one", _no_provider_access)
    if case == "success":
        assert _continue(path) == success_result()
        return
    expected = {
        "cross-binding": "binding",
        "schema": "state",
        "error-outcome": "not eligible for handover",
    }[case]
    with pytest.raises(kaggle_ci.KaggleError, match=expected):
        _ = _continue(path)


def test_stale_result_with_a_different_frozen_identity_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    result = success_result()
    result["run_id"] = "371234567.2"
    _ = path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(kaggle_ci.KaggleError, match="run_id"):
        _private_helper("validate_downloaded_result")(path, BINDING)


def test_existing_output_result_is_not_treated_as_new_run_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    _ = (tmp_path / "result.json").write_text("{}", encoding="utf-8")

    def fail_existing_output(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("existing results must fail before submission")

    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        fail_existing_output,
    )
    with pytest.raises(kaggle_ci.KaggleError, match="already contains"):
        _ = kaggle_ci.run(BINDING, "tpu", tmp_path)


def test_kernel_slug_stays_valid_with_long_pull_request_and_run_ids() -> None:
    binding = {**BINDING, "pr": 999999999999, "run_id": "37123456789.1"}
    slug = _private_helper("_kernel_slug")(binding)
    assert len(slug) <= 50
    assert re.fullmatch(r"jaxr-[0-9a-f]{16}-[0-9a-f]{10}", slug)


def test_cli_failure_diagnostic_redacts_the_api_token() -> None:
    token = SYNTHETIC_API_KEY
    args = ["kaggle", "quota"]
    environment = {"KAGGLE_API_TOKEN": token}
    with patch.object(
        kaggle_ci,
        "_run_cli",
        return_value=subprocess.CompletedProcess(args, 1, "", f"failed with {token}"),
    ):
        with pytest.raises(kaggle_ci.KaggleError) as error:
            _private_helper("_command_ok")(args, env=environment)
    assert token not in str(error.value)
    assert "[redacted]" in str(error.value)
