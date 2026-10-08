import hashlib
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import patch

import pytest
from tools import kaggle_ci

# Synthetic fixture catalogue: api-key-a (active), joey-private-v3.
SYNTHETIC_API_KEY = "codex_synth_v1_api_key_a"

BINDING = {
    "kind": "release",
    "head_sha": "a" * 40,
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "371234567.1",
}


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


def authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", SYNTHETIC_API_KEY)
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")


def test_quota_requires_unambiguous_tpu_headroom() -> None:
    assert (
        kaggle_ci._read_quota(
            '[{"resource":"GPU","remaining":"10.00h"},'
            '{"resource":"TPU","remaining":"0.75h"}]'
        )
        == 0.75
    )
    for response in (
        "not json",
        "[]",
        '[{"resource":"TPU","remaining":"unknown"}]',
        '[{"resource":"TPU","remaining":"0.74h"}]',
        '[{"resource":"TPU","remaining":"1h"},{"resource":"TPU","remaining":"2h"}]',
    ):
        with pytest.raises(kaggle_ci.KaggleError):
            kaggle_ci._read_quota(response)


def test_status_accepts_kaggle_worker_status_enum_output() -> None:
    assert kaggle_ci._status('has status "KernelWorkerStatus.QUEUED"') == "queued"
    assert kaggle_ci._status('has status "KernelWorkerStatus.COMPLETE"') == "complete"
    with pytest.raises(kaggle_ci.KaggleError, match="status type"):
        kaggle_ci._status('has status "UntrustedStatus.QUEUED"')


def test_missing_credentials_fail_before_any_kaggle_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    def unexpected_command(*args: object, **kwargs: object) -> None:
        pytest.fail("Kaggle command was reached without credentials")

    monkeypatch.setattr(kaggle_ci, "_command_ok", unexpected_command)
    with pytest.raises(kaggle_ci.KaggleError, match="KAGGLE_API_TOKEN"):
        kaggle_ci.run(BINDING, "tpu", tmp_path)
    assert "KAGGLE_API_TOKEN" in (tmp_path / "kaggle-controller.log").read_text()


def test_private_kernel_embeds_only_the_trusted_runner(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "accelerator_runner.py"
    runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    folder = tmp_path / "kernel"
    kaggle_ci._make_kernel(
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


def test_success_uses_a_fresh_private_slug_polls_and_downloads_bound_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        lambda duration: fake_time.__setitem__(0, fake_time[0] + duration),
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
                (output_dir / "kaggle-controller-state.json").write_text("{}")
            output = output_dir / "result.json"
            output.write_text(json.dumps(success_result()), encoding="utf-8")
            return completed(args)
        pytest.fail(f"unexpected Kaggle command: {args}")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    result = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

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
    runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "QUEUE_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 3)
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        lambda duration: fake_time.__setitem__(0, fake_time[0] + duration),
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
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
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
    runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
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
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

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
    runner.write_text(
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
            output.write_text("device backend was cpu", encoding="utf-8")
            return completed(args)
        pytest.fail(f"unexpected Kaggle command: {args}")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="status error") as raised:
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
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
    runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "QUEUE_TIMEOUT_SECONDS", 1000)
    monkeypatch.setattr(kaggle_ci, "EXECUTION_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(kaggle_ci, "POLL_INTERVAL_SECONDS", 3)
    fake_time = [0.0]
    monkeypatch.setattr(kaggle_ci.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(
        kaggle_ci.time,
        "sleep",
        lambda duration: fake_time.__setitem__(0, fake_time[0] + duration),
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
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

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
    runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
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
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
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
    runner.write_text("def run(binding, backend, output_dir):\n    return {}\n")
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
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
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
    resume_state.write_text(original_text)
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
    resume_state.write_text(
        json.dumps(
            {
                "binding": {**BINDING, "run_id": "371234567.2"},
                "kernel_id": "test-user/jaxr-0000000000000000-0123456789",
                "submitted_version": 1,
            }
        )
    )
    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        lambda *args, **kwargs: pytest.fail("mismatched binding reached Kaggle"),
    )
    with pytest.raises(kaggle_ci.KaggleError, match="binding does not exactly match"):
        kaggle_ci.resume_status(BINDING, resume_state, tmp_path / "collector")


def test_resume_unknown_status_fails_closed_without_gate_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    run_tag = hashlib.sha256(
        json.dumps(BINDING, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "binding": BINDING,
                "kernel_id": f"test-user/jaxr-{run_tag}-0123456789",
                "submitted_version": 1,
            }
        )
    )
    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        lambda args, **kwargs: completed(args, 'has status "WAITING"'),
    )
    with pytest.raises(kaggle_ci.KaggleError, match="unrecognised kernel status"):
        kaggle_ci.resume_status(BINDING, state_path, tmp_path / "collector")
    report = json.loads(
        (tmp_path / "collector" / "kaggle-resume-report.json").read_text()
    )
    assert report["remote_status"] == "unknown"
    assert report["outcome"] == "unknown_api_status"
    assert report["version_verified"] is False
    assert report["success"] is False


def test_stale_result_with_a_different_frozen_identity_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    result = success_result()
    result["run_id"] = "371234567.2"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(kaggle_ci.KaggleError, match="run_id"):
        kaggle_ci._validate_downloaded_result(path, BINDING)


def test_existing_output_result_is_not_treated_as_new_run_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        kaggle_ci,
        "_command_ok",
        lambda *args, **kwargs: pytest.fail(
            "existing results must fail before submission"
        ),
    )
    with pytest.raises(kaggle_ci.KaggleError, match="already contains"):
        kaggle_ci.run(BINDING, "tpu", tmp_path)


def test_kernel_slug_stays_valid_with_long_pull_request_and_run_ids() -> None:
    binding = {**BINDING, "pr": 999999999999, "run_id": "37123456789.1"}
    slug = kaggle_ci._kernel_slug(binding)
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
            kaggle_ci._command_ok(args, env=environment)
    assert token not in str(error.value)
    assert "[redacted]" in str(error.value)
