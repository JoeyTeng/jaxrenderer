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
            '{"resource":"TPU","remaining":"0.50h"}]'
        )
        == 0.5
    )
    for response in (
        "not json",
        "[]",
        '[{"resource":"TPU","remaining":"unknown"}]',
        '[{"resource":"TPU","remaining":"0.49h"}]',
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
    commands: list[list[str]] = []
    status_responses = iter(
        (
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
                kaggle_ci.KAGGLE_TIMEOUT_SECONDS
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
    monkeypatch.setattr(kaggle_ci.time, "sleep", lambda _: None)
    result = kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

    assert result == success_result()
    controller_state = json.loads(
        (tmp_path / "output" / "kaggle-controller-state.json").read_text(
            encoding="utf-8"
        )
    )
    assert controller_state["submitted_version"] == 1
    assert controller_state["binding"] == BINDING
    pushes = [command for command in commands if command[1:3] == ["kernels", "push"]]
    assert len(pushes) == 1
    # The private slug includes a random suffix so output from an older run cannot match.
    status_commands = [
        command for command in commands if command[1:3] == ["kernels", "status"]
    ]
    assert len(status_commands) == 2
    assert status_commands[0][-1] == status_commands[1][-1]
    assert controller_state["kernel_id"] == status_commands[0][-1]
    assert re.fullmatch(
        r"test-user/jaxr-[0-9a-f]{16}-[0-9a-f]{10}", status_commands[0][-1]
    )


def test_poll_timeout_fails_without_downloading_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    runner = tmp_path / "accelerator_runner.py"
    runner.write_text(
        "def run(binding, backend, output_dir):\n    return {}\n", encoding="utf-8"
    )
    monkeypatch.setattr(kaggle_ci, "__file__", str(tmp_path / "kaggle_ci.py"))
    monkeypatch.setattr(kaggle_ci, "KAGGLE_TIMEOUT_SECONDS", 0.01)
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
            return completed(args, 'has status "RUNNING"')
        pytest.fail("a timed-out kernel must not download results")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="polling timed out"):
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")
    assert sum(command[1:3] == ["kernels", "push"] for command in commands) == 1


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
            return completed(args, 'has status "KernelWorkerStatus.ERROR"')
        if args[1:3] == ["kernels", "output"]:
            output = Path(args[args.index("--path") + 1]) / "diagnostics.log"
            output.write_text("device backend was cpu", encoding="utf-8")
            return completed(args)
        pytest.fail(f"unexpected Kaggle command: {args}")

    monkeypatch.setattr(kaggle_ci, "_command_ok", fake_command)
    with pytest.raises(kaggle_ci.KaggleError, match="status error"):
        kaggle_ci.run(BINDING, "tpu", tmp_path / "output")

    output_commands = [
        command for command in commands if command[1:3] == ["kernels", "output"]
    ]
    assert len(output_commands) == 1
    assert (
        tmp_path / "output" / "diagnostics.log"
    ).read_text() == "device backend was cpu"


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
