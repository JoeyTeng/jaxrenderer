import json
from pathlib import Path
import subprocess

import pytest
from tools import kaggle_recovery

# Synthetic fixture catalogue: api-key-a (active), joey-private-v3.
SYNTHETIC_API_KEY = "codex_synth_v1_api_key_a"


def frozen_binding() -> dict[str, object]:
    return dict(kaggle_recovery.EXPECTED_BINDING)


def write_binding(path: Path, value: dict[str, object] | None = None) -> Path:
    path.write_text(json.dumps(value or frozen_binding()), encoding="utf-8")
    return path


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, 0, stdout, "")


def candidate(username: str = "joeyteng") -> str:
    return f"{username}/{kaggle_recovery._prefix(frozen_binding())}0123456789"


def success_result() -> dict[str, object]:
    binding = frozen_binding()
    return {
        "pr": binding["pr"],
        "head_sha": binding["head_sha"],
        "base_sha": binding["base_sha"],
        "run_id": binding["run_id"],
        "backend": "tpu",
        "device_backend": "tpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
        "versions": {"jax": "0.11.2"},
    }


def authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", SYNTHETIC_API_KEY)
    monkeypatch.setenv("KAGGLE_USERNAME", "joeyteng")


def test_recovery_refuses_a_different_frozen_binding_before_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding = {**frozen_binding(), "run_id": "37278198939.2"}
    path = write_binding(tmp_path / "binding.json", binding)
    monkeypatch.setattr(
        kaggle_recovery,
        "_command_ok",
        lambda *args, **kwargs: pytest.fail("wrong binding must not query Kaggle"),
    )
    with pytest.raises(kaggle_recovery.KaggleError, match="frozen PR 25"):
        kaggle_recovery.recover(path, tmp_path / "output")


def test_recovery_uses_narrow_search_and_leaves_pending_kernel_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    binding_path = write_binding(tmp_path / "binding.json")
    calls: list[list[str]] = []

    def fake_command(
        args: list[str], *, env: dict[str, str], timeout: float = 90
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        calls.append(args)
        if args[1:3] == ["kernels", "list"]:
            assert args[args.index("--mine")]
            assert args[args.index("--search") + 1] == kaggle_recovery._prefix(
                frozen_binding()
            )
            return completed(args, json.dumps([{"ref": candidate()}]))
        if args[1:3] == ["kernels", "status"]:
            return completed(args, 'has status "KernelWorkerStatus.QUEUED"')
        pytest.fail("pending recovery must not download or submit a kernel")

    monkeypatch.setattr(kaggle_recovery, "_command_ok", fake_command)
    with pytest.raises(kaggle_recovery.RecoveryPending, match="is queued"):
        kaggle_recovery.recover(binding_path, tmp_path / "output")

    report = json.loads((tmp_path / "output" / "recovery-state.json").read_text())
    assert report == {
        "binding": frozen_binding(),
        "kernel_id": candidate(),
        "status": "queued",
    }
    assert [call[1:3] for call in calls] == [["kernels", "list"], ["kernels", "status"]]


def test_recovery_downloads_and_validates_only_unique_terminal_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    binding_path = write_binding(tmp_path / "binding.json")
    calls: list[list[str]] = []

    def fake_command(
        args: list[str], *, env: dict[str, str], timeout: float = 90
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        calls.append(args)
        if args[1:3] == ["kernels", "list"]:
            return completed(
                args,
                json.dumps(
                    [
                        {"ref": "someone/other-notebook"},
                        {"ref": candidate()},
                    ]
                ),
            )
        if args[1:3] == ["kernels", "status"]:
            assert args[-1] == candidate()
            return completed(args, 'has status "KernelWorkerStatus.COMPLETE"')
        if args[1:3] == ["kernels", "output"]:
            assert args[3] == f"{candidate()}/1"
            assert (
                args[args.index("--file-pattern") + 1] == kaggle_recovery.OUTPUT_PATTERN
            )
            output = Path(args[args.index("--path") + 1])
            (output / "result.json").write_text(json.dumps(success_result()))
            (output / "render-artifacts").mkdir()
            (output / "render-artifacts" / "numeric-report.json").write_text("{}")
            return completed(args)
        pytest.fail("recovery must never create or submit a kernel")

    monkeypatch.setattr(kaggle_recovery, "_command_ok", fake_command)
    report = kaggle_recovery.recover(binding_path, tmp_path / "output")

    assert report["result_validated"] is True
    assert (tmp_path / "output" / "result.json").is_file()
    assert (tmp_path / "output" / "render-artifacts" / "numeric-report.json").is_file()
    assert all(call[1:3] != ["kernels", "push"] for call in calls)


def test_multiple_matching_private_kernels_fail_closed_before_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authenticated(monkeypatch)
    binding_path = write_binding(tmp_path / "binding.json")

    def fake_command(
        args: list[str], *, env: dict[str, str], timeout: float = 90
    ) -> subprocess.CompletedProcess[str]:
        del env, timeout
        if args[1:3] == ["kernels", "list"]:
            return completed(
                args,
                json.dumps(
                    [{"ref": candidate()}, {"ref": candidate()[:-10] + "fedcba9876"}]
                ),
            )
        pytest.fail("ambiguous candidates must not be queried or downloaded")

    monkeypatch.setattr(kaggle_recovery, "_command_ok", fake_command)
    with pytest.raises(kaggle_recovery.KaggleError, match="found 2"):
        kaggle_recovery.recover(binding_path, tmp_path / "output")
