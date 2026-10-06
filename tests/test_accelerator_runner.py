from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest
from tools import accelerator_runner as runner

# Synthetic fixture catalogue: access-a (active), joey-private-v3.
SYNTHETIC_ACCESS = "codex_synth_v1_access_a"
# Synthetic fixture catalogue: api-key-a (active), joey-private-v3.
SYNTHETIC_KAGGLE_API_KEY = "codex_synth_v1_api_key_a"

GOOD_BINDING = {
    "pr": 24,
    "head_sha": "a" * 40,
    "base_sha": "b" * 40,
    "base_ref": "master",
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "371234567.1",
}
RELEASE_BINDING = {
    "kind": "release",
    "head_sha": "c" * 40,
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "371234568.1",
}


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_archive_url_is_derived_from_frozen_public_head() -> None:
    assert runner.source_url_for_binding(GOOD_BINDING) == (
        f"https://github.com/JoeyTeng/jaxrenderer/archive/{'a' * 40}.tar.gz"
    )
    invalid = {**GOOD_BINDING, "head_repository": "github.com/attacker/repo?token=x"}
    with pytest.raises(runner.RunnerError, match="owner/repository"):
        runner.source_url_for_binding(invalid)
    assert runner.source_url_for_binding(RELEASE_BINDING) == (
        f"https://github.com/JoeyTeng/jaxrenderer/archive/{'c' * 40}.tar.gz"
    )


def test_release_binding_and_manifest_have_no_pr_placeholders() -> None:
    validated = runner._validate_binding(RELEASE_BINDING)
    manifest = runner._initial_result(validated, "tpu")
    assert manifest["kind"] == "release"
    assert manifest["head_sha"] == RELEASE_BINDING["head_sha"]
    assert manifest["head_repository"] == "JoeyTeng/jaxrenderer"
    assert "pr" not in manifest
    assert "base_sha" not in manifest
    for invalid in (
        {**RELEASE_BINDING, "head_sha": "short"},
        {**RELEASE_BINDING, "head_repository": "attacker/project"},
        {**RELEASE_BINDING, "pr": 24},
        {**RELEASE_BINDING, "kind": "pr"},
    ):
        with pytest.raises(runner.RunnerError):
            runner._validate_binding(invalid)


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = _archive({"project/../escape.txt": b"outside"})
    with pytest.raises(runner.RunnerError, match="unsafe path"):
        runner._extract_source_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


def test_minimal_environment_keeps_accelerator_runtime_and_drops_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/driver/lib")
    monkeypatch.setenv("GITHUB_TOKEN", SYNTHETIC_ACCESS)
    monkeypatch.setenv("KAGGLE_API_TOKEN", SYNTHETIC_KAGGLE_API_KEY)
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    environment = runner._minimal_environment(
        tmp_path / "home", tmp_path / "venv", "tpu", tmp_path / "artifacts"
    )
    assert environment["TPU_WORKER_ID"] == "0"
    assert environment["LD_LIBRARY_PATH"] == "/driver/lib"
    assert environment["JAX_PLATFORMS"] == "tpu"
    assert environment["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
    assert "GITHUB_TOKEN" not in environment
    assert "KAGGLE_API_TOKEN" not in environment
    assert "KAGGLE_USERNAME" not in environment
    assert "/opt/conda/bin" in environment["PATH"].split(os.pathsep)


def test_gpu_platform_selects_cuda_without_rocm_alias_expansion(tmp_path: Path) -> None:
    environment = runner._minimal_environment(
        tmp_path / "home", tmp_path / "venv", "gpu", tmp_path / "artifacts"
    )
    assert environment["JAX_PLATFORMS"] == "cuda"
    assert environment["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"


def test_output_directory_replacement_fails_identity_revalidation(
    tmp_path: Path,
) -> None:
    output, identity = runner._prepare_output(tmp_path / "out")
    moved = tmp_path / "old-out"
    output.rename(moved)
    output.mkdir()
    with pytest.raises(runner.RunnerError, match="identity or ownership changed"):
        runner._check_output_identity(output, identity)


def test_failed_archive_fetch_writes_failure_manifest_and_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_download(_: str) -> bytes:
        raise runner.RunnerError("archive unavailable")

    monkeypatch.setattr(runner, "_download_archive", fail_download)
    output = tmp_path / "output"
    result = runner.run(GOOD_BINDING, "gpu", output)
    manifest = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result == manifest
    assert result["success"] is False
    assert result["device_backend"] == "unknown"
    assert result["device_count"] == 0
    assert result["pr"] == 24
    assert "archive unavailable" in (output / "diagnostics.log").read_text(
        encoding="utf-8"
    )


def test_success_runs_both_suites_with_locked_jax_and_real_backend_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".venv" / "bin").mkdir(parents=True)
    (source_root / ".venv" / "bin" / "python").touch()
    baseline = {
        "jax": "0.11.2",
        "jaxlib": "0.11.2",
        "numpy": "2.1.3",
        "pytest": "9.1.1",
    }
    installed = {
        **baseline,
        "jax-cuda12-plugin": "0.11.2",
        "jax-cuda12-pjrt": "0.11.2",
    }
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(runner, "_download_archive", lambda _: b"frozen-archive")
    monkeypatch.setattr(
        runner, "_extract_source_archive", lambda _data, _dest: source_root
    )
    monkeypatch.setattr(runner.shutil, "which", lambda _name, path=None: "/trusted/uv")
    monkeypatch.setattr(runner, "_locked_jax_version", lambda _path: "0.11.2")
    monkeypatch.setattr(runner, "_uv_version", lambda *_: "uv 0.12.20")
    monkeypatch.setattr(runner, "_python_version", lambda *_: "3.14.1")

    version_calls = 0

    def installed_versions(*_: object) -> dict[str, str]:
        nonlocal version_calls
        version_calls += 1
        return baseline if version_calls == 1 else installed

    monkeypatch.setattr(runner, "_installed_versions", installed_versions)

    def probe(
        _python: Path,
        backend: str,
        _cwd: Path,
        environment: dict[str, str],
        _log: Path,
    ) -> dict[str, object]:
        assert backend == "gpu"
        assert environment["JAX_PLATFORMS"] == "cuda"
        assert environment["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
        return {
            "device_backend": "gpu",
            "devices": [{"platform": "gpu", "device_kind": "T4", "id": "0"}],
        }

    monkeypatch.setattr(runner, "_probe_accelerator", probe)

    def command(
        argv: list[str],
        _cwd: Path,
        environment: dict[str, str],
        log_path: Path,
    ) -> str:
        calls.append((argv, environment))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("completed\n", encoding="utf-8")
        return "completed\n"

    monkeypatch.setattr(runner, "_run_command", command)
    result = runner.run(GOOD_BINDING, "gpu", tmp_path / "output")

    assert result["success"] is True
    assert result["device_count"] == 1
    assert result["versions"]["python"] == "3.14.1"
    assert result["versions"]["jax-cuda12-plugin"] == "0.11.2"
    assert any("jax[cuda12]==0.11.2" in argv for argv, _ in calls)
    assert any("tests/" in argv and "--import-mode" in argv for argv, _ in calls)
    assert any("tests/render_regression.py" in argv for argv, _ in calls)
    assert all("GITHUB_TOKEN" not in environment for _, environment in calls)


def test_release_tpu_run_uses_release_identity_and_locked_tpu_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", SYNTHETIC_ACCESS)
    monkeypatch.setenv("GH_TOKEN", SYNTHETIC_ACCESS)
    monkeypatch.setenv("KAGGLE_API_TOKEN", SYNTHETIC_KAGGLE_API_KEY)
    monkeypatch.setenv("KAGGLE_USERNAME", "test-user")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / ".venv" / "bin").mkdir(parents=True)
    (source_root / ".venv" / "bin" / "python").touch()
    locked = {"jax": "0.11.2", "jaxlib": "0.11.2", "numpy": "2.5.3"}
    installed = {**locked, "libtpu": "0.0.48"}
    calls: list[tuple[list[str], dict[str, str]]] = []
    monkeypatch.setattr(runner, "_download_archive", lambda _: b"frozen-archive")
    monkeypatch.setattr(
        runner, "_extract_source_archive", lambda _data, _dest: source_root
    )

    def find_conda_uv(name: str, path: str | None = None) -> str | None:
        assert name == "uv"
        if path and "/opt/conda/bin" in path.split(os.pathsep):
            return "/opt/conda/bin/uv"
        return None

    monkeypatch.setattr(runner.shutil, "which", find_conda_uv)
    monkeypatch.setattr(runner, "_locked_jax_version", lambda _path: "0.11.2")
    monkeypatch.setattr(runner, "_uv_version", lambda uv, *_: f"{uv} 0.12.20")
    monkeypatch.setattr(runner, "_python_version", lambda *_: "3.13.15")
    version_calls = 0

    def versions(*_: object) -> dict[str, str]:
        nonlocal version_calls
        version_calls += 1
        return locked if version_calls == 1 else installed

    monkeypatch.setattr(runner, "_installed_versions", versions)

    def probe(
        _python: Path,
        backend: str,
        _cwd: Path,
        environment: dict[str, str],
        _log: Path,
    ) -> dict[str, object]:
        assert backend == "tpu"
        assert environment["JAX_PLATFORMS"] == "tpu"
        return {
            "device_backend": "tpu",
            "devices": [{"platform": "tpu", "device_kind": "TPU v5e-8", "id": "0"}],
        }

    monkeypatch.setattr(runner, "_probe_accelerator", probe)

    def command(
        argv: list[str], _cwd: Path, environment: dict[str, str], log: Path
    ) -> str:
        calls.append((argv, environment))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("completed\n", encoding="utf-8")
        return "completed\n"

    monkeypatch.setattr(runner, "_run_command", command)
    manifest = runner.run(RELEASE_BINDING, "tpu", tmp_path / "output")
    assert manifest["success"] is True
    assert manifest["kind"] == "release"
    assert manifest["head_sha"] == RELEASE_BINDING["head_sha"]
    assert "pr" not in manifest and "base_sha" not in manifest
    assert manifest["versions"]["python"] == "3.13.15"
    assert manifest["versions"]["uv"] == "/opt/conda/bin/uv 0.12.20"
    assert manifest["versions"]["libtpu"] == "0.0.48"
    assert any("jax[tpu]==0.11.2" in argv for argv, _ in calls)
    uv_calls = [argv for argv, _ in calls if "uv" in Path(argv[0]).name]
    assert len(uv_calls) == 3
    assert all(argv[0] == "/opt/conda/bin/uv" for argv in uv_calls)
    assert all(
        not {"GITHUB_TOKEN", "GH_TOKEN", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME"}
        & environment.keys()
        for _, environment in calls
    )


def test_command_stops_when_log_budget_is_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "MAX_LOG_BYTES", 64)
    log_path = tmp_path / "noisy.log"
    with pytest.raises(runner.RunnerError, match="log budget"):
        runner._run_command(
            (sys.executable, "-c", "print('x' * 1000)"),
            tmp_path,
            {"PATH": os.environ["PATH"]},
            log_path,
        )
    assert log_path.stat().st_size == 64


def test_runner_requires_frozen_binding_and_backend() -> None:
    with pytest.raises(runner.RunnerError, match="exact frozen PR or release"):
        runner._validate_binding({**GOOD_BINDING, "source_url": "https://example.com"})
    with pytest.raises(runner.RunnerError, match="run_id"):
        runner._validate_binding({**GOOD_BINDING, "run_id": "0.1"})


@pytest.mark.parametrize(
    ("binding", "backend"), [(GOOD_BINDING, "tpu"), (RELEASE_BINDING, "gpu")]
)
def test_runner_rejects_backend_binding_type_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: dict[str, object],
    backend: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "_download_archive",
        lambda _: pytest.fail("backend mismatch reached remote source download"),
    )
    with pytest.raises(runner.RunnerError, match="only backend allowed"):
        runner.run(binding, backend, tmp_path / "output")
