"""Run the locked test suites on an explicitly selected accelerator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import io
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_ID_RE = re.compile(r"^[1-9][0-9]*\.[1-9][0-9]*$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
COMMAND_TIMEOUT_SECONDS = 30 * 60
MAX_LOG_BYTES = 16 * 1024 * 1024
LOG_TAIL_BYTES = 64 * 1024


class RunnerError(RuntimeError):
    """A closed-gate validation or accelerator execution error."""


def _validate_binding(binding: Mapping[str, object]) -> dict[str, object]:
    required = {
        "pr",
        "head_sha",
        "base_sha",
        "base_ref",
        "head_repository",
        "run_id",
    }
    if set(binding) != required:
        raise RunnerError("binding must contain exactly the six frozen PR fields")
    if type(binding["pr"]) is not int or binding["pr"] <= 0:
        raise RunnerError("binding PR number must be a positive integer")
    for name in ("head_sha", "base_sha"):
        value = binding[name]
        if not isinstance(value, str) or not SHA_RE.fullmatch(value):
            raise RunnerError(f"binding {name} must be a full lowercase commit SHA")
    if not isinstance(binding["base_ref"], str) or not binding["base_ref"]:
        raise RunnerError("binding base_ref must be a non-empty string")
    repository = binding["head_repository"]
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise RunnerError("binding head_repository must be owner/repository")
    run_id = binding["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise RunnerError("binding run_id must be <positive-run-id>.<positive-attempt>")
    return dict(binding)


def source_url_for_binding(binding: Mapping[str, object]) -> str:
    """Return the only accepted unauthenticated archive URL for the frozen head."""
    value = _validate_binding(binding)
    return (
        f"https://github.com/{value['head_repository']}/archive/"
        f"{value['head_sha']}.tar.gz"
    )


def _check_output_identity(path: Path, identity: tuple[int, int, int]) -> None:
    try:
        current = path.lstat()
    except OSError as error:
        raise RunnerError(
            f"output directory could not be revalidated: {error}"
        ) from error
    expected_device, expected_inode, expected_uid = identity
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino, current.st_uid)
        != (expected_device, expected_inode, expected_uid)
        or current.st_uid != os.geteuid()
    ):
        raise RunnerError(
            "output directory identity or ownership changed during the run"
        )


def _prepare_output(path: Path) -> tuple[Path, tuple[int, int, int]]:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RunnerError("output directory must not be a symbolic link")
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise RunnerError("output directory must be a directory owned by this process")
    identity = (info.st_dev, info.st_ino, info.st_uid)
    _check_output_identity(resolved, identity)
    return resolved, identity


def _download_archive(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "jaxrenderer-ci",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise RunnerError("GitHub source archive exceeds the 100 MiB limit")
    return data


def _extract_source_archive(data: bytes, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise RunnerError(
            f"GitHub source archive is not a valid gzip tar: {error}"
        ) from error

    with archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise RunnerError("GitHub source archive contains too many members")
        total_size = 0
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RunnerError("GitHub source archive contains an unsafe path")
            if not (member.isdir() or member.isfile()):
                raise RunnerError(
                    "GitHub source archive contains a link or special file"
                )
            total_size += member.size
            if total_size > MAX_EXTRACTED_BYTES:
                raise RunnerError(
                    "GitHub source archive exceeds the 500 MiB expanded limit"
                )
        try:
            archive.extractall(destination, members=members, filter="data")
        except (tarfile.TarError, OSError) as error:
            raise RunnerError(
                f"could not extract GitHub source archive: {error}"
            ) from error

    roots = [
        child
        for child in destination.iterdir()
        if child.is_dir()
        and (child / "pyproject.toml").is_file()
        and (child / "uv.lock").is_file()
    ]
    if len(roots) != 1:
        raise RunnerError(
            "source archive must contain one project with pyproject.toml and uv.lock"
        )
    return roots[0]


def _minimal_environment(
    home: Path, venv: Path, backend: str, artifacts: Path
) -> dict[str, str]:
    jax_platform = "cuda" if backend == "gpu" else backend
    environment = {
        "HOME": str(home),
        "PATH": os.pathsep.join(
            (str(venv / "bin"), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "JAX_PLATFORMS": jax_platform,
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "MPLBACKEND": "Agg",
        "JAXRENDERER_ARTIFACT_DIR": str(artifacts),
        "UV_CACHE_DIR": str(home / ".cache" / "uv"),
    }
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    for name, value in os.environ.items():
        if name.startswith(("TPU_", "CLOUD_TPU_", "COLAB_TPU_", "NVIDIA_")) or name in {
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "LIBTPU_INIT_ARGS",
            "PJRT_DEVICE",
            "XLA_FLAGS",
            "XRT_TPU_CONFIG",
        }:
            environment[name] = value
    return environment


def _run_command(
    command: Sequence[str], cwd: Path, environment: Mapping[str, str], log_path: Path
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    current_size = log_path.stat().st_size if log_path.exists() else 0
    remaining = MAX_LOG_BYTES - current_size
    if remaining <= 0:
        raise RunnerError(f"log output budget exhausted; see {log_path.name}")
    process: subprocess.Popen[bytes] | None = None
    return_code: int | None = None
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        assert process.stdout is not None
        deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
        written = 0
        with selectors.DefaultSelector() as selector, log_path.open("ab") as log:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                time_left = deadline - time.monotonic()
                if time_left <= 0:
                    raise RunnerError(f"command timed out; see {log_path.name}")
                events = selector.select(min(time_left, 0.25))
                for key, _ in events:
                    chunk = os.read(key.fd, min(64 * 1024, remaining - written + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = remaining - written
                    log.write(chunk[:available])
                    written += min(len(chunk), available)
                    if len(chunk) > available:
                        raise RunnerError(
                            f"command exceeded the {MAX_LOG_BYTES}-byte log budget; "
                            f"see {log_path.name}"
                        )
            return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except RunnerError:
        if process is not None:
            _terminate_process(process)
        raise
    except subprocess.TimeoutExpired as error:
        if process is not None:
            _terminate_process(process)
        raise RunnerError(f"command timed out; see {log_path.name}") from error
    except OSError as error:
        if process is not None:
            _terminate_process(process)
        raise RunnerError(f"could not start command {command[0]!r}: {error}") from error
    if process is not None and process.stdout is not None:
        process.stdout.close()
    if return_code is None:
        raise RunnerError("command did not return an exit status")
    if return_code:
        raise RunnerError(
            f"command failed with exit {return_code}; see {log_path.name}"
        )
    with log_path.open("rb") as log:
        log.seek(max(0, log_path.stat().st_size - LOG_TAIL_BYTES))
        return log.read(LOG_TAIL_BYTES).decode("utf-8", errors="replace")


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _locked_jax_version(lock_path: Path) -> str:
    import tomllib

    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise RunnerError(f"could not read uv.lock: {error}") from error
    versions = {
        package.get("version")
        for package in lock.get("package", [])
        if package.get("name") == "jax" and isinstance(package.get("version"), str)
    }
    if len(versions) != 1:
        raise RunnerError(
            "uv.lock must resolve exactly one JAX version for the accelerator runner"
        )
    return versions.pop()


def _installed_versions(
    python: Path, cwd: Path, environment: Mapping[str, str], log: Path
) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json; "
        "print(json.dumps({d.metadata['Name'].lower(): d.version "
        "for d in m.distributions() if d.metadata.get('Name')}))"
    )
    output = _run_command((str(python), "-c", code), cwd, environment, log)
    try:
        value = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RunnerError("could not read installed package versions") from error
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(version, str)
        for name, version in value.items()
    ):
        raise RunnerError("installed package version report is malformed")
    return value


def _probe_accelerator(
    python: Path,
    expected_backend: str,
    cwd: Path,
    environment: Mapping[str, str],
    log: Path,
) -> dict[str, object]:
    code = "\n".join(
        (
            "import json",
            "import jax",
            "import jax.numpy as jnp",
            "devices = jax.devices()",
            "assert devices, 'JAX reported no devices'",
            "x = jax.device_put(jnp.arange(4, dtype=jnp.float32).reshape(2, 2), devices[0])",
            "y = jax.jit(lambda a: a @ a)(x)",
            "y.block_until_ready()",
            "backend = jax.default_backend()",
            "assert backend == EXPECTED, f'expected {EXPECTED}, got {backend}'",
            "assert y.device.platform == EXPECTED, f'computation ran on {y.device.platform}'",
            "items = [{'platform': d.platform, 'device_kind': str(d.device_kind), 'id': str(d.id)} for d in devices]",
            "assert all(item['platform'] == EXPECTED for item in items)",
            "print(json.dumps({'device_backend': backend, 'devices': items}))",
        )
    ).replace("EXPECTED", repr(expected_backend))
    output = _run_command((str(python), "-c", code), cwd, environment, log)
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RunnerError("accelerator probe returned malformed JSON") from error
    if (
        not isinstance(report, dict)
        or report.get("device_backend") != expected_backend
        or not isinstance(report.get("devices"), list)
        or not report["devices"]
    ):
        raise RunnerError("accelerator probe did not report the requested backend")
    for device in report["devices"]:
        if (
            not isinstance(device, dict)
            or set(device) != {"platform", "device_kind", "id"}
            or device["platform"] != expected_backend
            or not all(isinstance(device[key], str) and device[key] for key in device)
        ):
            raise RunnerError("accelerator probe returned malformed device metadata")
    return report


def _initial_result(binding: Mapping[str, object], backend: str) -> dict[str, object]:
    return {
        "pr": binding["pr"],
        "head_sha": binding["head_sha"],
        "base_sha": binding["base_sha"],
        "run_id": binding["run_id"],
        "backend": backend,
        "device_backend": "unknown",
        "device_count": 0,
        "success": False,
        "devices": [],
        "versions": {},
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(
    binding: Mapping[str, object], backend: str, output_dir: Path
) -> dict[str, object]:
    """Run the locked suite on a verified GPU or TPU and write its manifest."""
    frozen = _validate_binding(binding)
    if backend not in {"gpu", "tpu"}:
        raise RunnerError("backend must be gpu or tpu")
    output, output_identity = _prepare_output(Path(output_dir))
    result = _initial_result(frozen, backend)
    diagnostics: list[str] = []
    diagnostics_path = output / "diagnostics.log"
    try:
        source_url = source_url_for_binding(frozen)
        jax_version: str | None = None
        with tempfile.TemporaryDirectory(
            prefix="jaxrenderer-accelerator-"
        ) as temporary:
            temporary_root = Path(temporary)
            home = temporary_root / "home"
            home.mkdir()
            work = temporary_root / "work"
            source_root = _extract_source_archive(_download_archive(source_url), work)
            python_version = "3.14" if backend == "gpu" else "3.13"
            venv = source_root / ".venv"
            artifacts = output / "render-artifacts"
            artifacts.mkdir(exist_ok=True)
            environment = _minimal_environment(home, venv, backend, artifacts)
            uv = shutil.which("uv", path=environment["PATH"])
            if uv is None:
                raise RunnerError(
                    "uv executable is required in the trusted accelerator image"
                )
            setup_log = output / "setup.log"
            _run_command(
                (uv, "python", "install", python_version),
                source_root,
                environment,
                setup_log,
            )
            _run_command(
                (
                    uv,
                    "sync",
                    "--locked",
                    "--group",
                    "dev",
                    "--group",
                    "test",
                    "--no-install-project",
                    "--python",
                    python_version,
                ),
                source_root,
                environment,
                setup_log,
            )
            venv_python = venv / "bin" / "python"
            if not venv_python.is_file():
                raise RunnerError(
                    "uv sync did not create the expected virtual environment"
                )
            baseline_versions = _installed_versions(
                venv_python, source_root, environment, setup_log
            )
            jax_version = _locked_jax_version(source_root / "uv.lock")
            result["versions"] = {
                "python": _python_version(
                    venv_python, source_root, environment, setup_log
                ),
                "uv": _uv_version(uv, source_root, environment, setup_log),
                "jax": jax_version,
                "jaxlib": baseline_versions.get("jaxlib", "unknown"),
                "numpy": baseline_versions.get("numpy", "unknown"),
            }
            extra = "cuda12" if backend == "gpu" else "tpu"
            _run_command(
                (
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    f"jax[{extra}]=={jax_version}",
                ),
                source_root,
                environment,
                setup_log,
            )
            accelerator_versions = _installed_versions(
                venv_python, source_root, environment, setup_log
            )
            changed = {
                name: (version, accelerator_versions.get(name))
                for name, version in baseline_versions.items()
                if accelerator_versions.get(name) != version
            }
            if changed:
                raise RunnerError(
                    "accelerator extra changed locked packages: "
                    + ", ".join(
                        f"{name} {before} -> {after}"
                        for name, (before, after) in sorted(changed.items())
                    )
                )
            result["versions"] = {
                "python": _python_version(
                    venv_python, source_root, environment, setup_log
                ),
                "uv": _uv_version(uv, source_root, environment, setup_log),
                "jax": accelerator_versions.get("jax", "unknown"),
                "jaxlib": accelerator_versions.get("jaxlib", "unknown"),
                "numpy": accelerator_versions.get("numpy", "unknown"),
                **{
                    name: accelerator_versions[name]
                    for name in (
                        "jax-cuda12-plugin",
                        "jax-cuda12-pjrt",
                        "libtpu",
                    )
                    if name in accelerator_versions
                },
            }
            if result["versions"]["jax"] != jax_version:
                raise RunnerError("accelerator install changed the locked JAX version")
            if result["versions"]["jaxlib"] != baseline_versions.get("jaxlib"):
                raise RunnerError(
                    "accelerator install changed the locked jaxlib version"
                )
            if result["versions"]["numpy"] != baseline_versions.get("numpy"):
                raise RunnerError(
                    "accelerator install changed the locked NumPy version"
                )

            probe = _probe_accelerator(
                venv_python,
                backend,
                source_root,
                environment,
                output / "device-probe.log",
            )
            result["device_backend"] = probe["device_backend"]
            result["devices"] = probe["devices"]
            result["device_count"] = len(probe["devices"])

            _run_command(
                (
                    str(venv_python),
                    "-m",
                    "pytest",
                    "tests/",
                    "--import-mode",
                    "importlib",
                ),
                source_root,
                environment,
                output / "full-tests.log",
            )
            _run_command(
                (
                    str(venv_python),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/render_regression.py",
                    "tests/test_smoke_grad.py",
                ),
                source_root,
                environment,
                output / "render-gradient-tests.log",
            )
        result["success"] = True
    except Exception as error:
        diagnostics.append(f"{type(error).__name__}: {error}")
        try:
            diagnostics_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
        except OSError:
            pass

    _check_output_identity(output, output_identity)
    _write_json(output / "result.json", result)
    return result


def _uv_version(uv: str, cwd: Path, environment: Mapping[str, str], log: Path) -> str:
    output = _run_command((uv, "--version"), cwd, environment, log)
    return output.strip().splitlines()[-1]


def _python_version(
    python: Path, cwd: Path, environment: Mapping[str, str], log: Path
) -> str:
    output = _run_command(
        (str(python), "-c", "import sys; print(sys.version.split()[0])"),
        cwd,
        environment,
        log,
    )
    return output.strip().splitlines()[-1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "tpu"), required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        binding = json.loads(arguments.binding.read_text(encoding="utf-8"))
        if not isinstance(binding, dict):
            raise RunnerError("binding JSON must be an object")
        result = run(binding, arguments.backend, arguments.output_dir)
    except (OSError, UnicodeError, json.JSONDecodeError, RunnerError) as error:
        print(f"accelerator runner: {error}", file=sys.stderr)
        return 1
    return 0 if result["success"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
