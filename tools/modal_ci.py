"""Run a frozen public pull-request snapshot on a one-off Modal T4."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
from typing import Any

from tools import accelerator_runner

MAX_ARTIFACT_BYTES = 300 * 1024 * 1024
MAX_ARTIFACT_FILES = 1_000
MODAL_TOKEN_ENV = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


class ModalCIError(RuntimeError):
    """A binding, credential, or artifact validation error."""


def _preflight_credentials(environment: Mapping[str, str] | None = None) -> None:
    values = os.environ if environment is None else environment
    missing = [name for name in MODAL_TOKEN_ENV if not values.get(name)]
    if missing:
        raise ModalCIError(
            "Modal credentials must be available to the trusted controller: "
            + ", ".join(missing)
        )


def _load_binding(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModalCIError(f"could not read binding file: {error}") from error
    if not isinstance(value, dict):
        raise ModalCIError("binding JSON must be an object")
    try:
        return accelerator_runner._validate_binding(value)
    except accelerator_runner.RunnerError as error:
        raise ModalCIError(str(error)) from error


def _validate_result(
    result: object, binding: Mapping[str, object], backend: str = "gpu"
) -> dict[str, object]:
    identity_fields = (
        ("pr", "head_sha", "base_sha", "run_id")
        if "pr" in binding
        else ("kind", "head_sha", "head_repository", "run_id")
    )
    fields = set(identity_fields) | {
        "backend",
        "device_backend",
        "device_count",
        "success",
        "devices",
        "versions",
    }
    if not isinstance(result, dict) or set(result) != fields:
        raise ModalCIError("remote result has missing or unexpected fields")
    for name in identity_fields:
        if (
            type(result[name]) is not type(binding[name])
            or result[name] != binding[name]
        ):
            raise ModalCIError(
                f"remote result {name} does not match the frozen binding"
            )
    if result["backend"] != backend:
        raise ModalCIError(
            "remote result backend does not match the Modal accelerator request"
        )
    if type(result["success"]) is not bool:
        raise ModalCIError("remote result success must be a boolean")
    devices = result["devices"]
    if not isinstance(devices, list) or type(result["device_count"]) is not int:
        raise ModalCIError("remote result has malformed device information")
    if result["device_count"] != len(devices):
        raise ModalCIError("remote result device_count does not match its devices")
    if result["success"] is True:
        if not devices or result["device_backend"] != backend:
            raise ModalCIError(
                "successful result did not use the requested GPU backend"
            )
    elif result["device_backend"] not in {backend, "unknown"}:
        raise ModalCIError("failed result reports an unexpected device backend")
    for device in devices:
        if (
            not isinstance(device, dict)
            or set(device) != {"platform", "device_kind", "id"}
            or device["platform"] != backend
            or any(
                not isinstance(device[key], str) or not device[key] for key in device
            )
        ):
            raise ModalCIError("remote result contains malformed device metadata")
    versions = result["versions"]
    if not isinstance(versions, dict) or not all(
        isinstance(name, str) and name and isinstance(version, str) and version
        for name, version in versions.items()
    ):
        raise ModalCIError("remote result versions must map names to strings")
    if result["success"] is True and not versions:
        raise ModalCIError("successful remote result must include version metadata")
    return result


def _pack_artifacts(output_dir: Path) -> bytes:
    files: list[Path] = []
    total_size = 0
    for path in output_dir.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ModalCIError("remote artifacts must not contain symbolic links")
        if stat.S_ISREG(info.st_mode):
            files.append(path)
            total_size += info.st_size
    if len(files) > MAX_ARTIFACT_FILES:
        raise ModalCIError("remote run produced too many artifact files")
    if total_size > MAX_ARTIFACT_BYTES:
        raise ModalCIError("remote artifacts exceed the 300 MiB transfer limit")

    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for path in sorted(files):
            archive.add(
                path, arcname=path.relative_to(output_dir).as_posix(), recursive=False
            )
    result = archive_buffer.getvalue()
    if len(result) > MAX_ARTIFACT_BYTES:
        raise ModalCIError(
            "compressed remote artifacts exceed the 300 MiB transfer limit"
        )
    return result


def _remote_run(binding_data: str) -> tuple[bytes, dict[str, object]]:
    binding_value = json.loads(binding_data)
    if not isinstance(binding_value, dict):
        raise ModalCIError("remote binding must be a JSON object")
    binding = accelerator_runner._validate_binding(binding_value)
    with tempfile.TemporaryDirectory(prefix="jaxrenderer-modal-") as temporary:
        output = Path(temporary) / "output"
        result = accelerator_runner.run(binding, "gpu", output)
        archive_bytes = _pack_artifacts(output)
    return archive_bytes, result


def _create_modal_app(modal: Any) -> tuple[Any, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    image = (
        modal.Image.debian_slim(python_version="3.14")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install("uv==0.12.20")
        .add_local_file(
            repository_root / "tools" / "modal_ci.py",
            "/root/tools/modal_ci.py",
        )
        .add_local_file(
            repository_root / "tools" / "accelerator_runner.py",
            "/root/tools/accelerator_runner.py",
        )
    )
    app = modal.App("jaxrenderer-accelerator-ci", include_source=False)

    @app.function(
        image=image,
        gpu="T4",
        timeout=1_800,
        max_containers=1,
        serialized=True,
        include_source=False,
    )
    def run_frozen_snapshot(binding_data: str) -> tuple[bytes, dict[str, object]]:
        import sys

        sys.path.insert(0, "/root")
        from tools.modal_ci import _remote_run as remote_run

        return remote_run(binding_data)

    return app, run_frozen_snapshot


def _prepare_output(path: Path) -> tuple[Path, tuple[int, int, int]]:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ModalCIError("output directory must not be a symbolic link")
    resolved = path.resolve(strict=True)
    info = resolved.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ModalCIError("output directory must be a directory owned by this process")
    return resolved, (info.st_dev, info.st_ino, info.st_uid)


def _check_output_identity(path: Path, identity: tuple[int, int, int]) -> None:
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_dev, info.st_ino, info.st_uid) != identity
        or info.st_uid != os.geteuid()
    ):
        raise ModalCIError(
            "output directory identity or ownership changed during the run"
        )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute(binding: Mapping[str, object], output_dir: Path) -> dict[str, object]:
    frozen = accelerator_runner._validate_binding(binding)
    output, identity = _prepare_output(Path(output_dir))
    try:
        _preflight_credentials()
        import modal

        app, remote_function = _create_modal_app(modal)
        archive_bytes: bytes | None = None
        remote_result: dict[str, object] | None = None
        controller_error: str | None = None
        with modal.enable_output(), app.run():
            try:
                archive_bytes, raw_result = remote_function.remote(
                    json.dumps(frozen, sort_keys=True)
                )
                remote_result = _validate_result(raw_result, frozen)
            except Exception as error:
                controller_error = f"{type(error).__name__}: {error}"
        if remote_result is None:
            remote_result = accelerator_runner._initial_result(frozen, "gpu")
            _atomic_write(
                output / "diagnostics.log",
                (controller_error or "Modal invocation failed").encode("utf-8"),
            )
        if archive_bytes is not None:
            if (
                not isinstance(archive_bytes, bytes)
                or len(archive_bytes) > MAX_ARTIFACT_BYTES
            ):
                raise ModalCIError(
                    "Modal returned an invalid or oversized artifact archive"
                )
            _atomic_write(output / "artifacts.tar.gz", archive_bytes)
        _atomic_write(
            output / "result.json",
            (json.dumps(remote_result, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    except Exception as error:
        if not (output / "result.json").exists():
            failure = accelerator_runner._initial_result(frozen, "gpu")
            _atomic_write(
                output / "diagnostics.log",
                f"{type(error).__name__}: {error}\n".encode("utf-8"),
            )
            _atomic_write(
                output / "result.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        _check_output_identity(output, identity)
        raise
    _check_output_identity(output, identity)
    return remote_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        binding = _load_binding(arguments.binding)
        result = execute(binding, arguments.output_dir)
    except (ModalCIError, accelerator_runner.RunnerError, OSError) as error:
        print(f"Modal accelerator CI: {error}", file=sys.stderr)
        return 1
    return 0 if result["success"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
