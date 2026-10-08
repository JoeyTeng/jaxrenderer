"""Run the frozen accelerator regression on a private Kaggle TPU kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any

QUEUE_TIMEOUT_SECONDS = 4 * 60 * 60
EXECUTION_TIMEOUT_SECONDS = 45 * 60
POLL_INTERVAL_SECONDS = 20
CLI_TIMEOUT_SECONDS = 90
MAX_CLI_OUTPUT_BYTES = 64 * 1024
MIN_TPU_QUOTA_HOURS = 0.75
BINDING_FIELDS = {"kind", "head_sha", "head_repository", "run_id"}
LEGACY_CONTROLLER_STATE_FIELDS = {"binding", "kernel_id", "submitted_version"}
CONTROLLER_STATE_FIELDS = LEGACY_CONTROLLER_STATE_FIELDS | {
    "status",
    "phase",
    "phase_started_at",
    "submitted_at",
    "execution_started_at",
    "updated_at",
    "queue_elapsed_seconds",
    "execution_elapsed_seconds",
    "outcome",
    "last_error",
}
TERMINAL_SUCCESS = {"complete", "completed", "success", "succeeded"}
TERMINAL_FAILURE = {"error", "failed", "failure", "cancelled", "canceled", "aborted"}
QUEUED = {"queued"}
ACTIVE = {"running", "starting", "compiling", "initializing"}
IN_PROGRESS = QUEUED | ACTIVE
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
STATUS_RE = re.compile(r'has status "([^"]+)"', re.IGNORECASE)
VERSION_RE = re.compile(r"Kernel version (\d+) successfully pushed\.")
HOURS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*h\s*$", re.IGNORECASE)


class KaggleError(RuntimeError):
    """A Kaggle preflight, submission, polling, or output error."""


def _redact(message: str, env: dict[str, str]) -> str:
    token = env.get("KAGGLE_API_TOKEN", "")
    return message.replace(token, "[redacted]") if token else message


def _validate_binding(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != BINDING_FIELDS:
        raise KaggleError("binding has missing or unexpected fields")
    if value["kind"] != "release":
        raise KaggleError("binding kind is not release")
    if not isinstance(value["head_sha"], str) or not SHA_RE.fullmatch(
        value["head_sha"]
    ):
        raise KaggleError("binding head_sha is invalid")
    if value["head_repository"] != "JoeyTeng/jaxrenderer":
        raise KaggleError("binding head_repository is invalid")
    if not isinstance(value["run_id"], str) or not re.fullmatch(
        r"[1-9][0-9]*\.[1-9][0-9]*", value["run_id"]
    ):
        raise KaggleError("binding run_id is invalid")
    return value


def _load_binding_file(path: Path) -> dict[str, object]:
    try:
        return _validate_binding(json.loads(path.read_text(encoding="utf-8")))
    except KaggleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleError(f"could not read binding file: {error}") from error


def _require_credentials(env: dict[str, str]) -> str:
    # Check presence only; never include either credential in diagnostics.
    token = env.get("KAGGLE_API_TOKEN", "")
    if not token or token.isspace():
        raise KaggleError("KAGGLE_API_TOKEN is required")
    username = env.get("KAGGLE_USERNAME", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,50}", username):
        raise KaggleError(
            "KAGGLE_USERNAME is required and must be a valid account slug"
        )
    return username


def _read_quota(raw: str) -> float:
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as error:
        raise KaggleError("Kaggle quota response was not valid JSON") from error
    if not isinstance(rows, list):
        raise KaggleError("Kaggle quota response was not a row list")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("resource") == "TPU"
    ]
    if len(matches) != 1:
        raise KaggleError("Kaggle quota response did not contain exactly one TPU row")
    remaining = matches[0].get("remaining")
    if not isinstance(remaining, str):
        raise KaggleError("Kaggle TPU remaining quota was malformed")
    parsed = HOURS_RE.fullmatch(remaining)
    if parsed is None:
        raise KaggleError("Kaggle TPU remaining quota was malformed")
    hours = float(parsed.group(1))
    if hours < MIN_TPU_QUOTA_HOURS:
        raise KaggleError(
            f"Kaggle TPU quota must have at least {MIN_TPU_QUOTA_HOURS:.2f} hours remaining"
        )
    return hours


def _run_cli(
    args: list[str], *, env: dict[str, str], timeout: float = CLI_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run one CLI command with wall-clock and combined-output limits."""
    command = [sys.executable, "-m", "kaggle", *args[1:]]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=env,
            start_new_session=True,
        )
    except OSError as error:
        raise KaggleError(f"could not start Kaggle CLI: {error}") from error
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KaggleError("Kaggle CLI command exceeded its local timeout")
            for key, _ in selector.select(min(remaining, 1.0)):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > MAX_CLI_OUTPUT_BYTES:
                    raise KaggleError("Kaggle CLI output exceeded the 64 KiB limit")
                captured[key.data].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KaggleError("Kaggle CLI command exceeded its local timeout")
        return_code = process.wait(timeout=remaining)
    except KaggleError:
        process.kill()
        process.wait()
        raise
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise KaggleError("Kaggle CLI command exceeded its local timeout") from error
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return subprocess.CompletedProcess(
        command,
        return_code,
        captured["stdout"].decode("utf-8", errors="replace"),
        captured["stderr"].decode("utf-8", errors="replace"),
    )


def _command_ok(
    args: list[str], *, env: dict[str, str], timeout: float = CLI_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    result = _run_cli(args, env=env, timeout=timeout)
    if result.returncode:
        detail = result.stderr.strip()[:400] or result.stdout.strip()[:400]
        detail = _redact(detail, env)
        raise KaggleError(f"Kaggle CLI command failed: {detail or 'unknown error'}")
    return result


def _kernel_slug(binding: dict[str, object]) -> str:
    identity = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    run_tag = hashlib.sha256(identity.encode()).hexdigest()[:16]
    slug = f"jaxr-{run_tag}-{secrets.token_hex(5)}"
    if not SLUG_RE.fullmatch(slug):
        raise KaggleError("generated Kaggle kernel slug was invalid")
    return slug


def _write_controller_state(path: Path, value: dict[str, object]) -> None:
    """Atomically replace the local, secret-free record of one Kaggle submission."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".kaggle-state-",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _append_controller_log(path: Path, value: dict[str, object]) -> None:
    """Append and flush one secret-free controller event immediately."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bootstrap_source(binding: dict[str, object], runner_path: Path) -> str:
    try:
        runner_source = runner_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise KaggleError(
            f"could not read trusted accelerator runner: {error}"
        ) from error
    binding_literal = repr(json.dumps(binding, sort_keys=True))
    source_literal = repr(runner_source)
    return f"""# Embedded trusted runner from the workflow's default-branch checkout.
import json
import subprocess
import sys
import types
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.12.20"],
    check=True,
)
_name = "_jaxrenderer_accelerator_runner"
_module = types.ModuleType(_name)
_module.__file__ = "accelerator_runner.py"
sys.modules[_name] = _module
exec(compile({source_literal}, _module.__file__, "exec"), _module.__dict__)
_binding = json.loads({binding_literal})
_result = _module.run(_binding, "tpu", Path("/kaggle/working"))
if not isinstance(_result, dict):
    raise TypeError("accelerator runner returned a non-dictionary result")
Path("/kaggle/working/result.json").write_text(
    json.dumps(_result, sort_keys=True) + "\\n", encoding="utf-8"
)
if _result.get("success") is not True:
    raise RuntimeError("trusted accelerator runner reported failure")
"""


def _make_kernel(
    folder: Path,
    username: str,
    slug: str,
    binding: dict[str, object],
    runner_path: Path,
) -> None:
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "bootstrap.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "machine_shape": "TpuV5E8",
    }
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    source = _bootstrap_source(binding, runner_path)
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "id": "accelerator-ci",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (folder / "bootstrap.ipynb").write_text(
        json.dumps(notebook, sort_keys=True) + "\n", encoding="utf-8"
    )


def _status(stdout: str) -> str:
    match = STATUS_RE.search(stdout)
    if match is None:
        raise KaggleError("Kaggle returned an unrecognised kernel status")
    raw_status = match.group(1).strip().casefold()
    if raw_status.startswith("kernelworkerstatus."):
        return raw_status.rsplit(".", maxsplit=1)[1]
    if "." in raw_status:
        raise KaggleError("Kaggle returned an unrecognised kernel status type")
    return raw_status


def _validate_downloaded_result(
    path: Path, binding: dict[str, object]
) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleError(
            f"Kaggle output did not contain a readable result.json: {error}"
        ) from error
    if not isinstance(result, dict):
        raise KaggleError("Kaggle result.json was not an object")
    expected_fields = BINDING_FIELDS | {
        "backend",
        "device_backend",
        "device_count",
        "success",
        "devices",
        "versions",
    }
    if set(result) != expected_fields:
        raise KaggleError("Kaggle result.json has missing or unexpected fields")
    for field in ("kind", "head_sha", "head_repository", "run_id"):
        if result.get(field) != binding[field] or type(result.get(field)) is not type(
            binding[field]
        ):
            raise KaggleError(
                f"Kaggle result.json {field} did not match the frozen binding"
            )
    if result.get("backend") != "tpu" or result.get("device_backend") != "tpu":
        raise KaggleError("Kaggle result.json did not report a TPU device backend")
    if type(result.get("success")) is not bool or result["success"] is not True:
        raise KaggleError("Kaggle result.json did not report success")
    devices = result.get("devices")
    if not isinstance(devices, list) or not devices:
        raise KaggleError("Kaggle result.json has no TPU devices")
    if type(result.get("device_count")) is not int or result["device_count"] != len(
        devices
    ):
        raise KaggleError(
            "Kaggle result.json device_count does not match its device list"
        )
    for device in devices:
        if (
            not isinstance(device, dict)
            or set(device) != {"platform", "device_kind", "id"}
            or device.get("platform") != "tpu"
            or not all(
                isinstance(device.get(key), str) and device[key] for key in device
            )
        ):
            raise KaggleError("Kaggle result.json contains invalid TPU device metadata")
    versions = result.get("versions")
    if (
        not isinstance(versions, dict)
        or not versions
        or not all(
            isinstance(name, str) and name and isinstance(version, str) and version
            for name, version in versions.items()
        )
    ):
        raise KaggleError("Kaggle result.json versions are invalid")
    return result


def _run(binding: dict[str, object], backend: str, output_dir: Path) -> dict[str, Any]:
    """Submit one private TPU kernel and return its binding-checked result."""
    binding = _validate_binding(binding)
    if backend != "tpu":
        raise KaggleError("Kaggle adapter only supports the TPU backend")
    env = os.environ.copy()
    username = _require_credentials(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise KaggleError("output directory already contains result.json")
    state_path = output_dir / "kaggle-controller-state.json"
    if state_path.exists():
        raise KaggleError("output directory already contains controller state")
    log_path = output_dir / "kaggle-controller.log"
    runner_path = Path(__file__).with_name("accelerator_runner.py")
    slug = _kernel_slug(binding)
    kernel_id = f"{username}/{slug}"
    state: dict[str, object] = {
        "binding": binding,
        "kernel_id": kernel_id,
        "submitted_version": None,
        "status": "not_submitted",
        "phase": "quota_preflight",
        "phase_started_at": _timestamp(),
        "submitted_at": None,
        "execution_started_at": None,
        "updated_at": None,
        "queue_elapsed_seconds": 0.0,
        "execution_elapsed_seconds": 0.0,
        "outcome": "pending",
        "last_error": None,
    }
    queue_started: float | None = None
    queue_finished: float | None = None
    execution_started: float | None = None
    execution_finished: float | None = None
    last_logged_state: tuple[object, object, object] | None = None

    def persist(
        *,
        status: str,
        phase: str,
        outcome: str = "pending",
        now: float | None = None,
        submitted_at: str | None = None,
        execution_started_at: str | None = None,
        last_error: str | None = None,
    ) -> None:
        nonlocal last_logged_state
        current = time.monotonic() if now is None else now
        status = _redact(status, env)
        phase_changed = phase != state["phase"]
        if phase_changed:
            state["phase_started_at"] = _timestamp()
        state.update(
            status=status,
            phase=phase,
            outcome=outcome,
            updated_at=_timestamp(),
            queue_elapsed_seconds=(
                max(
                    0.0,
                    (queue_finished if queue_finished is not None else current)
                    - queue_started,
                )
                if queue_started is not None
                else 0.0
            ),
            execution_elapsed_seconds=(
                max(
                    0.0,
                    (execution_finished if execution_finished is not None else current)
                    - execution_started,
                )
                if execution_started is not None
                else 0.0
            ),
            last_error=last_error,
        )
        if isinstance(state["last_error"], str):
            state["last_error"] = _redact(state["last_error"], env)
        if submitted_at is not None:
            state["submitted_at"] = submitted_at
        if execution_started_at is not None:
            state["execution_started_at"] = execution_started_at
        _write_controller_state(state_path, state)
        key = (phase, status, outcome)
        if key != last_logged_state:
            event = {
                "event": "controller_state",
                "binding": binding,
                "kernel_id": kernel_id,
                "submitted_version": state["submitted_version"],
                "phase": phase,
                "status": status,
                "queue_elapsed_seconds": state["queue_elapsed_seconds"],
                "execution_elapsed_seconds": state["execution_elapsed_seconds"],
                "outcome": outcome,
                "last_error": state["last_error"],
            }
            line = json.dumps(event, sort_keys=True)
            _append_controller_log(log_path, event)
            print(line, flush=True)
            last_logged_state = key

    with tempfile.TemporaryDirectory(prefix="jaxrenderer-kaggle-") as temporary:
        config_dir = Path(temporary) / "config"
        config_dir.mkdir()
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        persist(status="not_submitted", phase="quota_preflight")
        try:
            quota = _command_ok(["kaggle", "quota", "--format", "json"], env=env)
            _read_quota(quota.stdout)
        except KaggleError as error:
            persist(
                status="preflight_error",
                phase="quota_preflight",
                outcome="preflight_error",
                last_error=str(error),
            )
            raise

        kernel_dir = Path(temporary) / "kernel"
        _make_kernel(kernel_dir, username, slug, binding, runner_path)
        try:
            pushed = _command_ok(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "--path",
                    str(kernel_dir),
                    "--accelerator",
                    "TpuV5E8",
                    "--timeout",
                    str(EXECUTION_TIMEOUT_SECONDS),
                ],
                env=env,
            )
        except KaggleError as error:
            persist(
                status="submission_error",
                phase="submission",
                outcome="submission_error_unknown",
                now=time.monotonic(),
                last_error=str(error),
            )
            raise
        version_match = VERSION_RE.search(pushed.stdout)
        if version_match is None or version_match.group(1) != "1":
            error = "fresh private Kaggle kernel did not report version 1"
            persist(
                status="submission_version_unknown",
                phase="submission",
                outcome="submission_version_unknown",
                now=time.monotonic(),
                last_error=error,
            )
            raise KaggleError(error)
        state["submitted_version"] = 1
        queue_started = time.monotonic()
        submitted_at = _timestamp()
        queue_deadline = queue_started + QUEUE_TIMEOUT_SECONDS
        execution_deadline: float | None = None
        persist(
            status="submitted",
            phase="queue",
            now=time.monotonic(),
            submitted_at=submitted_at,
        )
        remote_failure = ""
        terminal_outcome = ""
        while True:
            now = time.monotonic()
            phase = "execution" if execution_started is not None else "queue"
            deadline = (
                execution_deadline if execution_started is not None else queue_deadline
            )
            assert deadline is not None
            remaining = deadline - now
            if remaining <= 0:
                timeout_outcome = (
                    "execution_timeout"
                    if execution_started is not None
                    else "queue_timeout"
                )
                timeout_error = (
                    "Kaggle execution polling timed out"
                    if execution_started is not None
                    else "Kaggle queue polling timed out"
                )
                if execution_started is None:
                    queue_finished = now
                else:
                    execution_finished = now
                persist(
                    status=str(state["status"]),
                    phase=phase,
                    outcome=timeout_outcome,
                    now=now,
                    last_error=timeout_error,
                )
                raise KaggleError(
                    f"{timeout_error}; the remote TPU session may continue because the CLI has no cancellation command"
                )
            try:
                status_result = _command_ok(
                    ["kaggle", "kernels", "status", kernel_id],
                    env=env,
                    timeout=min(CLI_TIMEOUT_SECONDS, remaining),
                )
            except KaggleError as error:
                now = time.monotonic()
                if now >= deadline:
                    timeout_outcome = (
                        "execution_timeout"
                        if execution_started is not None
                        else "queue_timeout"
                    )
                    timeout_label = (
                        "execution" if execution_started is not None else "queue"
                    )
                    timeout_error = f"Kaggle {timeout_label} polling timed out"
                    if execution_started is None:
                        queue_finished = now
                    else:
                        execution_finished = now
                    persist(
                        status=str(state["status"]),
                        phase=phase,
                        outcome=timeout_outcome,
                        now=now,
                        last_error=timeout_error,
                    )
                    raise KaggleError(
                        f"{timeout_error}; the remote TPU session may continue because the CLI has no cancellation command"
                    ) from error
                persist(
                    status="cli_error",
                    phase=phase,
                    outcome="cli_error",
                    now=now,
                    last_error=str(error),
                )
                raise
            try:
                remote_status = _status(status_result.stdout)
            except KaggleError as error:
                persist(
                    status="unknown_api_status",
                    phase=phase,
                    outcome="unknown_api_status",
                    now=time.monotonic(),
                    last_error=str(error),
                )
                raise
            now = time.monotonic()
            if now >= deadline:
                timeout_outcome = (
                    "execution_timeout"
                    if execution_started is not None
                    else "queue_timeout"
                )
                timeout_label = (
                    "execution" if execution_started is not None else "queue"
                )
                timeout_error = f"Kaggle {timeout_label} polling timed out"
                if execution_started is None:
                    queue_finished = now
                persist(
                    status=remote_status,
                    phase=phase,
                    outcome=timeout_outcome,
                    now=now,
                    last_error=timeout_error,
                )
                raise KaggleError(
                    f"{timeout_error}; the remote TPU session may continue because the CLI has no cancellation command"
                )
            if execution_started is None and remote_status in ACTIVE:
                execution_started = now
                queue_finished = now
                execution_deadline = execution_started + EXECUTION_TIMEOUT_SECONDS
                execution_started_at = _timestamp()
            if execution_started is None:
                current_phase = "queue"
            else:
                current_phase = "execution"
            persist(
                status=remote_status,
                phase=current_phase,
                now=now,
                execution_started_at=(
                    execution_started_at
                    if execution_started is not None
                    and state["execution_started_at"] is None
                    else None
                ),
            )
            if remote_status in TERMINAL_SUCCESS:
                terminal_outcome = "remote_terminal_success"
                if execution_started is None:
                    queue_finished = now
                else:
                    execution_finished = now
                persist(
                    status=remote_status,
                    phase=current_phase,
                    outcome=terminal_outcome,
                    now=now,
                )
                break
            if remote_status in TERMINAL_FAILURE:
                detail = status_result.stdout.strip()[-400:]
                remote_failure = (
                    f"Kaggle TPU kernel ended with status {remote_status}: {detail}"
                )
                terminal_outcome = "remote_terminal_failure"
                if execution_started is None:
                    queue_finished = now
                else:
                    execution_finished = now
                persist(
                    status=remote_status,
                    phase=current_phase,
                    outcome=terminal_outcome,
                    now=now,
                    last_error=remote_failure,
                )
                break
            if remote_status not in IN_PROGRESS:
                unknown_error = (
                    f"Kaggle TPU kernel returned unknown status {remote_status!r}"
                )
                persist(
                    status=remote_status,
                    phase=current_phase,
                    outcome="unknown_api_status",
                    now=now,
                    last_error=unknown_error,
                )
                raise KaggleError(unknown_error)
            phase_deadline = (
                execution_deadline if execution_started is not None else queue_deadline
            )
            assert phase_deadline is not None
            remaining = phase_deadline - time.monotonic()
            if remaining <= 0:
                continue
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

        # CLI 2.2.4 accepts a version suffix but fetches output by slug. The
        # random one-use slug and verified first version bind it to this run.
        try:
            _command_ok(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    f"{kernel_id}/1",
                    "--path",
                    str(output_dir),
                    "--force",
                    "--file-pattern",
                    r"^(?:result\.json|(?:diagnostics|setup|device-probe|full-tests|render-gradient-tests)\.log|render-artifacts/(?:numeric-report\.json|[A-Za-z0-9._-]+\.png))$",
                ],
                env=env,
            )
        except KaggleError as error:
            if remote_failure:
                persist(
                    status=str(state["status"]),
                    phase=str(state["phase"]),
                    outcome=terminal_outcome,
                    last_error=f"{remote_failure}; output retrieval failed: {error}",
                )
                raise KaggleError(
                    f"{remote_failure}; output retrieval failed: {error}"
                ) from error
            persist(
                status=str(state["status"]),
                phase=str(state["phase"]),
                outcome="output_retrieval_error",
                last_error=str(error),
            )
            raise
        if remote_failure:
            raise KaggleError(remote_failure)
    try:
        result = _validate_downloaded_result(output_dir / "result.json", binding)
    except KaggleError as error:
        persist(
            status=str(state["status"]),
            phase=str(state["phase"]),
            outcome="result_validation_error",
            last_error=str(error),
        )
        raise
    persist(
        status=str(state["status"]),
        phase=str(state["phase"]),
        outcome="success",
        last_error=None,
    )
    return result


def run(binding: dict[str, object], backend: str, output_dir: Path) -> dict[str, Any]:
    """Run Kaggle and preserve a short failure diagnostic with its artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _run(binding, backend, output_dir)
    except KaggleError as error:
        message = _redact(str(error), os.environ)
        try:
            _append_controller_log(
                output_dir / "kaggle-controller.log",
                {"event": "controller_error", "message": message[:400]},
            )
        except OSError:
            pass
        raise KaggleError(message) from error


def resume_status(
    binding: dict[str, object], resume_state_path: Path, output_dir: Path
) -> dict[str, object]:
    """Report one latest-status query without treating it as gate evidence."""
    binding = _validate_binding(binding)
    env = os.environ.copy()
    username = _require_credentials(env)
    try:
        if resume_state_path.resolve().parent == output_dir.resolve():
            raise KaggleError("resume output must use a separate collector directory")
        stored = json.loads(resume_state_path.read_text(encoding="utf-8"))
    except KaggleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleError(f"could not read resume state: {error}") from error
    if not isinstance(stored, dict) or set(stored) not in (
        LEGACY_CONTROLLER_STATE_FIELDS,
        CONTROLLER_STATE_FIELDS,
    ):
        raise KaggleError("resume state has an unsupported controller schema")
    if stored["binding"] != binding:
        raise KaggleError("resume state binding does not exactly match --binding")
    kernel_id = stored["kernel_id"]
    run_tag = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if (
        type(stored["submitted_version"]) is not int
        or stored["submitted_version"] != 1
        or not isinstance(kernel_id, str)
        or re.fullmatch(
            rf"{re.escape(username)}/jaxr-{run_tag}-[0-9a-f]{{10}}", kernel_id
        )
        is None
    ):
        raise KaggleError("resume state kernel identity or version is invalid")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "kaggle-resume-report.json"
    if report_path.exists():
        raise KaggleError("collector directory already contains a resume report")
    report: dict[str, object] = {
        "binding": binding,
        "kernel_id": kernel_id,
        "submitted_version": 1,
        "remote_status": "unknown",
        "version_verified": False,
        "success": False,
        "outcome": "status_only_version_unverified",
        "checked_at": _timestamp(),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="jaxrenderer-kaggle-resume-") as tmp:
            env["KAGGLE_CONFIG_DIR"] = str(Path(tmp) / "config")
            Path(env["KAGGLE_CONFIG_DIR"]).mkdir()
            result = _command_ok(
                ["kaggle", "kernels", "status", kernel_id],
                env=env,
                timeout=CLI_TIMEOUT_SECONDS,
            )
        try:
            remote_status = _status(result.stdout)
            if remote_status not in IN_PROGRESS | TERMINAL_SUCCESS | TERMINAL_FAILURE:
                raise KaggleError("Kaggle returned an unrecognised kernel status")
            report["remote_status"] = remote_status
        except KaggleError as error:
            report.update(
                outcome="unknown_api_status",
                remote_status="unknown",
                last_error=str(error)[:400],
            )
            _write_controller_state(report_path, report)
            _append_controller_log(
                output_dir / "kaggle-resume.log",
                {"event": "unknown_api_status", "message": str(error)[:400]},
            )
            raise
    except KaggleError as error:
        message = _redact(str(error), env)
        if report["outcome"] != "unknown_api_status":
            report.update(outcome="status_query_error", last_error=message[:400])
            _write_controller_state(report_path, report)
            _append_controller_log(
                output_dir / "kaggle-resume.log",
                {"event": "status_query_error", "message": message[:400]},
            )
        raise KaggleError(message) from error
    _write_controller_state(report_path, report)
    _append_controller_log(
        output_dir / "kaggle-resume.log",
        {
            "event": "status_only_version_unverified",
            "binding": binding,
            "kernel_id": kernel_id,
            "submitted_version": 1,
            "remote_status": report["remote_status"],
            "version_verified": False,
            "success": False,
        },
    )
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--binding", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--resume-state", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        binding = _load_binding_file(args.binding)
        if args.resume_state is not None:
            report = resume_status(binding, args.resume_state, args.output_dir)
            print(json.dumps(report, sort_keys=True))
            return 0
        result = run(binding, "tpu", args.output_dir)
        print(json.dumps(result, sort_keys=True))
        return 0
    except KaggleError as error:
        print(f"Kaggle TPU gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
