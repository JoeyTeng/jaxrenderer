"""Run the frozen accelerator regression on a private Kaggle TPU kernel."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any, cast

QUEUE_TIMEOUT_SECONDS = 285 * 60
EXECUTION_TIMEOUT_SECONDS = 45 * 60
RELAY_QUEUE_TIMEOUT_SECONDS = 22 * 60 * 60
RELAY_SEGMENT_TIMEOUT_SECONDS = 345 * 60
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


type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)


def _redact(message: str, env: Mapping[str, str]) -> str:
    token = env.get("KAGGLE_API_TOKEN", "")
    return message.replace(token, "[redacted]") if token else message


def validate_binding(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise KaggleError("binding has missing or unexpected fields")
    value = cast(dict[str, object], value)
    if set(value) != BINDING_FIELDS:
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
        return validate_binding(json.loads(path.read_text(encoding="utf-8")))
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
    response = raw.strip()
    if not response:
        raise KaggleError("Kaggle quota response was empty")
    if re.fullmatch(
        r"no\s+quota\s+information\s+available\.?", response, re.IGNORECASE
    ):
        raise KaggleError("Kaggle quota information is unavailable")
    try:
        rows = cast(JsonValue, json.loads(response))
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
    command = [sys.executable, "-m", "tools.kaggle_cli", "--no-warn", *args[1:]]
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
    _ = selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    _ = selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise KaggleError("Kaggle CLI command exceeded its local timeout")
            for key, _ in selector.select(min(remaining, 1.0)):
                chunk = os.read(key.fd, 4096)
                if not chunk:
                    _ = selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                if total > MAX_CLI_OUTPUT_BYTES:
                    raise KaggleError("Kaggle CLI output exceeded the 64 KiB limit")
                captured[cast(str, key.data)].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise KaggleError("Kaggle CLI command exceeded its local timeout")
        return_code = process.wait(timeout=remaining)
    except KaggleError:
        process.kill()
        _ = process.wait()
        raise
    except subprocess.TimeoutExpired as error:
        process.kill()
        _ = process.wait()
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
            _ = stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _append_controller_log(path: Path, value: dict[str, object]) -> None:
    """Append and flush one secret-free controller event immediately."""
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise KaggleError(f"resume state {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise KaggleError(f"resume state {field} is invalid") from error
    if parsed.tzinfo is None:
        raise KaggleError(f"resume state {field} is invalid")
    return parsed.astimezone(timezone.utc)


def _verify_version_one(binding: dict[str, object], state_path: Path) -> None:
    """Run the collector's existing, version-pinned metadata check lazily."""
    try:
        collector = importlib.import_module("tools.kaggle_collect")
        _ = collector.verify_version_one(binding, state_path)
    except Exception as error:
        if type(error).__name__ == "CollectionError":
            raise KaggleError(str(error)) from error
        if isinstance(error, KaggleError):
            raise
        raise KaggleError(
            f"Kaggle version verification failed ({type(error).__name__})"
        ) from error


def _load_continuation_state(
    binding: dict[str, object],
    state_path: Path,
    continue_state_path: Path,
    output_dir: Path,
    username: str,
) -> tuple[dict[str, object], dict[str, Any] | None, float, float]:
    if continue_state_path.resolve() != state_path.resolve():
        raise KaggleError(
            "--continue-state must be the controller state in --output-dir"
        )
    try:
        stored_value: object = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleError(f"could not read continuation state: {error}") from error
    if not isinstance(stored_value, dict):
        raise KaggleError("continuation state has an unsupported controller schema")
    state = cast(dict[str, object], stored_value)
    if set(state) != CONTROLLER_STATE_FIELDS:
        raise KaggleError("continuation state has an unsupported controller schema")
    if state["binding"] != binding:
        raise KaggleError("continuation state binding does not exactly match --binding")
    kernel_id = state["kernel_id"]
    run_tag = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if (
        type(state["submitted_version"]) is not int
        or state["submitted_version"] != 1
        or not isinstance(kernel_id, str)
        or re.fullmatch(
            rf"{re.escape(username)}/jaxr-{run_tag}-[0-9a-f]{{10}}", kernel_id
        )
        is None
    ):
        raise KaggleError("continuation state kernel identity or version is invalid")

    phase = state["phase"]
    status = state["status"]
    outcome = state["outcome"]
    if not isinstance(status, str):
        raise KaggleError("continuation state status is invalid")
    if not isinstance(phase, str) or phase not in {"queue", "execution"}:
        raise KaggleError("continuation state phase is invalid")
    if not isinstance(outcome, str):
        raise KaggleError("continuation state outcome is invalid")
    if state["last_error"] is not None:
        raise KaggleError("continuation state contains a prior error")
    for field in ("queue_elapsed_seconds", "execution_elapsed_seconds"):
        value = state[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KaggleError(f"continuation state {field} is invalid")
        number = value
        if not math.isfinite(number) or number < 0:
            raise KaggleError(f"continuation state {field} is invalid")
    queue_elapsed = float(cast(int | float, state["queue_elapsed_seconds"]))
    execution_elapsed = float(cast(int | float, state["execution_elapsed_seconds"]))
    submitted_at = _parse_timestamp(state["submitted_at"], "submitted_at")
    updated_at = _parse_timestamp(state["updated_at"], "updated_at")
    phase_started_at = _parse_timestamp(state["phase_started_at"], "phase_started_at")
    execution_started_value = state["execution_started_at"]
    execution_started_at = (
        None
        if execution_started_value is None
        else _parse_timestamp(execution_started_value, "execution_started_at")
    )
    if (
        submitted_at > updated_at
        or phase_started_at > updated_at
        or (execution_started_at is not None and execution_started_at > updated_at)
    ):
        raise KaggleError("continuation state timestamps are inconsistent")
    now = _utc_now()
    if updated_at > now:
        raise KaggleError("continuation state updated_at is in the future")

    if outcome == "success":
        if status not in TERMINAL_SUCCESS:
            raise KaggleError("successful continuation state has no terminal success")
        result = validate_downloaded_result(output_dir / "result.json", binding)
        return state, result, queue_elapsed, execution_elapsed
    if outcome != "handover":
        raise KaggleError("continuation state is not eligible for handover")
    if status not in IN_PROGRESS | {"submitted"}:
        raise KaggleError("handover state does not contain an in-progress kernel")
    if phase != ("execution" if execution_started_at is not None else "queue"):
        raise KaggleError("handover state phase does not match its execution history")
    if (output_dir / "result.json").exists():
        raise KaggleError("handover output directory already contains result.json")
    elapsed_gap = (now - updated_at).total_seconds()
    if execution_started_at is None:
        queue_elapsed = max(
            queue_elapsed + elapsed_gap, (now - submitted_at).total_seconds()
        )
    else:
        execution_elapsed = max(
            execution_elapsed + elapsed_gap,
            (now - execution_started_at).total_seconds(),
        )
    return state, None, queue_elapsed, execution_elapsed


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
    _ = (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    source = _bootstrap_source(binding, runner_path)
    notebook = cast(
        JsonValue,
        {
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
        },
    )
    _ = (folder / "bootstrap.ipynb").write_text(
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


def validate_downloaded_result(
    path: Path, binding: dict[str, object]
) -> dict[str, Any]:
    try:
        result = cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
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
            name and isinstance(version, str) and version
            for name, version in versions.items()
        )
    ):
        raise KaggleError("Kaggle result.json versions are invalid")
    return cast(dict[str, Any], result)


def _run(
    binding: dict[str, object],
    backend: str,
    output_dir: Path,
    *,
    relay: bool = False,
    submit_only: bool = False,
    continue_state_path: Path | None = None,
) -> dict[str, Any]:
    """Submit or continue one private TPU kernel and validate its bound result."""
    binding = validate_binding(binding)
    if backend != "tpu":
        raise KaggleError("Kaggle adapter only supports the TPU backend")
    if submit_only and not relay:
        raise KaggleError("--submit-only requires --relay")
    if submit_only and continue_state_path is not None:
        raise KaggleError("--submit-only cannot be combined with continuation options")
    if not relay and continue_state_path is not None:
        raise KaggleError("continuation options require --relay")
    env = os.environ.copy()
    username = _require_credentials(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "kaggle-controller-state.json"
    log_path = output_dir / "kaggle-controller.log"
    segment_started = time.monotonic()
    segment_deadline = (
        segment_started + RELAY_SEGMENT_TIMEOUT_SECONDS if relay else None
    )
    runner_path = Path(__file__).with_name("accelerator_runner.py")
    continuation = continue_state_path is not None
    queue_started: float | None = None
    queue_finished: float | None = None
    execution_started: float | None = None
    execution_started_at: str | None = None
    execution_finished: float | None = None
    queue_elapsed_base = 0.0
    execution_elapsed_base = 0.0
    last_logged_state: tuple[object, object, object] | None = None

    if continuation:
        assert continue_state_path is not None
        state, prior_result, queue_elapsed_base, execution_elapsed_base = (
            _load_continuation_state(
                binding,
                state_path,
                continue_state_path,
                output_dir,
                username,
            )
        )
        if prior_result is not None:
            return prior_result
        kernel_id = cast(str, state["kernel_id"])
        execution_started_at = cast(str | None, state["execution_started_at"])
        if execution_started_at is None:
            queue_started = segment_started
        else:
            execution_started = segment_started
            queue_finished = segment_started
    else:
        if (output_dir / "result.json").exists():
            raise KaggleError("output directory already contains result.json")
        if state_path.exists():
            raise KaggleError("output directory already contains controller state")
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
        if phase != state["phase"]:
            state["phase_started_at"] = _timestamp()
        elapsed_queue = queue_elapsed_base
        if queue_started is not None:
            queue_end = queue_finished if queue_finished is not None else current
            elapsed_queue += max(0.0, queue_end - queue_started)
        elapsed_execution = execution_elapsed_base
        if execution_started is not None:
            execution_end = (
                execution_finished if execution_finished is not None else current
            )
            elapsed_execution += max(0.0, execution_end - execution_started)
        state.update(
            status=status,
            phase=phase,
            outcome=outcome,
            updated_at=_timestamp(),
            queue_elapsed_seconds=elapsed_queue,
            execution_elapsed_seconds=elapsed_execution,
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

    def segment_handover(
        now: float,
        phase: str,
        *,
        status: str | None = None,
        submitted_at: str | None = None,
    ) -> dict[str, Any]:
        handover_status = str(state["status"]) if status is None else status
        persist(
            status=handover_status,
            phase=phase,
            outcome="handover",
            now=now,
            submitted_at=submitted_at,
        )
        return {
            "binding": binding,
            "kernel_id": kernel_id,
            "submitted_version": 1,
            "success": False,
            "pending": True,
            "outcome": "handover",
        }

    with tempfile.TemporaryDirectory(prefix="jaxrenderer-kaggle-") as temporary:
        config_dir = Path(temporary) / "config"
        config_dir.mkdir()
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        if not continuation:
            persist(status="not_submitted", phase="quota_preflight")
            try:
                quota = _command_ok(["kaggle", "quota", "--format", "json"], env=env)
                _ = _read_quota(quota.stdout)
            except KaggleError as error:
                persist(
                    status="preflight_error",
                    phase="quota_preflight",
                    outcome="preflight_error",
                    last_error=str(error),
                )
                raise

            kernel_dir = Path(temporary) / "kernel"
            _make_kernel(
                kernel_dir,
                username,
                kernel_id.split("/", maxsplit=1)[1],
                binding,
                runner_path,
            )
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
            queue_deadline = queue_started + (
                RELAY_QUEUE_TIMEOUT_SECONDS if relay else QUEUE_TIMEOUT_SECONDS
            )
            execution_deadline: float | None = None
            if submit_only:
                return segment_handover(
                    time.monotonic(),
                    "queue",
                    status="submitted",
                    submitted_at=submitted_at,
                )
            persist(
                status="submitted",
                phase="queue",
                now=time.monotonic(),
                submitted_at=submitted_at,
            )
        else:
            queue_deadline = (
                queue_started
                + max(0.0, RELAY_QUEUE_TIMEOUT_SECONDS - queue_elapsed_base)
                if queue_started is not None
                else None
            )
            execution_deadline = (
                execution_started
                + max(0.0, EXECUTION_TIMEOUT_SECONDS - execution_elapsed_base)
                if execution_started is not None
                else None
            )

        if relay:
            try:
                _verify_version_one(binding, state_path)
            except KaggleError as error:
                persist(
                    status=str(state["status"]),
                    phase=str(state["phase"]),
                    outcome="version_verification_error",
                    last_error=str(error),
                )
                raise

        remote_failure = ""
        terminal_outcome = ""
        while True:
            now = time.monotonic()
            phase = "execution" if execution_started is not None else "queue"
            deadline = (
                execution_deadline if execution_started is not None else queue_deadline
            )
            if deadline is None:
                raise KaggleError("Kaggle controller deadline was not initialised")
            phase_remaining = deadline - now
            if phase_remaining <= 0:
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
            segment_remaining = (
                segment_deadline - now if segment_deadline is not None else None
            )
            if segment_remaining is not None and segment_remaining <= 0:
                return segment_handover(now, phase)
            remaining = min(
                phase_remaining,
                segment_remaining if segment_remaining is not None else phase_remaining,
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
                if (
                    segment_deadline is not None
                    and now >= segment_deadline
                    and str(error) == "Kaggle CLI command exceeded its local timeout"
                ):
                    return segment_handover(now, phase)
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
                else:
                    execution_finished = now
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
                execution_deadline = execution_started + max(
                    0.0, EXECUTION_TIMEOUT_SECONDS - execution_elapsed_base
                )
                execution_started_at = _timestamp()
            current_phase = "execution" if execution_started is not None else "queue"
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
            if segment_deadline is not None:
                remaining = min(remaining, segment_deadline - time.monotonic())
            if remaining <= 0:
                continue
            time.sleep(min(POLL_INTERVAL_SECONDS, remaining))

        # The CLI accepts a version suffix but fetches output by slug. Relay
        # mode checks Kaggle metadata before and after this same scoped fetch.
        if relay:
            try:
                _verify_version_one(binding, state_path)
            except KaggleError as error:
                persist(
                    status=str(state["status"]),
                    phase=str(state["phase"]),
                    outcome="version_verification_error",
                    last_error=str(error),
                )
                raise
        output_error: KaggleError | None = None
        try:
            _ = _command_ok(
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
            output_error = error
        if relay:
            try:
                _verify_version_one(binding, state_path)
            except KaggleError as error:
                persist(
                    status=str(state["status"]),
                    phase=str(state["phase"]),
                    outcome="version_verification_error",
                    last_error=str(error),
                )
                raise
        if output_error is not None:
            if remote_failure:
                persist(
                    status=str(state["status"]),
                    phase=str(state["phase"]),
                    outcome=terminal_outcome,
                    last_error=f"{remote_failure}; output retrieval failed: {output_error}",
                )
                raise KaggleError(
                    f"{remote_failure}; output retrieval failed: {output_error}"
                ) from output_error
            persist(
                status=str(state["status"]),
                phase=str(state["phase"]),
                outcome="output_retrieval_error",
                last_error=str(output_error),
            )
            raise output_error
        if remote_failure:
            raise KaggleError(remote_failure)
    try:
        result = validate_downloaded_result(output_dir / "result.json", binding)
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


def run(
    binding: dict[str, object],
    backend: str,
    output_dir: Path,
    *,
    relay: bool = False,
    submit_only: bool = False,
    continue_state_path: Path | None = None,
) -> dict[str, Any]:
    """Run Kaggle and preserve a short failure diagnostic with its artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _run(
            binding,
            backend,
            output_dir,
            relay=relay,
            submit_only=submit_only,
            continue_state_path=continue_state_path,
        )
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
    binding = validate_binding(binding)
    env = os.environ.copy()
    username = _require_credentials(env)
    try:
        if resume_state_path.resolve().parent == output_dir.resolve():
            raise KaggleError("resume output must use a separate collector directory")
        stored: object = json.loads(resume_state_path.read_text(encoding="utf-8"))
    except KaggleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KaggleError(f"could not read resume state: {error}") from error
    if not isinstance(stored, dict):
        raise KaggleError("resume state has an unsupported controller schema")
    stored = cast(dict[str, object], stored)
    if set(stored) not in (
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
    _ = root.add_argument("--binding", type=Path, required=True)
    _ = root.add_argument("--output-dir", type=Path, required=True)
    _ = root.add_argument("--resume-state", type=Path)
    _ = root.add_argument("--relay", action="store_true")
    _ = root.add_argument("--submit-only", action="store_true")
    _ = root.add_argument("--continue-state", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        binding = _load_binding_file(args.binding)
        if args.resume_state is not None:
            if args.relay or args.submit_only or args.continue_state is not None:
                raise KaggleError("--resume-state remains a status-only query")
            report = resume_status(binding, args.resume_state, args.output_dir)
            print(json.dumps(report, sort_keys=True))
            return 0
        result = run(
            binding,
            "tpu",
            args.output_dir,
            relay=args.relay,
            submit_only=args.submit_only,
            continue_state_path=args.continue_state,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except KaggleError as error:
        print(f"Kaggle TPU gate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
