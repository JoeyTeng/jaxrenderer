"""Acknowledge the first Kaggle queue wait and continue collecting its kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import cast

from tools import kaggle_ci, kaggle_collect

FIRST_QUEUE_WAIT_SECONDS = 4 * 60 * 60
POLL_INTERVAL_SECONDS = kaggle_ci.POLL_INTERVAL_SECONDS
MAX_HANDOFF_FILES = kaggle_collect.MAX_FILES + 4
MAX_STATE_BYTES = 1024 * 1024
MAX_HANDOFF_BYTES = kaggle_collect.MAX_TOTAL_BYTES + MAX_STATE_BYTES + 1024 * 1024
ACTIVE = {"running", "starting", "compiling", "initializing"}
PENDING = ACTIVE | {"queued", "cancel_requested"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_timestamp(value: object, field: str, now: datetime) -> datetime:
    if not isinstance(value, str):
        raise kaggle_ci.KaggleError(f"controller state {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise kaggle_ci.KaggleError(f"controller state {field} is invalid") from error
    if parsed.tzinfo is None:
        raise kaggle_ci.KaggleError(f"controller state {field} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed > now:
        raise kaggle_ci.KaggleError(f"controller state {field} is in the future")
    return parsed


def _elapsed(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise kaggle_ci.KaggleError(f"controller state {field} is invalid")
    try:
        if not math.isfinite(value) or value < 0:
            raise kaggle_ci.KaggleError(f"controller state {field} is invalid")
    except OverflowError:
        raise kaggle_ci.KaggleError(f"controller state {field} is invalid") from None
    return float(value)


def _read_state(path: Path) -> dict[str, object]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STATE_BYTES:
            raise kaggle_ci.KaggleError("controller state file is unsafe or too large")
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except kaggle_ci.KaggleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise kaggle_ci.KaggleError("could not read controller state") from error
    if not isinstance(value, dict):
        raise kaggle_ci.KaggleError("controller state has an unsupported schema")
    state = cast(dict[str, object], value)
    if set(state) != kaggle_ci.CONTROLLER_STATE_FIELDS:
        raise kaggle_ci.KaggleError("controller state has an unsupported schema")
    return state


def _write_state(path: Path, state: dict[str, object]) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".kaggle-wait-",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(state, stream, sort_keys=True)
            _ = stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _append_log(path: Path, event: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        _ = stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()


def _validate_handoff(
    binding: dict[str, object], state: dict[str, object], now: datetime
) -> datetime:
    if state["binding"] != binding:
        raise kaggle_ci.KaggleError("controller state binding does not exactly match")
    if type(state["submitted_version"]) is not int or state["submitted_version"] != 1:
        raise kaggle_ci.KaggleError("controller state version is invalid")
    if (
        state["status"] != "queued"
        or state["phase"] != "queue"
        or state["outcome"] != "queue_timeout"
        or state["execution_started_at"] is not None
        or state["last_error"] != "Kaggle queue polling timed out"
    ):
        raise kaggle_ci.KaggleError("controller state is not a queued timeout handoff")
    username = os.environ.get("KAGGLE_USERNAME", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,50}", username):
        raise kaggle_ci.KaggleError(
            "KAGGLE_USERNAME is required to validate the kernel"
        )
    try:
        owner, _ = kaggle_collect.kernel_identity(binding, state["kernel_id"])
    except kaggle_collect.CollectionError as error:
        raise kaggle_ci.KaggleError(
            "controller state kernel identity is invalid"
        ) from error
    if owner != username:
        raise kaggle_ci.KaggleError("controller state kernel identity does not match")

    submitted = _parse_timestamp(state["submitted_at"], "submitted_at", now)
    phase_started = _parse_timestamp(state["phase_started_at"], "phase_started_at", now)
    updated = _parse_timestamp(state["updated_at"], "updated_at", now)
    if phase_started < submitted or updated < phase_started:
        raise kaggle_ci.KaggleError("controller state timestamps are out of order")
    queued = _elapsed(state["queue_elapsed_seconds"], "queue_elapsed_seconds")
    execution = _elapsed(
        state["execution_elapsed_seconds"], "execution_elapsed_seconds"
    )
    if (
        queued < FIRST_QUEUE_WAIT_SECONDS
        or queued > kaggle_ci.QUEUE_TIMEOUT_SECONDS + kaggle_ci.CLI_TIMEOUT_SECONDS
        or execution != 0
    ):
        raise kaggle_ci.KaggleError("controller state queue timeout timing is invalid")
    return submitted


def _safe_files(root: Path) -> list[Path]:
    try:
        root_info = root.lstat()
    except OSError as error:
        raise kaggle_ci.KaggleError("handoff directory could not be read") from error
    if not stat.S_ISDIR(root_info.st_mode):
        raise kaggle_ci.KaggleError("handoff path is not a directory")

    files: list[Path] = []
    total = 0
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            path = current_path / directory
            try:
                info = path.lstat()
            except OSError as error:
                raise kaggle_ci.KaggleError(
                    "handoff directory entry is unreadable"
                ) from error
            if (
                not stat.S_ISDIR(info.st_mode)
                or path.relative_to(root).as_posix() != "render-artifacts"
            ):
                raise kaggle_ci.KaggleError("handoff contains an unsafe directory")
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
            except OSError as error:
                raise kaggle_ci.KaggleError("handoff file is unreadable") from error
            allowed = (
                relative
                in {
                    "kaggle-controller-state.json",
                    "kaggle-controller.log",
                    "kaggle-collection-report.json",
                }
                or kaggle_collect.FILE_RE.fullmatch(relative) is not None
            )
            if not allowed or not stat.S_ISREG(info.st_mode):
                raise kaggle_ci.KaggleError("handoff contains an unsafe file")
            if info.st_size > kaggle_collect.MAX_FILE_BYTES:
                raise kaggle_ci.KaggleError("handoff file exceeds its size limit")
            files.append(path)
            total += info.st_size
            if len(files) > MAX_HANDOFF_FILES or total > MAX_HANDOFF_BYTES:
                raise kaggle_ci.KaggleError("handoff exceeds its aggregate size limit")
    return files


def _copy_files(
    source: Path, destination: Path, *, replace: frozenset[str] | None = None
) -> None:
    for path in _safe_files(source):
        relative = path.relative_to(source)
        target = destination / relative
        if target.exists() and (replace is None or relative.as_posix() not in replace):
            raise kaggle_ci.KaggleError(
                "output directory already contains handoff files"
            )
        _ = target.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.copyfile(path, target)


def _prepare_output(path: Path, resume_path: Path) -> None:
    if path.resolve() == resume_path.resolve():
        raise kaggle_ci.KaggleError(
            "output directory must differ from resume directory"
        )
    if path.exists():
        if not stat.S_ISDIR(path.lstat().st_mode) or any(path.iterdir()):
            raise kaggle_ci.KaggleError("output directory must be fresh")
    else:
        _ = path.mkdir(parents=True)


def _prepare_submit_output(path: Path) -> None:
    if path.exists():
        if not stat.S_ISDIR(path.lstat().st_mode) or any(path.iterdir()):
            raise kaggle_ci.KaggleError("submission output directory must be fresh")
    else:
        _ = path.mkdir(parents=True)


def submit(binding_path: Path, output_dir: Path) -> int:
    try:
        binding = kaggle_ci.validate_binding(
            json.loads(binding_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise kaggle_ci.KaggleError("could not read binding file") from error
    _prepare_submit_output(output_dir)
    try:
        result = kaggle_ci.run(
            binding,
            "tpu",
            output_dir,
            queue_timeout_seconds=FIRST_QUEUE_WAIT_SECONDS,
        )
        if result.get("success") is not True:
            raise kaggle_ci.KaggleError("Kaggle did not return a validated result")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except kaggle_ci.KaggleError as error:
        try:
            state = _read_state(output_dir / "kaggle-controller-state.json")
            _ = _validate_handoff(binding, state, _utc_now())
        except kaggle_ci.KaggleError:
            raise error
        print("Kaggle queue wait saved a validated continuation handoff.", flush=True)
        return 0


def continue_wait(binding_path: Path, resume_dir: Path, output_dir: Path) -> int:
    try:
        binding = kaggle_ci.validate_binding(
            json.loads(binding_path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise kaggle_ci.KaggleError("could not read binding file") from error
    _prepare_output(output_dir, resume_dir)
    result_path = resume_dir / "result.json"
    if result_path.exists():
        info = result_path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > kaggle_collect.MAX_FILE_BYTES
        ):
            raise kaggle_ci.KaggleError("handoff result is unsafe or too large")
        _ = kaggle_ci.validate_downloaded_result(result_path, binding)
        _copy_files(resume_dir, output_dir)
        print(
            "Kaggle result was already validated; copied bounded outputs.", flush=True
        )
        return 0

    state_path = resume_dir / "kaggle-controller-state.json"
    state = _read_state(state_path)
    now = _utc_now().astimezone(timezone.utc)
    submitted = _validate_handoff(binding, state, now)
    initial_elapsed = max(0.0, (now - submitted).total_seconds())
    start_monotonic = _monotonic()
    _copy_files(resume_dir, output_dir)

    execution_started: datetime | None = None
    execution_started_mono: float | None = None
    queue_finished_elapsed: float | None = None
    last_event: tuple[str, str, str] | None = None
    status = "queued"
    phase = "queue"
    last_updated = now

    def record(outcome: str, last_error: str | None = None) -> None:
        nonlocal last_event, last_updated
        last_updated = max(last_updated, _utc_now().astimezone(timezone.utc), submitted)
        queue_elapsed_now = initial_elapsed + max(0.0, _monotonic() - start_monotonic)
        execution_elapsed = (
            max(0.0, _monotonic() - execution_started_mono)
            if execution_started_mono is not None
            else 0.0
        )
        if phase != state["phase"]:
            state["phase_started_at"] = _timestamp(last_updated)
        state.update(
            status=status,
            phase=phase,
            outcome=outcome,
            updated_at=_timestamp(last_updated),
            queue_elapsed_seconds=max(
                0.0,
                queue_elapsed_now
                if queue_finished_elapsed is None
                else queue_finished_elapsed,
            ),
            execution_elapsed_seconds=execution_elapsed,
            execution_started_at=(
                _timestamp(execution_started) if execution_started is not None else None
            ),
            last_error=last_error,
        )
        _write_state(output_dir / "kaggle-controller-state.json", state)
        key = (phase, status, outcome)
        if key != last_event:
            event = {
                "event": "controller_state",
                "phase": phase,
                "status": status,
                "outcome": outcome,
                "queue_elapsed_seconds": state["queue_elapsed_seconds"],
                "execution_elapsed_seconds": execution_elapsed,
            }
            _append_log(output_dir / "kaggle-controller.log", event)
            print(json.dumps(event, sort_keys=True), flush=True)
            last_event = key

    def check_deadline() -> None:
        elapsed = initial_elapsed + max(0.0, _monotonic() - start_monotonic)
        execution_elapsed = (
            max(0.0, _monotonic() - execution_started_mono)
            if execution_started_mono is not None
            else 0.0
        )
        if execution_started is None and elapsed >= kaggle_ci.QUEUE_TIMEOUT_SECONDS:
            record("queue_timeout", "Kaggle queue polling timed out")
            raise kaggle_ci.KaggleError("Kaggle total queue wait timed out")
        if (
            execution_started_mono is not None
            and execution_elapsed >= kaggle_ci.EXECUTION_TIMEOUT_SECONDS
        ):
            record("execution_timeout", "Kaggle execution polling timed out")
            raise kaggle_ci.KaggleError("Kaggle execution wait timed out")

    while True:
        check_deadline()
        record("pending")

        try:
            with tempfile.TemporaryDirectory(
                prefix="jaxrenderer-kaggle-wait-"
            ) as temporary:
                collector_dir = Path(temporary) / "collector"
                report = kaggle_collect.collect(binding, state_path, collector_dir)
                _copy_files(
                    collector_dir,
                    output_dir,
                    replace=frozenset({"kaggle-collection-report.json"}),
                )
        except Exception as error:
            record("collection_error", "Kaggle collection failed")
            raise kaggle_ci.KaggleError("Kaggle collection failed") from error

        now = max(last_updated, _utc_now().astimezone(timezone.utc), submitted)
        elapsed = initial_elapsed + max(0.0, _monotonic() - start_monotonic)
        execution_elapsed = (
            max(0.0, _monotonic() - execution_started_mono)
            if execution_started_mono is not None
            else 0.0
        )
        if (
            report.get("binding") != binding
            or report.get("kernel_id") != state["kernel_id"]
        ):
            record("collection_error", "Kaggle collection identity did not match")
            raise kaggle_ci.KaggleError(
                "Kaggle collection identity did not match the handoff"
            )
        if (
            report.get("submitted_version") != 1
            or report.get("version_verified") is not True
        ):
            record("collection_error", "Kaggle collection version was not verified")
            raise kaggle_ci.KaggleError(
                "Kaggle collection did not verify controller version 1"
            )
        check_deadline()

        outcome = report.get("outcome")
        remote_status = report.get("remote_status")
        if outcome == "success" and report.get("success") is True:
            _ = kaggle_ci.validate_downloaded_result(
                output_dir / "result.json", binding
            )
            status = cast(str, remote_status)
            if execution_started is None:
                queue_finished_elapsed = elapsed
            record("success")
            print("Kaggle continuation collected a validated result.", flush=True)
            return 0
        if outcome == "remote_failure" or outcome == "collection_error":
            if isinstance(remote_status, str):
                status = remote_status
            record(
                "remote_terminal_failure"
                if outcome == "remote_failure"
                else "collection_error",
                "Kaggle continuation failed",
            )
            raise kaggle_ci.KaggleError("Kaggle continuation failed")
        if outcome != "pending" or report.get("success") is not False:
            record("collection_error", "Kaggle collection report was invalid")
            raise kaggle_ci.KaggleError("Kaggle collection report was invalid")
        if not isinstance(remote_status, str) or remote_status not in PENDING:
            record("collection_error", "Kaggle returned an unsupported status")
            raise kaggle_ci.KaggleError(
                "Kaggle continuation returned an unsupported status"
            )
        if remote_status in ACTIVE and execution_started is None:
            execution_started = now
            execution_started_mono = _monotonic()
            queue_finished_elapsed = elapsed
            phase = "execution"
        status = remote_status
        record("pending")
        remaining = kaggle_ci.EXECUTION_TIMEOUT_SECONDS - execution_elapsed
        if execution_started_mono is None:
            remaining = kaggle_ci.QUEUE_TIMEOUT_SECONDS - elapsed
        _sleep(min(POLL_INTERVAL_SECONDS, remaining))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    first = commands.add_parser("submit")
    _ = first.add_argument("--binding", type=Path, required=True)
    _ = first.add_argument("--output-dir", type=Path, required=True)
    following = commands.add_parser("continue")
    _ = following.add_argument("--binding", type=Path, required=True)
    _ = following.add_argument("--resume-dir", type=Path, required=True)
    _ = following.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "submit":
            return submit(args.binding, args.output_dir)
        return continue_wait(args.binding, args.resume_dir, args.output_dir)
    except kaggle_ci.KaggleError as error:
        print(f"Kaggle TPU continuation: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
