"""Recover the read-only status and output of the first PR 25 Kaggle run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from tools.accelerator_gate import GateError, _load_result
from tools.kaggle_ci import (
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
    KaggleError,
    _command_ok,
    _load_binding_file,
    _require_credentials,
    _status,
)

EXPECTED_BINDING = {
    "pr": 25,
    "head_sha": "e84b28fd8f83b13b332c7f602409c907f5bb0715",
    "base_sha": "90d1d450fca92b6b0ae3ae053a63a3c73a7fb3a3",
    "base_ref": "master",
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "37278198939.1",
}
OUTPUT_PATTERN = (
    r"^(?:result\.json|(?:diagnostics|setup|device-probe|full-tests|"
    r"render-gradient-tests)\.log|render-artifacts/(?:numeric-report\.json|"
    r"[A-Za-z0-9._-]+\.png))$"
)


class RecoveryPending(KaggleError):
    """The original remote kernel is still queued or running."""


def _prefix(binding: dict[str, object]) -> str:
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
    return f"jaxr-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}-"


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".recovery-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(state, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def recover(binding_path: Path, output_dir: Path) -> dict[str, object]:
    binding = _load_binding_file(binding_path)
    if binding != EXPECTED_BINDING:
        raise KaggleError("binding does not match the frozen PR 25 bootstrap run")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        raise KaggleError("recovery output directory already contains result.json")
    env = os.environ.copy()
    username = _require_credentials(env)
    tag = _prefix(binding)
    pattern = re.compile(rf"^{re.escape(username)}/{re.escape(tag)}[0-9a-f]{{10}}$")
    with tempfile.TemporaryDirectory(
        prefix="jaxrenderer-kaggle-recovery-"
    ) as temporary:
        config = Path(temporary) / "config"
        config.mkdir()
        env["KAGGLE_CONFIG_DIR"] = str(config)
        listed = _command_ok(
            [
                "kaggle",
                "kernels",
                "list",
                "--mine",
                "--search",
                tag,
                "--page-size",
                "200",
                "--format",
                "json",
            ],
            env=env,
        ).stdout
        if "Next Page Token" in listed:
            raise KaggleError(
                "Kaggle search was paginated; refusing an incomplete candidate set"
            )
        try:
            rows = json.loads(listed)
        except json.JSONDecodeError as error:
            if listed.strip() == "Not found":
                rows = []
            else:
                raise KaggleError(
                    "Kaggle kernel search returned malformed JSON"
                ) from error
        if not isinstance(rows, list):
            raise KaggleError("Kaggle kernel search did not return a row list")
        candidates = [
            row["ref"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("ref"), str)
            and pattern.fullmatch(row["ref"])
        ]
        if len(candidates) != 1:
            raise KaggleError(
                f"expected one matching old kernel, found {len(candidates)}"
            )
        kernel_id = candidates[0]
        status_output = _command_ok(
            ["kaggle", "kernels", "status", kernel_id], env=env
        ).stdout
        state = _status(status_output)
        report: dict[str, object] = {
            "binding": binding,
            "kernel_id": kernel_id,
            "status": state,
        }
        report_path = output_dir / "recovery-state.json"
        _write_state(report_path, report)
        if state not in TERMINAL_SUCCESS | TERMINAL_FAILURE:
            raise RecoveryPending(
                f"old kernel {kernel_id} is {state}; no output was downloaded"
            )
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
                OUTPUT_PATTERN,
            ],
            env=env,
        )
        try:
            _load_result(output_dir / "result.json", binding, "tpu")
        except GateError as error:
            report["result_validation_error"] = str(error)
            _write_state(report_path, report)
            raise KaggleError(
                f"old kernel output did not prove bound TPU success: {error}"
            ) from error
        report["result_validated"] = True
        _write_state(report_path, report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(recover(args.binding, args.output_dir), sort_keys=True))
        return 0
    except RecoveryPending as error:
        print(f"Kaggle recovery: {error}", file=sys.stderr)
        return 2
    except (KaggleError, GateError) as error:
        print(f"Kaggle recovery: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
