"""Check that the built and installed package exposes its public types."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import cast
import zipfile

PYRIGHT_TIMEOUT_SECONDS = 60
MARKER_PATH = "renderer/py.typed"


def _check_wheel(wheel_path: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        if MARKER_PATH not in names:
            raise RuntimeError(f"wheel is missing {MARKER_PATH}")

        record_paths = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(record_paths) != 1:
            raise RuntimeError("wheel must contain exactly one dist-info/RECORD")
        with wheel.open(record_paths[0], "r") as record_file:
            rows = csv.reader(line.decode("utf-8") for line in record_file)
            if MARKER_PATH not in {row[0] for row in rows if row}:
                raise RuntimeError(f"wheel RECORD does not list {MARKER_PATH}")


def _check_installed_marker() -> None:
    distribution = importlib.metadata.distribution("jaxrenderer")
    installed_files = distribution.files or ()
    marker = next(
        (path for path in installed_files if path.as_posix() == MARKER_PATH), None
    )
    if marker is None:
        raise RuntimeError(f"installed distribution is missing {MARKER_PATH}")
    marker_path = Path(str(distribution.locate_file(marker))).resolve()
    if not marker_path.is_file():
        raise RuntimeError(f"installed marker is not a file: {marker_path}")

    package_spec = importlib.util.find_spec("renderer")
    if package_spec is None or package_spec.origin is None:
        raise RuntimeError("could not locate the installed renderer package")
    installed_init = Path(
        str(distribution.locate_file("renderer/__init__.py"))
    ).resolve()
    if Path(package_spec.origin).resolve() != installed_init:
        raise RuntimeError(
            "renderer resolves outside the installed distribution: "
            f"{package_spec.origin}"
        )


def _check_consumer_types() -> None:
    consumer_source = """\
from renderer.types import DtypeInfo

info: DtypeInfo[int] = DtypeInfo(min=0, max=1, bits=8, dtype=int)
valid_bits: int = info.bits
invalid_bits: str = info.bits
"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    with tempfile.TemporaryDirectory(prefix="jaxrenderer-typing-") as temp_name:
        temp_dir = Path(str(temp_name))
        consumer_path = temp_dir / "consumer.py"
        _ = consumer_path.write_text(consumer_source, encoding="utf-8")
        config_path = temp_dir / "pyrightconfig.json"
        _ = config_path.write_text(
            json.dumps(
                {
                    "include": ["consumer.py"],
                    "typeCheckingMode": "strict",
                    "useLibraryCodeForTypes": False,
                }
            ),
            encoding="utf-8",
        )

        environment = os.environ.copy()
        _ = environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyright",
                "--outputjson",
                "--pythonpath",
                sys.executable,
                "--pythonversion",
                python_version,
                "--project",
                str(config_path),
            ],
            cwd=temp_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=PYRIGHT_TIMEOUT_SECONDS,
            check=False,
        )

    if result.stderr.strip():
        raise RuntimeError(f"Pyright wrote to stderr: {result.stderr.strip()}")
    try:
        report: object = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Pyright did not return valid JSON diagnostics") from error

    if not isinstance(report, dict):
        raise RuntimeError("Pyright JSON report must be an object")
    report_object = cast(dict[str, object], report)
    diagnostics = report_object.get("generalDiagnostics")
    if not isinstance(diagnostics, list):
        raise RuntimeError("Pyright JSON omitted generalDiagnostics")
    diagnostic_values = cast(list[object], diagnostics)
    expected: list[dict[str, object]] = []
    for diagnostic_value in diagnostic_values:
        diagnostic: object = diagnostic_value
        if not isinstance(diagnostic, dict):
            raise RuntimeError("Pyright JSON contains a malformed diagnostic")
        diagnostic_object = cast(dict[str, object], diagnostic)
        location = diagnostic_object.get("range")
        if not isinstance(location, dict):
            continue
        location_object = cast(dict[str, object], location)
        start = location_object.get("start")
        if not isinstance(start, dict):
            continue
        start_object = cast(dict[str, object], start)
        if (
            diagnostic_object.get("rule") == "reportAssignmentType"
            and start_object.get("line") == 4
        ):
            expected.append(diagnostic_object)
    if len(expected) != 1 or len(diagnostic_values) != 1:
        rendered = json.dumps(diagnostic_values, sort_keys=True)
        raise RuntimeError(
            "expected exactly one reportAssignmentType diagnostic on the "
            f"intentional invalid assignment; got {rendered}"
        )
    if result.returncode != 1:
        raise RuntimeError(
            "expected Pyright to report one type error and exit 1; "
            f"got exit status {result.returncode}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("wheel", type=Path, help="built jaxrenderer wheel")
    args = parser.parse_args()

    _check_wheel(args.wheel)
    _check_installed_marker()
    _check_consumer_types()
    print("Installed wheel exposes and enforces renderer type information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
