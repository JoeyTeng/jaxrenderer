"""Exercise the built-wheel version check used by release preparation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


def test_candidate_version_drives_the_installed_wheel_smoke_check(
    tmp_path: Path,
) -> None:
    workflow = yaml.load(
        (WORKFLOWS / "pypi.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(workflow, dict)
    prepare = cast(dict[str, Any], workflow)["jobs"]["prepare"]
    assert prepare["outputs"]["version"] == "${{ steps.candidate.outputs.version }}"

    smoke = next(
        step
        for step in cast(dict[str, Any], workflow)["jobs"]["build"]["steps"]
        if "PACKAGE_VERSION" in step.get("env", {})
    )
    assert smoke["env"]["PACKAGE_VERSION"] == "${{ needs.prepare.outputs.version }}"

    smoke_lines = smoke["run"].splitlines()
    heredoc_index = next(
        index for index, line in enumerate(smoke_lines) if "<<" in line
    )
    delimiter = smoke_lines[heredoc_index].partition("<<")[2].strip().strip("'\"")
    body_start = heredoc_index + 1
    body_end = next(
        index
        for index in range(body_start, len(smoke_lines))
        if smoke_lines[index] == delimiter
    )
    smoke_script = "\n".join(smoke_lines[body_start:body_end])
    site_packages = tmp_path / "site-packages"
    renderer = site_packages / "renderer"
    renderer.mkdir(parents=True)
    _ = (renderer / "__init__.py").write_text("", encoding="utf-8")
    distribution = site_packages / "jaxrenderer-1.2.3.dist-info"
    distribution.mkdir()
    _ = (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: jaxrenderer\nVersion: 1.2.3\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(site_packages)
    environment["RELEASE_TAG"] = ""
    for expected, returncode in (("1.2.3", 0), ("9.9.9", 1)):
        environment["PACKAGE_VERSION"] = expected
        result = subprocess.run(
            [sys.executable, "-c", smoke_script],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert result.returncode == returncode, result.stderr
        if returncode:
            assert "AssertionError" in result.stderr
