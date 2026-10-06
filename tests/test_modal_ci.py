from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any

import pytest
from tools import modal_ci

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


def _result(**overrides: Any) -> dict[str, object]:
    result: dict[str, object] = {
        "pr": 24,
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "run_id": "371234567.1",
        "backend": "gpu",
        "device_backend": "gpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "gpu", "device_kind": "T4", "id": "0"}],
        "versions": {"python": "3.14.1", "jax": "0.11.2"},
    }
    result.update(overrides)
    return result


def _release_result(**overrides: Any) -> dict[str, object]:
    result: dict[str, object] = {
        **RELEASE_BINDING,
        "backend": "gpu",
        "device_backend": "gpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "gpu", "device_kind": "T4", "id": "0"}],
        "versions": {"python": "3.14.1", "jax": "0.11.2"},
    }
    result.update(overrides)
    return result


def test_preflight_requires_modal_credentials_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(modal_ci.ModalCIError, match="Modal credentials"):
        modal_ci._preflight_credentials({})


def test_result_must_match_frozen_binding_and_report_real_gpu() -> None:
    accepted = modal_ci._validate_result(_result(), GOOD_BINDING)
    assert accepted["success"] is True
    with pytest.raises(modal_ci.ModalCIError, match="head_sha"):
        modal_ci._validate_result(_result(head_sha="c" * 40), GOOD_BINDING)
    with pytest.raises(modal_ci.ModalCIError, match="requested GPU"):
        modal_ci._validate_result(
            _result(device_backend="cpu", devices=[], device_count=0), GOOD_BINDING
        )


def test_result_accepts_exact_release_binding_and_requires_versions() -> None:
    accepted = modal_ci._validate_result(_release_result(), RELEASE_BINDING)
    assert accepted["success"] is True
    with pytest.raises(modal_ci.ModalCIError, match="head_sha"):
        modal_ci._validate_result(_release_result(head_sha="d" * 40), RELEASE_BINDING)
    with pytest.raises(modal_ci.ModalCIError, match="unexpected fields"):
        modal_ci._validate_result(
            _release_result(pr=25, base_sha="b" * 40), RELEASE_BINDING
        )
    with pytest.raises(modal_ci.ModalCIError, match="version metadata"):
        modal_ci._validate_result(_release_result(versions={}), RELEASE_BINDING)


def test_failed_remote_result_is_preserved_for_controller_failure_status() -> None:
    result = _result(
        device_backend="unknown",
        device_count=0,
        success=False,
        devices=[],
        versions={},
    )
    assert modal_ci._validate_result(result, GOOD_BINDING)["success"] is False


def test_artifact_archive_contains_files_and_rejects_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "remote-output"
    output.mkdir()
    (output / "result.json").write_text("{}\n", encoding="utf-8")
    archive_bytes = modal_ci._pack_artifacts(output)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        assert [member.name for member in archive.getmembers()] == ["result.json"]

    (output / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(modal_ci.ModalCIError, match="symbolic links"):
        modal_ci._pack_artifacts(output)


def test_output_identity_detects_replacement(tmp_path: Path) -> None:
    output, identity = modal_ci._prepare_output(tmp_path / "out")
    moved = tmp_path / "old-out"
    output.rename(moved)
    output.mkdir()
    with pytest.raises(modal_ci.ModalCIError, match="identity or ownership changed"):
        modal_ci._check_output_identity(output, identity)


def test_app_factory_requests_only_trusted_sources_and_one_t4() -> None:
    class FakeImage:
        def __init__(self) -> None:
            self.files: list[tuple[Path, str]] = []

        @classmethod
        def debian_slim(cls, *, python_version: str) -> FakeImage:
            assert python_version == "3.14"
            return cls()

        def apt_install(self, *_packages: str) -> FakeImage:
            return self

        def pip_install(self, package: str) -> FakeImage:
            assert package == "uv==0.12.20"
            return self

        def add_local_file(self, local_path: Path, remote_path: str) -> FakeImage:
            self.files.append((local_path, remote_path))
            return self

    class FakeApp:
        def __init__(self, name: str, *, include_source: bool) -> None:
            assert name == "jaxrenderer-accelerator-ci"
            assert include_source is False
            self.options: dict[str, object] = {}

        def function(self, **options: object):
            self.options = options

            def decorate(function):
                return function

            return decorate

    class FakeModal:
        Image = FakeImage

        def App(self, name: str, *, include_source: bool) -> FakeApp:
            return FakeApp(name, include_source=include_source)

    app, function = modal_ci._create_modal_app(FakeModal())
    assert callable(function)
    assert app.options["gpu"] == "T4"
    assert app.options["timeout"] == 1_800
    assert app.options["include_source"] is False
    source_files = dict(app.options["image"].files)
    assert set(source_files.values()) == {
        "/root/tools/modal_ci.py",
        "/root/tools/accelerator_runner.py",
    }
    assert all(path.is_file() for path in source_files)


def test_execute_records_credential_preflight_failure(tmp_path: Path) -> None:
    with pytest.raises(modal_ci.ModalCIError, match="Modal credentials"):
        modal_ci.execute(GOOD_BINDING, tmp_path / "output")
    result = json.loads(
        (tmp_path / "output" / "result.json").read_text(encoding="utf-8")
    )
    assert result["success"] is False
    assert (tmp_path / "output" / "diagnostics.log").is_file()


def test_release_execute_failure_manifest_keeps_release_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(modal_ci.ModalCIError, match="Modal credentials"):
        modal_ci.execute(RELEASE_BINDING, tmp_path / "release-output")
    result = json.loads(
        (tmp_path / "release-output" / "result.json").read_text(encoding="utf-8")
    )
    assert result["kind"] == "release"
    assert result["head_sha"] == RELEASE_BINDING["head_sha"]
    assert "pr" not in result
    assert "base_sha" not in result
