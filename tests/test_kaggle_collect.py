from __future__ import annotations

import builtins
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
from tools import kaggle_collect as collector

BINDING: dict[str, object] = {
    "kind": "release",
    "head_sha": "a" * 40,
    "head_repository": "JoeyTeng/jaxrenderer",
    "run_id": "12345.1",
}
# Catalog fixture: api-key-a from joey-private-v3.
TOKEN = "codex_synth_v1_api_key_a"
FILES = [
    "result.json",
    "diagnostics.log",
    "setup.log",
    "device-probe.log",
    "full-tests.log",
    "render-gradient-tests.log",
    "render-artifacts/numeric-report.json",
    "render-artifacts/regression.png",
]


def _helper(name: str) -> Callable[..., Any]:
    return cast(Callable[..., Any], getattr(collector, name))


def _fake_requests_module(session_factory: Callable[[], Any]) -> ModuleType:
    module = ModuleType("requests")
    setattr(module, "Session", session_factory)
    return module


def test_module_import_does_not_require_requests_or_kagglesdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_optional_dependencies(
        name: str, *args: Any, **kwargs: Any
    ) -> Any:
        if name.split(".", maxsplit=1)[0] in {"requests", "kagglesdk"}:
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_optional_dependencies)
    namespace = runpy.run_path(
        str(Path(collector.__file__)), run_name="collector_probe"
    )

    assert callable(namespace["collect"])


def _kernel_id() -> str:
    run_tag = hashlib.sha256(
        json.dumps(BINDING, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"joey/jaxr-{run_tag}-0123456789"


def _resume_state(
    path: Path, *, kernel_id: str | None = None, version: Any = 1
) -> None:
    path.write_text(
        json.dumps(
            {
                "binding": BINDING,
                "kernel_id": kernel_id or _kernel_id(),
                "submitted_version": version,
            }
        ),
        encoding="utf-8",
    )


class _Request:
    def __init__(self) -> None:
        self.user_name = ""
        self.kernel_slug = ""
        self.version_label = ""
        self.page_size = 0
        self.page_token = ""


class _Api:
    def __init__(self, files: list[str], status: str = "COMPLETE") -> None:
        self.files = files
        self.status = status
        self.metadata_requests: list[_Request] = []
        self.status_requests: list[_Request] = []
        self.output_requests: list[_Request] = []

    def get_kernel(self, request: _Request) -> Any:
        self.metadata_requests.append(request)
        return SimpleNamespace(metadata=SimpleNamespace(current_version_number=1))

    def get_kernel_session_status(self, request: _Request) -> Any:
        self.status_requests.append(request)
        return SimpleNamespace(status=self.status)

    def list_kernel_session_output(self, request: _Request) -> Any:
        self.output_requests.append(request)
        return SimpleNamespace(
            files=[
                SimpleNamespace(
                    file_name=name,
                    url=f"https://storage.googleapis.com/kaggle-output/{name.replace('/', '_')}",
                )
                for name in self.files
            ],
            next_page_token="",
        )


def _success_result() -> dict[str, Any]:
    return {
        **BINDING,
        "backend": "tpu",
        "device_backend": "tpu",
        "device_count": 1,
        "success": True,
        "devices": [{"platform": "tpu", "device_kind": "TPU v5e", "id": "0"}],
        "versions": {"jax": "0.0"},
    }


def _install_fake_api(
    monkeypatch: pytest.MonkeyPatch,
    files: list[str] | None = None,
    status: str = "COMPLETE",
) -> _Api:
    api = _Api(FILES if files is None else files, status)
    client = SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=api))

    def load_sdk(
        token: str,
    ) -> tuple[Any, type[_Request], type[_Request], type[_Request]]:
        return client, _Request, _Request, _Request

    def bounded_client(value: Any) -> Any:
        return value

    monkeypatch.setattr(collector, "_load_sdk", load_sdk)
    monkeypatch.setattr(collector, "_bounded_client", bounded_client)
    return api


def _install_files(monkeypatch: pytest.MonkeyPatch) -> None:
    def download(url: str, destination: Path, remaining: int) -> int:
        name = destination.relative_to(destination.parents[1]).as_posix()
        if destination.name == "result.json":
            data = json.dumps(_success_result()).encode()
        elif destination.suffix == ".json":
            data = b'{"passed": true}\n'
        elif destination.suffix == ".png":
            data = b"png-data"
        else:
            data = f"{name}: passed\n".encode()
        destination.write_bytes(data)
        return len(data)

    monkeypatch.setattr(collector, "_download_file", download)


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    monkeypatch.setenv("KAGGLE_API_TOKEN", TOKEN)
    monkeypatch.setenv("KAGGLE_USERNAME", "joey")
    state = tmp_path / "resume.json"
    output = tmp_path / "collected"
    _resume_state(state)
    return state, output


def test_collect_requests_version_one_and_keeps_frozen_result_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _install_fake_api(monkeypatch)
    _install_files(monkeypatch)
    state, output = _prepare(monkeypatch, tmp_path)

    report = collector.collect(BINDING, state, output)

    assert report["success"] is True
    assert report["requested_version"] == 1
    assert report["version_verified"] is True
    assert report["outcome"] == "success"
    assert len(api.metadata_requests) == 2
    assert all(request.version_label == "1" for request in api.metadata_requests)
    assert len(api.status_requests) == 1
    assert api.status_requests[0].version_label == "1"
    assert len(api.output_requests) == 1
    assert api.output_requests[0].version_label == "1"
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["head_sha"] == BINDING["head_sha"]
    assert (
        json.loads((output / "kaggle-collection-report.json").read_text())["success"]
        is True
    )


@pytest.mark.parametrize(
    ("remote_status", "outcome"),
    [("QUEUED", "pending"), ("RUNNING", "pending"), ("ERROR", "remote_failure")],
)
def test_collect_pending_and_remote_failure_do_not_fetch_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_status: str,
    outcome: str,
) -> None:
    api = _install_fake_api(monkeypatch, status=remote_status)
    state, output = _prepare(monkeypatch, tmp_path)

    report = collector.collect(BINDING, state, output)

    assert report["outcome"] == outcome
    assert report["success"] is False
    assert api.output_requests == []
    assert (output / "kaggle-collection-report.json").is_file()


@pytest.mark.parametrize(
    ("kernel_id", "version"),
    [("other/jaxr-invalid-0123456789", 1), (_kernel_id(), True)],
)
def test_resume_state_identity_mismatch_fails_before_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kernel_id: str,
    version: Any,
) -> None:
    state, output = _prepare(monkeypatch, tmp_path)
    _resume_state(state, kernel_id=kernel_id, version=version)

    report = collector.collect(BINDING, state, output)

    assert report["success"] is False
    assert "resume state kernel identity" in report["last_error"]
    assert report["kernel_id"] is None


def test_full_controller_state_schema_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_api(monkeypatch, status="QUEUED")
    state, output = _prepare(monkeypatch, tmp_path)
    contents = json.loads(state.read_text(encoding="utf-8"))
    contents.update(
        {
            "status": "running",
            "phase": "execution",
            "phase_started_at": "2026-10-09T00:00:00Z",
            "submitted_at": "2026-10-09T00:00:00Z",
            "execution_started_at": None,
            "updated_at": "2026-10-09T00:00:00Z",
            "queue_elapsed_seconds": 0,
            "execution_elapsed_seconds": 0,
            "outcome": "submitted",
            "last_error": None,
        }
    )
    state.write_text(json.dumps(contents), encoding="utf-8")

    report = collector.collect(BINDING, state, output)

    assert report["outcome"] == "pending"
    assert report["kernel_id"] == _kernel_id()


@pytest.mark.parametrize(
    "name",
    [
        "../result.json",
        "result.json/../../x",
        "render-artifacts/extra.txt",
        "C:\\result.json",
    ],
)
def test_output_paths_are_strictly_allowlisted(name: str) -> None:
    with pytest.raises(collector.CollectionError):
        _helper("_allowed_name")(name)


@pytest.mark.parametrize(
    "url",
    [
        "http://storage.googleapis.com/file",
        "https://evil.example/file",
        "https://user@storage.googleapis.com/file",
        "https://storage.googleapis.com:444/file",
    ],
)
def test_download_urls_require_trusted_https_hosts(url: str) -> None:
    with pytest.raises(collector.CollectionError):
        _helper("_safe_download_url")(url)


def test_duplicate_and_missing_outputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _install_fake_api(monkeypatch, files=FILES + ["setup.log"])
    state, output = _prepare(monkeypatch, tmp_path)

    duplicate_report = collector.collect(BINDING, state, output)

    assert duplicate_report["success"] is False
    assert "duplicate file names" in duplicate_report["last_error"]
    assert api.output_requests[0].version_label == "1"

    state2 = tmp_path / "resume-2.json"
    _resume_state(state2)
    api2 = _install_fake_api(
        monkeypatch, files=[name for name in FILES if name != "setup.log"]
    )
    output2 = tmp_path / "collected-2"
    missing_report = collector.collect(BINDING, state2, output2)
    assert missing_report["success"] is False
    assert "missing required files" in missing_report["last_error"]
    assert api2.output_requests


def test_bad_kernel_metadata_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = _install_fake_api(monkeypatch)
    api.get_kernel = lambda request: SimpleNamespace(
        metadata=SimpleNamespace(current_version_number=2)
    )
    state, output = _prepare(monkeypatch, tmp_path)

    report = collector.collect(BINDING, state, output)

    assert report["success"] is False
    assert "latest version is not controller version 1" in report["last_error"]
    assert api.status_requests == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "999.1", "run_id did not match"),
        ("backend", "cpu", "TPU device backend"),
        ("devices", [], "no TPU devices"),
        ("success", False, "did not report success"),
    ],
)
def test_result_binding_backend_devices_and_success_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    _install_fake_api(monkeypatch)

    def download(url: str, destination: Path, remaining: int) -> int:
        if destination.name == "result.json":
            result = _success_result()
            result[field] = value
            if field == "devices":
                result["device_count"] = len(value)
            data = json.dumps(result).encode()
        else:
            data = b"ok"
        destination.write_bytes(data)
        return len(data)

    monkeypatch.setattr(collector, "_download_file", download)
    state, output = _prepare(monkeypatch, tmp_path)

    report = collector.collect(BINDING, state, output)

    assert report["success"] is False
    assert message in report["last_error"]


def test_collector_refuses_existing_output_and_redacts_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, output = _prepare(monkeypatch, tmp_path)
    output.mkdir()
    (output / "kaggle-collection-report.json").write_text("old", encoding="utf-8")
    with pytest.raises(collector.CollectionError, match="already exists"):
        collector.collect(BINDING, state, output)
    assert (output / "kaggle-collection-report.json").read_text() == "old"

    api = _install_fake_api(monkeypatch)
    api.get_kernel = lambda request: (_ for _ in ()).throw(
        RuntimeError(f"request failed with {TOKEN}")
    )
    output2 = tmp_path / "fresh"
    report = collector.collect(BINDING, state, output2)
    assert TOKEN not in json.dumps(report)
    assert TOKEN not in (output2 / "kaggle-collection-report.json").read_text()


def test_download_size_limit_is_enforced_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Response:
        is_redirect = False
        headers = {"Content-Length": str(collector.MAX_FILE_BYTES + 1)}

        def raise_for_status(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _Session:
        def get(self, *args: Any, **kwargs: Any) -> _Response:
            return _Response()

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(_Session))
    destination = tmp_path / "oversized.log"
    with pytest.raises(collector.CollectionError, match="size limit"):
        _helper("_download_file")(
            "https://storage.googleapis.com/file",
            destination,
            collector.MAX_TOTAL_BYTES,
        )
    assert not destination.exists()


def test_sdk_uses_explicit_production_environment_and_serializes_version_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("kagglesdk")
    monkeypatch.setenv("KAGGLE_API_ENVIRONMENT", "STAGING")
    client, get_type, status_type, output_type = _helper("_load_sdk")(TOKEN)
    http_client = client.http_client()
    assert http_client._endpoint == "https://api.kaggle.com"
    assert http_client._api_token == TOKEN

    for request_type in (get_type, status_type, output_type):
        request = request_type()
        request.user_name = "joey"
        request.kernel_slug = "jaxr-test-0123456789"
        request.version_label = "1"
        serialized = request_type.to_dict(request)
        assert serialized["versionLabel"] == "1"
    client.__exit__(None, None, None)


def test_sdk_http_is_bounded_without_ambient_redirect_or_environment_auth() -> None:
    class _Response:
        url = "https://api.kaggle.com/api/v1/kernels/status"
        headers = {"Content-Length": "2"}
        is_redirect = False
        closed = False
        _content = b""

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b"{}"]

        def close(self) -> None:
            self.closed = True

    class _Session:
        trust_env = True
        sent: list[tuple[Any, dict[str, Any]]] = []

        def send(self, request: Any, **kwargs: Any) -> _Response:
            self.sent.append((request, kwargs))
            return _Response()

    session = _Session()
    http_client = SimpleNamespace(_session=session, _init_session=lambda: None)
    client = SimpleNamespace(http_client=lambda: http_client)
    _helper("_bounded_client")(client)
    request = SimpleNamespace(url="https://api.kaggle.com/api/v1/kernels/status")
    response = session.send(request)

    assert session.trust_env is False
    assert session.sent[0][1]["allow_redirects"] is False
    assert session.sent[0][1]["timeout"] == collector.REQUEST_TIMEOUT
    assert session.sent[0][1]["stream"] is True
    assert getattr(response, "_content") == b"{}"


def test_sdk_redirect_is_rejected_before_following() -> None:
    class _Response:
        url = "https://api.kaggle.com/api/v1/kernels/status"
        headers = {}
        is_redirect = True
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Session:
        trust_env = True
        response = _Response()
        kwargs: dict[str, Any] = {}

        def send(self, request: Any, **kwargs: Any) -> _Response:
            self.kwargs = kwargs
            return self.response

    session = _Session()
    client = SimpleNamespace(
        http_client=lambda: SimpleNamespace(
            _session=session, _init_session=lambda: None
        )
    )
    _helper("_bounded_client")(client)
    request = SimpleNamespace(url="https://api.kaggle.com/api/v1/kernels/status")

    with pytest.raises(collector.CollectionError, match="redirects are not permitted"):
        session.send(request)
    assert session.kwargs["allow_redirects"] is False
    assert session.response.closed is True


def test_sdk_rejects_unexpected_destination_before_network_call() -> None:
    class _Session:
        trust_env = True

        def send(self, request: Any, **kwargs: Any) -> Any:
            raise AssertionError("unexpected destination reached the network")

    session = _Session()
    client = SimpleNamespace(
        http_client=lambda: SimpleNamespace(
            _session=session, _init_session=lambda: None
        )
    )
    _helper("_bounded_client")(client)

    with pytest.raises(collector.CollectionError, match="destination was unexpected"):
        session.send(SimpleNamespace(url="https://staging.kaggle.com/api"))


def test_download_session_disables_environment_auth_and_closes_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Response:
        is_redirect = False
        headers = {"Content-Length": "4"}
        closed = False

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b"data"]

        def close(self) -> None:
            self.closed = True

    class _Session:
        trust_env = True
        auth = None

        def __init__(self) -> None:
            self.response = _Response()
            self.call: tuple[tuple[Any, ...], dict[str, Any]] | None = None

        def get(self, *args: Any, **kwargs: Any) -> _Response:
            self.call = (args, kwargs)
            return self.response

        def close(self) -> None:
            pass

    session = _Session()
    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(lambda: session))
    destination = tmp_path / "download.log"

    count = _helper("_download_file")(
        "https://storage.googleapis.com/file", destination, collector.MAX_TOTAL_BYTES
    )

    assert count == 4
    assert session.trust_env is False
    assert session.auth is None
    assert session.call is not None
    assert "auth" not in session.call[1]
    assert session.call[1]["allow_redirects"] is False
    assert session.call[1]["timeout"] == collector.REQUEST_TIMEOUT
    assert session.response.closed is True


def test_download_redirect_is_validated_before_following(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        is_redirect = True
        headers = {"Location": "https://evil.example/output"}
        closed = False

        def close(self) -> None:
            self.closed = True

    class _Session:
        trust_env = True
        response = _Response()
        calls = 0

        def get(self, *args: Any, **kwargs: Any) -> _Response:
            self.calls += 1
            assert kwargs["allow_redirects"] is False
            return self.response

        def close(self) -> None:
            pass

    session = _Session()
    monkeypatch.setitem(sys.modules, "requests", _fake_requests_module(lambda: session))
    with pytest.raises(collector.CollectionError, match="untrusted host"):
        _helper("_download_file")(
            "https://storage.googleapis.com/file",
            Path("unused.log"),
            collector.MAX_TOTAL_BYTES,
        )
    assert session.calls == 1
    assert session.response.closed is True
