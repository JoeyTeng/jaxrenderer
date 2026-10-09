"""Collect result files from the already submitted, frozen Kaggle TPU kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any
from urllib.parse import urljoin, urlsplit

from tools.kaggle_ci import (
    CONTROLLER_STATE_FIELDS,
    LEGACY_CONTROLLER_STATE_FIELDS,
    KaggleError,
    _validate_binding,
    _validate_downloaded_result,
)

VERSION = "v1"
REQUEST_TIMEOUT = (8, 25)
MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_FILES = 32
MAX_PAGES = 8
MAX_REDIRECTS = 3
MAX_REPORT_ERROR = 400
REQUIRED_FILES = {
    "result.json",
    "setup.log",
    "device-probe.log",
    "full-tests.log",
    "render-gradient-tests.log",
    "render-artifacts/numeric-report.json",
}
FILE_RE = re.compile(
    r"^(?:result\.json|(?:diagnostics|setup|device-probe|full-tests|render-gradient-tests)\.log|render-artifacts/(?:numeric-report\.json|[A-Za-z0-9._-]+\.png))$"
)
KERNEL_ID_RE = re.compile(
    r"^(?P<username>[A-Za-z0-9_.-]{1,50})/(?P<slug>[a-z0-9][a-z0-9-]{2,49})$"
)
ALLOWED_DOWNLOAD_HOSTS = ("kaggleusercontent.com", "storage.googleapis.com")


class CollectionError(RuntimeError):
    """A bounded collection or validation failure."""


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _redact(message: str, token: str) -> str:
    if token:
        message = message.replace(token, "[redacted]")
    return message[:MAX_REPORT_ERROR]


def _load_sdk(token: str) -> tuple[Any, Any, Any, Any]:
    """Import the installed Kaggle SDK only after all local identity checks."""
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_env import KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest,
        ApiGetKernelSessionStatusRequest,
        ApiListKernelSessionOutputRequest,
    )

    return (
        KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False),
        ApiGetKernelRequest,
        ApiGetKernelSessionStatusRequest,
        ApiListKernelSessionOutputRequest,
    )


def _bounded_client(client: Any) -> Any:
    """Apply finite time and response-size bounds to SDK HTTP calls."""
    http_client = client.http_client()
    http_client._init_session()
    session = http_client._session
    session.trust_env = False
    original_send = session.send

    def bounded_send(request: Any, **kwargs: Any) -> Any:
        parsed_request_url = urlsplit(request.url)
        if (
            parsed_request_url.scheme != "https"
            or parsed_request_url.hostname != "api.kaggle.com"
            or parsed_request_url.port not in (None, 443)
        ):
            raise CollectionError("Kaggle API request destination was unexpected")
        kwargs["timeout"] = REQUEST_TIMEOUT
        kwargs["stream"] = True
        kwargs["allow_redirects"] = False
        response = original_send(request, **kwargs)
        try:
            if response.is_redirect:
                raise CollectionError("Kaggle API redirects are not permitted")
            if response.url and urlsplit(response.url).hostname != "api.kaggle.com":
                raise CollectionError("Kaggle API redirected to an unexpected host")
            length = response.headers.get("Content-Length")
            if length:
                try:
                    if int(length) > MAX_API_RESPONSE_BYTES:
                        raise CollectionError(
                            "Kaggle API response exceeded its size limit"
                        )
                except ValueError as error:
                    raise CollectionError(
                        "Kaggle API response length was malformed"
                    ) from error
            body = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                if len(body) + len(chunk) > MAX_API_RESPONSE_BYTES:
                    raise CollectionError("Kaggle API response exceeded its size limit")
                body.extend(chunk)
            response._content = bytes(body)
            response._content_consumed = True
            return response
        finally:
            response.close()

    session.send = bounded_send
    return client


def _credentials(env: dict[str, str]) -> tuple[str, str]:
    token = env.get("KAGGLE_API_TOKEN", "")
    username = env.get("KAGGLE_USERNAME", "")
    if not token or token.isspace():
        raise CollectionError("KAGGLE_API_TOKEN is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,50}", username):
        raise CollectionError(
            "KAGGLE_USERNAME is required and must be a valid account slug"
        )
    return token, username


def _controller_identity(
    binding: dict[str, object], state_path: Path, username: str
) -> tuple[str, int]:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CollectionError("could not read controller resume state") from error
    if not isinstance(state, dict) or set(state) not in (
        LEGACY_CONTROLLER_STATE_FIELDS,
        CONTROLLER_STATE_FIELDS,
    ):
        raise CollectionError("resume state has an unsupported controller schema")
    if state["binding"] != binding:
        raise CollectionError("resume state binding does not exactly match --binding")
    submitted_version = state.get("submitted_version")
    if type(submitted_version) is not int or submitted_version != 1:
        raise CollectionError("resume state kernel identity or version is invalid")
    kernel_id = state.get("kernel_id")
    if not isinstance(kernel_id, str):
        raise CollectionError("resume state kernel identity is invalid")
    owner, _ = _kernel_identity(binding, kernel_id)
    if owner != username:
        raise CollectionError("resume state kernel identity does not match credentials")
    return kernel_id, submitted_version


def _kernel_identity(binding: dict[str, object], kernel_id: object) -> tuple[str, str]:
    """Validate the binding-derived kernel ID independently of submission state."""
    if not isinstance(kernel_id, str):
        raise CollectionError("resume state kernel identity is invalid")
    match = KERNEL_ID_RE.fullmatch(kernel_id)
    if match is None:
        raise CollectionError("resume state kernel identity is invalid")
    run_tag = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    slug = match.group("slug")
    if re.fullmatch(rf"jaxr-{run_tag}-[0-9a-f]{{10}}", slug) is None:
        raise CollectionError("resume state kernel identity is invalid")
    return match.group("username"), slug


def _versioned_metadata(api: Any, request_type: Any, username: str, slug: str) -> int:
    request = request_type()
    request.user_name = username
    request.kernel_slug = slug
    request.version_label = VERSION
    response = api.get_kernel(request)
    metadata = getattr(response, "metadata", None)
    version_number = getattr(metadata, "current_version_number", None)
    if type(version_number) is not int or version_number != 1:
        raise CollectionError(
            "Kaggle kernel latest version is not controller version 1"
        )
    return version_number


def _remote_status(api: Any, request_type: Any, username: str, slug: str) -> str:
    request = request_type()
    request.user_name = username
    request.kernel_slug = slug
    request.version_label = VERSION
    response = api.get_kernel_session_status(request)
    status = getattr(response, "status", None)
    if hasattr(status, "name"):
        raw = status.name
    else:
        raw = str(status)
    raw = raw.rsplit(".", maxsplit=1)[-1].strip().casefold()
    if raw.startswith("kernelworkerstatus_"):
        raw = raw.removeprefix("kernelworkerstatus_")
    if raw == "cancel_acknowledged":
        raw = "cancelled"
    if raw not in {
        "queued",
        "running",
        "starting",
        "compiling",
        "initializing",
        "cancel_requested",
        "complete",
        "completed",
        "success",
        "succeeded",
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "aborted",
    }:
        raise CollectionError("Kaggle returned an unrecognised kernel status")
    return raw


def _allowed_name(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise CollectionError("Kaggle output contained an unsafe file name")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise CollectionError("Kaggle output contained an unsafe file name")
    normalised = pure.as_posix()
    if normalised != name or not FILE_RE.fullmatch(normalised):
        raise CollectionError("Kaggle output contained a non-allowlisted file")
    return normalised


def _safe_download_url(url: object) -> str:
    if not isinstance(url, str):
        raise CollectionError("Kaggle output URL was malformed")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold().rstrip(".")
    allowed = any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ALLOWED_DOWNLOAD_HOSTS
    )
    try:
        valid_port = parsed.port in (None, 443)
    except ValueError as error:
        raise CollectionError("Kaggle output URL used an invalid port") from error
    if (
        parsed.scheme != "https"
        or not allowed
        or parsed.username is not None
        or parsed.password is not None
        or not valid_port
    ):
        raise CollectionError("Kaggle output URL used an untrusted host")
    return url


def _download_file(url: str, destination: Path, remaining: int) -> int:
    import requests

    current = _safe_download_url(url)
    session = requests.Session()
    session.trust_env = False
    try:
        for redirect in range(MAX_REDIRECTS + 1):
            response = session.get(
                current,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )
            if response.is_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location or redirect >= MAX_REDIRECTS:
                    raise CollectionError("Kaggle output redirect limit was exceeded")
                current = _safe_download_url(urljoin(current, location))
                continue
            try:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > min(MAX_FILE_BYTES, remaining):
                            raise CollectionError(
                                "Kaggle output file exceeded its size limit"
                            )
                    except ValueError as error:
                        raise CollectionError(
                            "Kaggle output length was malformed"
                        ) from error
                total = 0
                with destination.open("xb") as output:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > min(MAX_FILE_BYTES, remaining):
                            raise CollectionError(
                                "Kaggle output file exceeded its size limit"
                            )
                        output.write(chunk)
                return total
            finally:
                response.close()
        raise CollectionError("Kaggle output redirect limit was exceeded")
    finally:
        session.close()


def _collect_pages(api: Any, request_type: Any, username: str, slug: str) -> list[Any]:
    files: list[Any] = []
    seen_names: set[str] = set()
    seen_tokens: set[str] = set()
    page_token = ""
    for _ in range(MAX_PAGES):
        request = request_type()
        request.user_name = username
        request.kernel_slug = slug
        request.version_label = VERSION
        request.page_size = min(MAX_FILES, 20)
        request.page_token = page_token
        response = api.list_kernel_session_output(request)
        page_files = getattr(response, "files", None)
        if not isinstance(page_files, list):
            raise CollectionError("Kaggle output listing was malformed")
        if len(files) + len(page_files) > MAX_FILES:
            raise CollectionError("Kaggle output listing exceeded its file limit")
        for item in page_files:
            name = _allowed_name(getattr(item, "file_name", None))
            if name in seen_names:
                raise CollectionError(
                    "Kaggle output listing contained duplicate file names"
                )
            _safe_download_url(getattr(item, "url", None))
            seen_names.add(name)
            files.append(item)
        next_token = getattr(response, "next_page_token", "") or ""
        if not isinstance(next_token, str):
            raise CollectionError("Kaggle output page token was malformed")
        if not next_token:
            return files
        if next_token in seen_tokens:
            raise CollectionError("Kaggle output listing repeated a page token")
        seen_tokens.add(next_token)
        page_token = next_token
    raise CollectionError("Kaggle output listing exceeded its page limit")


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    path = output_dir / "kaggle-collection-report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def collect(
    binding: dict[str, object], resume_state_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Collect only version 1 outputs from the frozen, already-submitted kernel."""
    output_dir = Path(output_dir)
    report: dict[str, Any] = {
        "binding": binding,
        "kernel_id": None,
        "submitted_version": 1,
        "status": "unknown",
        "remote_status": "unknown",
        "requested_version": 1,
        "version_verified": False,
        "version_verification_rationale": (
            "Status and output requests explicitly specify versionLabel=v1; Kaggle "
            "responses do not echo a session or version identity. Metadata current_version_number "
            "is checked before status and after output collection."
        ),
        "success": False,
        "outcome": "collection_error",
        "checked_at": _timestamp(),
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise CollectionError("collector output directory already exists")
    token = ""
    client = None
    try:
        binding = _validate_binding(binding)
        token, username = _credentials(os.environ)
        if Path(resume_state_path).resolve() == output_dir.resolve() or (
            Path(resume_state_path).resolve().parent == output_dir.resolve()
        ):
            raise CollectionError(
                "resume state must be outside the fresh collector directory"
            )
        kernel_id, submitted_version = _controller_identity(
            binding, Path(resume_state_path), username
        )
        report.update(kernel_id=kernel_id, submitted_version=submitted_version)
        match = KERNEL_ID_RE.fullmatch(kernel_id)
        if match is None or match.group("username") != username:
            raise CollectionError(
                "resume state kernel identity does not match credentials"
            )
        slug = match.group("slug")
        client, get_request, status_request, output_request = _load_sdk(token)
        client = _bounded_client(client)
        api = client.kernels.kernels_api_client
        _versioned_metadata(api, get_request, username, slug)
        status = _remote_status(api, status_request, username, slug)
        report.update(status=status, remote_status=status, version_verified=True)
        if status in {
            "queued",
            "running",
            "starting",
            "compiling",
            "initializing",
            "cancel_requested",
        }:
            report.update(outcome="pending", success=False)
        elif status in {
            "error",
            "failed",
            "failure",
            "cancelled",
            "canceled",
            "aborted",
        }:
            report.update(outcome="remote_failure", success=False)
        else:
            entries = _collect_pages(api, output_request, username, slug)
            names = {
                _allowed_name(getattr(item, "file_name", None)) for item in entries
            }
            missing = REQUIRED_FILES - names
            if missing:
                raise CollectionError(
                    "Kaggle output was missing required files: "
                    + ", ".join(sorted(missing))
                )
            if not any(name.endswith(".png") for name in names):
                raise CollectionError("Kaggle output was missing render PNG files")
            total_bytes = 0
            for item in entries:
                name = _allowed_name(getattr(item, "file_name", None))
                destination = output_dir.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                total_bytes += _download_file(
                    getattr(item, "url", None),
                    destination,
                    MAX_TOTAL_BYTES - total_bytes,
                )
                if total_bytes > MAX_TOTAL_BYTES:
                    raise CollectionError(
                        "Kaggle output exceeded its aggregate size limit"
                    )
            _versioned_metadata(api, get_request, username, slug)
            _validate_downloaded_result(output_dir / "result.json", binding)
            report.update(outcome="success", success=True)
    except Exception as error:
        if isinstance(error, (CollectionError, KaggleError)):
            detail = str(error)
        else:
            detail = f"Kaggle collection operation failed ({type(error).__name__})"
        message = _redact(detail, token)
        report.update(outcome="collection_error", success=False, last_error=message)
    finally:
        if client is not None and hasattr(client, "http_client"):
            http_client = client.http_client()
            session = getattr(http_client, "_session", None)
            if session is not None:
                session.close()
    _write_report(output_dir, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--binding", type=Path, required=True)
    root.add_argument("--resume-state", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    try:
        try:
            binding = _validate_binding(
                json.loads(args.binding.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CollectionError("could not read binding file") from error
        report = collect(binding, args.resume_state, args.output_dir)
        print(json.dumps(report, sort_keys=True))
        return (
            0
            if report.get("success")
            else (2 if report.get("outcome") == "pending" else 1)
        )
    except Exception as error:
        print(_redact(str(error), token), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
