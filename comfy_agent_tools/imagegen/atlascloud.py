"""Atlas Cloud image generation client for comfy-imagegen."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import ipaddress
import json
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from PIL import Image


ATLAS_PROVIDER = "atlascloud"
DEFAULT_ATLAS_API_BASE = "https://api.atlascloud.ai/api/v1"
DEFAULT_ATLAS_MODEL = "bytedance/seedream-v5.0-lite"
DEFAULT_ATLAS_SIZE = "2048*2048"
DEFAULT_ATLAS_OUTPUT_FORMAT = "jpeg"
DEFAULT_ATLAS_MAX_POLLS = 60
DEFAULT_ATLAS_POLL_INTERVAL = 3.0
DEFAULT_ATLAS_REQUEST_TIMEOUT = 60.0
ATLAS_OUTPUT_FORMATS = ("jpeg", "png")
ATLAS_SIZES = (
    "2048*2048",
    "2304*1728",
    "1728*2304",
    "2848*1600",
    "1600*2848",
    "2496*1664",
    "1664*2496",
    "3136*1344",
    "3072*3072",
    "3456*2592",
    "2592*3456",
    "4096*2304",
    "2304*4096",
    "2496*3744",
    "3744*2496",
    "4704*2016",
)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
USER_AGENT = "comfy-agent-tools/atlascloud"


class AtlasCloudError(RuntimeError):
    """Base Atlas Cloud error with a stable error_type."""

    error_type = "remote_api_error"


class AtlasCloudAuthRequiredError(AtlasCloudError):
    error_type = "auth_required"


class AtlasCloudGenerationFailedError(AtlasCloudError):
    error_type = "remote_generation_failed"


class AtlasCloudTimeoutError(AtlasCloudError):
    error_type = "remote_generation_timeout"


@dataclass(frozen=True)
class AtlasCloudConfig:
    """Runtime configuration for Atlas Cloud image generation."""

    model: str = DEFAULT_ATLAS_MODEL
    size: str = DEFAULT_ATLAS_SIZE
    output_format: str = DEFAULT_ATLAS_OUTPUT_FORMAT
    max_polls: int = DEFAULT_ATLAS_MAX_POLLS
    poll_interval: float = DEFAULT_ATLAS_POLL_INTERVAL
    request_timeout: float = DEFAULT_ATLAS_REQUEST_TIMEOUT
    api_base: str = DEFAULT_ATLAS_API_BASE
    api_key: str | None = None

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("Atlas Cloud model must not be empty")
        if self.size not in ATLAS_SIZES:
            raise ValueError(f"unsupported Atlas Cloud image size: {self.size}")
        if self.output_format not in ATLAS_OUTPUT_FORMATS:
            raise ValueError(f"unsupported Atlas Cloud output format: {self.output_format}")
        if self.max_polls < 1:
            raise ValueError("Atlas Cloud max_polls must be at least 1")
        if self.poll_interval < 0:
            raise ValueError("Atlas Cloud poll_interval must not be negative")
        if self.request_timeout <= 0:
            raise ValueError("Atlas Cloud request_timeout must be positive")
        if not self.effective_api_key:
            raise AtlasCloudAuthRequiredError(
                "ATLASCLOUD_API_KEY is required for Atlas Cloud image generation"
            )

    @property
    def effective_api_key(self) -> str:
        return (self.api_key or os.environ.get("ATLASCLOUD_API_KEY", "")).strip()


def run_generate(
    *,
    prompt: str,
    config: AtlasCloudConfig,
    opener: Callable[..., Any] = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
    download_opener: Any | None = None,
) -> tuple[list[Image.Image], str]:
    """Submit once, poll the prediction, and return downloaded images."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    config.validate()

    prediction = _submit(prompt=prompt, config=config, opener=opener)
    prediction_id = prediction.get("id")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise AtlasCloudError("Atlas Cloud submission did not return a prediction ID")

    completed = _poll(
        prediction=prediction,
        prediction_id=prediction_id,
        config=config,
        opener=opener,
        sleep_fn=sleep_fn,
    )
    outputs = completed.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise AtlasCloudError("Atlas Cloud prediction completed without output URLs")

    images = [
        _download_image(str(output), config=config, opener=download_opener)
        for output in outputs
    ]
    return images, prediction_id


def _submit(*, prompt: str, config: AtlasCloudConfig, opener: Callable[..., Any]) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "prompt": prompt,
        "size": config.size,
        "output_format": config.output_format,
    }
    request = Request(
        f"{config.api_base.rstrip('/')}/model/generateImage",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(config.effective_api_key, json_body=True),
        method="POST",
    )
    return _api_request(request, config=config, opener=opener)


def _poll(
    *,
    prediction: dict[str, Any],
    prediction_id: str,
    config: AtlasCloudConfig,
    opener: Callable[..., Any],
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    current = prediction
    prediction_url = (
        f"{config.api_base.rstrip('/')}/model/prediction/"
        f"{quote(prediction_id, safe='')}"
    )
    for poll_number in range(config.max_polls):
        status = str(current.get("status", "")).lower()
        if status in {"completed", "succeeded"}:
            return current
        if status in {"failed", "canceled", "cancelled"}:
            detail = current.get("error") or current.get("message") or "no error details"
            raise AtlasCloudGenerationFailedError(
                f"Atlas Cloud prediction {status}: {detail}"
            )
        if poll_number:
            sleep_fn(config.poll_interval)

        request = Request(
            prediction_url,
            headers=_headers(config.effective_api_key),
            method="GET",
        )
        current = _prediction_request(
            request,
            config=config,
            opener=opener,
            sleep_fn=sleep_fn,
        )

    raise AtlasCloudTimeoutError(
        f"Atlas Cloud prediction did not complete after {config.max_polls} polls"
    )


def _prediction_request(
    request: Request,
    *,
    config: AtlasCloudConfig,
    opener: Callable[..., Any],
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    for attempt in range(3):
        try:
            return _api_request(request, config=config, opener=opener)
        except AtlasCloudError as exc:
            cause = exc.__cause__
            transient = (
                isinstance(cause, HTTPError)
                and (cause.code == 429 or 500 <= cause.code < 600)
            ) or (
                isinstance(cause, URLError) and not isinstance(cause, HTTPError)
            )
            if not transient or attempt == 2:
                raise
            sleep_fn(float(2**attempt))
    raise AtlasCloudError("Atlas Cloud prediction request exhausted retries")


def _api_request(
    request: Request,
    *,
    config: AtlasCloudConfig,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    try:
        with opener(request, timeout=config.request_timeout) as response:
            payload = _read_json(response)
    except HTTPError as exc:
        raise AtlasCloudError(f"Atlas Cloud request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise AtlasCloudError(f"Atlas Cloud request failed: {exc.reason}") from exc
    return _response_data(payload)


def _read_json(response: Any) -> dict[str, Any]:
    body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise AtlasCloudError("Atlas Cloud response exceeded the 2 MiB limit")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtlasCloudError("Atlas Cloud returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AtlasCloudError("Atlas Cloud returned an invalid response object")
    return payload


def _response_data(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("code") not in (None, 0, 200, "200"):
        message = payload.get("message") or "unknown API error"
        raise AtlasCloudError(f"Atlas Cloud request failed: {message}")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise AtlasCloudError("Atlas Cloud response did not contain prediction data")
    return data


def _download_image(
    raw_url: str,
    *,
    config: AtlasCloudConfig,
    opener: Any | None,
) -> Image.Image:
    url = _validate_output_url(raw_url)
    client = opener or build_opener(_NoRedirectHandler())
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with client.open(request, timeout=config.request_timeout) as response:
            declared = int(response.headers.get("Content-Length", "0") or "0")
            if declared > MAX_IMAGE_BYTES:
                raise AtlasCloudError("Atlas Cloud output exceeded the 64 MiB limit")
            body = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPError as exc:
        raise AtlasCloudError(f"Atlas Cloud output download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise AtlasCloudError(f"Atlas Cloud output download failed: {exc.reason}") from exc
    if len(body) > MAX_IMAGE_BYTES:
        raise AtlasCloudError("Atlas Cloud output exceeded the 64 MiB limit")
    try:
        with Image.open(BytesIO(body)) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise AtlasCloudError("Atlas Cloud output was not a valid image") from exc


def _validate_output_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise AtlasCloudError("Atlas Cloud output URL must be credential-free HTTPS")
    if parsed.port not in (None, 443):
        raise AtlasCloudError("Atlas Cloud output URL must use the default HTTPS port")
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise AtlasCloudError("Atlas Cloud output URL cannot target localhost")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise AtlasCloudError("Atlas Cloud output URL cannot target a non-public address")
    return raw_url


def _headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": USER_AGENT,
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None
