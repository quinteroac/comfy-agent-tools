"""MuAPI image generation client for comfy-imagegen."""

from __future__ import annotations

import ipaddress
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from PIL import Image

MUAPI_PROVIDER = "muapi"
DEFAULT_MUAPI_API_BASE = "https://api.muapi.ai/api/v1"
DEFAULT_MUAPI_MODEL = "flux-dev"
DEFAULT_MUAPI_WIDTH = 1024
DEFAULT_MUAPI_HEIGHT = 1024
DEFAULT_MUAPI_NUMBER_OF_IMAGES = 1
DEFAULT_MUAPI_MAX_POLLS = 60
DEFAULT_MUAPI_POLL_INTERVAL = 3.0
DEFAULT_MUAPI_REQUEST_TIMEOUT = 60.0
MUAPI_MODELS = ("flux-dev",)
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
USER_AGENT = "comfy-agent-tools/muapi"


class MuAPIError(RuntimeError):
    """Base MuAPI error with a stable error_type."""

    error_type = "remote_api_error"


class MuAPIAuthRequiredError(MuAPIError):
    error_type = "auth_required"


class MuAPIGenerationFailedError(MuAPIError):
    error_type = "remote_generation_failed"


class MuAPITimeoutError(MuAPIError):
    error_type = "remote_generation_timeout"


@dataclass(frozen=True)
class MuAPIConfig:
    """Runtime configuration for MuAPI image generation."""

    model: str = DEFAULT_MUAPI_MODEL
    width: int = DEFAULT_MUAPI_WIDTH
    height: int = DEFAULT_MUAPI_HEIGHT
    number_of_images: int = DEFAULT_MUAPI_NUMBER_OF_IMAGES
    max_polls: int = DEFAULT_MUAPI_MAX_POLLS
    poll_interval: float = DEFAULT_MUAPI_POLL_INTERVAL
    request_timeout: float = DEFAULT_MUAPI_REQUEST_TIMEOUT
    api_base: str = DEFAULT_MUAPI_API_BASE
    api_key: str | None = None

    def validate(self) -> None:
        if self.model not in MUAPI_MODELS:
            raise ValueError(f"unsupported MuAPI model: {self.model}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MuAPI width and height must be positive")
        if self.width > 8192 or self.height > 8192:
            raise ValueError("MuAPI width and height must not exceed 8192")
        if self.number_of_images not in (1, 2, 3, 4):
            raise ValueError("MuAPI number_of_images must be between 1 and 4")
        if self.max_polls < 1:
            raise ValueError("MuAPI max_polls must be at least 1")
        if self.poll_interval < 0:
            raise ValueError("MuAPI poll_interval must not be negative")
        if self.request_timeout <= 0:
            raise ValueError("MuAPI request_timeout must be positive")
        if not self.effective_api_key:
            raise MuAPIAuthRequiredError("MUAPI_API_KEY is required for MuAPI image generation")

    @property
    def effective_api_key(self) -> str:
        return (self.api_key or os.environ.get("MUAPI_API_KEY", "")).strip()


def run_generate(
    *,
    prompt: str,
    config: MuAPIConfig,
    opener: Callable[..., Any] = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
    download_opener: Any | None = None,
) -> tuple[list[Image.Image], str]:
    """Submit once, poll the prediction, and return downloaded images."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    config.validate()

    prediction = _submit(prompt=prompt, config=config, opener=opener)
    prediction_id = _prediction_value(prediction, "request_id", "id", "prediction_id")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise MuAPIError("MuAPI submission did not return a request ID")

    completed = _poll(
        prediction=prediction,
        prediction_id=prediction_id,
        config=config,
        opener=opener,
        sleep_fn=sleep_fn,
    )
    output_urls = _output_urls(completed)
    if not output_urls:
        raise MuAPIError("MuAPI prediction completed without output URLs")

    images = [
        _download_image(raw_url, config=config, opener=download_opener)
        for raw_url in output_urls
    ]
    return images, prediction_id


def _endpoint(model: str) -> str:
    return {
        "flux-dev": "flux-dev-image",
    }[model]


def _submit(*, prompt: str, config: MuAPIConfig, opener: Callable[..., Any]) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "width": config.width,
        "height": config.height,
        "num_images": config.number_of_images,
    }
    request = Request(
        f"{config.api_base.rstrip('/')}/{_endpoint(config.model)}",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(config.effective_api_key, json_body=True),
        method="POST",
    )
    return _api_request(request, config=config, opener=opener)


def _poll(
    *,
    prediction: dict[str, Any],
    prediction_id: str,
    config: MuAPIConfig,
    opener: Callable[..., Any],
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    current = prediction
    prediction_url = (
        f"{config.api_base.rstrip('/')}/predictions/"
        f"{quote(prediction_id, safe='')}/result"
    )
    for poll_number in range(config.max_polls):
        current = _unwrap(current)
        status = str(current.get("status", "")).lower()
        if status in {"completed", "succeeded"}:
            return current
        if status in {"failed", "error", "canceled", "cancelled"}:
            detail = current.get("error") or current.get("message") or "no error details"
            raise MuAPIGenerationFailedError(f"MuAPI prediction {status}: {detail}")
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

    raise MuAPITimeoutError(
        f"MuAPI prediction did not complete after {config.max_polls} polls"
    )


def _prediction_request(
    request: Request,
    *,
    config: MuAPIConfig,
    opener: Callable[..., Any],
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    for attempt in range(3):
        try:
            return _api_request(request, config=config, opener=opener)
        except MuAPIError as exc:
            cause = exc.__cause__
            transient = (
                isinstance(cause, HTTPError)
                and (cause.code == 429 or 500 <= cause.code < 600)
            ) or isinstance(cause, URLError)
            if not transient or attempt == 2:
                raise
            sleep_fn(float(2**attempt))
    raise MuAPIError("MuAPI prediction request exhausted retries")


def _api_request(
    request: Request,
    *,
    config: MuAPIConfig,
    opener: Callable[..., Any],
) -> dict[str, Any]:
    try:
        with opener(request, timeout=config.request_timeout) as response:
            payload = _read_json(response)
    except HTTPError as exc:
        error_type = MuAPIAuthRequiredError if exc.code in (401, 403) else MuAPIError
        raise error_type(f"MuAPI request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise MuAPIError(f"MuAPI request failed: {exc.reason}") from exc
    return _unwrap(payload)


def _read_json(response: Any) -> dict[str, Any]:
    body = response.read(MAX_JSON_BYTES + 1)
    if len(body) > MAX_JSON_BYTES:
        raise MuAPIError("MuAPI response exceeded the 2 MiB limit")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuAPIError("MuAPI returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MuAPIError("MuAPI returned an invalid response object")
    return payload


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and (
        "status" in data
        or "request_id" in data
        or "id" in data
        or "outputs" in data
    ):
        return data
    if payload.get("error") and payload.get("status") not in {"completed", "succeeded"}:
        raise MuAPIError(f"MuAPI request failed: {payload['error']}")
    return payload


def _prediction_value(payload: dict[str, Any], *keys: str) -> Any:
    current = _unwrap(payload)
    for key in keys:
        if current.get(key) is not None:
            return current[key]
    return None


def _output_urls(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    _collect_output_urls(_unwrap(payload), urls, allow_generic=False)
    return list(dict.fromkeys(urls))


def _collect_output_urls(value: Any, result: list[str], *, allow_generic: bool) -> None:
    if isinstance(value, str):
        if allow_generic and value.startswith(("https://", "http://")):
            result.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_output_urls(item, result, allow_generic=True)
        return
    if not isinstance(value, dict):
        return

    media_keys = ("outputs", "images", "image", "output", "result", "media")
    for key in media_keys:
        if key in value:
            _collect_output_urls(value[key], result, allow_generic=True)

    if allow_generic:
        for key in ("url", "image_url", "uri"):
            if key in value:
                _collect_output_urls(value[key], result, allow_generic=True)
        for key, item in value.items():
            if key not in media_keys and key not in {"url", "image_url", "uri"}:
                _collect_output_urls(item, result, allow_generic=True)


def _download_image(
    raw_url: str,
    *,
    config: MuAPIConfig,
    opener: Any | None,
) -> Image.Image:
    url = _validate_output_url(raw_url)
    client = opener or build_opener(_NoRedirectHandler())
    request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with client.open(request, timeout=config.request_timeout) as response:
            declared = int(response.headers.get("Content-Length", "0") or "0")
            if declared > MAX_IMAGE_BYTES:
                raise MuAPIError("MuAPI output exceeded the 64 MiB limit")
            body = response.read(MAX_IMAGE_BYTES + 1)
    except HTTPError as exc:
        raise MuAPIError(f"MuAPI output download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise MuAPIError(f"MuAPI output download failed: {exc.reason}") from exc
    if len(body) > MAX_IMAGE_BYTES:
        raise MuAPIError("MuAPI output exceeded the 64 MiB limit")
    try:
        with Image.open(BytesIO(body)) as image:
            image.load()
            return image.copy()
    except Exception as exc:
        raise MuAPIError("MuAPI output was not a valid image") from exc


def _validate_output_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise MuAPIError("MuAPI output URL has an invalid port") from exc
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MuAPIError("MuAPI output URL must be credential-free HTTPS")
    if port not in (None, 443):
        raise MuAPIError("MuAPI output URL must use the default HTTPS port")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise MuAPIError("MuAPI output URL cannot target localhost")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise MuAPIError("MuAPI output URL cannot target a non-public address")
    return raw_url


def _headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {
        "x-api-key": api_key,
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
