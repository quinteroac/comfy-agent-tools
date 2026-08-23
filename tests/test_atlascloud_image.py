from __future__ import annotations

from io import BytesIO
import json
from types import SimpleNamespace
from urllib.error import HTTPError

from PIL import Image
import pytest

from comfy_agent_tools.imagegen.atlascloud import (
    AtlasCloudAuthRequiredError,
    AtlasCloudConfig,
    AtlasCloudError,
    run_generate,
)


class FakeResponse:
    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int = -1) -> bytes:
        return self.body if limit < 0 else self.body[:limit]


def json_response(payload: dict[str, object]) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"))


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 8), "orange").save(output, format="PNG")
    return output.getvalue()


def test_run_generate_submits_once_polls_and_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    requests: list[object] = []
    responses = iter(
        [
            json_response({"code": 200, "data": {"id": "pred-1", "status": "starting"}}),
            json_response(
                {
                    "code": "200",
                    "data": {
                        "id": "pred-1",
                        "status": "completed",
                        "outputs": ["https://cdn.example.com/image.png"],
                    },
                }
            ),
        ]
    )

    def api_opener(request: object, *, timeout: float) -> FakeResponse:
        requests.append(request)
        assert timeout == 15
        return next(responses)

    download_opener = SimpleNamespace(
        open=lambda request, timeout: FakeResponse(
            png_bytes(), headers={"Content-Length": str(len(png_bytes()))}
        )
    )
    images, prediction_id = run_generate(
        prompt="an orange square",
        config=AtlasCloudConfig(request_timeout=15, poll_interval=0),
        opener=api_opener,
        sleep_fn=lambda _: None,
        download_opener=download_opener,
    )

    assert prediction_id == "pred-1"
    assert len(images) == 1
    assert images[0].size == (12, 8)
    assert [request.get_method() for request in requests] == ["POST", "GET"]
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload == {
        "model": "bytedance/seedream-v5.0-lite",
        "prompt": "an orange square",
        "size": "2048*2048",
        "output_format": "jpeg",
    }
    assert requests[1].full_url.endswith("/model/prediction/pred-1")


def test_run_generate_does_not_retry_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    calls = 0

    def fail_submit(request: object, *, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        raise HTTPError("https://api.atlascloud.ai", 503, "unavailable", {}, None)

    with pytest.raises(AtlasCloudError, match="HTTP 503"):
        run_generate(prompt="test", config=AtlasCloudConfig(), opener=fail_submit)

    assert calls == 1


def test_prediction_get_retries_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    calls = 0
    sleeps: list[float] = []

    def api_opener(request: object, *, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response({"data": {"id": "pred-2", "status": "processing"}})
        if calls == 2:
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)
        return json_response(
            {
                "data": {
                    "id": "pred-2",
                    "status": "completed",
                    "outputs": ["https://cdn.example.com/image.png"],
                }
            }
        )

    download_opener = SimpleNamespace(
        open=lambda request, timeout: FakeResponse(png_bytes())
    )
    images, prediction_id = run_generate(
        prompt="test",
        config=AtlasCloudConfig(poll_interval=0),
        opener=api_opener,
        sleep_fn=sleeps.append,
        download_opener=download_opener,
    )

    assert prediction_id == "pred-2"
    assert images[0].size == (12, 8)
    assert calls == 3
    assert sleeps == [1.0]


def test_prediction_get_does_not_retry_non_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-key")
    calls = 0

    def api_opener(request: object, *, timeout: float) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return json_response({"data": {"id": "pred-3", "status": "processing"}})
        raise HTTPError(request.full_url, 400, "bad request", {}, None)

    with pytest.raises(AtlasCloudError, match="HTTP 400"):
        run_generate(
            prompt="test",
            config=AtlasCloudConfig(poll_interval=0),
            opener=api_opener,
            sleep_fn=lambda _: None,
        )

    assert calls == 2


def test_run_generate_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)

    with pytest.raises(AtlasCloudAuthRequiredError):
        run_generate(prompt="test", config=AtlasCloudConfig())
