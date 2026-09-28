from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from comfy_agent_tools.imagegen.muapi import (
    MuAPIAuthRequiredError,
    MuAPIConfig,
    run_generate,
)


class FakeResponse:
    def __init__(self, body: bytes, *, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self):
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
    monkeypatch.setenv("MUAPI_API_KEY", "test-key")
    requests: list[object] = []
    responses = iter(
        [
            json_response({"request_id": "req-1", "status": "processing"}),
            json_response(
                {
                    "id": "req-1",
                    "status": "completed",
                    "outputs": ["https://cdn.example.com/image.png"],
                    "urls": {"get": "https://api.muapi.ai/api/v1/predictions/req-1/result"},
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
        config=MuAPIConfig(
            width=768,
            height=512,
            number_of_images=2,
            request_timeout=15,
            poll_interval=0,
        ),
        opener=api_opener,
        sleep_fn=lambda _: None,
        download_opener=download_opener,
    )

    assert prediction_id == "req-1"
    assert len(images) == 1
    assert images[0].size == (12, 8)
    assert [request.get_method() for request in requests] == ["POST", "GET"]
    assert requests[0].get_header("X-api-key") == "test-key"
    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload == {
        "prompt": "an orange square",
        "width": 768,
        "height": 512,
        "num_images": 2,
    }


def test_run_generate_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUAPI_API_KEY", raising=False)

    with pytest.raises(MuAPIAuthRequiredError):
        run_generate(prompt="an image", config=MuAPIConfig(), opener=lambda *args, **kwargs: None)
