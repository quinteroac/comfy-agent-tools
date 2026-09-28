from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from comfy_agent_tools.cli import imagegen


def test_parser_muapi_defaults() -> None:
    args = imagegen.build_parser().parse_args(["muapi-generate", "--prompt", "hello"])

    assert args.command == "muapi-generate"
    assert args.model is None
    assert args.width is None
    assert args.height is None
    assert args.number_of_images is None
    assert args.max_polls is None
    assert args.poll_interval is None
    assert args.request_timeout is None
    assert args.no_manifest is False


def test_muapi_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    produced = Image.new("RGB", (18, 10), "orange")
    seen: dict[str, object] = {}

    def fake_run_muapi_generate(*, prompt: str, config: object) -> tuple[list[Image.Image], str]:
        seen["prompt"] = prompt
        seen["model"] = config.model
        seen["width"] = config.width
        seen["height"] = config.height
        seen["number_of_images"] = config.number_of_images
        return [produced], "req-123"

    monkeypatch.setattr(imagegen, "run_muapi_generate", fake_run_muapi_generate)

    rc = imagegen.main(
        [
            "muapi-generate",
            "--prompt",
            "remote image",
            "--model",
            "flux-dev",
            "--width",
            "768",
            "--height",
            "512",
            "--number-of-images",
            "2",
            "--out",
            str(tmp_path),
            "--no-manifest",
        ]
    )

    assert rc == 0
    assert seen == {
        "prompt": "remote image",
        "model": "flux-dev",
        "width": 768,
        "height": 512,
        "number_of_images": 2,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "muapi-generate"
    assert payload["remote"] is True
    assert payload["provider"] == "muapi"
    assert payload["prediction_id"] == "req-123"
    assert payload["capability"] == "imagegen.muapi-generate"
    assert payload["model_profile"] == "muapi-image-api"
    assert payload["architecture"] == "muapi-image-api"
    assert payload["outputs"] == [{"width": 18, "height": 10, "mode": "RGB"}]
    assert Path(payload["artifacts"][0]).is_file()
