from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from comfy_agent_tools.cli import imagedescribe


def _make_image(path: Path) -> Path:
    image = Image.new("RGB", (8, 8), (123, 45, 67))
    image.save(path, format="PNG")
    return path


def test_parser_describe_defaults() -> None:
    args = imagedescribe.build_parser().parse_args(
        ["describe", "--input", "image.png"]
    )

    assert args.command == "describe"
    assert args.input == Path("image.png")
    assert args.prompt == "Describe this image in detail."
    assert args.models_dir is None
    assert args.llm is None
    assert args.seed == 0
    assert args.max_length is None
    assert args.temperature is None
    assert args.top_k is None
    assert args.top_p is None
    assert args.min_p is None
    assert args.repetition_penalty is None
    assert args.greedy is False
    assert args.verbose is False
    assert args.no_manifest is False


def test_parser_describe_overrides() -> None:
    args = imagedescribe.build_parser().parse_args(
        [
            "describe",
            "--input",
            "image.png",
            "--prompt",
            "List tags.",
            "--max-length",
            "128",
            "--temperature",
            "0.3",
            "--top-k",
            "32",
            "--top-p",
            "0.9",
            "--min-p",
            "0.1",
            "--repetition-penalty",
            "1.1",
            "--seed",
            "42",
            "--greedy",
            "--verbose",
        ]
    )

    assert args.prompt == "List tags."
    assert args.max_length == 128
    assert args.temperature == 0.3
    assert args.top_k == 32
    assert args.top_p == 0.9
    assert args.min_p == 0.1
    assert args.repetition_penalty == 1.1
    assert args.seed == 42
    assert args.greedy is True
    assert args.verbose is True


def test_describe_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def fake_run(*, image: object, prompt: str, config: object) -> str:
        assert prompt == "Describe this image in detail."
        assert config.do_sample is True
        return "A small square image with a brownish color."

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", fake_run)

    rc = imagedescribe.main(
        [
            "describe",
            "--input",
            str(image_path),
            "--models-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["kind"] == "text"
    assert payload["mode"] == "describe"
    assert payload["input"] == str(image_path)
    assert payload["description"] == "A small square image with a brownish color."
    assert payload["model"] == "Qwen3-VL-2B-Instruct"
    assert payload["capability"] == "imagedescribe.describe"
    assert payload["model_profile"] == "qwen3vl-2b-instruct"
    assert payload["architecture"] == "qwen3-vl"
    assert payload["max_length"] == 512
    assert payload["do_sample"] is True
    assert payload["temperature"] == 0.7
    assert payload["resolved_models"]["llm"].endswith("LLM/Qwen-VL/Qwen3-VL-2B-Instruct")
    assert Path(payload["manifests"][0]).is_file()


def test_describe_greedy_disables_sampling(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def fake_run(*, image: object, prompt: str, config: object) -> str:
        assert config.do_sample is False
        return "greedy output"

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", fake_run)

    rc = imagedescribe.main(
        [
            "describe",
            "--input",
            str(image_path),
            "--greedy",
            "--models-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["do_sample"] is False


def test_describe_no_manifest(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    image_path = _make_image(tmp_path / "input.png")
    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", lambda *, image, prompt, config: "text")

    rc = imagedescribe.main(
        [
            "describe",
            "--input",
            str(image_path),
            "--no-manifest",
            "--models-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "manifests" not in payload


def test_describe_suppresses_output_by_default(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def noisy_run(*, image: object, prompt: str, config: object) -> str:
        print("progress should not leak")
        return "text"

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", noisy_run)

    rc = imagedescribe.main(
        ["describe", "--input", str(image_path), "--models-dir", str(tmp_path), "--out", str(tmp_path)]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "progress should not leak" not in captured.out
    assert json.loads(captured.out)["ok"] is True


def test_describe_verbose_allows_output(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def noisy_run(*, image: object, prompt: str, config: object) -> str:
        print("progress is visible")
        return "text"

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", noisy_run)

    rc = imagedescribe.main(
        [
            "describe",
            "--input",
            str(image_path),
            "--verbose",
            "--models-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert "progress is visible" in capsys.readouterr().out


def test_describe_missing_input_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", lambda *, image, prompt, config: "text")

    rc = imagedescribe.main(
        ["describe", "--input", str(tmp_path / "missing.png"), "--models-dir", str(tmp_path), "--out", str(tmp_path)]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "not_found"


def test_describe_runtime_error_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def fail(*, image: object, prompt: str, config: object) -> str:
        raise RuntimeError("ComfyUI runtime not available: missing dependency")

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", fail)

    rc = imagedescribe.main(
        ["describe", "--input", str(image_path), "--models-dir", str(tmp_path), "--out", str(tmp_path)]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "runtime"


def test_describe_empty_prompt_error(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    image_path = _make_image(tmp_path / "input.png")

    def fail(*, image: object, prompt: str, config: object) -> str:
        raise ValueError("prompt must not be empty")

    monkeypatch.setattr(imagedescribe, "run_qwen3vl_describe", fail)

    rc = imagedescribe.main(
        [
            "describe",
            "--input",
            str(image_path),
            "--prompt",
            "",
            "--models-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
