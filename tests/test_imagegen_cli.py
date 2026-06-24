from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from comfy_agent_tools.loras import ExtraLora
from comfy_agent_tools.cli import imagegen
from comfy_agent_tools.imagegen.artifacts import create_seed_image
from comfy_agent_tools.imagegen.config import ImagegenConfig
from comfy_agent_tools.imagegen.flux_klein import _encode_flux2_prompt, run_flux_klein_edit
from comfy_agent_tools.imagegen.ideogram4 import Ideogram4Config, build_prompt, run_ideogram4_t2i


def test_parser_generate_defaults() -> None:
    args = imagegen.build_parser().parse_args(["generate", "--prompt", "hello"])

    assert args.command == "generate"
    assert args.width is None
    assert args.height is None
    assert args.models_dir is None
    assert args.extra_lora == []
    assert args.verbose is False
    assert args.no_manifest is False


def test_parser_edit_accepts_optional_dimensions(tmp_path: Path) -> None:
    args = imagegen.build_parser().parse_args(
        [
            "edit",
            "--input",
            str(tmp_path / "input.png"),
            "--prompt",
            "hello",
            "--width",
            "768",
            "--height",
            "512",
        ]
    )

    assert args.width == 768
    assert args.height == 512


def test_parser_accepts_verbose_for_all_modes(tmp_path: Path) -> None:
    parser = imagegen.build_parser()

    generate = parser.parse_args(["generate", "--prompt", "hello", "--verbose"])
    edit = parser.parse_args(
        ["edit", "--input", str(tmp_path / "input.png"), "--prompt", "hello", "--verbose"]
    )
    upscale = parser.parse_args(["upscale", "--input", str(tmp_path / "input.png"), "--verbose"])
    grok_generate = parser.parse_args(["grok-generate", "--prompt", "hello", "--verbose"])
    grok_edit = parser.parse_args(["grok-edit", "--input", str(tmp_path / "input.png"), "--prompt", "hello", "--verbose"])

    assert generate.verbose is True
    assert edit.verbose is True
    assert upscale.verbose is True
    assert grok_generate.verbose is True
    assert grok_edit.verbose is True


def test_parser_grok_defaults(tmp_path: Path) -> None:
    parser = imagegen.build_parser()

    generate = parser.parse_args(["grok-generate", "--prompt", "hello"])
    edit = parser.parse_args(["grok-edit", "--input", str(tmp_path / "input.png"), "--prompt", "hello"])

    assert generate.command == "grok-generate"
    assert generate.model is None
    assert generate.resolution is None
    assert generate.aspect_ratio is None
    assert generate.number_of_images is None
    assert generate.seed is None
    assert edit.command == "grok-edit"
    assert edit.aspect_ratio is None


def test_parser_ideogram4_generate_accepts_core_and_builder_flags(tmp_path: Path) -> None:
    args = imagegen.build_parser().parse_args(
        [
            "ideogram4-generate",
            "--prompt",
            "A poster for a jazz night.",
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--background",
            "Black paper.",
            "--width",
            "1536",
            "--height",
            "1024",
            "--steps",
            "48",
            "--cfg",
            "7.0",
            "--cfg-override-value",
            "3.0",
            "--cfg-override-start",
            "0.7",
            "--cfg-override-end",
            "1.0",
            "--seed",
            "123",
            "--mu",
            "0.0",
            "--std",
            "1.5",
            "--sampler",
            "euler",
            "--style-color",
            "#101010",
            "--style-color",
            "#f4d35e",
            "--object",
            "420,120,900,880|A golden saxophone.",
            "--text",
            "80,120,220,880|JAZZ NIGHT|Large headline.",
            "--extra-lora",
            "loras/ideogram4/poster-detail.safetensors:0.8:0.2",
            "--output-json",
            str(tmp_path / "prompt.json"),
        ]
    )

    assert args.command == "ideogram4-generate"
    assert args.prompt == "A poster for a jazz night."
    assert args.width == 1536
    assert args.height == 1024
    assert args.steps == 48
    assert args.cfg == 7.0
    assert args.cfg_override_value == 3.0
    assert args.cfg_override_start == 0.7
    assert args.cfg_override_end == 1.0
    assert args.seed == 123
    assert args.mu == 0.0
    assert args.std == 1.5
    assert args.sampler == "euler"
    assert args.style_color == ["#101010", "#f4d35e"]
    assert args.object == ["420,120,900,880|A golden saxophone."]
    assert args.text == ["80,120,220,880|JAZZ NIGHT|Large headline."]
    assert args.extra_lora[0].path == Path("loras/ideogram4/poster-detail.safetensors")
    assert args.extra_lora[0].strength_model == 0.8
    assert args.extra_lora[0].strength_clip == 0.2
    assert args.output_json == tmp_path / "prompt.json"


def test_ideogram4_prompt_builder_preserves_json_shape_and_key_order() -> None:
    prompt = build_prompt(
        high_level_description="A modern concert poster for a jazz trio.",
        style_aesthetics="minimal, elegant, high contrast",
        style_lighting="flat graphic design lighting",
        style_medium="graphic_design",
        style_photo=None,
        style_art_style="clean vector poster, sans-serif typography",
        style_colors=["#101010", "#f4d35e"],
        background="Solid black background with subtle paper texture.",
        objects=["420,120,900,880|A golden saxophone centered in the lower half."],
        texts=["80,120,220,880|JAZZ NIGHT|Large condensed yellow headline."],
    )

    assert (
        prompt
        == '{"high_level_description":"A modern concert poster for a jazz trio.",'
        '"style_description":{"aesthetics":"minimal, elegant, high contrast",'
        '"lighting":"flat graphic design lighting","medium":"graphic_design",'
        '"art_style":"clean vector poster, sans-serif typography",'
        '"color_palette":["#101010","#F4D35E"]},'
        '"compositional_deconstruction":{"background":"Solid black background with subtle paper texture.",'
        '"elements":[{"type":"obj","bbox":[420,120,900,880],'
        '"desc":"A golden saxophone centered in the lower half."},'
        '{"type":"text","bbox":[80,120,220,880],"text":"JAZZ NIGHT",'
        '"desc":"Large condensed yellow headline."}]}}'
    )


def test_ideogram4_prompt_builder_validates_bbox() -> None:
    import pytest

    with pytest.raises(ValueError, match="y_min < y_max"):
        build_prompt(
            high_level_description="A poster.",
            style_aesthetics="minimal",
            style_lighting="flat",
            style_medium="graphic_design",
            style_photo=None,
            style_art_style="vector",
            style_colors=[],
            background="background",
            objects=["900,120,420,880|Invalid saxophone."],
            texts=[],
        )


def test_generate_creates_seed_image_with_requested_dimensions() -> None:
    image = create_seed_image(320, 240)

    assert image.size == (320, 240)
    assert image.mode == "RGB"


def test_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    produced = Image.new("RGB", (8, 8), "red")
    lora_path = tmp_path / "loras" / "anima" / "realism.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")

    def fake_run_anima_t2i(
        *, prompt: str, width: int, height: int, config: object
    ) -> list[Image.Image]:
        assert prompt == "make an icon"
        assert width == 64
        assert height == 32
        assert config.extra_loras[0].path == Path("loras/anima/realism.safetensors")
        return [produced]

    monkeypatch.setattr(imagegen, "run_anima_t2i", fake_run_anima_t2i)

    rc = imagegen.main(
        [
            "generate",
            "--prompt",
            "make an icon",
            "--width",
            "64",
            "--height",
            "32",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/anima/realism.safetensors:0.8:0.0",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["kind"] == "image"
    assert payload["mode"] == "generate"
    assert payload["upscaled"] is False
    assert payload["requested_width"] == 64
    assert payload["requested_height"] == 32
    assert payload["steps"] == 8
    assert payload["cfg"] == 1.0
    assert payload["capability"] == "imagegen.generate"
    assert payload["model_profile"] == "anima-base"
    assert payload["architecture"] == "anima"
    assert payload["models_dir"] == str(tmp_path)
    assert payload["extra_loras"] == [
        {"path": str(lora_path), "strength_model": 0.8, "strength_clip": 0.0}
    ]
    assert payload["resolved_models"]["unet"].endswith("diffusion_models/anima-base-v1.0.safetensors")
    assert payload["resolved_models"]["lora"].endswith("loras/anima/anima-turbo-lora-v0.1.safetensors")
    assert payload["outputs"] == [{"width": 8, "height": 8, "mode": "RGB"}]
    assert len(payload["artifacts"]) == 1
    assert Path(payload["artifacts"][0]).is_file()
    assert len(payload["manifests"]) == 1
    assert Path(payload["manifests"][0]).is_file()


def test_generate_accepts_no_manifest(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setattr(
        imagegen,
        "run_anima_t2i",
        lambda *, prompt, width, height, config: [Image.new("RGB", (8, 8), "red")],
    )

    rc = imagegen.main(["generate", "--prompt", "no manifest", "--out", str(tmp_path), "--no-manifest"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "manifests" not in payload
    assert not (tmp_path / ".comfy-media" / "runs").exists()


def test_generate_suppresses_inference_output_by_default(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def noisy_run_anima_t2i(
        *, prompt: str, width: int, height: int, config: object
    ) -> list[Image.Image]:
        print("progress should not leak")
        return [Image.new("RGB", (8, 8), "red")]

    monkeypatch.setattr(imagegen, "run_anima_t2i", noisy_run_anima_t2i)

    rc = imagegen.main(["generate", "--prompt", "quiet please", "--out", str(tmp_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "progress should not leak" not in captured.out
    payload = json.loads(captured.out)
    assert payload["ok"] is True


def test_generate_verbose_allows_inference_output(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def noisy_run_anima_t2i(
        *, prompt: str, width: int, height: int, config: object
    ) -> list[Image.Image]:
        print("progress is visible")
        return [Image.new("RGB", (8, 8), "red")]

    monkeypatch.setattr(imagegen, "run_anima_t2i", noisy_run_anima_t2i)

    rc = imagegen.main(
        ["generate", "--prompt", "verbose please", "--out", str(tmp_path), "--verbose"]
    )

    assert rc == 0
    assert "progress is visible" in capsys.readouterr().out


def test_edit_requires_existing_input(tmp_path: Path, capsys: MagicMock) -> None:
    rc = imagegen.main(
        [
            "edit",
            "--input",
            str(tmp_path / "missing.png"),
            "--prompt",
            "make it brighter",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["mode"] == "edit"
    assert payload["error_type"] == "not_found"
    assert "input image not found" in payload["error"]


def test_runtime_exception_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    def failing_run_anima_t2i(
        *, prompt: str, width: int, height: int, config: object
    ) -> list[Image.Image]:
        raise RuntimeError("ComfyUI runtime not available: missing psutil")

    monkeypatch.setattr(imagegen, "run_anima_t2i", failing_run_anima_t2i)

    rc = imagegen.main(["generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "generate"
    assert payload["error_type"] == "runtime"


def test_grok_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    produced = Image.new("RGB", (16, 12), "purple")
    seen: dict[str, object] = {}

    def fake_run_grok_generate(*, prompt: str, config: object) -> list[Image.Image]:
        seen["prompt"] = prompt
        seen["model"] = config.model
        seen["aspect_ratio"] = config.aspect_ratio
        seen["number_of_images"] = config.number_of_images
        return [produced]

    monkeypatch.setattr(imagegen, "run_grok_generate", fake_run_grok_generate)

    rc = imagegen.main(
        [
            "grok-generate",
            "--prompt",
            "remote image",
            "--model",
            "grok-imagine-image-pro",
            "--resolution",
            "2K",
            "--aspect-ratio",
            "16:9",
            "--number-of-images",
            "1",
            "--seed",
            "123",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen == {
        "prompt": "remote image",
        "model": "grok-imagine-image-pro",
        "aspect_ratio": "16:9",
        "number_of_images": 1,
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["kind"] == "image"
    assert payload["mode"] == "grok-generate"
    assert payload["remote"] is True
    assert payload["provider"] == "comfy-api"
    assert payload["capability"] == "imagegen.grok-generate"
    assert payload["model_profile"] == "grok-imagine-api"
    assert payload["architecture"] == "grok-imagine-api"
    assert payload["resolved_models"] == {}
    assert payload["model"] == "grok-imagine-image-pro"
    assert payload["resolution"] == "2K"
    assert payload["aspect_ratio"] == "16:9"
    assert payload["seed"] == 123
    assert payload["outputs"] == [{"width": 16, "height": 12, "mode": "RGB"}]
    assert Path(payload["artifacts"][0]).is_file()


def test_grok_edit_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(input_path)
    seen: dict[str, object] = {}

    def fake_run_grok_edit(*, image: Path, prompt: str, config: object) -> list[Image.Image]:
        seen["image"] = image
        seen["prompt"] = prompt
        seen["aspect_ratio"] = config.aspect_ratio
        return [Image.new("RGB", (10, 10), "green")]

    monkeypatch.setattr(imagegen, "run_grok_edit", fake_run_grok_edit)

    rc = imagegen.main(["grok-edit", "--input", str(input_path), "--prompt", "make it green", "--out", str(tmp_path)])

    assert rc == 0
    assert seen == {"image": input_path, "prompt": "make it green", "aspect_ratio": "auto"}
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "grok-edit"
    assert payload["capability"] == "imagegen.grok-edit"
    assert payload["input"] == str(input_path)
    assert payload["aspect_ratio"] == "auto"


def test_grok_missing_api_key_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    from comfy_agent_tools.imagegen.grok import GrokImagineAuthRequiredError

    def fail(*, prompt: str, config: object) -> list[Image.Image]:
        raise GrokImagineAuthRequiredError("COMFY_ORG_API_KEY is required for Grok Imagine API generation")

    monkeypatch.setattr(imagegen, "run_grok_generate", fail)

    rc = imagegen.main(["grok-generate", "--prompt", "remote", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "grok-generate"
    assert payload["error_type"] == "auth_required"


def test_ideogram4_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    seen: dict[str, object] = {}
    lora_path = tmp_path / "models" / "loras" / "ideogram4" / "poster-detail.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")

    def fake_run_ideogram4_t2i(*, prompt: str, config: object) -> list[Image.Image]:
        seen["prompt"] = prompt
        seen["width"] = config.width
        seen["height"] = config.height
        seen["uncond_unet"] = config.uncond_unet
        seen["extra_loras"] = config.extra_loras
        return [Image.new("RGB", (18, 10), "yellow")]

    monkeypatch.setattr(imagegen, "run_ideogram4_t2i", fake_run_ideogram4_t2i)

    rc = imagegen.main(
        [
            "ideogram4-generate",
            "--prompt",
            "A poster for a jazz night",
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--background",
            "black paper",
            "--text",
            "100,100,250,900|JAZZ NIGHT|large headline",
            "--models-dir",
            str(tmp_path / "models"),
            "--width",
            "768",
            "--height",
            "512",
            "--steps",
            "12",
            "--seed",
            "99",
            "--extra-lora",
            "loras/ideogram4/poster-detail.safetensors:0.6:0.1",
            "--out",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "generated-prompt.json"),
        ]
    )

    assert rc == 0
    assert seen == {
        "prompt": (
            '{"high_level_description":"A poster for a jazz night",'
            '"style_description":{"aesthetics":"minimal","lighting":"flat",'
            '"medium":"graphic_design","art_style":"vector"},'
            '"compositional_deconstruction":{"background":"black paper",'
            '"elements":[{"type":"text","bbox":[100,100,250,900],'
            '"text":"JAZZ NIGHT","desc":"large headline"}]}}'
        ),
        "width": 768,
        "height": 512,
        "uncond_unet": Path("diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"),
        "extra_loras": seen["extra_loras"],
    }
    assert seen["extra_loras"][0].path == Path("loras/ideogram4/poster-detail.safetensors")
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "ideogram4-generate"
    assert payload["capability"] == "imagegen.ideogram4-generate"
    assert payload["model_profile"] == "ideogram4-fp8"
    assert payload["architecture"] == "ideogram4"
    assert payload["seed"] == 99
    assert payload["requested_width"] == 768
    assert payload["requested_height"] == 512
    assert payload["resolved_models"]["unet"].endswith("diffusion_models/ideogram4_fp8_scaled.safetensors")
    assert payload["resolved_models"]["uncond_unet"].endswith(
        "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"
    )
    assert payload["extra_loras"] == [
        {"path": str(lora_path), "strength_model": 0.6, "strength_clip": 0.1}
    ]
    assert payload["outputs"] == [{"width": 18, "height": 10, "mode": "RGB"}]
    assert Path(payload["artifacts"][0]).is_file()
    assert payload["prompt_json"] == str(tmp_path / "generated-prompt.json")
    assert Path(payload["prompt_json"]).read_text(encoding="utf-8") == str(seen["prompt"]) + "\n"


def test_ideogram4_generate_builder_prompt(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    seen: dict[str, object] = {}

    def fake_run_ideogram4_t2i(*, prompt: str, config: object) -> list[Image.Image]:
        seen["prompt"] = prompt
        return [Image.new("RGB", (8, 8), "black")]

    monkeypatch.setattr(imagegen, "run_ideogram4_t2i", fake_run_ideogram4_t2i)

    rc = imagegen.main(
        [
            "ideogram4-generate",
            "--prompt",
            "A poster.",
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--style-color",
            "#abcdef",
            "--background",
            "Black paper.",
            "--object",
            "100,100,900,900|A saxophone.",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert json.loads(str(seen["prompt"]))["style_description"]["color_palette"] == ["#ABCDEF"]


def test_ideogram4_rejects_raw_json_prompt(tmp_path: Path, capsys: MagicMock) -> None:
    rc = imagegen.main(
        [
            "ideogram4-generate",
            "--prompt",
            '{"high_level_description":"raw json"}',
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--background",
            "background",
            "--object",
            "100,100,900,900|Object.",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "ideogram4-generate"
    assert payload["error_type"] == "error"
    assert "does not accept raw JSON" in payload["error"]


def test_ideogram4_rejects_missing_elements(tmp_path: Path, capsys: MagicMock) -> None:
    rc = imagegen.main(
        [
            "ideogram4-generate",
            "--prompt",
            "A poster for a jazz night",
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--background",
            "black paper",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "ideogram4-generate"
    assert payload["error_type"] == "error"
    assert "at least one --object or --text" in payload["error"]


def test_ideogram4_extra_lora_missing_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    rc = imagegen.main(
        [
            "ideogram4-generate",
            "--prompt",
            "A poster",
            "--style-aesthetics",
            "minimal",
            "--style-lighting",
            "flat",
            "--style-medium",
            "graphic_design",
            "--style-art-style",
            "vector",
            "--background",
            "black paper",
            "--object",
            "100,100,900,900|A saxophone.",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/ideogram4/missing.safetensors",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "ideogram4-generate"
    assert payload["error_type"] == "not_found"
    assert "extra LoRA file not found" in payload["error"]


def test_ideogram4_runtime_applies_extra_loras_to_both_unets(
    monkeypatch: MagicMock, tmp_path: Path
) -> None:
    lora_path = tmp_path / "loras" / "ideogram4" / "poster-detail.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")
    calls: dict[str, object] = {}

    class FakeModelManager:
        def __init__(self, models_dir: Path) -> None:
            calls["models_dir"] = models_dir

        def load_unet(self, path: Path) -> str:
            return f"unet:{path.name}"

        def load_clip(self, path: Path, *, clip_type: str) -> str:
            calls["clip_type"] = clip_type
            return f"clip:{path.name}"

        def load_vae(self, path: Path) -> str:
            return f"vae:{path.name}"

    conditioning = types.ModuleType("comfy_diffusion.conditioning")
    conditioning.conditioning_zero_out = lambda cond: ("zero", cond)
    conditioning.encode_prompt = lambda clip, prompt: ("positive", clip, prompt)

    latent = types.ModuleType("comfy_diffusion.latent")
    latent.empty_flux2_latent_image = lambda width, height, batch_size: ("latent", width, height, batch_size)

    models = types.ModuleType("comfy_diffusion.models")
    models.ModelManager = FakeModelManager

    runtime = types.ModuleType("comfy_diffusion.runtime")
    runtime.check_runtime = lambda: {}

    sampling = types.ModuleType("comfy_diffusion.sampling")
    sampling.cfg_override = lambda model, value, start, end: f"cfg:{model}:{value}:{start}:{end}"
    sampling.dual_model_guider = lambda model, positive, cfg, *, model_negative, negative: (
        "guider",
        model,
        positive,
        cfg,
        model_negative,
        negative,
    )
    sampling.get_sampler = lambda sampler: ("sampler", sampler)
    sampling.ideogram4_scheduler = lambda steps, width, height, mu, std: (
        "sigmas",
        steps,
        width,
        height,
        mu,
        std,
    )
    sampling.random_noise = lambda seed: ("noise", seed)
    sampling.sample_custom = lambda noise, guider, sampler, sigmas, latent: (("latent_out", guider), None)

    vae = types.ModuleType("comfy_diffusion.vae")
    vae.vae_decode = lambda vae_model, latent_out: Image.new("RGB", (12, 10), "orange")

    monkeypatch.setitem(sys.modules, "comfy_diffusion", types.ModuleType("comfy_diffusion"))
    monkeypatch.setitem(sys.modules, "comfy_diffusion.conditioning", conditioning)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.latent", latent)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.models", models)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.runtime", runtime)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.sampling", sampling)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.vae", vae)

    def fake_apply_extra_loras_to_models(
        model_list: list[str],
        clip: str,
        loras: list[ExtraLora],
    ) -> tuple[list[str], str]:
        calls["lora_models"] = model_list
        calls["lora_clip"] = clip
        calls["loras"] = loras
        return ["patched-main", "patched-uncond"], "patched-clip"

    monkeypatch.setattr(
        "comfy_agent_tools.imagegen.ideogram4.apply_extra_loras_to_models",
        fake_apply_extra_loras_to_models,
    )

    images = run_ideogram4_t2i(
        prompt="poster",
        config=Ideogram4Config(
            models_dir=tmp_path,
            width=64,
            height=48,
            steps=5,
            cfg=6.5,
            extra_loras=[ExtraLora(Path("loras/ideogram4/poster-detail.safetensors"), 0.7, 0.2)],
        ),
    )

    assert images[0].size == (12, 10)
    assert calls["clip_type"] == "ideogram4"
    assert calls["lora_models"] == [
        "unet:ideogram4_fp8_scaled.safetensors",
        "unet:ideogram4_unconditional_fp8_scaled.safetensors",
    ]
    assert calls["lora_clip"] == "clip:qwen3vl_8b_fp8_scaled.safetensors"
    assert calls["loras"] == [ExtraLora(lora_path, 0.7, 0.2)]


def test_edit_uses_qwen_default(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(input_path)
    seen: dict[str, object] = {}

    def fake_run_qwen_edit(*, prompt: str, image: Image.Image, config: object) -> list[Image.Image]:
        seen["prompt"] = prompt
        seen["unet"] = str(config.unet)
        return [Image.new("RGB", (8, 8), "green")]

    monkeypatch.setattr(imagegen, "run_qwen_edit", fake_run_qwen_edit)

    rc = imagegen.main(
        ["edit", "--input", str(input_path), "--prompt", "make it green", "--out", str(tmp_path)]
    )

    assert rc == 0
    assert seen["prompt"] == "make it green"
    assert seen["unet"] == "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_profile"] == "qwen-edit2511"
    assert payload["architecture"] == "qwen-image-edit"


def test_generate_uses_flux_klein_profile(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        json.dumps(
            {
                "models_dir": str(tmp_path),
                "defaults": {"imagegen.generate": "flux-klein-9b-snofs"},
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_run_flux_klein_t2i(
        *, prompt: str, width: int, height: int, config: object
    ) -> list[Image.Image]:
        assert prompt == "make it snofs"
        assert width == 128
        assert height == 128
        assert str(config.lora) == "loras/flux-klein/klein_snofs_v1_1.safetensors"
        return [Image.new("RGB", (16, 16), "purple")]

    monkeypatch.setattr(imagegen, "run_flux_klein_t2i", fake_run_flux_klein_t2i)

    rc = imagegen.main(
        [
            "generate",
            "--prompt",
            "make it snofs",
            "--width",
            "128",
            "--height",
            "128",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_profile"] == "flux-klein-9b-snofs"
    assert payload["architecture"] == "flux-klein"
    assert payload["steps"] == 4
    assert payload["cfg"] == 1.0
    assert payload["resolved_models"]["lora"].endswith("loras/flux-klein/klein_snofs_v1_1.safetensors")


def test_edit_uses_flux_klein_profile(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        json.dumps(
            {
                "models_dir": str(tmp_path),
                "defaults": {"imagegen.edit": "flux-klein-9b-snofs"},
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "blue").save(input_path)

    def fake_run_flux_klein_edit(
        *, prompt: str, image: Image.Image, width: int, height: int, config: object
    ) -> list[Image.Image]:
        assert prompt == "edit with snofs"
        assert image.size == (32, 32)
        assert width == 32
        assert height == 32
        assert str(config.clip) == "text_encoders/qwen_3_8b_fp8mixed.safetensors"
        return [Image.new("RGB", (32, 32), "pink")]

    monkeypatch.setattr(imagegen, "run_flux_klein_edit", fake_run_flux_klein_edit)

    rc = imagegen.main(
        ["edit", "--input", str(input_path), "--prompt", "edit with snofs", "--out", str(tmp_path)]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_profile"] == "flux-klein-9b-snofs"
    assert payload["architecture"] == "flux-klein"
    assert payload["input"] == str(input_path)
    assert payload["requested_width"] == 32
    assert payload["requested_height"] == 32


def test_edit_flux_klein_accepts_requested_dimensions(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        json.dumps(
            {
                "models_dir": str(tmp_path),
                "defaults": {"imagegen.edit": "flux-klein-9b-snofs"},
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 48), "blue").save(input_path)

    def fake_run_flux_klein_edit(
        *, prompt: str, image: Image.Image, width: int, height: int, config: object
    ) -> list[Image.Image]:
        assert prompt == "edit wide"
        assert image.size == (32, 48)
        assert width == 64
        assert height == 48
        return [Image.new("RGB", (64, 48), "pink")]

    monkeypatch.setattr(imagegen, "run_flux_klein_edit", fake_run_flux_klein_edit)

    rc = imagegen.main(
        [
            "edit",
            "--input",
            str(input_path),
            "--prompt",
            "edit wide",
            "--width",
            "64",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_width"] == 64
    assert payload["requested_height"] == 48
    assert payload["outputs"] == [{"width": 64, "height": 48, "mode": "RGB"}]


def test_flux_klein_encoder_uses_qwen3_tokens() -> None:
    class FakeClip:
        def __init__(self) -> None:
            self.add_dict: dict[str, object] | None = None

        def tokenize(self, text: str) -> dict[str, list[int]]:
            assert text == " "
            return {"qwen3_8b": [1, 2, 3]}

        def encode_from_tokens_scheduled(
            self, tokens: dict[str, list[int]], *, add_dict: dict[str, object]
        ) -> dict[str, object]:
            self.add_dict = add_dict
            return {"tokens": tokens, "guidance": add_dict["guidance"]}

    clip = FakeClip()

    result = _encode_flux2_prompt(clip, "", guidance=1.0)

    assert result["tokens"] == {"qwen3_8b": [1, 2, 3]}
    assert result["guidance"] == 1.0


def test_flux_klein_edit_applies_model_sampling(
    monkeypatch: MagicMock, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeModelManager:
        def __init__(self, models_dir: Path) -> None:
            calls["models_dir"] = models_dir

        def load_unet(self, path: Path) -> str:
            return f"unet:{path.name}"

        def load_clip(self, path: Path, *, clip_type: str) -> str:
            calls["clip_type"] = clip_type
            return f"clip:{path.name}"

        def load_vae(self, path: Path) -> str:
            return f"vae:{path.name}"

    conditioning = types.ModuleType("comfy_diffusion.conditioning")
    conditioning.reference_latent = lambda cond, ref: ("reference", cond, ref)
    conditioning.conditioning_zero_out = lambda cond: ("zero", cond)
    conditioning.encode_prompt = lambda clip, prompt, negative: (("positive", prompt), ("negative", negative))

    latent = types.ModuleType("comfy_diffusion.latent")
    latent.empty_flux2_latent_image = lambda width, height, batch_size: ("latent", width, height, batch_size)

    lora = types.ModuleType("comfy_diffusion.lora")
    lora.apply_lora = lambda model, clip, path, model_strength, clip_strength: (model, clip)

    models = types.ModuleType("comfy_diffusion.models")
    models.ModelManager = FakeModelManager

    def fake_model_sampling_flux(
        model: str, max_shift: float, min_shift: float, width: int, height: int
    ) -> str:
        calls["model_sampling_flux"] = (model, max_shift, min_shift, width, height)
        return f"sampled:{model}:{width}x{height}"

    models.model_sampling_flux = fake_model_sampling_flux

    sampling = types.ModuleType("comfy_diffusion.sampling")
    sampling.flux2_scheduler = lambda steps, width, height: ("sigmas", steps, width, height)
    sampling.get_sampler = lambda sampler: ("sampler", sampler)
    sampling.cfg_guider = lambda model, positive, negative, cfg: ("guider", model, positive, negative, cfg)
    sampling.random_noise = lambda seed: ("noise", seed)
    sampling.sample_custom = lambda noise, guider, sampler, sigmas, latent: (("latent_out", guider), None)
    sampling.sample_custom_simple = lambda *args: ("latent_out_simple", args)

    vae = types.ModuleType("comfy_diffusion.vae")
    vae.vae_encode = lambda vae_model, image: ("ref_latent", image.size)
    vae.vae_decode = lambda vae_model, latent_out: Image.new("RGB", (16, 16), "pink")

    monkeypatch.setitem(sys.modules, "comfy_diffusion", types.ModuleType("comfy_diffusion"))
    monkeypatch.setitem(sys.modules, "comfy_diffusion.conditioning", conditioning)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.latent", latent)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.lora", lora)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.models", models)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.sampling", sampling)
    monkeypatch.setitem(sys.modules, "comfy_diffusion.vae", vae)
    monkeypatch.setattr("comfy_agent_tools.imagegen.flux_klein.require_comfy_runtime", lambda: None)
    monkeypatch.setattr(
        "comfy_agent_tools.imagegen.flux_klein.apply_extra_loras",
        lambda model, clip, loras: (model, clip),
    )

    images = run_flux_klein_edit(
        prompt="edit",
        image=Image.new("RGB", (32, 32), "blue"),
        width=64,
        height=48,
        config=ImagegenConfig(models_dir=tmp_path, steps=4, cfg=1.0),
    )

    assert images[0].size == (16, 16)
    assert calls["model_sampling_flux"] == (
        "unet:qwen_image_edit_2511_fp8mixed.safetensors",
        1.15,
        0.5,
        64,
        48,
    )


def test_extra_lora_missing_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    rc = imagegen.main(
        [
            "generate",
            "--prompt",
            "missing lora",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/anima/missing.safetensors",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "not_found"
    assert "extra LoRA file not found" in payload["error"]


def test_upscale_uses_clear_reality_default(monkeypatch: MagicMock, tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(input_path)
    seen: dict[str, object] = {}

    def fake_run_upscale(*, image: Image.Image, config: object) -> list[Image.Image]:
        seen["upscaler"] = str(config.upscaler)
        return [image]

    monkeypatch.setattr(imagegen, "run_upscale", fake_run_upscale)

    rc = imagegen.main(["upscale", "--input", str(input_path), "--out", str(tmp_path)])

    assert rc == 0
    assert seen["upscaler"] == "upscale_models/4x-ClearRealityV1.pth"


def test_upscale_success_metadata(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "blue").save(input_path)

    def fake_run_upscale(*, image: Image.Image, config: object) -> list[Image.Image]:
        return [Image.new("RGB", (16, 16), "blue")]

    monkeypatch.setattr(imagegen, "run_upscale", fake_run_upscale)

    rc = imagegen.main(["upscale", "--input", str(input_path), "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input"] == str(input_path)
    assert payload["upscaled"] is True
    assert payload["upscaler"] == "4x-ClearRealityV1.pth"
    assert payload["capability"] == "imagegen.upscale"
    assert payload["model_profile"] == "clear-reality"
    assert payload["architecture"] == "upscale-model"
    assert payload["outputs"] == [{"width": 16, "height": 16, "mode": "RGB"}]


def test_parser_krea2_generate_defaults() -> None:
    args = imagegen.build_parser().parse_args(["krea2-generate", "--prompt", "hi"])

    assert args.command == "krea2-generate"
    assert args.prompt == "hi"
    assert args.models_dir is None
    assert args.unet is None
    assert args.clip is None
    assert args.vae is None
    assert args.width is None
    assert args.height is None
    assert args.steps is None
    assert args.cfg is None
    assert args.seed is None
    assert args.sampler is None
    assert args.scheduler is None
    assert args.rebalance_multiplier is None
    assert args.verbose is False
    assert args.no_manifest is False


def test_parser_krea2_generate_overrides() -> None:
    args = imagegen.build_parser().parse_args(
        [
            "krea2-generate",
            "--prompt",
            "hi",
            "--width",
            "768",
            "--height",
            "512",
            "--steps",
            "6",
            "--cfg",
            "1.5",
            "--seed",
            "42",
            "--sampler",
            "euler",
            "--scheduler",
            "normal",
            "--rebalance-multiplier",
            "3.0",
            "--verbose",
        ]
    )

    assert args.width == 768
    assert args.height == 512
    assert args.steps == 6
    assert args.cfg == 1.5
    assert args.seed == 42
    assert args.sampler == "euler"
    assert args.scheduler == "normal"
    assert args.rebalance_multiplier == 3.0
    assert args.verbose is True


def test_krea2_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    seen: dict[str, object] = {}

    def fake_run_krea2_t2i(*, prompt: str, width: int, height: int, config: object) -> list[Image.Image]:
        seen["prompt"] = prompt
        seen["width"] = width
        seen["height"] = height
        seen["steps"] = config.steps
        seen["cfg"] = config.cfg
        seen["rebalance_multiplier"] = config.rebalance_multiplier
        seen["unet"] = config.unet
        seen["clip"] = config.clip
        seen["vae"] = config.vae
        return [Image.new("RGB", (24, 16), "purple")]

    monkeypatch.setattr(imagegen, "run_krea2_t2i", fake_run_krea2_t2i)

    rc = imagegen.main(
        [
            "krea2-generate",
            "--prompt",
            "a cinematic portrait",
            "--models-dir",
            str(tmp_path / "models"),
            "--width",
            "768",
            "--height",
            "512",
            "--steps",
            "6",
            "--cfg",
            "1.5",
            "--seed",
            "42",
            "--rebalance-multiplier",
            "3.0",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["prompt"] == "a cinematic portrait"
    assert seen["width"] == 768
    assert seen["height"] == 512
    assert seen["steps"] == 6
    assert seen["cfg"] == 1.5
    assert seen["rebalance_multiplier"] == 3.0
    assert seen["unet"] == Path("diffusion_models/krea2_turbo_fp8_scaled.safetensors")
    assert seen["clip"] == Path("text_encoders/qwen3vl_4b_fp8_scaled.safetensors")
    assert seen["vae"] == Path("vae/qwen_image_vae.safetensors")
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "krea2-generate"
    assert payload["capability"] == "imagegen.krea2-generate"
    assert payload["model_profile"] == "krea2-turbo"
    assert payload["architecture"] == "krea2"
    assert payload["seed"] == 42
    assert payload["steps"] == 6
    assert payload["cfg"] == 1.5
    assert payload["rebalance_multiplier"] == 3.0
    assert payload["requested_width"] == 768
    assert payload["requested_height"] == 512
    assert payload["resolved_models"]["unet"].endswith("diffusion_models/krea2_turbo_fp8_scaled.safetensors")
    assert payload["resolved_models"]["clip"].endswith("text_encoders/qwen3vl_4b_fp8_scaled.safetensors")
    assert payload["resolved_models"]["vae"].endswith("vae/qwen_image_vae.safetensors")
    assert payload["outputs"] == [{"width": 24, "height": 16, "mode": "RGB"}]
    assert Path(payload["artifacts"][0]).is_file()
    assert Path(payload["manifests"][0]).is_file()


def test_krea2_generate_suppresses_output_by_default(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def noisy_run(*, prompt: str, width: int, height: int, config: object) -> list[Image.Image]:
        print("progress should not leak")
        return [Image.new("RGB", (8, 8), "red")]

    monkeypatch.setattr(imagegen, "run_krea2_t2i", noisy_run)

    rc = imagegen.main(["krea2-generate", "--prompt", "quiet", "--out", str(tmp_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "progress should not leak" not in captured.out
    assert json.loads(captured.out)["ok"] is True


def test_krea2_generate_missing_model_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def fail(*, prompt: str, width: int, height: int, config: object) -> list[Image.Image]:
        raise FileNotFoundError("unet model file not found: missing.safetensors")

    monkeypatch.setattr(imagegen, "run_krea2_t2i", fail)

    rc = imagegen.main(["krea2-generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "krea2-generate"
    assert payload["error_type"] == "not_found"


def test_krea2_generate_runtime_error_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def fail(*, prompt: str, width: int, height: int, config: object) -> list[Image.Image]:
        raise RuntimeError("ComfyUI runtime not available: missing dependency")

    monkeypatch.setattr(imagegen, "run_krea2_t2i", fail)

    rc = imagegen.main(["krea2-generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "runtime"


def test_parser_rtx_upscale_defaults() -> None:
    args = imagegen.build_parser().parse_args(["rtx-upscale", "--input", "img.png"])

    assert args.command == "rtx-upscale"
    assert args.input == Path("img.png")
    assert args.resolution is None
    assert args.width is None
    assert args.height is None
    assert args.scale is None
    assert args.quality is None
    assert args.verbose is False
    assert args.no_manifest is False


def test_parser_rtx_upscale_overrides() -> None:
    args = imagegen.build_parser().parse_args(
        [
            "rtx-upscale",
            "--input",
            "img.png",
            "--resolution",
            "4k",
            "--quality",
            "HIGH",
            "--verbose",
        ]
    )

    assert args.resolution == "4k"
    assert args.quality == "HIGH"
    assert args.verbose is True


def test_rtx_upscale_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (480, 270), "blue").save(input_path)
    seen: dict[str, object] = {}

    def fake_run_rtx(*, image: Image.Image, config: object) -> dict[str, object]:
        seen["quality"] = config.quality
        return {
            "image": Image.new("RGB", (1920, 1080), "blue"),
            "input_width": 480,
            "input_height": 270,
            "target": "1080p",
            "target_width": 1920,
            "target_height": 1080,
            "quality": config.quality,
        }

    monkeypatch.setattr(imagegen, "run_rtx_upscale_image", fake_run_rtx)

    rc = imagegen.main(
        [
            "rtx-upscale",
            "--input",
            str(input_path),
            "--resolution",
            "4k",
            "--quality",
            "HIGH",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["quality"] == "HIGH"
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "rtx-upscale"
    assert payload["capability"] == "imagegen.rtx-upscale"
    assert payload["model_profile"] == "rtx-vsr"
    assert payload["architecture"] == "rtx-vsr"
    assert payload["input_width"] == 480
    assert payload["input_height"] == 270
    assert payload["target_width"] == 1920
    assert payload["target_height"] == 1080
    assert payload["quality"] == "HIGH"
    assert payload["outputs"] == [{"width": 1920, "height": 1080, "mode": "RGB"}]
    assert Path(payload["artifacts"][0]).is_file()
    assert Path(payload["manifests"][0]).is_file()


def test_rtx_upscale_missing_input_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.setattr(imagegen, "run_rtx_upscale_image", lambda *, image, config: {})

    rc = imagegen.main(["rtx-upscale", "--input", str(tmp_path / "missing.png"), "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "rtx-upscale"
    assert payload["error_type"] == "not_found"


def test_rtx_upscale_missing_dependency_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "red").save(input_path)

    def fail(*, image: Image.Image, config: object) -> dict[str, object]:
        raise ModuleNotFoundError("nvidia-vfx is required for RTX image upscaling")

    monkeypatch.setattr(imagegen, "run_rtx_upscale_image", fail)

    rc = imagegen.main(["rtx-upscale", "--input", str(input_path), "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "missing_dependency"
