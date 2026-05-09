from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from comfy_agent_tools.cli import imagegen
from comfy_agent_tools.imagegen.artifacts import create_seed_image
from comfy_agent_tools.imagegen.config import ImagegenConfig
from comfy_agent_tools.imagegen.flux_klein import _encode_flux2_prompt, run_flux_klein_edit


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


def test_generate_creates_seed_image_with_requested_dimensions() -> None:
    image = create_seed_image(320, 240)

    assert image.size == (320, 240)
    assert image.mode == "RGB"


def test_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
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
    assert payload["model_profile"] == "anima-preview3-turbo"
    assert payload["architecture"] == "anima"
    assert payload["models_dir"] == str(tmp_path)
    assert payload["extra_loras"] == [
        {"path": str(lora_path), "strength_model": 0.8, "strength_clip": 0.0}
    ]
    assert payload["resolved_models"]["unet"].endswith("diffusion_models/animaOfficial_preview3Base.safetensors")
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
