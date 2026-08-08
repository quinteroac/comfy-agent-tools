"""Local MiniMax H3 reference-to-video generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util
from contextlib import contextmanager, nullcontext

from comfy_agent_tools.videogen.artifacts import make_video_path

DEFAULT_MINIMAX_WIDTH = 1344
DEFAULT_MINIMAX_HEIGHT = 768
DEFAULT_MINIMAX_LENGTH = 124
DEFAULT_MINIMAX_STEPS = 4
DEFAULT_MINIMAX_SEED = 0
DEFAULT_MINIMAX_REF_IMAGE_SIZE = "match"
DEFAULT_MINIMAX_FPS = 24
DEFAULT_MINIMAX_EASYCACHE_REUSE_THRESHOLD = 0.20
DEFAULT_MINIMAX_EASYCACHE_START_PERCENT = 0.15
DEFAULT_MINIMAX_EASYCACHE_END_PERCENT = 0.95
DEFAULT_MINIMAX_FL2VA_UNET = Path("diffusion_models/minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
DEFAULT_MINIMAX_REF2VA_UNET = Path("diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
DEFAULT_MINIMAX_TURBO_LORA = Path("loras/minimax/minimax_h3_turbo_4step_ckpt850.safetensors")
DEFAULT_MINIMAX_TURBO_NODE = Path("/home/victor/AI/Comfy/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo")


@dataclass(frozen=True)
class MiniMaxH3Config:
    """Runtime options for the local MiniMax H3 pipeline."""

    models_dir: Path
    width: int = DEFAULT_MINIMAX_WIDTH
    height: int = DEFAULT_MINIMAX_HEIGHT
    length: int = DEFAULT_MINIMAX_LENGTH
    steps: int = DEFAULT_MINIMAX_STEPS
    seed: int = DEFAULT_MINIMAX_SEED
    ref_image_size: str = DEFAULT_MINIMAX_REF_IMAGE_SIZE
    unet: Path = DEFAULT_MINIMAX_REF2VA_UNET
    fl2va_unet: Path = DEFAULT_MINIMAX_FL2VA_UNET
    text_encoder: Path | None = None
    audio_vae: Path | None = None
    video_vae: Path | None = None
    turbo_lora: Path = DEFAULT_MINIMAX_TURBO_LORA
    turbo_node: Path = DEFAULT_MINIMAX_TURBO_NODE
    turbo_lora_strength: float = 1.0
    sage_attention: bool = False
    easycache: bool = False
    easycache_reuse_threshold: float = DEFAULT_MINIMAX_EASYCACHE_REUSE_THRESHOLD
    easycache_start_percent: float = DEFAULT_MINIMAX_EASYCACHE_START_PERCENT
    easycache_end_percent: float = DEFAULT_MINIMAX_EASYCACHE_END_PERCENT

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MiniMax H3 width and height must be positive")
        if self.length <= 0 or self.steps <= 0:
            raise ValueError("MiniMax H3 length and steps must be positive")
        if self.ref_image_size not in {"match", "max"}:
            raise ValueError("MiniMax H3 ref_image_size must be 'match' or 'max'")

    def model_path(self, value: Path | None, default: str) -> Path:
        path = value if value is not None else Path(default)
        return path if path.is_absolute() else self.models_dir / path


def run_r2v(*, images: list[Path], prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate a MiniMax H3 video with native synchronized audio."""
    config.validate()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not images:
        raise ValueError("at least one reference image is required")
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference image not found: {missing[0]}")

    try:
        from comfy_diffusion.image import load_image
        from comfy_diffusion.pipelines.video.minimax import h3
    except Exception as exc:
        raise RuntimeError("comfy-diffusion 2.6.0 with MiniMax H3 support is required") from exc

    reference_images = [load_image(path)[0] for path in images]
    artifact = make_video_path(out_dir, prefix="comfy-videogen-minimax-h3-r2v")
    # comfy-diffusion treats explicit override paths as already-resolved paths;
    # profile paths are intentionally relative to the configured models root.
    def resolved_override(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else config.models_dir / path

    _configure_optimizations(config)
    # The R2V helper owns its sampler internally. Temporarily decorating the
    # UNet loader lets us apply the same native EasyCache patch without
    # changing the public comfy-diffusion API.
    original_load_unet = None
    turbo_context = _turbo_runtime(config) if config.turbo_lora is not None else nullcontext()
    if config.easycache:
        from comfy_diffusion.models import ModelManager

        original_load_unet = ModelManager.load_unet

        def load_unet_with_easycache(manager: Any, path: Path) -> Any:
            return _apply_easycache(original_load_unet(manager, path), config)

        ModelManager.load_unet = load_unet_with_easycache  # type: ignore[method-assign]
    try:
        with turbo_context:
            result = h3.run(
                prompt,
                reference_images,
                models_dir=config.models_dir,
                width=config.width,
                height=config.height,
                length=config.length,
                steps=config.steps,
                seed=config.seed,
                ref_image_size=config.ref_image_size,
                unet_filename=resolved_override(config.unet),
                text_encoder_filename=resolved_override(config.text_encoder),
                audio_vae_filename=resolved_override(config.audio_vae),
                video_vae_filename=resolved_override(config.video_vae),
                output_path=artifact,
            )
            return {"artifact": artifact, "result": result}
    finally:
        if original_load_unet is not None:
            ModelManager.load_unet = original_load_unet  # type: ignore[method-assign]

def run_t2v(*, prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate MiniMax H3 text-to-video with synchronized audio."""
    return _run_fl2va(prompt=prompt, first_image=None, config=config, out_dir=out_dir, mode="t2v")


def run_i2v(*, image: Path, prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate MiniMax H3 image-to-video with synchronized audio."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    return _run_fl2va(prompt=prompt, first_image=image, config=config, out_dir=out_dir, mode="i2v")


def _run_fl2va(
    *,
    prompt: str,
    first_image: Path | None,
    config: MiniMaxH3Config,
    out_dir: Path,
    mode: str,
) -> dict[str, Any]:
    config.validate()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    try:
        from comfy_diffusion._runtime import ensure_comfyui_on_path
        from comfy_diffusion.audio import vae_decode_audio
        from comfy_diffusion.image import load_image
        from comfy_diffusion.models import ModelManager
        from comfy_diffusion.runtime import check_runtime
        from comfy_diffusion.sampling import basic_guider, basic_scheduler, get_sampler, random_noise, sample_custom
        from comfy_diffusion.vae import vae_decode_batch
        from comfy_diffusion.video import save_video_with_audio

        runtime = check_runtime()
        if runtime.get("error"):
            raise RuntimeError(f"ComfyUI runtime not available: {runtime['error']}")
        comfy_root = ensure_comfyui_on_path()
        _configure_optimizations(config)
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, _resize
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "runtime not available" in str(exc):
            raise
        raise RuntimeError("comfy-diffusion 2.6.0 with MiniMax H3 T2V/I2V support is required") from exc

    models_root = Path(config.models_dir)
    manager = ModelManager(models_root)
    first_tensor = load_image(first_image)[0] if first_image is not None else None
    preencoded_keyframe = None
    if first_tensor is not None:
        # Encode the keyframe before loading the 15 GB Qwen3-VL text encoder.
        # This avoids overlapping the image VAE and CLIP allocations on 16 GB
        # GPUs. The node below receives a tiny adapter that reuses this latent.
        conditioning_vae = manager.load_vae(
            _resolve_path(models_root, config.video_vae or Path("vae/minimax/minimax_h3_video_vae_fp16.safetensors"))
        )
        resized = _resize(first_tensor[:1], config.width, config.height, "disabled")
        preencoded_keyframe = conditioning_vae.encode(resized)
        del conditioning_vae, resized
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    clip = manager.load_clip(
        _resolve_path(models_root, config.text_encoder or Path("text_encoders/minimax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")),
        clip_type="minimax",
    )

    class _PreencodedVAE:
        def encode(self, _image: Any) -> Any:
            return preencoded_keyframe

    node_result = MiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=_PreencodedVAE() if preencoded_keyframe is not None else None,
        prompt=prompt,
        width=config.width,
        height=config.height,
        length=config.length,
        first_frame=first_tensor,
    )
    positive, latent = getattr(node_result, "result", node_result)
    # The FL2VA checkpoint nearly fills a 16 GB card. The conditioning node
    # only needs CLIP/VAE, so release those references before loading the UNet
    # for sampling and reload the VAEs for the final decode.
    del clip, first_tensor, preencoded_keyframe
    model = manager.load_unet(_resolve_path(models_root, config.fl2va_unet))
    model = _apply_turbo_lora(model, config)
    model = _apply_easycache(model, config)
    guider = basic_guider(model, positive)
    sampled = sample_custom(
        random_noise(config.seed),
        guider,
        _get_turbo_sampler(config),
        basic_scheduler(model, "simple", config.steps),
        latent_image=latent,
    )[0]
    audio_samples = sampled["samples"].unbind()[-1]
    video_vae = manager.load_vae(
        _resolve_path(models_root, config.video_vae or Path("vae/minimax/minimax_h3_video_vae_fp16.safetensors"))
    )
    audio_vae = manager.load_vae(
        _resolve_path(models_root, config.audio_vae or Path("vae/minimax/minimax_h3_audio_vae_fp32.safetensors"))
    )
    audio = vae_decode_audio(audio_vae, {"samples": audio_samples})
    if audio.ndim == 3 and audio.shape[1] > 8 and audio.shape[-1] <= 8:
        audio = audio.movedim(-1, 1)
    # Match ComfyUI's VAEDecodeAudio node. The shared comfy-diffusion helper
    # decodes and rearranges the waveform but does not apply this normalization.
    import torch

    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    frames = vae_decode_batch(video_vae, {"samples": sampled["samples"]})
    artifact = make_video_path(out_dir, prefix=f"comfy-videogen-minimax-h3-{mode}")
    save_video_with_audio(
        frames,
        {
            "waveform": audio,
            "sample_rate": int(
                getattr(
                    audio_vae,
                    "audio_sample_rate_output",
                    getattr(audio_vae, "audio_sample_rate", 32000),
                )
            ),
        },
        artifact,
        fps=DEFAULT_MINIMAX_FPS,
    )
    return {"artifact": artifact, "frames": frames}


def _resolve_path(models_dir: Path, path: Path) -> Path:
    """Resolve profile-relative model paths for comfy-diffusion overrides."""
    return path if path.is_absolute() else models_dir / path


@contextmanager
def _turbo_runtime(config: MiniMaxH3Config):
    """Apply the custom node's loader and sampler around the R2V helper."""
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy_diffusion.sampling as sampling
    from comfy_diffusion.models import ModelManager

    original_load_unet = ModelManager.load_unet
    original_get_sampler = sampling.get_sampler
    original_sample_custom = sampling.sample_custom

    def load_unet(manager: Any, path: Path) -> Any:
        return _apply_turbo_lora(original_load_unet(manager, path), config)

    def get_sampler(name: str) -> Any:
        return _load_turbo_node(config).MiniMaxH3TurboSampler().get_sampler()[0]

    def sample_custom(*args: Any, **kwargs: Any) -> Any:
        # comfy-diffusion's H3 helper assigns the second return value to the
        # decoded latent, but the Turbo workflow decodes SamplerCustomAdvanced
        # output[0]. Keep R2V aligned with the official Turbo workflow.
        output, denoised_output = original_sample_custom(*args, **kwargs)
        return denoised_output, output

    ModelManager.load_unet = load_unet  # type: ignore[method-assign]
    sampling.get_sampler = get_sampler
    sampling.sample_custom = sample_custom
    try:
        yield
    finally:
        ModelManager.load_unet = original_load_unet  # type: ignore[method-assign]
        sampling.get_sampler = original_get_sampler
        sampling.sample_custom = original_sample_custom


_TURBO_NODE_MODULES: dict[Path, Any] = {}


def _load_turbo_node(config: MiniMaxH3Config) -> Any:
    node_dir = Path(config.turbo_node)
    init = node_dir / "__init__.py"
    if not init.is_file():
        raise RuntimeError(
            "MiniMax H3 Turbo custom node is required at "
            f"{node_dir}; install ComfyUI-MiniMax-H3-Turbo or set turbo_node"
        )
    module = _TURBO_NODE_MODULES.get(node_dir)
    if module is None:
        name = f"comfy_agent_tools_minimax_turbo_{abs(hash(node_dir))}"
        spec = importlib.util.spec_from_file_location(name, init)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load MiniMax H3 Turbo custom node: {init}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _patch_turbo_node_device_handling(module)
        _TURBO_NODE_MODULES[node_dir] = module
    return module


def _patch_turbo_node_device_handling(module: Any) -> None:
    """Keep the custom node's curve-mode LoRA math on the model device."""
    adaln_delta = getattr(module, "_AdalnDelta", None)
    if adaln_delta is None:
        return

    import torch
    import torch.nn.functional as F

    def forward(self: Any, t_emb: Any) -> Any:
        base = self.base
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        st = self.shared.get("silu_temb")
        if st is not None:
            dtype = x.dtype
            device = x.device
            a = self.a.to(device=device, dtype=dtype)
            b = self.b.to(device=device, dtype=dtype)
            x = x + (b @ (a @ st.to(device=device, dtype=dtype).T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    adaln_delta.forward = forward


def _apply_turbo_lora(model: Any, config: MiniMaxH3Config) -> Any:
    lora_path = _resolve_path(config.models_dir, config.turbo_lora)
    if not lora_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 Turbo LoRA not found: {lora_path}")
    node = _load_turbo_node(config)
    import folder_paths

    lora_root = config.models_dir / "loras"
    try:
        lora_name = lora_path.relative_to(lora_root).as_posix()
        folder_paths.add_model_folder_path("loras", str(lora_root))
    except ValueError:
        folder_paths.add_model_folder_path("loras", str(lora_path.parent))
        lora_name = lora_path.name
    return node.MiniMaxH3TurboLoRA().apply_lora(
        model, lora_name, config.turbo_lora_strength
    )[0]


def _get_turbo_sampler(config: MiniMaxH3Config) -> Any:
    return _load_turbo_node(config).MiniMaxH3TurboSampler().get_sampler()[0]


def _configure_optimizations(config: MiniMaxH3Config) -> None:
    """Enable H3 attention optimizations for the current ComfyUI process."""
    if not config.sage_attention:
        return
    try:
        from comfy.ldm.modules import attention
        if not attention.SAGE_ATTENTION_IS_AVAILABLE:
            raise RuntimeError("SageAttention is enabled but the sageattention package is not installed")
        attention.optimized_attention = attention.attention_sage
        attention.optimized_attention_masked = attention.attention_sage
    except ImportError as exc:
        raise RuntimeError("SageAttention requires the sageattention package") from exc


def _apply_easycache(model: Any, config: MiniMaxH3Config) -> Any:
    if not config.easycache:
        return model
    try:
        import comfy.patcher_extension
        from comfy_extras.nodes_easycache import (
            EasyCacheHolder,
            easycache_calc_cond_batch_wrapper,
            easycache_forward_wrapper,
            easycache_sample_wrapper,
        )
    except ImportError as exc:
        raise RuntimeError("EasyCache is not available in the installed ComfyUI runtime") from exc
    model = model.clone()
    model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
        config.easycache_reuse_threshold,
        config.easycache_start_percent,
        config.easycache_end_percent,
        subsample_factor=9,
        offload_cache_diff=False,
        output_channels=model.model.latent_format.latent_channels,
    )
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, "easycache", easycache_calc_cond_batch_wrapper)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
    return model
