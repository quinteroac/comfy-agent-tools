"""Run MiniMax H3 on Modal.

Examples:
    modal run modal/minimax_h3.py --mode t2v --prompt "..." --prepare-models
    modal run modal/minimax_h3.py --mode i2v --input first.png --prompt "..."
    modal run modal/minimax_h3.py --mode r2v --input reference.png --prompt "..."
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import modal

from comfy_agent_tools.modal_minimax import (
    DEFAULT_MODAL_GPU,
    DEFAULT_MODAL_VOLUME,
    MODEL_MOUNT,
    prepare_models as prepare_modal_models,
    validate_modal_auth,
)
from comfy_agent_tools.videogen.multishot import plan_shots, run_multishot_t2v


APP_NAME = "comfy-minimax-h3"
GPU = os.environ.get("MINIMAX_MODAL_GPU", DEFAULT_MODAL_GPU)
VOLUME_NAME = os.environ.get("MINIMAX_MODAL_VOLUME", DEFAULT_MODAL_VOLUME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime")
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "comfy-diffusion[video,audio] @ git+https://github.com/quinteroac/comfy-diffusion.git@v2.6.0",
        "psutil",
        "sageattention==1.0.6",
        "comfy-aimdo==0.4.10",
        "comfy-kitchen>=0.2.26",
        "comfyui-embedded-docs==0.5.5",
        "comfyui-frontend-package==1.44.19",
        "comfyui-workflow-templates==0.10.3",
        "gguf",
        "glfw",
        "kornia>=0.7.1",
        "matplotlib",
        "omegaconf>=2.3.0",
        "pydantic-settings~=2.0",
        "pydantic~=2.0",
        "pyopengl",
        "rotary_embedding_torch>=0.5.3",
        "torchsde",
        "spandrel",
        "sqlalchemy>=2.0",
    )
    .add_local_python_source("comfy_agent_tools")
)
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
    volumes={str(MODEL_MOUNT): models_volume.with_mount_options(read_only=True)},
)
def generate(
    *,
    mode: str,
    prompt: str,
    inputs: list[bytes] | None = None,
    width: int = 1344,
    height: int = 768,
    length: int = 124,
    steps: int = 20,
    seed: int = 0,
    ref_image_size: str = "match",
    sage_attention: bool = False,
    easycache: bool = False,
    multishot: bool = False,
    duration: float = 0.0,
    shot_duration: float = 10.0,
    shot_count: int = 0,
    first_mode: str = "auto",
) -> bytes:
    """Generate one MP4 and return its bytes to the local Modal entrypoint."""
    from comfy_agent_tools.videogen.minimax import MiniMaxH3Config, run_i2v, run_r2v, run_t2v

    if mode not in {"t2v", "i2v", "r2v"}:
        raise ValueError(f"unsupported MiniMax H3 mode: {mode}")
    if mode != "t2v" and not inputs:
        raise ValueError(f"{mode} requires at least one input image")
    with tempfile.TemporaryDirectory(prefix="minimax-h3-modal-") as temp_dir:
        temp_root = Path(temp_dir)
        input_paths: list[Path] = []
        for index, data in enumerate(inputs or []):
            path = temp_root / f"input-{index}.png"
            path.write_bytes(data)
            input_paths.append(path)
        config = MiniMaxH3Config(
            models_dir=MODEL_MOUNT,
            width=width,
            height=height,
            length=length,
            steps=steps,
            seed=seed,
            ref_image_size=ref_image_size,
            sage_attention=sage_attention,
            easycache=easycache,
        )
        if multishot:
            if mode != "t2v":
                raise ValueError("multishot is only supported with --mode t2v")
            if duration <= 0:
                raise ValueError("multishot requires --duration greater than zero")
            result = run_multishot_t2v(
                script=prompt,
                duration=duration,
                shot_duration=shot_duration,
                shot_count=shot_count or None,
                start_images=input_paths,
                first_mode=first_mode,
                config=config,
                out_dir=temp_root,
            )
        elif mode == "t2v":
            result = run_t2v(prompt=prompt, config=config, out_dir=temp_root)
        elif mode == "i2v":
            result = run_i2v(image=input_paths[0], prompt=prompt, config=config, out_dir=temp_root)
        else:
            result = run_r2v(images=input_paths, prompt=prompt, config=config, out_dir=temp_root)
        return Path(result["artifact"]).read_bytes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MiniMax H3 on Modal.")
    parser.add_argument("--mode", choices=("t2v", "i2v", "r2v"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--volume", default=VOLUME_NAME)
    parser.add_argument(
        "--prepare-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the Volume and download/upload missing H3 models (default: enabled).",
    )
    parser.add_argument("--force-upload", action="store_true")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--length", type=int, default=124)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref-image-size", choices=("match", "max"), default="match")
    parser.add_argument("--sage-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--easycache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--multishot", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--shot-duration", type=float, default=10.0)
    parser.add_argument("--shot-count", type=int, default=0)
    parser.add_argument("--first-mode", choices=("auto", "t2v", "i2v", "r2v"), default="auto")
    return parser


def _artifact_path(out_dir: Path, mode: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"comfy-videogen-minimax-h3-modal-{mode}.mp4"


@app.local_entrypoint()
def main(
    mode: str,
    prompt: str,
    input: str = "",
    out: str = "outputs",
    models_dir: str = "",
    volume: str = VOLUME_NAME,
    prepare_models: bool = True,
    force_upload: bool = False,
    width: int = 1344,
    height: int = 768,
    length: int = 124,
    steps: int = 20,
    seed: int = 0,
    ref_image_size: str = "match",
    sage_attention: bool = False,
    easycache: bool = False,
    multishot: bool = False,
    duration: float = 0.0,
    shot_duration: float = 10.0,
    shot_count: int = 0,
    first_mode: str = "auto",
) -> None:
    """Modal CLI entrypoint; arguments are exposed by Modal from this signature."""
    args = argparse.Namespace(
        mode=mode,
        prompt=prompt,
        input=[Path(value) for value in input.split(",") if value],
        out=Path(out),
        models_dir=Path(models_dir) if models_dir else None,
        volume=volume,
        prepare_models=prepare_models,
        force_upload=force_upload,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=seed,
        ref_image_size=ref_image_size,
        sage_attention=sage_attention,
        easycache=easycache,
        multishot=multishot,
        duration=duration,
        shot_duration=shot_duration,
        shot_count=shot_count,
        first_mode=first_mode,
    )
    validate_modal_auth()
    if args.mode == "t2v" and args.input and not args.multishot:
        raise ValueError("t2v does not accept --input")
    if args.mode != "t2v" and not args.input:
        raise ValueError(f"{args.mode} requires --input; repeat it for r2v references")
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(f"input image not found: {path}")
    multishot_plan = None
    if args.multishot:
        multishot_plan = plan_shots(
            args.prompt,
            duration=args.duration,
            shot_duration=args.shot_duration,
            shot_count=args.shot_count or None,
        )
    preparation: dict[str, object] | None = None
    if args.volume != VOLUME_NAME:
        raise ValueError(
            f"--volume {args.volume!r} differs from the mounted volume {VOLUME_NAME!r}; "
            "set MINIMAX_MODAL_VOLUME before invoking modal run"
        )
    if args.prepare_models:
        preparation_modes = [args.mode]
        if args.multishot and (args.first_mode == "r2v" or (args.first_mode == "auto" and len(args.input) > 1)):
            preparation_modes = ["r2v", "t2v"]
        prepared = [
            prepare_modal_models(
                mode,
                models_dir=args.models_dir,
                volume=args.volume,
                force_upload=args.force_upload,
            )
            for mode in preparation_modes
        ]
        preparation = prepared[0] if len(prepared) == 1 else {"modes": preparation_modes, "runs": prepared}
    elif not args.volume:
        raise ValueError("--volume must not be empty")

    data = [path.read_bytes() for path in args.input]
    artifact = _artifact_path(args.out, args.mode)
    video = generate.remote(
        mode=args.mode,
        prompt=args.prompt,
        inputs=data,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        ref_image_size=args.ref_image_size,
        sage_attention=args.sage_attention,
        easycache=args.easycache,
        multishot=args.multishot,
        duration=args.duration,
        shot_duration=args.shot_duration,
        shot_count=args.shot_count,
        first_mode=args.first_mode,
    )
    artifact.write_bytes(video)
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": f"minimax-h3-{'multishot-t2v' if args.multishot else args.mode}",
        "remote": True,
        "provider": "modal",
        "gpu": GPU,
        "volume": args.volume,
        "prompt": args.prompt,
        "artifacts": [str(artifact)],
        "width": args.width,
        "height": args.height,
        "frames_requested": (
            round(args.duration * 24) if args.multishot else args.length
        ),
        "steps": args.steps,
        "seed": args.seed,
        "audio_muxed": True,
        "multishot": args.multishot,
    }
    if args.multishot:
        payload.update({
            "duration": args.duration,
            "shot_duration": args.shot_duration,
            "shot_count": len(multishot_plan[0]) if multishot_plan else None,
            "frames_per_shot": multishot_plan[1] if multishot_plan else None,
            "first_mode": args.first_mode,
            "start_images": [str(path) for path in args.input],
        })
    if args.input:
        payload["inputs"] = [str(path) for path in args.input]
    if preparation is not None:
        payload["preparation"] = preparation
    print(json.dumps(payload, sort_keys=True))
