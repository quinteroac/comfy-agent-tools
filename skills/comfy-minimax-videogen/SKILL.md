---
name: comfy-minimax-videogen
description: Generate local MiniMax H3 text-to-video, image-to-video, or reference-to-video clips with synchronized native audio using comfy-videogen. Use when the user wants local MiniMax H3 video generation. Do not use for MiniMax hosted APIs, Seedance, image-only generation, or ComfyUI server workflows.
---

# comfy-minimax-videogen

Use this skill for local MiniMax H3 text-to-video, image-to-video, and
reference-to-video generation through the `comfy-videogen` CLI. The pipeline is
included in `comfy-diffusion` 2.6.0 and generates an MP4 with synchronized audio.
All MiniMax modes use the MiniMax-H3 Turbo LoRA and its four-step audio/video
sampler. The LoRA is expected at
`loras/minimax/minimax_h3_turbo_4step_ckpt850.safetensors` under the
configured models directory. The custom node directory defaults to
`/home/victor/AI/Comfy/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo`; use
`--turbo-node` when ComfyUI is installed elsewhere.

For MiniMax H3 generations, enable both accelerators by default by passing
`--sage-attention --easycache`. Only omit either flag when the user explicitly
requests it or prioritizes quality/compatibility over speed. EasyCache remains
an opt-in CLI default internally; this skill makes it the agent's default
choice.

If `comfy-videogen` is unavailable, use `comfy-tools-setup` first. In this
repository, prefer `uv run comfy-videogen`.

Before the first run, make sure the model directory is configured. If the
MiniMax H3 files are missing, download only this capability:

```bash
uv run comfy-models download videogen.minimax-h3-r2v --dry-run
uv run comfy-models download videogen.minimax-h3-r2v --yes
```

Generate from one reference image:

```bash
uv run comfy-videogen minimax-h3-r2v \
  --input path/to/reference.png \
  --prompt 'Use <Picture 1> as the character reference; a cinematic shot of the character walking through rain, with natural ambience and dialogue.' \
  --out outputs
```

Text to video uses the FL2VA checkpoint without an input image:

```bash
uv run comfy-videogen minimax-h3-t2v \
  --prompt 'A cinematic aerial shot of a misty mountain valley at sunrise, with wind and distant birds.' \
  --out outputs
```

Image to video uses the first frame as a visual anchor:

```bash
uv run comfy-videogen minimax-h3-i2v \
  --input path/to/first-frame.png \
  --prompt 'Slow camera push-in, subtle natural motion, and synchronized ambient sound.' \
  --out outputs
```

## Long videos with chained shots

MiniMax H3 is best used as a chain of short shots. The CLI renders the first
shot as T2V, feeds its final frame into the next shot as I2V, removes the
duplicated seam frame, and stitches the audio. Use one prompt per shot,
separated by `---`, and let `--duration` determine the number of shots:

```bash
uv run comfy-videogen minimax-h3-multishot-t2v \
  --prompt '{"prompts":["Wide establishing shot of a rainy neon street.","A courier runs past the camera, then turns into a narrow alley.","The courier reaches a warm doorway and looks back at the storm."]}' \
  --duration 30 \
  --shot-duration 10 \
  --width 896 --height 576 \
  --sage-attention --easycache \
  --out outputs
```

For example, `--duration 30 --shot-duration 5` renders six chained shots;
`--duration 30 --shot-duration 10` renders three. The final MP4 is trimmed to
the requested duration because H3 snaps each shot to its valid temporal grid.
Use `--shot-count` to force a count, or provide JSON as
`{"prompts": ["shot 1", "shot 2"]}`. Keep EasyCache opt-in because it can
reduce quality; SageAttention is also opt-in. Add one `--input` and use
`--first-mode i2v` to seed the first shot from an image. Repeat `--input` and
use `--first-mode r2v` for multiple first-shot references; with `--first-mode
auto` (the default), no input means T2V, one input means I2V, and multiple
inputs mean R2V.

Multiple reference images can be supplied by repeating `--input`. Address them
in the prompt in the same order as the arguments: `<Picture 1>`, `<Picture 2>`,
and so on.

Important options:

- `--width` and `--height`: output dimensions; defaults are 1344x768.
- `--length`: requested frame count; H3 snaps it to its valid `17k + 5` grid.
- `--steps`: sampling steps, default 4 for the Turbo LoRA.
- `--ref-image-size match|max`: `max` retains larger reference detail but uses
  more memory and time.
- `--seed`: reproducible sampling seed.
- `--unet`, `--text-encoder`, `--audio-vae`, and `--video-vae`: explicit model
  overrides when the configured profile paths are not used.
- `--turbo-lora`, `--turbo-lora-strength`, and `--turbo-node`: Turbo overrides.

The default profile is `minimax-h3`, architecture `minimax-h3`, and capabilities
`videogen.minimax-h3-t2v`, `videogen.minimax-h3-i2v`, and
`videogen.minimax-h3-r2v`. It is local and does not use `COMFY_ORG_API_KEY`.
Models are stored under the configured ComfyUI models directory:

- `diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `diffusion_models/minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `text_encoders/minimax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `vae/minimax/minimax_h3_audio_vae_fp32.safetensors`
- `vae/minimax/minimax_h3_video_vae_fp16.safetensors`

Successful runs write an MP4 and a comfy-media run manifest. After generation,
use `comfy-media` to index or review the output.
