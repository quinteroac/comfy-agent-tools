---
name: comfy-imagegen
description: Generate, edit, or upscale raster images locally with comfy-diffusion, Anima Preview3, Qwen Image Edit 2511, FLUX.2 Klein 9B SNOFS, and ClearReality. Use when the user wants local GPU-backed image generation or image editing from the current machine, especially when outputs should be saved into the workspace. Do not use for hosted OpenAI image generation, vector/SVG work, video, music, voice, model downloads, custom node installation, or ComfyUI server workflows.
---

# comfy-imagegen

Use this skill for local image generation, image editing, and image upscaling through
the `comfy-imagegen` CLI. The CLI uses models under `/mnt/models/comfyui`.
If a supported built-in model is missing, use `comfy-model-downloader` to fetch
only the requested capability before running inference.

The CLI is quiet by default and prints only final JSON. Use `--verbose` only when
debugging ComfyUI runtime output, warnings, or progress bars.

If `comfy-imagegen` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-imagegen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

At the start of every image workflow, start or reuse the local Comfy Media
gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use `comfy-media --help` only if the CLI is missing or behaves unexpectedly; do
not skip the gallery just because generation can run headless.

If `.comfy-agent-tools.json` is missing or the user wants to configure a new
checkpoint/fine-tune/default, use `comfy-model-onboarding` first.

If model validation fails with `missing_model_file`, use `comfy-model-downloader`:
`imagegen.generate` for the active generation profile, `imagegen.edit` for the
active edit profile, or `imagegen.upscale` for ClearReality upscale.

If the user asks to use or organize a LoRA by name or purpose, use
`comfy-lora-onboarding` to search `loras/<architecture>/` first and pass the
chosen file with `--extra-lora`.

## Modes

- `generate`: text prompt to image with Anima Preview3 + turbo LoRA. Use for
  anime, illustration, and non-photorealistic art by default. When the active
  profile is `flux-klein-9b-snofs`, use FLUX.2 Klein 9B FP8 plus SNOFS.
- `edit`: input image plus prompt to edited image with Qwen Image Edit 2511.
  When the active profile is `flux-klein-9b-snofs`, use FLUX.2 Klein 9B plus
  SNOFS for single-reference editing.
- `upscale`: input image to 4x upscale with ClearReality.

## Commands

Generate:

```bash
uv run comfy-imagegen generate \
  --prompt "masterpiece, best quality, score_7, safe, 1girl, anime style, cinematic lighting, detailed background" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --extra-lora /mnt/models/comfyui/loras/anima/realism-portrait.safetensors:0.8:0.0 \
  --out outputs
```

Edit:

```bash
uv run comfy-imagegen edit \
  --input path/to/input.png \
  --prompt "Transform this photo into a polished animated film still style while preserving the main subject and composition" \
  --seed 43 \
  --out outputs
```

For FLUX.2 Klein edit profiles, pass `--width` and/or `--height` to choose the
output canvas. Any omitted dimension defaults to the input image dimension.

Upscale:

```bash
uv run comfy-imagegen upscale \
  --input path/to/input.png \
  --out outputs
```

## Prompt Guidance

For generation, Anima accepts Danbooru-style tags, natural language, or a mix.
Use lowercase tags with spaces instead of underscores except score tags.
Recommended positive prefix: `masterpiece, best quality, score_7, safe, ...`.
Recommended negative guidance when adapting prompts: `worst quality, low quality,
score_1, score_2, score_3, artist name`.

Anima is not a realism model. It is intended for anime, illustration, and art.
Keep generation around 1MP, such as 1024x1024, 896x1152, or 1152x896.

For editing, write an instruction that names what should change and what should stay
fixed. Preserve identity, layout, pose, and important object details unless the user
explicitly asks to change them.

Qwen Image Edit may rescale internally, so read `outputs[].width` and
`outputs[].height` from the final JSON instead of assuming requested or input
dimensions match the saved PNG.

For upscale, do not add prompt text; use the input image only.

FLUX.2 Klein 9B SNOFS uses natural-language prompts, supports both generation
and editing, and is step-distilled for `steps=4`, `cfg=1.0`. Keep dimensions
divisible by 16. It requires gated Black Forest Labs weights and SNOFS has a
separate personal-use license: local image generation is allowed, generated
images may be sold, but public/commercial generation services, derivative model
creation, and weight redistribution are not allowed without a separate license.
For edits, Flux Klein follows the official distilled image-edit workflow:
reference-image VAE encoding, reference latents on both positive and negative
conditioning, `Flux2Scheduler`, `CFGGuider`, and `SamplerCustomAdvanced`.

## Defaults

- Models directory: `/mnt/models/comfyui`
- Generate profile: `anima-preview3-turbo`
- Anima diffusion model: `diffusion_models/animaOfficial_preview3Base.safetensors`
- Anima text encoder: `text_encoders/qwen_3_06b_base.safetensors`
- VAE: `vae/qwen_image_vae.safetensors`
- Anima turbo LoRA: `loras/anima/anima-turbo-lora-v0.1.safetensors`
- Anima params: `steps=8`, `cfg=1.0`, `seed=0`
- Edit profile: `qwen-edit2511`
- Qwen diffusion model: `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors`
- Qwen text encoder: `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- Qwen LoRA: `loras/qwen-image-edit/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
- Qwen params: `steps=4`, `cfg=3.0`, `seed=0`
- FLUX profile: `flux-klein-9b-snofs`
- FLUX architecture: `flux-klein`
- FLUX diffusion model: `diffusion_models/flux-2-klein-9b-fp8.safetensors`
- FLUX text encoder: `text_encoders/qwen_3_8b_fp8mixed.safetensors`
- FLUX VAE: `vae/flux2-vae.safetensors`
- SNOFS LoRA: `loras/flux-klein/klein_snofs_v1_1.safetensors`
- FLUX params: `steps=4`, `cfg=1.0`, `sampler=euler`, `seed=0`
- Upscaler: `upscale_models/4x-ClearRealityV1.pth`
- Dependency: `comfy-diffusion[comfyui,video]`

The Anima turbo LoRA expects `cfg=1.0`; increasing CFG can degrade or break the
expected turbo behavior.

Extra LoRAs are optional and ad hoc. Use repeatable
`--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]` after resolving the file
through the LoRA folder convention:

- `loras/anima/` for Anima generation LoRAs.
- `loras/qwen-image-edit/` for Qwen edit LoRAs.
- `loras/flux-klein/` for FLUX.2 Klein LoRAs such as SNOFS.
- fallback to loose files in `loras/` only when no architecture folder match is clear.

Warnings about missing `torchaudio` can appear in verbose mode because ComfyUI imports
audio-capable modules. They are expected while this skill is only using image workflows.

## Output Handling

The CLI prints JSON to stdout. On success, read `artifacts` for the saved PNG paths
and `outputs` for final image dimensions. On failure, read `error` and `error_type`;
do not parse logs for control flow.

Project-bound images should be saved under the current workspace. Do not overwrite
existing assets unless the user explicitly asks for replacement.

After every successful image generation, edit, or upscale command, immediately
index the same output directory so the new artifact appears in Comfy Media:

```bash
uv run comfy-media index --out outputs
```
