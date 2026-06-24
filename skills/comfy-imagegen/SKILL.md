---
name: comfy-imagegen
description: Generate, edit, or upscale raster images with comfy-diffusion, including local Anima Base v1.0 with turbo LoRA, Qwen Image Edit 2511, FLUX.2 Klein 9B SNOFS, local Ideogram 4 structured prompt/bbox generation, local Krea2 Turbo, ClearReality, and remote Grok Imagine API nodes. Use when the user wants image generation or image editing from the current machine with outputs saved into the workspace. Do not use for hosted OpenAI image generation, vector/SVG work, video, music, voice, model downloads, custom node installation, or ComfyUI server workflows.
---

# comfy-imagegen

Use this skill for image generation, image editing, and image upscaling through
the `comfy-imagegen` CLI. Local modes use models under `/mnt/models/comfyui`.
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
active edit profile, `imagegen.ideogram4-generate` for Ideogram 4, `imagegen.krea2-generate` for Krea2 Turbo, or
`imagegen.upscale` for ClearReality upscale.

If the user asks to use or organize a LoRA by name or purpose, use
`comfy-lora-onboarding` to search `loras/<architecture>/` first and pass the
chosen file with `--extra-lora`.

## Modes

- `generate`: text prompt to image with Anima Base v1.0 + turbo LoRA. Use for
  anime, illustration, and non-photorealistic art by default. When the active
  profile is `flux-klein-9b-snofs`, use FLUX.2 Klein 9B FP8 plus SNOFS.
- `edit`: input image plus prompt to edited image with Qwen Image Edit 2511.
  When the active profile is `flux-klein-9b-snofs`, use FLUX.2 Klein 9B plus
  SNOFS for single-reference editing.
- `upscale`: input image to 4x upscale with ClearReality.
- `rtx-upscale`: input image to a target resolution (480p/720p/1080p/1440p/4k/8k),
  custom width/height, or 1.0-4.0 scale with NVIDIA RTX Video Super Resolution.
  Requires an NVIDIA RTX GPU and the `nvidia-vfx` package.
- `ideogram4-generate`: local Ideogram 4 text-to-image. Use this for poster,
  typography, graphic design, color palette, and bbox/layout-controlled images.
- `krea2-generate`: local Krea2 Turbo text-to-image. Use this for fast, high-
  fidelity prompt-following generation with a Qwen3-VL text encoder and the
  Krea2 conditioning rebalance.
- `grok-generate`: remote Grok Imagine text prompt to PNG through Comfy API
  nodes. Requires `COMFY_ORG_API_KEY`, not local model files.
- `grok-edit`: remote Grok Imagine input image plus prompt to PNG through Comfy
  API nodes. Requires `COMFY_ORG_API_KEY`, not local model files.

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

RTX upscale to a target resolution:

```bash
uv run comfy-imagegen rtx-upscale \
  --input path/to/input.png \
  --resolution 4k \
  --quality ULTRA \
  --out outputs
```

RTX upscale by scale factor or custom size (requires an NVIDIA RTX GPU and
`nvidia-vfx`):

```bash
uv run comfy-imagegen rtx-upscale --input path/to/input.png --scale 2.0 --out outputs
uv run comfy-imagegen rtx-upscale --input path/to/input.png --width 2560 --height 1440 --out outputs
```

Use exactly one of `--resolution`, `--scale`, or `--width`/`--height`. The
aspect ratio is preserved for `--resolution` and `--scale`; `--width` and
`--height` must be provided together. Quality presets are `LOW`, `MEDIUM`,
`HIGH`, and `ULTRA` (default `ULTRA`).

Ideogram 4:

```bash
uv run comfy-imagegen ideogram4-generate \
  --prompt "A poster for a jazz night" \
  --style-aesthetics "minimal, geometric, high contrast" \
  --style-lighting "flat graphic design lighting" \
  --style-medium "graphic_design" \
  --style-art-style "clean vector poster, sans-serif typography" \
  --background "Matte black paper background with subtle grain." \
  --text "90,120,260,880|JAZZ NIGHT|Large cream uppercase headline." \
  --output-json outputs/ideogram4-prompt.json \
  --width 1024 \
  --height 1024 \
  --out outputs
```

Ideogram 4 bbox builder:

```bash
uv run comfy-imagegen ideogram4-generate \
  --prompt "A modern concert poster for a jazz trio." \
  --style-aesthetics "minimal, elegant, high contrast" \
  --style-lighting "flat graphic design lighting" \
  --style-medium "graphic_design" \
  --style-art-style "clean vector poster, sans-serif typography" \
  --style-color "#101010" \
  --style-color "#F4D35E" \
  --background "Solid black background with subtle paper texture." \
  --object "420,120,900,880|A golden saxophone centered in the lower half." \
  --text "80,120,220,880|JAZZ NIGHT|Large condensed yellow headline." \
  --output-json outputs/ideogram4-poster-prompt.json \
  --out outputs
```

Krea2 Turbo:

```bash
uv run comfy-imagegen krea2-generate \
  --prompt "a cinematic portrait of an astronaut floating in a nebula, dramatic rim light" \
  --width 1024 \
  --height 1024 \
  --steps 8 \
  --cfg 1 \
  --rebalance-multiplier 4.0 \
  --out outputs
```

Grok Imagine:

```bash
COMFY_ORG_API_KEY=... uv run comfy-imagegen grok-generate \
  --prompt "A cinematic product photo of a translucent orange cassette player on wet asphalt" \
  --model grok-imagine-image \
  --resolution 1K \
  --aspect-ratio 1:1 \
  --out outputs
```

Grok Imagine edit:

```bash
COMFY_ORG_API_KEY=... uv run comfy-imagegen grok-edit \
  --input path/to/input.png \
  --prompt "Keep the subject and composition, change the background to a clean moonlit studio" \
  --resolution 1K \
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

Ideogram 4 is trained for structured JSON captions, and the CLI builds that JSON
from flags. Do not pass raw JSON. Required fields are `--prompt`,
`--style-aesthetics`, `--style-lighting`, `--style-medium`, `--background`,
exactly one of `--style-photo` or `--style-art-style`, and at least one
`--object` or `--text`. Bboxes are `y_min,x_min,y_max,x_max` in normalized
`0..1000` coordinates, not pixels. Use `--object "BBOX|DESCRIPTION"` for visual
subjects and `--text "BBOX|TEXT|DESCRIPTION"` for literal rendered text. Use
uppercase `#RRGGBB` color values; the CLI normalizes `--style-color` values to
uppercase. Use `--output-json PATH` when the generated Ideogram prompt JSON
should be saved for inspection or reuse.

Grok Imagine uses remote Comfy API nodes and is separate from local model
profiles. Do not route `grok-imagine-api` through `models_dir`,
`comfy-model-downloader`, local LoRAs, or checkpoint onboarding. Supported image
models are `grok-imagine-image-pro`, `grok-imagine-image`, and
`grok-imagine-image-beta`; supported resolutions are `1K` and `2K`.

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
- Generate profile: `anima-base`
- Anima diffusion model: `diffusion_models/anima-base-v1.0.safetensors`
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
- Ideogram profile: `ideogram4-fp8`
- Ideogram diffusion model: `diffusion_models/ideogram4_fp8_scaled.safetensors`
- Ideogram unconditional model: `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors`
- Ideogram text encoder: `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`
- Ideogram VAE: `vae/flux2-vae.safetensors`
- Ideogram params: `steps=20`, `cfg=7.0`, `cfg_override_value=3.0`, `sampler=euler`, `seed=0`
- Krea2 profile: `krea2-turbo`
- Krea2 diffusion model: `diffusion_models/krea2_turbo_fp8_scaled.safetensors`
- Krea2 text encoder: `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
- Krea2 VAE: `vae/qwen_image_vae.safetensors`
- Krea2 params: `steps=8`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`, `rebalance_multiplier=4.0`, `seed=0`
- Upscaler: `upscale_models/4x-ClearRealityV1.pth`
- RTX upscale profile: `rtx-vsr` (shared with `videogen.rtx-upscale`)
- RTX upscale params: `resolution=1080p`, `quality=ULTRA`; presets `480p`, `720p`, `1080p`, `1440p`, `4k`, `8k`; `--scale` 1.0-4.0
- RTX upscale dependency: `nvidia-vfx` plus a CUDA-capable NVIDIA RTX GPU
- Grok profile: `grok-imagine-api`
- Grok provider: `comfy-api`
- Grok model: `grok-imagine-image`
- Grok params: `resolution=1K`, `aspect_ratio=1:1`, `number_of_images=1`, `seed=0`
- Dependency: `comfy-diffusion[video,audio]` v2.4.5 or newer plus the vendored
  ComfyUI requirements

The Anima turbo LoRA expects `cfg=1.0`; increasing CFG can degrade or break the
expected turbo behavior.

Extra LoRAs are optional and ad hoc. Use repeatable
`--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]` after resolving the file
through the LoRA folder convention:

- `loras/anima/` for Anima generation LoRAs.
- `loras/qwen-image-edit/` for Qwen edit LoRAs.
- `loras/flux-klein/` for FLUX.2 Klein LoRAs such as SNOFS.
- `loras/ideogram4/` for Ideogram 4 style or poster/detail LoRAs.
- fallback to loose files in `loras/` only when no architecture folder match is clear.

For Ideogram 4, `--extra-lora` patches both the main and unconditional UNets.
Nonzero `CLIP_STRENGTH` patches the shared Ideogram text encoder once.

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
