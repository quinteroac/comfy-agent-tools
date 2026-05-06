---
name: comfy-model-onboarding
description: Configure local comfy-agent-tools model profiles and defaults. Use when the user has no .comfy-agent-tools.json config, wants to set models_dir, change a default capability profile, add a new checkpoint or fine-tune of a supported architecture, or diagnose profile/config errors such as unknown_profile, unsupported_capability, architecture_mismatch, config_error, missing_model_file, or unsupported_architecture.
---

# comfy-model-onboarding

Use this skill to help configure model profiles for `comfy-imagegen`,
`comfy-videogen`, and `comfy-musicgen`. Profiles are selected by capability, not
by an entire tool. A profile is not an architecture: for example `ltx23` is the
architecture/adapter, while `ltx23-10eros` is one profile of that architecture.

For loose LoRAs, naming, moving, or selecting ad hoc LoRAs, use
`comfy-lora-onboarding`. Do not add one-off style LoRAs to model profiles unless
the user wants that LoRA to become a persistent default.

If `comfy-models` is not available, use `comfy-tools-setup` first. In this
repository, prefer `uv run comfy-models`; outside the repo, let
`comfy-tools-setup` install the CLIs with `uv tool`.

## Setup

If `.comfy-agent-tools.json` is missing, initialize and validate the local config:

```bash
uv run comfy-models init
uv run comfy-models validate
uv run comfy-models show
```

If models live somewhere else:

```bash
uv run comfy-models set-models-dir /mnt/models/comfyui
```

## Supported Architectures

Only configure these architectures in v1:

- `qwen-image-edit`: base profile `qwen-edit2511`, capabilities `imagegen.generate`, `imagegen.edit`.
- `anima`: base profile `anima-preview3-turbo`, capability `imagegen.generate`.
- `upscale-model`: base profile `clear-reality`, capability `imagegen.upscale`.
- `ltx23`: base profile `ltx23-10eros`, capabilities `videogen.t2v`, `videogen.i2v`, `videogen.flf2v`, `videogen.ia2av`.
- `ace-step-1.5`: base profile `ace15-base`, capability `musicgen.generate`.

If a model is SDXL, Flux, Wan, a new audio architecture, or anything else, do not
pretend it is supported. Explain that a new architecture adapter is required.

## Onboarding Flow

1. Inspect the user-provided path and current models dir with non-destructive
   shell commands such as `ls`, `find`, or safetensors metadata reads.
2. Identify architecture first, then choose a base profile.
3. For fine-tunes, prefer `extends` and override only changed model paths.
4. Validate the profile before making it a default.
5. Set defaults per capability only where the profile supports that capability.
6. Recommend a small smoke test.

## Fine-Tune Examples

LTX 2.3 checkpoint/fine-tune:

```bash
uv run comfy-models add-profile my-ltx23-finetune \
  --extends ltx23-10eros \
  --checkpoint checkpoints/my_ltx23_finetune.safetensors

uv run comfy-models validate-profile my-ltx23-finetune
uv run comfy-models set-default videogen.t2v my-ltx23-finetune
```

Qwen Image Edit fine-tune:

```bash
uv run comfy-models add-profile my-qwen-edit \
  --extends qwen-edit2511 \
  --unet diffusion_models/my_qwen_edit.safetensors

uv run comfy-models validate-profile my-qwen-edit
uv run comfy-models set-default imagegen.generate my-qwen-edit
uv run comfy-models set-default imagegen.edit my-qwen-edit
```

Anima fine-tune or compatible LoRA:

```bash
uv run comfy-models add-profile my-anima \
  --extends anima-preview3-turbo \
  --unet diffusion_models/my_anima.safetensors \
  --lora loras/my_anima_turbo_lora.safetensors

uv run comfy-models validate-profile my-anima
uv run comfy-models set-default imagegen.generate my-anima
```

Anima profiles generate images only. They do not support `imagegen.edit`. Keep
turbo Anima profiles at `steps=8` and `cfg=1.0` unless the user explicitly wants
to test a slower non-turbo profile.

ACE-Step 1.5 variant:

```bash
uv run comfy-models add-profile my-ace15 \
  --extends ace15-base \
  --unet diffusion_models/my_ace15.safetensors

uv run comfy-models validate-profile my-ace15
uv run comfy-models set-default musicgen.generate my-ace15
```

## Smoke Tests

- Image generate: `uv run comfy-imagegen generate --prompt "simple cinematic portrait" --width 512 --height 512`
- Image edit: use a small existing PNG and a simple prompt.
- Upscale: use a small existing PNG.
- Video: use `--length 49 --fps 24`.
- Music: use `--duration 30 --steps 8 --cfg 1`.

Do not install custom nodes or add unsupported adapters in this skill. For
missing files in supported built-in profiles, use `comfy-model-downloader`.
Keep changes scoped to `.comfy-agent-tools.json` and validated profile defaults.
