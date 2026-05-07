---
name: comfy-model-downloader
description: Download missing built-in comfy-agent-tools model files on demand by capability. Use when a generation/edit/upscale/music/video request needs local models that are missing, when comfy-models reports missing_model_file, or when the user asks to download supported base models. Do not use to download all models unless explicitly requested.
---

# comfy-model-downloader

Use this skill to download missing model files for supported built-in profiles.
Downloads are capability-scoped: download only what the user's current request
needs.

If `comfy-models` is not available, use `comfy-tools-setup` first. In this repo,
prefer `uv run comfy-models`; outside the repo, use the installed CLI.

## Capability Mapping

- Image generation: `imagegen.generate`
- Image editing: `imagegen.edit`
- Image upscale: `imagegen.upscale`
- Text to video: `videogen.t2v`
- Image to video: `videogen.i2v`
- First/last frame video: `videogen.flf2v`
- Image plus audio to video: `videogen.ia2av`
- Music generation: `musicgen.generate`

Do not map Seedance 2.0 API requests to downloads. `videogen.seedance2-t2v`,
`videogen.seedance2-r2v`, and `videogen.seedance2-flf2v` use the remote
`seedance2-api` profile and require `COMFY_ORG_API_KEY`, not local model files.

## Flow

1. Resolve the capability from the user's request.
2. Preview the download:

```bash
uv run comfy-models download imagegen.generate --dry-run
```

3. Tell the user which profile, files, sources, and approximate size will be
   used, especially for large video/music models.
4. Download only after the request implies the capability is needed:

```bash
uv run comfy-models download imagegen.generate --yes
```

5. Validate the profile, then return to the original generation skill:

```bash
uv run comfy-models validate-profile anima-preview3-turbo
```

## Supported Built-In Profiles

- `anima-preview3-turbo`: Anima Preview3 image generation.
- `qwen-edit2511`: Qwen Image Edit 2511 generation/editing.
- `clear-reality`: ClearReality image upscaling.
- `ltx23-10eros`: LTX 2.3 video and IA2AV.
- `ace15-base`: ACE-Step 1.5 music generation.

`seedance2-api` is intentionally excluded: it is a remote Comfy API profile, not
a downloadable model profile.

Do not download models for custom local profiles unless the CLI provides a
source. For unknown custom checkpoints, use `comfy-model-onboarding` and ask the
user for the file.

## Auth And Sources

Supported sources are Hugging Face and direct HTTP model links. Set `HF_TOKEN`
for gated Hugging Face repositories. Set `CIVITAI_API_TOKEN` if Civitai requires
authenticated downloads.

The tool writes into the active `models_dir`, usually `/mnt/models/comfyui`, and
creates missing subfolders. Partial downloads use `*.part` files and are renamed
only after validation.

## Limits

- Do not download every model unless the user explicitly asks for all supported
  models.
- Do not download ad hoc LoRAs; `--extra-lora` files are managed by
  `comfy-lora-onboarding`.
- Do not change model defaults as part of downloading.
- Do not install ComfyUI custom nodes or start a server.
- Do not download or validate local files for Seedance 2.0 API generation.
