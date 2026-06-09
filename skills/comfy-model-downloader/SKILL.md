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
- Ideogram 4 local generation: `imagegen.ideogram4-generate`
- Image upscale: `imagegen.upscale`
- Text to video: `videogen.t2v`
- Image to video: `videogen.i2v`
- First/last frame video: `videogen.flf2v`
- Image plus audio to video: `videogen.ia2av`
- WAN 2.2 image to video: `videogen.wan22-i2v`
- WAN 2.2 first/last frame video: `videogen.wan22-flf2v`
- WAN 2.2 sound to video: `videogen.wan22-s2v`
- WAN 2.2 video plus audio: `videogen.wan22-video-audio`
- Music generation: `musicgen.generate`

Do not map Seedance 2.0 API requests to downloads. `videogen.seedance2-t2v`,
`videogen.seedance2-r2v`, and `videogen.seedance2-flf2v` use the remote
`seedance2-api` profile and require `COMFY_ORG_API_KEY`, not local model files.
Do not map Grok Imagine API requests to downloads either. `imagegen.grok-generate`
and `imagegen.grok-edit` use the remote `grok-imagine-api` profile.
Ideogram 4 is local and should be downloaded with `imagegen.ideogram4-generate`
when requested.

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
uv run comfy-models validate-profile anima-base
```

## Supported Built-In Profiles

- `anima-base`: Anima Base v1.0 image generation with the turbo LoRA.
- `qwen-edit2511`: Qwen Image Edit 2511 generation/editing.
- `flux-klein-9b-snofs`: FLUX.2 Klein 9B FP8 generation/editing with SNOFS.
- `ideogram4-fp8`: local Ideogram 4 FP8 text-to-image with structured JSON/bbox prompting.
- `clear-reality`: ClearReality image upscaling.
- `ltx23-10eros`: LTX 2.3 video and IA2AV.
- `ltx23-dasiwa-golden-lace-v3`: Dasiwa LTX 2.3 Golden Lace v3 video and IA2AV.
- `wan22-t2v`: WAN 2.2 text-to-video.
- `wan22-i2v`: WAN 2.2 image and first/last-frame video.
- `wan22-dasiwa-tastysin-t2v`: Dasiwa TastySin WAN 2.2 T2V local UNets plus shared UMT5/VAE.
- `wan22-dasiwa-boundbite-t2v`: Dasiwa BoundBite WAN 2.2 T2V local UNets plus shared UMT5/VAE.
- `wan22-s2v`: WAN 2.2 sound-to-video with wav2vec2 audio encoder.
- `wan22-dasiwa-littledemon-v2-s2v`: Dasiwa LittleDemon v2 S2V, 4-step profile.
- `wan22-dasiwa-littledemon-v2-video-audio`: Dasiwa LittleDemon v2 S2V for 16 fps video+audio, audio-driven, and external-mask lipsync.
- `ace15-base`: ACE-Step 1.5 music generation.

`seedance2-api` is intentionally excluded: it is a remote Comfy API profile, not
a downloadable model profile. `grok-imagine-api` is excluded for the same reason.
`ideogram4-fp8` is not remote; it uses Hugging Face model files under
`models_dir`.

Do not download models for custom local profiles unless the CLI provides a
source. For unknown custom checkpoints, use `comfy-model-onboarding` and ask the
user for the file.

## Auth And Sources

Supported sources are Hugging Face and direct HTTP model links. Set `HF_TOKEN`
for gated Hugging Face repositories; FLUX.2 Klein 9B requires accepting the
Black Forest Labs license terms before download. Ideogram 4 may require
accepting the Ideogram model terms before download. Set `CIVITAI_API_TOKEN` if
Civitai requires authenticated downloads.

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
- Do not download or validate local files for Grok Imagine API generation.
- Do not redistribute FLUX.2 Klein or SNOFS weights, and do not use SNOFS for a
  public/commercial generation service without separate licensing.
