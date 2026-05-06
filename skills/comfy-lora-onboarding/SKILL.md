---
name: comfy-lora-onboarding
description: Help organize and select local ComfyUI LoRAs for comfy-agent-tools. Use when the user wants to add, move, rename, organize, inspect, or apply LoRAs; when a requested LoRA is ambiguous; or when LoRAs are loose in /mnt/models/comfyui/loras and should be arranged by architecture and purpose.
---

# comfy-lora-onboarding

Use this skill to help the user organize and select LoRAs for `comfy-imagegen`,
`comfy-videogen`, and `comfy-musicgen`. Do not create a catalog. Prefer a simple
folder convention and descriptive filenames.

If validation requires `comfy-models` and that CLI is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-models`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

## Folder Convention

Recommend this structure under the active `models_dir`, usually
`/mnt/models/comfyui`:

```text
loras/
  anima/
    realism-portrait.safetensors
    blue-dawn-style.safetensors
    animation-sheet.safetensors
  qwen-image-edit/
    multi-angle.safetensors
    product-retouch.safetensors
  ltx23/
    camera-static.safetensors
    camera-dolly-left.safetensors
    detailer.safetensors
  ace-step-1.5/
    vocal-polish.safetensors
```

Filenames should describe purpose or intent. Prefer names like
`realism-portrait.safetensors`, `blue-dawn-style.safetensors`, and
`camera-dolly-left.safetensors`. Avoid vague names like `final.safetensors`,
`test2.safetensors`, or `lora_new.safetensors`.

## Selection Flow

When the user asks to apply a LoRA by name, style, or purpose:

1. Identify the active architecture from the command/profile: `anima`,
   `qwen-image-edit`, `ltx23`, or `ace-step-1.5`.
2. Search first in `${models_dir}/loras/<architecture>/`.
3. Search `${models_dir}/loras/` as fallback for loose LoRAs.
4. Rank candidates by matching filename tokens against the user request.
5. If one candidate is clear, pass it to the CLI with `--extra-lora`.
6. If several candidates are plausible, show 2-4 choices and ask the user.
7. If no candidate matches, say so and do not invent a path.

Use safetensors metadata when helpful to detect architecture, but do not trust
metadata over an obvious folder/name mismatch without telling the user.

## CLI Usage

`--extra-lora` is repeatable and accepts:

```text
PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]
```

Defaults are `MODEL_STRENGTH=1.0` and `CLIP_STRENGTH=0.0`.

Example:

```bash
uv run comfy-imagegen generate \
  --prompt "masterpiece, best quality, score_7, safe, realistic portrait..." \
  --extra-lora /mnt/models/comfyui/loras/anima/realism-portrait.safetensors:0.8:0.0 \
  --out outputs
```

## Organization

If a LoRA is loose in `loras/`, suggest the target architecture folder and a
descriptive filename. Do not move, rename, or delete files unless the user
explicitly confirms that exact operation.

When moving is approved, preserve the `.safetensors` file exactly. Move matching
preview or metadata sidecar files only if the user confirms and their names
clearly belong to the same LoRA.

## Limits

LoRA compatibility depends on the architecture and ComfyUI. If a command returns
a clean JSON error after applying a LoRA, report that the selected LoRA may not
be compatible with the active architecture or pipeline.
