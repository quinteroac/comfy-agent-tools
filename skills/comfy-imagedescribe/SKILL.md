---
name: comfy-imagedescribe
description: Describe or caption local images with the Qwen3-VL 2B Instruct vision-language model through transformers. Use when the user wants a local GPU-backed image description, caption, tagging, or VLM question-answering over an image file. Do not use for image generation, editing, upscaling, video generation, music generation, model downloads, ComfyUI server workflows, UI work, or custom node installation.
---

# comfy-imagedescribe

Use this skill for local image description through the `comfy-imagedescribe` CLI.
The CLI uses the Qwen3-VL 2B Instruct HuggingFace model directory under
`/mnt/models/comfyui/LLM/Qwen-VL/Qwen3-VL-2B-Instruct`. The model is a full
vision-language model loaded with `transformers` (`Qwen3VLForConditionalGeneration`),
not a diffusion text encoder, so the image is actually grounded in the generated
description. It is not distributed by `comfy-models download`; obtain the model
directory from Hugging Face (`Qwen/Qwen3-VL-2B-Instruct`) and place it under
`LLM/Qwen-VL/` if missing.

The CLI is quiet by default and prints only final JSON. Use `--verbose` only when
debugging ComfyUI runtime output, warnings, or progress bars.

If `comfy-imagedescribe` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer
`uv run comfy-imagedescribe`; outside the repo, let `comfy-tools-setup` install
the CLIs with `uv tool`.

At the start of every image-description workflow, start or reuse the local Comfy
Media gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use `comfy-media --help` only if the CLI is missing or behaves unexpectedly; do
not skip the gallery just because description can run headless.

If `.comfy-agent-tools.json` is missing or the user wants to configure a new
Qwen3-VL model path or default, use `comfy-model-onboarding` first. The built-in
capability is `imagedescribe.describe` and the default profile is
`qwen3vl-2b-instruct`.

If model validation fails with a missing model directory, obtain the
`Qwen3-VL-2B-Instruct` HuggingFace folder and place it under
`LLM/Qwen-VL/`. Do not point `--llm` at a single `text_encoders/qwen3vl_*`
safetensors file: those are diffusion text encoders and will hallucinate
because they do not connect the vision tower for generation.

## Mode

- `describe`: send an input image plus an instruction to Qwen3-VL 2B and return
  generated text (a description, caption, tags, or an answer).

## Command

Describe an image in detail:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/my-image.png \
  --prompt "Describe this image in detail." \
  --out outputs
```

Generate a short caption:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/my-image.png \
  --prompt "Write a concise one-sentence caption for this image." \
  --max-length 128 \
  --out outputs
```

Ask a question about the image (visual QA):

```bash
uv run comfy-imagedescribe describe \
  --input outputs/my-image.png \
  --prompt "What is the dominant color palette, and how many people are visible?" \
  --out outputs
```

Deterministic output for reproducible descriptions:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/my-image.png \
  --prompt "Describe this image in detail." \
  --greedy \
  --seed 0 \
  --out outputs
```

## Prompt Guidance

`--prompt` is the instruction sent to the model alongside the image. Qwen3-VL 2B
Instruct follows natural-language instructions well. Be specific about the
desired output shape:

- Detailed description: `"Describe this image in detail."`
- Short caption: `"Write a concise one-sentence caption for this image."`
- Tags: `"List comma-separated descriptive tags for this image."`
- Visual QA: `"How many objects are on the table? Answer briefly."`
- Style analysis: `"Describe the art style, lighting, and composition."`

Keep prompts focused. For long-form descriptions, raise `--max-length` (default
512). For short captions or tags, lower `--max-length` to keep output tight.

## Sampling

Defaults are tuned for coherent, descriptive output:

- `do_sample=true`, `temperature=0.7`, `top_k=64`, `top_p=0.95`, `min_p=0.05`
- `repetition_penalty=1.05`, `max_length=512`, `seed=0`

For reproducible descriptions, pass `--greedy` (disables sampling and ignores
temperature/top-k/top-p). For more varied output, raise `--temperature` up to
~1.0. For more focused output, lower `--temperature` toward 0.3.

## Defaults

- Models directory: `/mnt/models/comfyui`
- Model: `LLM/Qwen-VL/Qwen3-VL-2B-Instruct` (HuggingFace directory)
- Sampling: `max_length=512`, `temperature=0.7`, `top_k=64`, `top_p=0.95`,
  `min_p=0.05`, `repetition_penalty=1.05`, `do_sample=true`, `seed=0`
- Output: JSON to stdout with the generated `description` field
- Dependency: `transformers` plus the vendored ComfyUI requirements

## Output Handling

The CLI prints JSON to stdout. On success, read `description` for the generated
text and `input` for the source image path. On failure, read `error` and
`error_type`; do not parse logs for control flow.

Project-bound descriptions should be saved under the current workspace. Do not
overwrite existing assets unless the user explicitly asks for replacement.

After every successful description command, immediately index the same output
directory so the manifest appears in Comfy Media:

```bash
uv run comfy-media index --out outputs
```
