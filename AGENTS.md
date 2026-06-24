# AGENTS.md

Guidance for agents working in this repository.

## Project Shape

`comfy-agent-tools` is a skills-first local media generation toolkit backed by
`comfy-diffusion`. Users install the skills, and agents bootstrap the Python CLIs
when needed.

Primary surfaces:

- `skills/`: agent-facing instructions and workflows.
- `comfy_agent_tools/`: Python CLI/runtime implementation.
- `tests/`: pytest coverage for CLIs, profiles, LoRA handling, and static skill docs.
- `README.md`: public installation and usage docs.

Do not distribute models, generated outputs, API keys, or local config. Local
models live outside the repo. On a new machine, ask the user which `models_dir`
they want; do not assume `/mnt/models/comfyui` during setup.

## Setup

Use `uv` for development:

```bash
uv sync --extra dev
uv run pytest
```

In this repo, prefer `uv run <command>` for CLI checks:

```bash
uv run comfy-models validate
uv run comfy-imagegen generate --prompt "masterpiece, best quality, score_7, safe, anime style" --width 512 --height 512 --out outputs
```

For public/agent installation, the intended flow is skills-first:

```bash
npx skills add quinteroac/comfy-agent-tools
```

Then the `comfy-tools-setup` skill installs the Python CLIs on demand with:

```bash
uv tool install git+https://github.com/quinteroac/comfy-agent-tools
```

## Testing

Before finishing code or docs changes, run:

```bash
uv run pytest
```

For profile/path changes, also run:

```bash
uv run comfy-models validate
```

GPU smoke tests are optional and can be expensive. Only run them when the task
needs real inference validation. Keep generated files under `outputs/`, which is
ignored by git.

## Skills

When editing any skill:

- Keep YAML frontmatter valid.
- Keep instructions agent-actionable and concise.
- If a skill depends on a CLI, mention `comfy-tools-setup` as the bootstrap path.
- If model/profile config is involved, route to `comfy-model-onboarding`.
- If LoRA discovery, naming, or organization is involved, route to `comfy-lora-onboarding`.

Current skills:

- `comfy-tools-setup`: installs/validates the Python CLIs.
- `comfy-imagegen`: local image generation/edit/upscale, including Krea2 Turbo and NVIDIA RTX image upscale.
- `comfy-imagedescribe`: local Qwen3-VL 2B Instruct image description/captioning/visual QA.
- `comfy-videogen`: local LTX 2.3/WAN 2.2 video generation and remote Seedance 2.0 API video.
- `comfy-bernini-videoedit`: Bernini WAN 2.2 video edit workflows.
- `comfy-motion-track-control`: LTX 2.3 Motion Track IC-LoRA control video workflows.
- `comfy-musicgen`: local ACE-Step music generation.
- `comfy-model-onboarding`: persistent model profiles/defaults.
- `comfy-model-downloader`: on-demand downloads for supported built-in models.
- `comfy-lora-onboarding`: ad hoc LoRA organization and selection.

## Models And Profiles

The repo should not download models automatically. Built-in defaults include:

- models dir fallback: `/mnt/models/comfyui`
- image generation: Anima Base v1.0 turbo profile
- image editing: Qwen Image Edit 2511
- optional image generation/editing: FLUX.2 Klein 9B FP8 with SNOFS LoRA
- video: LTX 2.3 10Eros profile
- WAN text video: WAN 2.2 T2V profile
- motion-track video: LTX 2.3 10Eros plus Motion Track IC-LoRA
- music: ACE-Step 1.5 base
- image description: Qwen3-VL 2B Instruct (HuggingFace model directory)
- remote video API: Seedance 2.0 via `COMFY_ORG_API_KEY`

Use `comfy-models` for persistent profile/default changes. Keep architecture and
profile separate: for example, `ltx23` is the architecture and `ltx23-10eros` is
a profile.

When `.comfy-agent-tools.json` is missing, ask the user for the model directory
before running:

```bash
uv run comfy-models init
uv run comfy-models set-models-dir <models_dir>
```

When a user requests generation and the built-in profile files are missing, use
`comfy-model-downloader` and download only the requested capability:

```bash
uv run comfy-models download imagegen.generate --dry-run
uv run comfy-models download imagegen.generate --yes
```

Do not download all supported models unless explicitly asked. Use `HF_TOKEN` for
gated Hugging Face repositories and `CIVITAI_API_TOKEN` for Civitai when needed.

Seedance 2.0 is not a local model profile. Do not route `seedance2-api` through
the downloader, `models_dir`, local LoRAs, or checkpoint onboarding. It requires
`COMFY_ORG_API_KEY`, uses ComfyUI API Nodes vendored by `comfy-diffusion`, and
must remain limited to Seedance 2.0 only.

## LoRAs

Ad hoc LoRAs are passed per command with:

```text
--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]
```

There is no LoRA catalog. Prefer this folder convention under
`/mnt/models/comfyui/loras`:

```text
loras/
  anima/
  qwen-image-edit/
  flux-klein/
  ltx23/
  ace-step-1.5/
```

Use descriptive filenames. Do not move or rename user model files unless the user
explicitly approves the exact operation.

## Coding Guidelines

- Keep CLI output JSON-clean by default; use `--verbose` as the escape hatch.
- Return stable JSON on failures with `ok:false` and an `error_type`.
- Preserve existing command flags unless the user explicitly asks for a breaking change.
- Keep changes scoped; avoid broad refactors while adding a capability.
- Use shared helpers for repeated parsing/metadata behavior.
- Update tests and skill docs together when public behavior changes.

## Git Hygiene

The worktree may contain generated outputs or local config. Do not commit:

- `.venv/`
- `.pytest_cache/`
- `*.egg-info/`
- `outputs/`
- `.comfy-agent-tools.json`

Do not revert user changes unless explicitly asked.
