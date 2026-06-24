---
name: comfy-tools-setup
description: Bootstrap and validate the comfy-agent-tools Python CLIs for agent use. Use when the user asks to setup, install, update, or diagnose comfy-agent-tools; when a required CLI such as comfy-imagegen, comfy-imagedescribe, comfy-videogen, comfy-musicgen, or comfy-models is missing; or before another comfy skill runs a CLI on a new machine.
---

# comfy-tools-setup

Use this skill to install or validate the Python tools that back the
`comfy-agent-tools` skills. The public install flow is skills-first; the agent
installs the CLIs only when they are needed.

## Detect Mode

First determine whether you are inside the `comfy-agent-tools` repo:

```bash
test -f pyproject.toml && test -d comfy_agent_tools && test -d skills
```

If true, prefer repo/dev commands:

```bash
uv sync --extra dev
uv run comfy-models validate
```

If not in this repo, use globally installed tools from `uv tool`.

## Install Tools

Check for `uv` before installing:

```bash
command -v uv
```

If `uv` is missing, ask the user to install `uv` first. Do not install package
managers or system packages without explicit user approval.

Install the CLIs:

```bash
uv tool install git+https://github.com/quinteroac/comfy-agent-tools
```

Upgrade the CLIs:

```bash
uv tool upgrade comfy-agent-tools
```

Validate that the expected commands exist:

```bash
command -v comfy-imagegen
command -v comfy-imagedescribe
command -v comfy-videogen
command -v comfy-musicgen
command -v comfy-models
```

## Model Config

If `.comfy-agent-tools.json` is missing in the current project, ask the user
where they want to store or use ComfyUI model files for these tools. Do not pick
`/mnt/models/comfyui` automatically; mention it only as an example.

After the user provides a path, initialize config and set that exact path:

```bash
comfy-models init
comfy-models set-models-dir <models_dir>
```

Then validate:

```bash
comfy-models validate
comfy-models show
```

In repo/dev mode, prefix CLI commands with `uv run`, for example:

```bash
uv run comfy-models init
uv run comfy-models set-models-dir <models_dir>
uv run comfy-models validate
```

If `.comfy-agent-tools.json` already exists, respect its configured `models_dir`
and do not ask again unless validation fails or the user wants to change it.

If validation reports missing models, switch to `comfy-model-downloader` when the
user is trying to run a supported built-in capability. Use
`comfy-model-onboarding` for custom checkpoints, changed defaults, or unsupported
profiles.

## Skill Installation

Users install the agent guidance first:

```bash
npx skills add quinteroac/comfy-agent-tools
```

After that, the skills can bootstrap the Python tools on demand.
