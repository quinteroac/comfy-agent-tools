# Contributing

Thanks for helping improve `comfy-agent-tools`.

This project is a skills-first local media generation toolkit. The public
surface is split between agent skills under `skills/` and Python CLIs under
`comfy_agent_tools/`.

## Development Setup

Use Python 3.12 or newer and `uv`:

```bash
uv sync --extra dev
uv run pytest
```

For model profile or path changes, also run:

```bash
uv run comfy-models validate
```

For downloader changes, test dry-run and mocked download behavior. Do not run
large real downloads as part of normal validation.

GPU smoke tests are optional and can be expensive. Keep generated media under
`outputs/`.

## Contribution Flow

Contributions should use fork-based pull requests:

1. Fork `quinteroac/comfy-agent-tools`.
2. Create a feature branch in your fork.
3. Make your changes.
4. Run tests:

```bash
uv run pytest
```

5. Update `CHANGELOG.md` for user-visible changes.
6. Open a pull request against `quinteroac/comfy-agent-tools:main`.

Please do not push directly to `main`. Maintainers may use direct commits only
for small administrative changes.

Agent-assisted contributions are welcome. The contributor opening the PR is
responsible for understanding the generated code, explaining the approach during
review, and maintaining the change after merge.

## Git Hooks

Install local hooks with:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook checks that meaningful project changes include a staged
`CHANGELOG.md` update.

To add a changelog entry:

```bash
uv run python scripts/update_changelog.py --entry "Add support for a new model profile"
```

## Changelog

Update `CHANGELOG.md` for user-visible changes:

- new skills
- new CLI commands or flags
- model/profile defaults
- behavior changes
- bug fixes
- packaging or setup changes
- supported downloader sources or model download behavior

Small internal refactors, test-only changes, and generated outputs do not need a
changelog entry.

## Skills

When editing skills:

- Keep YAML frontmatter valid.
- Keep instructions concise and agent-actionable.
- Route CLI bootstrap through `comfy-tools-setup`.
- Route model/profile setup through `comfy-model-onboarding`.
- Route LoRA organization through `comfy-lora-onboarding`.

## Code Guidelines

- Keep CLI output JSON-clean by default.
- Use `--verbose` for logs and progress output.
- Return stable failure JSON with `ok:false` and an `error_type`.
- Preserve existing public flags unless a breaking change is intentional.
- Keep changes scoped and add tests for public behavior.

## Models And Outputs

Do not commit models, generated outputs, local config, or virtualenv artifacts.
Models are expected to live outside the repo, usually under
`/mnt/models/comfyui`.
