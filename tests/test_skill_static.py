from __future__ import annotations

from pathlib import Path


def test_comfy_imagegen_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-imagegen/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-imagegen" in frontmatter
    assert "description:" in frontmatter
    assert "Anima" in content
    assert "anima-turbo-lora-v0.1.safetensors" in content
    assert "steps=8" in content
    assert "cfg=1.0" in content
    assert "--extra-lora" in content
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content
    assert "Qwen Image Edit 2511" in content
    assert "4x-ClearRealityV1.pth" in content


def test_comfy_videogen_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-videogen/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-videogen" in frontmatter
    assert "description:" in frontmatter
    assert "10Eros_v1-fp8mixed_learned.safetensors" in content
    assert "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in content
    assert "ia2av" in content
    assert "--extra-lora" in content
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content


def test_comfy_musicgen_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-musicgen/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-musicgen" in frontmatter
    assert "description:" in frontmatter
    assert "acestep_v1.5_base.safetensors" in content
    assert "qwen_1.7b_ace15.safetensors" in content
    assert "steps=32" in content
    assert "cfg=7.0" in content
    assert "--extra-lora" in content
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content


def test_comfy_model_onboarding_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-model-onboarding/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-model-onboarding" in frontmatter
    assert "description:" in frontmatter
    assert ".comfy-agent-tools.json" in content
    assert "ltx23-10eros" in content
    assert "anima-preview3-turbo" in content
    assert "architecture" in content
    assert "validate-profile" in content
    assert "set-models-dir <models_dir>" in content
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content


def test_comfy_model_downloader_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-model-downloader/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-model-downloader" in frontmatter
    assert "description:" in frontmatter
    assert "comfy-models download imagegen.generate --dry-run" in content
    assert "HF_TOKEN" in content
    assert "CIVITAI_API_TOKEN" in content
    assert "Do not download every model" in content


def test_comfy_lora_onboarding_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-lora-onboarding/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-lora-onboarding" in frontmatter
    assert "description:" in frontmatter
    assert "loras/<architecture>" in content
    assert "descriptive filename" in content or "Filenames should describe" in content
    assert "Do not move" in content
    assert "--extra-lora" in content
    assert "comfy-tools-setup" in content


def test_comfy_tools_setup_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-tools-setup/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-tools-setup" in frontmatter
    assert "description:" in frontmatter
    assert "uv tool install git+https://github.com/quinteroac/comfy-agent-tools" in content
    assert "uv tool upgrade comfy-agent-tools" in content
    assert "uv run comfy-models validate" in content
    assert "comfy-models init" in content
    assert "comfy-models set-models-dir <models_dir>" in content
    assert "Do not pick" in content
    assert "comfy-models validate" in content
    assert "comfy-model-downloader" in content


def test_readme_documents_skills_first_installation() -> None:
    content = Path("README.md").read_text(encoding="utf-8")

    assert "npx skills add quinteroac/comfy-agent-tools" in content
    assert "uv tool install git+https://github.com/quinteroac/comfy-agent-tools" in content
    assert "Python CLIs on demand" in content
    assert "Models are always local" in content
    assert "comfy-models download imagegen.generate --dry-run" in content
    assert "comfy-model-downloader" in content
