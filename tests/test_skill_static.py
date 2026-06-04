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
    assert "flux-klein-9b-snofs" in content
    assert "klein_snofs_v1_1.safetensors" in content
    assert "uv run comfy-media gallery --out outputs" in content
    assert "uv run comfy-media index --out outputs" in content


def test_comfy_videogen_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-videogen/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-videogen" in frontmatter
    assert "description:" in frontmatter
    assert "10Eros_v1-fp8mixed_learned.safetensors" in content
    assert "ltx23-dasiwa-golden-lace-v3" in content
    assert "DasiwaLTX23_goldenLaceV3.safetensors" in content
    assert "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in content
    assert "ia2av" in content
    assert "seedance2-t2v" in content
    assert "Seedance 2.0" in content
    assert "wan22-s2v" in content
    assert "wan22-video-audio" in content
    assert "wav2vec2_large_english_fp16.safetensors" in content
    assert "wan22-dasiwa-littledemon-v2-s2v" in content
    assert "wan22-dasiwa-littledemon-v2-video-audio" in content
    assert "DasiwaWan2214BS2V_littledemonV2.safetensors" in content
    assert "COMFY_ORG_API_KEY" in content
    assert "missing_dependency" in content
    assert "--extra-lora" in content
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content
    assert "two-step pipeline" in content
    assert "--width 540 --height 360" in content
    assert "uv run comfy-media gallery --out outputs" in content
    assert "uv run comfy-media index --out outputs" in content


def test_comfy_motion_track_control_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-motion-track-control/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-motion-track-control" in frontmatter
    assert "description:" in frontmatter
    assert "comfy-tools-setup" in content
    assert "comfy-model-downloader" in content
    assert "comfy-model-onboarding" in content
    assert "HyperFrames" in content
    assert "LTXICLoRALoaderModelOnly" in content
    assert "LTXAddVideoICLoRAGuide" in content
    assert "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors" in content
    assert "uv run comfy-media gallery --out outputs" in content
    assert "uv run comfy-media index --out outputs" in content


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
    assert "uv run comfy-media gallery --out outputs" in content
    assert "uv run comfy-media index --out outputs" in content


def test_comfy_media_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-media/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-media" in frontmatter
    assert "description:" in frontmatter
    assert "comfy-tools-setup" in content
    assert "uv run comfy-media index" in content
    assert "uv run comfy-media gallery" in content
    assert "export-hyperframes" in content
    assert "HyperFrames" in content
    assert "begin by starting or reusing the" in content
    assert "after each successful generation command" in content


def test_comfy_model_onboarding_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-model-onboarding/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-model-onboarding" in frontmatter
    assert "description:" in frontmatter
    assert ".comfy-agent-tools.json" in content
    assert "ltx23-10eros" in content
    assert "ltx23-dasiwa-golden-lace-v3" in content
    assert "anima-base" in content
    assert "architecture" in content
    assert "validate-profile" in content
    assert "set-models-dir <models_dir>" in content
    assert "seedance2-api" in content
    assert "COMFY_ORG_API_KEY" in content
    assert "flux-klein-9b-snofs" in content
    assert "flux-klein" in content
    assert "wan22-dasiwa-littledemon-v2-video-audio" in content
    assert "videogen.wan22-video-audio" in content
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
    assert "flux-klein-9b-snofs" in content
    assert "SNOFS" in content
    assert "ltx23-dasiwa-golden-lace-v3" in content
    assert "seedance2-api" in content
    assert "COMFY_ORG_API_KEY" in content
    assert "Do not download every model" in content


def test_comfy_lora_onboarding_skill_frontmatter() -> None:
    skill_path = Path("skills/comfy-lora-onboarding/SKILL.md")
    content = skill_path.read_text(encoding="utf-8")

    assert content.startswith("---\n")
    frontmatter = content.split("---", 2)[1]
    assert "name: comfy-lora-onboarding" in frontmatter
    assert "description:" in frontmatter
    assert "loras/<architecture>" in content
    assert "flux-klein" in content
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
    assert "Seedance 2.0" in content
    assert "COMFY_ORG_API_KEY" in content
    assert "seedance2-api" in content
    assert "comfy-models download imagegen.generate --dry-run" in content
    assert "flux-klein-9b-snofs" in content
    assert "klein_snofs_v1_1.safetensors" in content
    assert "comfy-model-downloader" in content
    assert "comfy-media" in content
