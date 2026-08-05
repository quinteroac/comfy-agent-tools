from __future__ import annotations

import pytest

from comfy_agent_tools.videogen.multishot import parse_shot_script, plan_shots
from comfy_agent_tools.cli.videogen import build_parser


def test_parse_shot_script_accepts_separators() -> None:
    assert parse_shot_script("one\n---\ntwo") == ["one", "two"]


def test_parse_shot_script_accepts_json() -> None:
    assert parse_shot_script('{"prompts": ["one", "two"]}') == ["one", "two"]


def test_plan_shots_uses_duration_and_repeats_last_prompt() -> None:
    prompts, frames = plan_shots("one", duration=30, shot_duration=5)

    assert len(prompts) == 6
    assert prompts == ["one"] * 6
    assert frames == 120


def test_plan_shots_truncates_extra_prompts_when_count_is_forced() -> None:
    prompts, frames = plan_shots("one\n---\ntwo\n---\nthree", duration=30, shot_duration=10, shot_count=2)

    assert prompts == ["one", "two"]
    assert frames == 240


@pytest.mark.parametrize("duration", [0, -1])
def test_plan_shots_requires_positive_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        plan_shots("one", duration=duration)


def test_multishot_cli_accepts_first_shot_inputs() -> None:
    args = build_parser().parse_args(
        [
            "minimax-h3-multishot-t2v",
            "--prompt",
            "shot",
            "--duration",
            "30",
            "--shot-duration",
            "5",
            "--input",
            "a.png",
            "--input",
            "b.png",
            "--first-mode",
            "r2v",
        ]
    )

    assert args.duration == 30
    assert args.shot_duration == 5
    assert len(args.input) == 2
    assert args.first_mode == "r2v"
