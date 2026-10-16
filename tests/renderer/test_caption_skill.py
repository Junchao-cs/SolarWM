from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "skills" / "solarrenderer-caption" / "scripts"


@pytest.fixture(scope="module")
def caption_helpers():
    loaded = {}
    for name in ("conditions", "check", "first_frame"):
        spec = importlib.util.spec_from_file_location(
            f"caption_skill_{name}", SCRIPTS / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


@pytest.fixture
def free_caption():
    return {
        "summary": "A blue coupe follows the river through a sunlit town.",
        "detailed_description": "Natural afternoon light falls across the street. [Shot 1] "
        + "A blue coupe passes stone buildings beside a quiet river. " * 38,
        "overall_soundscape": "Soft tyre noise and wind accompany the passing car.",
        "non_diegetic_music": "N/A",
    }


CONDITIONS = [
    ("scene_graybox", "untextured grey block-out"),
    ("semantic", "one flat colour per class"),
    ("scene_whitebox", "untextured grey rendering"),
    ("mask_skeleton", "skeletons of the characters"),
    ("smpl_mesh", "3D body mesh"),
    ("near_depth", "depth within about 30 metres"),
]


@pytest.mark.parametrize("condition,description", CONDITIONS)
@pytest.mark.parametrize("inputs", ["full", "structure", "first", "text"])
def test_trained_conditions_assemble_valid_prompts_for_each_input_set(
    caption_helpers,
    free_caption,
    condition,
    description,
    inputs,
):
    prompt = caption_helpers.conditions.assemble(free_caption, inputs=inputs, condition=condition)
    assert caption_helpers.check.check_prompt(prompt) == []
    if inputs == "text":
        assert prompt.startswith("integrated_multimodal_description:")
        assert "subject_definitions:" not in prompt
        assert "<Video" not in prompt
    else:
        assert prompt.startswith("subject_definitions:\n")
    if inputs in ("full", "structure"):
        assert description in prompt
        assert "<Video 1>" in prompt
        assert "attribute_transfer" in prompt
    else:
        assert description not in prompt
        assert "attribute_transfer" not in prompt
    if inputs in ("full", "first"):
        first_label = "<Video 2>" if inputs == "full" else "<Video 1>"
        assert f"{first_label} is the first frame of [Shot 1], held still." in prompt
        assert f"[Shot 1] The shot begins from {first_label}." in prompt
        assert "fully_preserved" in prompt
    else:
        assert "fully_preserved" not in prompt
    if inputs == "full":
        assert "[keyframe completion + reference generation]" in prompt
    elif inputs == "structure":
        assert "[reference generation]" in prompt
        assert "<Video 2>" not in prompt
    elif inputs == "first":
        assert "[keyframe completion]" in prompt
        assert "<Video 2>" not in prompt


@pytest.mark.parametrize("inputs", ["full", "structure", "first", "text"])
def test_custom_structure_caption_preserves_its_visual_contract(
    caption_helpers, free_caption, inputs
):
    custom = caption_helpers.conditions.custom_condition(
        "a pencil sketch of the street on white paper",
        "the pencil lines and white paper",
    )
    prompt = caption_helpers.conditions.assemble(free_caption, inputs=inputs, condition=custom)
    assert caption_helpers.check.check_prompt(prompt) == []
    if inputs in ("full", "structure"):
        assert "a pencil sketch of the street on white paper" in prompt
        assert "the pencil lines and white paper of <Video 1> is not shown" in prompt
        assert "{v}" not in prompt
    else:
        assert "pencil" not in prompt


def test_assembly_defaults_missing_sound_fields_to_explicit_na(caption_helpers, free_caption):
    free_caption.pop("overall_soundscape")
    free_caption["non_diegetic_music"] = "   "
    prompt = caption_helpers.conditions.assemble(free_caption)
    assert "overall_soundscape:\nN/A" in prompt
    assert prompt.endswith("non_diegetic_music:\nN/A")
    assert caption_helpers.check.check_prompt(prompt) == []


@pytest.mark.parametrize("words,valid", [(349, False), (350, True), (500, True), (501, False)])
def test_free_caption_enforces_word_count_boundaries(caption_helpers, free_caption, words, valid):
    free_caption["detailed_description"] = "[Shot 1] " + " ".join(["street"] * (words - 2))
    _, checks = caption_helpers.check.check_free(free_caption, duration=6.583)
    assert checks["words"] == words
    assert checks["length"] is valid
    assert checks["pass"] is valid


@pytest.mark.parametrize("label", ["<Video 1>", "<Picture 2>", "<Subject 3>", "<Audio 1>"])
def test_free_caption_rejects_reference_labels(caption_helpers, free_caption, label):
    free_caption["summary"] += f" The clip follows {label}."
    _, checks = caption_helpers.check.check_free(free_caption, duration=6.583)
    assert checks["no_labels"] is False
    assert checks["pass"] is False
    assert any("reference label" in message for message in caption_helpers.check.problems(checks))


@pytest.mark.parametrize(
    "summary,valid",
    [
        ('A road sign reads "東京".', True),
        ("東京 appears beside the road.", False),
    ],
)
def test_english_rule_allows_only_quoted_non_english_text(
    caption_helpers, free_caption, summary, valid
):
    free_caption["summary"] = summary
    _, checks = caption_helpers.check.check_free(free_caption, duration=6.583)
    assert checks["english"] is valid
    assert checks["pass"] is valid


def test_free_caption_normalizes_task_prefixes_and_missing_cut_markers(
    caption_helpers, free_caption
):
    free_caption["summary"] = "[reference generation] [keyframe completion]: A car follows a river."
    free_caption["subject"] = "  A blue coupe. "
    free_caption["detailed_description"] = (
        "[Shot 1] At 00:00.000, the car follows the river. "
        + "The stone buildings catch warm afternoon light. " * 26
        + "At 00:03.000, the shot cuts to a courtyard. "
        + "The stone buildings catch warm afternoon light. " * 26
    )
    normalized, checks = caption_helpers.check.check_free(free_caption, duration=6.583)
    assert normalized["summary"] == "A car follows a river."
    assert normalized["subject"] == "A blue coupe"
    assert normalized["detailed_description"].startswith("[Shot 1] The car follows")
    assert "[Shot 2] At 00:03.000, the shot cuts to" in normalized["detailed_description"]
    assert checks["shots"] == 2
    assert checks["cuts"] is True
    assert checks["pass"] is True


@pytest.mark.parametrize(
    "second_time,valid",
    [
        ("00:00.000", False),
        ("00:03.000", True),
        ("00:06.582", True),
        ("00:06.583", False),
        ("00:06.584", False),
    ],
)
def test_finished_prompt_requires_shot_cut_inside_duration(
    caption_helpers,
    free_caption,
    second_time,
    valid,
):
    free_caption["detailed_description"] = (
        "[Shot 1] "
        + "A quiet street stretches beside the river. " * 25
        + f"[Shot 2] At {second_time}, the shot cuts to a courtyard. "
        + "Soft afternoon light shines over the stone. " * 25
    )
    prompt = caption_helpers.conditions.assemble(free_caption, inputs="structure")
    problems = caption_helpers.check.check_prompt(prompt, duration=6.583)
    assert (not problems) is valid
    if not valid:
        assert any("shot markers" in problem for problem in problems)


@pytest.mark.parametrize(
    "invalid_detail",
    [
        "[Shot 1] At 00:00.100, the car moves.",
        "[Shot 1] A car moves. [Shot 3] At 00:03.000, the shot cuts to a street.",
        (
            "[Shot 1] A car moves. [Shot 2] At 00:03.000, the shot cuts to a street. "
            "[Shot 3] At 00:02.000, the shot cuts again."
        ),
        "[Shot 1] A car moves. At 00:02.000 it stops beside the river.",
    ],
)
def test_finished_prompt_rejects_bad_shot_numbers_or_stray_timestamps(
    caption_helpers,
    free_caption,
    invalid_detail,
):
    free_caption["detailed_description"] = invalid_detail + " The buildings are warm stone." * 65
    prompt = caption_helpers.conditions.assemble(free_caption, inputs="structure")
    assert any("shot markers" in problem for problem in caption_helpers.check.check_prompt(prompt))


def test_finished_prompt_rejects_unknown_labels_and_reordered_sections(
    caption_helpers, free_caption
):
    prompt = caption_helpers.conditions.assemble(free_caption)
    undefined = prompt.replace("A blue coupe follows", "<Video 3> shows a blue coupe following", 1)
    assert any(
        "not defined" in problem for problem in caption_helpers.check.check_prompt(undefined)
    )
    reordered = (
        prompt.replace("summary:\n", "temporary:\n")
        .replace(
            "retention_analysis:\n",
            "summary:\n",
        )
        .replace("temporary:\n", "retention_analysis:\n")
    )
    assert any("sections" in problem for problem in caption_helpers.check.check_prompt(reordered))


def test_text_prompt_rejects_input_reference(caption_helpers, free_caption):
    prompt = caption_helpers.conditions.assemble(free_caption, inputs="text")
    prompt = prompt.replace("A blue coupe", "<Video 1> shows a blue coupe", 1)
    assert any("no inputs" in problem for problem in caption_helpers.check.check_prompt(prompt))


@pytest.mark.parametrize("inputs", ["full", "text"])
@pytest.mark.parametrize("missing", ["overall_soundscape", "non_diegetic_music"])
def test_finished_prompt_rejects_missing_required_audio_sections(
    caption_helpers,
    free_caption,
    inputs,
    missing,
):
    prompt = caption_helpers.conditions.assemble(free_caption, inputs=inputs)
    heading = f"\n\n{missing}:"
    start = prompt.index(heading)
    end = prompt.find("\n\n", start + len(heading))
    broken = prompt[:start] + (prompt[end:] if end >= 0 else "")
    assert caption_helpers.check.check_prompt(broken)


@pytest.mark.parametrize(
    "inputs,section",
    [
        ("full", "subject_definitions"),
        ("full", "summary"),
        ("full", "retention_analysis"),
        ("full", "detailed_description"),
        ("full", "overall_soundscape"),
        ("full", "non_diegetic_music"),
        ("text", "integrated_multimodal_description"),
        ("text", "overall_soundscape"),
        ("text", "non_diegetic_music"),
    ],
)
def test_finished_prompt_requires_nonempty_sections(caption_helpers, free_caption, inputs, section):
    prompt = caption_helpers.conditions.assemble(free_caption, inputs=inputs)
    heading = f"{section}:"
    start = prompt.index(heading) + len(heading)
    end = prompt.find("\n\n", start)
    broken = prompt[:start] + "\n \t " + (prompt[end:] if end >= 0 else "")
    assert any("filled" in problem for problem in caption_helpers.check.check_prompt(broken))


def test_first_frame_prompt_keeps_style_and_exactly_three_opening_sentences(
    caption_helpers, free_caption
):
    free_caption["summary"] = "SUMMARY_ONLY_TOKEN"
    free_caption["detailed_description"] = (
        "A hand-painted animation uses a limited blue palette. [Shot 1] "
        "A blue coupe waits beside a river. Sunlight catches its windscreen. "
        "A passenger watches the water. FOURTH_SENTENCE_TOKEN is later action."
    )
    prompt = caption_helpers.conditions.assemble(free_caption)
    image = caption_helpers.first_frame.image_prompt(prompt)
    assert "A hand-painted animation uses a limited blue palette." in image
    assert "A blue coupe waits beside a river." in image
    assert "Sunlight catches its windscreen." in image
    assert "A passenger watches the water." in image
    assert "SUMMARY_ONLY_TOKEN" not in image
    assert "FOURTH_SENTENCE_TOKEN" not in image
    assert "<Video" not in image
    assert "overall_soundscape" not in image
    assert "IDENTICAL camera" in image


def test_first_frame_prompt_does_not_cross_into_second_shot(caption_helpers, free_caption):
    free_caption["detailed_description"] = (
        "A hand-painted animation. [Shot 1] A blue coupe waits by the river. "
        "[Shot 2] At 00:03.000, the shot cuts to SECOND_SHOT_TOKEN. Birds fly overhead."
    )
    image = caption_helpers.first_frame.image_prompt(
        caption_helpers.conditions.assemble(free_caption)
    )
    assert "A blue coupe waits by the river." in image
    assert "[Shot 2]" not in image
    assert "SECOND_SHOT_TOKEN" not in image
    assert "00:03.000" not in image


def test_first_frame_quality_suffix_respects_stylised_output(caption_helpers, free_caption):
    prompt = caption_helpers.conditions.assemble(free_caption)
    realistic = caption_helpers.first_frame.image_prompt(prompt)
    stylised = caption_helpers.first_frame.image_prompt(prompt, stylised=True)
    assert "physically based materials" in realistic
    assert "do not make it photorealistic" in stylised
    assert "physically based materials" not in stylised
    assert "No text, no logos, no UI" in realistic and "No text, no logos, no UI" in stylised


def test_included_semantic_example_passes_caption_validation(caption_helpers):
    prompt = (
        ROOT / "examples" / "solar_renderer" / "sem-highway_synthwave" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert caption_helpers.check.check_prompt(prompt) == []
    image = caption_helpers.first_frame.image_prompt(prompt, stylised=True)
    assert "synthwave" in image
    assert "<Video" not in image
