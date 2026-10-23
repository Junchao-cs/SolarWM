"""VideoScore2 (TIGER-Lab/VideoScore2), as eval/solar_renderer/metrics.py reads it: visual quality, prompt alignment and physical
plausibility on 1-5. Decoding is greedy (the released script samples at temperature 0.7; greedy decoding gives the same score
for the same video, so methods can be compared); a score is read off the distribution over the levels:
the authors' discounted level and the plain expectation (*_expected, the one the summary uses).
Needs transformers 4.53 and qwen-vl-utils 0.0.14 (its own environment, see eval/solar_renderer/README.md).
"""

from __future__ import annotations

import os
import re

import numpy as np
import torch

MODEL = os.environ.get("VIDEOSCORE2", "TIGER-Lab/VideoScore2")
QUERY = """
    You are an expert for evaluating AI-generated videos from three dimensions:
    (1) visual quality - clarity, smoothness, artifacts;
    (2) text-to-video alignment - fidelity to the prompt;
    (3) physical/common-sense consistency - naturalness and physics plausibility.

    Video prompt: {prompt}

    Please output in this format:
    visual quality: <v_score>;
    text-to-video alignment: <t_score>,
    physical/common-sense consistency: <p_score>
    """

DIMENSIONS = (
    ("visual_quality", "visual quality:"),
    ("alignment", "text-to-video alignment:"),
    ("physics", "physical/common-sense consistency:"),
)


def soft_score(
    hard: int | None, index: int, scores, tokenizer
) -> tuple[float | None, float | None]:
    """Two readings of the same distribution over the levels 1-5.

    The first is the authors' own: the level it settled on, discounted by how much of the mass sat
    elsewhere. It is what their published numbers mean, so it stays. It is also discontinuous - the
    argmax hops - and two cards can disagree on it by a third of a level. The second is the plain
    expectation over the levels, which moves smoothly and is the one to read a trend off.
    """

    if hard is None or index < 0:
        return None, None
    logits = torch.log_softmax(scores[index][0], dim=-1)
    probabilities = []
    for level in range(1, 6):
        ids = tokenizer.encode(str(level), add_special_tokens=False)
        if len(ids) == 1:
            probabilities.append((level, float(np.exp(logits[ids[0]].item()))))
    if not probabilities:
        return None, None
    levels, values = zip(*probabilities, strict=False)
    total = sum(values)
    best = max(values)
    expected = sum(level * value for level, value in probabilities) / total
    return round(levels[values.index(best)] * best / total, 4), round(expected, 4)


def score_token(label: str, tokenizer, generated: list[int]) -> int:
    text = tokenizer.decode(generated, skip_special_tokens=False)
    match = re.search(r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(label), text, flags=re.IGNORECASE)
    if not match:
        return -1
    digit = re.search(r"\d", text[match.end() :])
    if not digit:
        return -1
    wanted = text[: match.end() + digit.start() + 1]
    for cut in range(len(generated)):
        if tokenizer.decode(generated[: cut + 1], skip_special_tokens=False) == wanted:
            return cut
    return -1


# Some answers leave the requested format ("Dim-1 Visual Quality: 4", "Physical Consistency: 5"): the score is then
# read where the answer last states it for that dimension.
FALLBACK = {
    "visual_quality": r"visual quality",
    "alignment": r"text-to-video alignment",
    "physics": r"physical(?:/common-sense)? consistency",
}


def stated_score(pattern: str, tokenizer, generated: list[int]) -> tuple[int | None, int]:
    text = tokenizer.decode(generated, skip_special_tokens=False)
    hits = list(re.finditer(pattern + r"[^\d\n]{0,60}?([1-5])\b", text, flags=re.IGNORECASE))
    if not hits:
        return None, -1
    hit = hits[-1]
    wanted = text[: hit.end(1)]
    for cut in range(len(generated)):
        if tokenizer.decode(generated[: cut + 1], skip_special_tokens=False) == wanted:
            return int(hit.group(1)), cut
    return None, -1


def build():
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

    model = (
        AutoModelForVision2Seq.from_pretrained(MODEL, trust_remote_code=True, torch_dtype="auto")
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
        MODEL, trust_remote_code=True, use_fast=False
    )
    return model, processor, tokenizer


def judge(model, processor, tokenizer, video: str, prompt: str, fps: float = 2.0) -> dict:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video, "fps": fps},
                {"type": "text", "text": QUERY.format(prompt=prompt)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, fps=fps, padding=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
    length = inputs["input_ids"].shape[1]
    generated = out.sequences[0, length:].tolist()
    spoken = processor.batch_decode(out.sequences[:, length:], skip_special_tokens=True)[0]
    stated = re.search(
        r"visual quality:\s*(\d+).*?text-to-video alignment:\s*(\d+)"
        r".*?physical/common-sense consistency:\s*(\d+)",
        spoken,
        re.DOTALL | re.IGNORECASE,
    )
    result = {"reason": spoken.strip()[-600:]}
    for position, (name, label) in enumerate(DIMENSIONS):
        if stated:
            hard, index = int(stated.group(position + 1)), score_token(label, tokenizer, generated)
        else:
            hard, index = stated_score(FALLBACK[name], tokenizer, generated)
        soft, expected = soft_score(hard, index, out.scores, tokenizer)
        result[name], result[f"{name}_expected"] = soft, expected
    return result
