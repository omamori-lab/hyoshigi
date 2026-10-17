"""The models a monitor asks: their media, prompts, questions, decisions and names."""

import math

import pytest

from hyoshigi import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    ConfigurationError,
    Decision,
    DecisionModel,
    GenerativeModel,
    MediaTypes,
    OpaquePart,
    Prompt,
    Question,
    Refused,
)


async def build_refusal(prompt_or_question: Prompt | Question) -> Refused:
    return Refused()


# RFC 6838, section 4.2: media type names are case-insensitive; RFC 9110, section 12.5.1:
# a range such as image/* stands for every subtype.
@pytest.mark.xfail(strict=True, reason="the model interface is not checked yet")
@pytest.mark.parametrize(
    ("patterns", "media_type"),
    [
        (("image/*",), "image/png"),
        (("image/*",), "image/*"),
        (("image/png",), "image/png"),
        (("IMAGE/PNG",), "image/png"),
        (("image/*",), "IMAGE/PNG"),
    ],
    ids=[
        "range-covers-a-subtype",
        "range-covers-itself",
        "type-covers-itself",
        "pattern-case-is-ignored",
        "media-type-case-is-ignored",
    ],
)
def test_a_model_perceives_the_media_its_patterns_cover(
    patterns: tuple[str, ...], media_type: str
) -> None:
    media = MediaTypes(*patterns)

    assert media.covers(media_type)


@pytest.mark.parametrize(
    ("patterns", "media_type"),
    [
        (("image/png",), "image/*"),
        (("image/png",), "image/jpeg"),
        (("image/*",), "application/pdf"),
        ((), "image/png"),
    ],
    ids=[
        "type-does-not-cover-its-range",
        "type-does-not-cover-another",
        "range-covers-only-its-top-level-type",
        "text-only-perceives-no-media",
    ],
)
def test_a_model_does_not_perceive_media_its_patterns_miss(
    patterns: tuple[str, ...], media_type: str
) -> None:
    media = MediaTypes(*patterns)

    assert not media.covers(media_type)


@pytest.mark.xfail(strict=True, reason="the model interface is not checked yet")
def test_a_pattern_that_is_not_a_media_type_raises() -> None:
    with pytest.raises(ConfigurationError):
        MediaTypes("image/*", "png")


@pytest.mark.xfail(strict=True, reason="the model interface is not checked yet")
@pytest.mark.parametrize(
    ("kind", "arguments"),
    [
        (Prompt, {"system": "Rate the action.", "content": (OpaquePart(kind="audio"),)}),
        (Question, {"statement": "The action is safe.", "state": (OpaquePart(kind="audio"),)}),
    ],
    ids=["prompt", "question"],
)
def test_content_a_model_cannot_be_given_raises(kind: type, arguments: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        kind(**arguments)


@pytest.mark.xfail(strict=True, reason="the model interface is not checked yet")
@pytest.mark.parametrize(
    ("probability", "confidence"),
    [(1.5, None), (-0.1, None), (math.nan, None), (0.5, 1.5), (0.5, -0.1), (0.5, math.nan)],
    ids=[
        "probability-above-one",
        "probability-below-zero",
        "probability-not-a-number",
        "confidence-above-one",
        "confidence-below-zero",
        "confidence-not-a-number",
    ],
)
def test_a_decision_outside_zero_and_one_raises(
    probability: float, confidence: float | None
) -> None:
    with pytest.raises(ConfigurationError):
        Decision(probability=probability, confidence=confidence)


@pytest.mark.parametrize(
    ("probability", "confidence"), [(0.0, 1.0), (1.0, 0.0)], ids=["lower-bound", "upper-bound"]
)
def test_a_decision_holds_probabilities_at_the_bounds(
    probability: float, confidence: float
) -> None:
    decision = Decision(probability=probability, confidence=confidence)

    assert (decision.probability, decision.confidence) == (probability, confidence)


@pytest.mark.xfail(strict=True, reason="the model interface is not checked yet")
@pytest.mark.parametrize(
    ("kind", "function"),
    [
        (GenerativeModel, {"generate": build_refusal}),
        (AsyncGenerativeModel, {"generate": build_refusal}),
        (DecisionModel, {"decide": build_refusal}),
        (AsyncDecisionModel, {"decide": build_refusal}),
    ],
    ids=["generative", "async-generative", "decision", "async-decision"],
)
def test_a_model_without_a_name_raises(kind: type, function: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        kind(name="", **function)
