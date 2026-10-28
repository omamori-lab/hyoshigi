"""The models a monitor asks: their media, prompts, questions, decisions and names."""

import math
import numbers
from decimal import Decimal
from fractions import Fraction

import pytest

from hyoshigi import (
    AlternativeToken,
    AsyncDecisionModel,
    AsyncGenerativeModel,
    ConfigurationError,
    Decision,
    DecisionModel,
    Generation,
    GenerativeModel,
    MediaTypes,
    OpaquePart,
    Prompt,
    Question,
    Refused,
    TokenLogProbability,
)

NOT_YET = pytest.mark.xfail(strict=True)


async def build_refusal(prompt_or_question: Prompt | Question) -> Refused:
    return Refused()


# RFC 6838, section 4.2: media type names are case-insensitive; RFC 9110, section 12.5.1:
# a range such as image/* stands for every subtype.
@pytest.mark.parametrize(
    ("patterns", "media_type"),
    [
        (
            ("image/*",),
            "image/png",
        ),
        (
            ("image/*",),
            "image/*",
        ),
        (
            ("image/png",),
            "image/png",
        ),
        (
            ("IMAGE/PNG",),
            "image/png",
        ),
        (
            ("image/*",),
            "IMAGE/PNG",
        ),
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
    patterns: tuple[str, ...],
    media_type: str,
) -> None:
    media = MediaTypes(*patterns)

    assert media.covers(media_type)


@pytest.mark.parametrize(
    ("patterns", "media_type"),
    [
        (
            ("image/png",),
            "image/*",
        ),
        (
            ("image/png",),
            "image/jpeg",
        ),
        (
            ("image/*",),
            "application/pdf",
        ),
        (
            (),
            "image/png",
        ),
    ],
    ids=[
        "type-does-not-cover-its-range",
        "type-does-not-cover-another",
        "range-covers-only-its-top-level-type",
        "text-only-perceives-no-media",
    ],
)
def test_a_model_does_not_perceive_media_its_patterns_miss(
    patterns: tuple[str, ...],
    media_type: str,
) -> None:
    media = MediaTypes(*patterns)

    assert not media.covers(media_type)


def test_a_pattern_that_is_not_a_media_type_raises() -> None:
    with pytest.raises(ConfigurationError) as raised:
        MediaTypes("image/*", "png")

    assert str(raised.value) == "each pattern must be an IANA media type, such as image/*"


@pytest.mark.parametrize(
    ("kind", "arguments"),
    [
        (
            Prompt,
            {
                "system": "Rate the action.",
                "content": (OpaquePart(kind="audio"),),
            },
        ),
        (
            Question,
            {
                "statement": "The action is safe.",
                "state": (OpaquePart(kind="audio"),),
            },
        ),
    ],
    ids=["prompt", "question"],
)
def test_content_a_model_cannot_be_given_raises(
    kind: type,
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**arguments)

    assert str(raised.value) == "a model is given text and media only"


@pytest.mark.parametrize(
    ("probability", "confidence", "message"),
    [
        (1.5, None, "probability must be between 0 and 1, got 1.5"),
        (-0.1, None, "probability must be between 0 and 1, got -0.1"),
        (
            math.nan,
            None,
            "probability must be between 0 and 1, got nan",
        ),
        (0.5, 1.5, "confidence must be between 0 and 1, got 1.5"),
        (0.5, -0.1, "confidence must be between 0 and 1, got -0.1"),
        (
            0.5,
            math.nan,
            "confidence must be between 0 and 1, got nan",
        ),
        (True, None, "probability must be a number between 0 and 1, got True"),
        ("0.5", None, "probability must be a number between 0 and 1, got '0.5'"),
        (None, None, "probability must be a number between 0 and 1, got None"),
        (0.5, True, "confidence must be a number between 0 and 1, got True"),
        (0.5, "0.5", "confidence must be a number between 0 and 1, got '0.5'"),
    ],
    ids=[
        "probability-above-one",
        "probability-below-zero",
        "probability-not-a-number",
        "confidence-above-one",
        "confidence-below-zero",
        "confidence-not-a-number",
        "probability-bool",
        "probability-string",
        "probability-none",
        "confidence-bool",
        "confidence-string",
    ],
)
def test_a_decision_that_is_not_a_number_from_zero_to_one_raises_naming_the_field(
    probability: object,
    confidence: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        Decision(
            probability=probability,  # ty: ignore[invalid-argument-type]
            confidence=confidence,  # ty: ignore[invalid-argument-type]
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("probability", "confidence", "kept"),
    [
        (
            Fraction(1, 2),
            Fraction(1, 4),
            ("0.5", "0.25"),
        ),
        (
            Decimal("0.25"),
            Decimal("0.75"),
            ("0.25", "0.75"),
        ),
        (
            1,
            0,
            ("1.0", "0.0"),
        ),
        (
            -0.0,
            -0.0,
            ("0.0", "0.0"),
        ),
    ],
    ids=["fractions", "decimals", "integers", "negative-zeros"],
)
def test_a_decision_keeps_its_probability_and_confidence_as_floats(
    probability: object,
    confidence: object,
    kept: tuple[str, str],
) -> None:
    decision = Decision(
        probability=probability,  # ty: ignore[invalid-argument-type]
        confidence=confidence,  # ty: ignore[invalid-argument-type]
    )

    assert (
        repr(decision.probability),
        repr(decision.confidence),
    ) == kept


@pytest.mark.parametrize(
    ("probability", "confidence"),
    [
        (0.0, 1.0),
        (1.0, 0.0),
    ],
    ids=["lower-bound", "upper-bound"],
)
def test_a_decision_holds_probabilities_at_the_bounds(
    probability: float,
    confidence: float,
) -> None:
    decision = Decision(
        probability=probability,
        confidence=confidence,
    )

    assert (
        decision.probability,
        decision.confidence,
    ) == (
        probability,
        confidence,
    )


@pytest.mark.parametrize(
    ("kind", "function"),
    [
        (
            GenerativeModel,
            {"generate": build_refusal},
        ),
        (
            AsyncGenerativeModel,
            {"generate": build_refusal},
        ),
        (
            DecisionModel,
            {"decide": build_refusal},
        ),
        (
            AsyncDecisionModel,
            {"decide": build_refusal},
        ),
    ],
    ids=["generative", "async-generative", "decision", "async-decision"],
)
def test_a_model_without_a_name_raises(
    kind: type,
    function: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(
            name="",
            **function,
        )

    assert str(raised.value) == "a model needs a name, such as the provider's model id"


@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        (
            GenerativeModel,
            {
                "name": "m",
                "generate": "x",
            },
            "generate must be callable, got an instance of str",
        ),
        (
            AsyncGenerativeModel,
            {
                "name": "m",
                "generate": None,
            },
            "generate must be callable, got None",
        ),
        (
            DecisionModel,
            {
                "name": "m",
                "decide": "x",
            },
            "decide must be callable, got an instance of str",
        ),
        (
            AsyncDecisionModel,
            {
                "name": "m",
                "decide": None,
            },
            "decide must be callable, got None",
        ),
        (
            GenerativeModel,
            {
                "name": "m",
                "generate": build_refusal,
                "perceives": "image/png",
            },
            "perceives must be a MediaTypes, got an instance of str",
        ),
        (
            AsyncDecisionModel,
            {
                "name": "m",
                "decide": build_refusal,
                "perceives": ("image/png",),
            },
            "perceives must be a MediaTypes, got an instance of tuple",
        ),
        (
            AsyncGenerativeModel,
            {
                "name": 3,
                "generate": build_refusal,
            },
            "name must be a str, got an instance of int",
        ),
        (
            DecisionModel,
            {
                "name": None,
                "decide": build_refusal,
            },
            "name must be a str, got None",
        ),
    ],
    ids=[
        "generate-not-callable",
        "async-generate-not-callable",
        "decide-not-callable",
        "async-decide-not-callable",
        "perceives-a-string",
        "async-perceives-a-tuple",
        "async-name-a-number",
        "name-none",
    ],
)
def test_a_model_field_of_the_wrong_type_raises_naming_the_field_when_built(
    kind: type,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**fields)

    assert str(raised.value) == message


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(
        self,
        value: int,
    ) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


@NOT_YET
@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        (
            AlternativeToken,
            {
                "token": 1,
                "log_probability": -1.0,
            },
            "token must be a str, got an instance of int",
        ),
        (
            AlternativeToken,
            {
                "token": "a",
                "log_probability": 0.5,
            },
            "log_probability must be a number of at most 0, got 0.5",
        ),
        (
            AlternativeToken,
            {
                "token": "a",
                "log_probability": True,
            },
            "log_probability must be a number of at most 0, got True",
        ),
        (
            TokenLogProbability,
            {
                "token": None,
                "log_probability": -1.0,
            },
            "token must be a str, got None",
        ),
        (
            TokenLogProbability,
            {
                "token": "a",
                "log_probability": math.nan,
            },
            "log_probability must be a number of at most 0, got nan",
        ),
        (
            TokenLogProbability,
            {
                "token": "a",
                "log_probability": "-1",
            },
            "log_probability must be a number of at most 0, got '-1'",
        ),
        (
            TokenLogProbability,
            {
                "token": "a",
                "log_probability": -1.0,
                "alternatives": [],
            },
            "alternatives must be a tuple, got an instance of list",
        ),
        (
            TokenLogProbability,
            {
                "token": "a",
                "log_probability": -1.0,
                "alternatives": ({"token": "b"},),
            },
            "alternatives[0] must be an AlternativeToken, got an instance of dict",
        ),
        (
            Generation,
            {
                "text": "a",
                "log_probabilities": [],
            },
            "log_probabilities must be a tuple, got an instance of list",
        ),
        (
            Generation,
            {
                "text": "a",
                "log_probabilities": ("a",),
            },
            "log_probabilities[0] must be a TokenLogProbability, got an instance of str",
        ),
    ],
    ids=[
        "alternative-token-not-a-string",
        "alternative-log-probability-above-zero",
        "alternative-log-probability-bool",
        "position-token-none",
        "position-log-probability-not-a-number",
        "position-log-probability-string",
        "position-alternatives-a-list",
        "position-alternative-a-dict",
        "generation-log-probabilities-a-list",
        "generation-log-probability-a-string",
    ],
)
def test_a_log_probability_field_of_the_wrong_type_or_value_raises_naming_the_field(
    kind: type,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**fields)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    "kind",
    [
        AlternativeToken,
        TokenLogProbability,
    ],
    ids=["alternative-token", "position"],
)
@pytest.mark.parametrize(
    ("log_probability", "kept"),
    [
        pytest.param(
            0,
            "0.0",
            id="an-integer-zero",
            marks=NOT_YET,
        ),
        pytest.param(
            Fraction(-1, 2),
            "-0.5",
            id="a-fraction",
            marks=NOT_YET,
        ),
        pytest.param(
            -math.inf,
            "-inf",
            id="minus-infinity",
        ),
        pytest.param(
            -(10**400),
            "-inf",
            id="an-integer-too-small-for-a-float",
            marks=NOT_YET,
        ),
    ],
)
def test_a_log_probability_at_most_zero_is_kept_as_a_float(
    kind: type,
    log_probability: object,
    kept: str,
) -> None:
    held = kind(
        token="a",
        log_probability=log_probability,
    )

    assert repr(held.log_probability) == kept


@NOT_YET
@pytest.mark.parametrize(
    ("kind", "value", "message"),
    [
        (
            GenerativeModel,
            -1,
            "top_log_probabilities must be at least 0, got -1",
        ),
        (
            AsyncGenerativeModel,
            1.5,
            "top_log_probabilities must be a whole number of at least 0, got 1.5",
        ),
        (
            GenerativeModel,
            True,
            "top_log_probabilities must be a whole number of at least 0, got True",
        ),
    ],
    ids=["negative", "float", "bool"],
)
def test_alternatives_that_are_not_a_count_raise_naming_the_field_when_a_model_is_built(
    kind: type,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(
            name="m",
            generate=build_refusal,
            top_log_probabilities=value,
        )

    assert str(raised.value) == message


@NOT_YET
@pytest.mark.parametrize(
    "kind",
    [
        GenerativeModel,
        AsyncGenerativeModel,
    ],
    ids=["generative", "async-generative"],
)
def test_a_whole_number_of_alternatives_that_is_not_an_int_is_kept_as_an_int(kind: type) -> None:
    model = kind(
        name="m",
        generate=build_refusal,
        top_log_probabilities=ArrayInteger(3),
    )

    assert (
        type(model.top_log_probabilities),
        model.top_log_probabilities,
    ) == (
        int,
        3,
    )
