"""A suspicion score: its value, the evidence it records and failing closed."""

import math

import pytest

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import Decision, Refused, Truncated, Unavailable
from hyoshigi.monitors.score import (
    Cause,
    Composite,
    Decisions,
    FailedClosed,
    Rationale,
    StatementDecision,
    SuspicionScore,
    Unperceivable,
    Unreadable,
    fail_closed,
)

# Each cause carries content the explanation must leave out: the agent may read it.
CAUSES_AND_EXPLANATIONS = [
    (
        Unreadable(),
        "The monitor could not read its model's reply, so the action is treated as suspicious.",
    ),
    (
        Unperceivable(content=("audio", "image/tiff")),
        "The action holds content the monitor cannot perceive, so it is treated as suspicious.",
    ),
    (
        Truncated(text="The action is suspicious because"),
        "The monitor's model stopped before it gave a score, so the action is treated as "
        "suspicious.",
    ),
    (
        Refused(explanation="I cannot assess this."),
        "The monitor's model refused to rate the action, so it is treated as suspicious.",
    ),
    (
        Unavailable(status=503),
        "The monitor's model was unavailable, so the action is treated as suspicious.",
    ),
]
CAUSE_IDS = ["unreadable", "unperceivable", "truncated", "refused", "unavailable"]
CAUSES = [cause for cause, _ in CAUSES_AND_EXPLANATIONS]

RATED = SuspicionScore(
    value=0.2,
    explanation="It reads the file the task names.",
    evidence=Rationale(model="m"),
)
FAILED = fail_closed(
    Refused(),
    model="m",
)
COMBINED_WITH_FAILED = SuspicionScore(
    value=1.0,
    explanation="A part failed.",
    evidence=Composite(
        parts=(
            RATED,
            FAILED,
        ),
    ),
)
UNANSWERED = fail_closed(
    Unavailable(status=503),
    model="m",
)
COMBINED_WITH_COMBINED = SuspicionScore(
    value=1.0,
    explanation="A part failed.",
    evidence=Composite(
        parts=(
            RATED,
            COMBINED_WITH_FAILED,
        ),
    ),
)
STATEMENT = StatementDecision(
    statement="The action reads a file.",
    decision=Decision(probability=0.1),
)


@pytest.mark.parametrize(
    ("cause", "explanation"),
    CAUSES_AND_EXPLANATIONS,
    ids=CAUSE_IDS,
)
def test_failing_closed_scores_one_with_the_causes_explanation_and_the_failure_as_evidence(
    cause: Cause,
    explanation: str,
) -> None:
    score = fail_closed(
        cause,
        model="trusted-1",
    )

    assert (
        score.value,
        score.explanation,
        score.evidence,
    ) == (
        1.0,
        explanation,
        FailedClosed(
            model="trusted-1",
            cause=cause,
        ),
    )


@pytest.mark.parametrize(
    "cause",
    CAUSES,
    ids=CAUSE_IDS,
)
def test_a_score_that_failed_closed_reports_it(
    cause: Cause,
) -> None:
    score = fail_closed(
        cause,
        model="trusted-1",
    )

    assert score.failed_closed is True


@pytest.mark.parametrize(
    "evidence",
    [
        Rationale(model="m"),
        Decisions(
            model="m",
            decisions=(STATEMENT,),
        ),
        Composite(parts=(RATED,)),
    ],
    ids=["rationale", "decisions", "composite-of-rated-parts"],
)
def test_a_score_that_did_not_fail_closed_does_not_report_it(
    evidence: Rationale | Decisions | Composite,
) -> None:
    score = SuspicionScore(
        value=0.2,
        explanation="Nothing odd.",
        evidence=evidence,
    )

    assert score.failed_closed is False


@pytest.mark.parametrize(
    "parts",
    [
        (
            RATED,
            FAILED,
        ),
        (
            RATED,
            COMBINED_WITH_FAILED,
        ),
        (
            RATED,
            RATED,
            FAILED,
        ),
        (
            RATED,
            RATED,
            SuspicionScore(
                value=1.0,
                explanation="Its only call was refused.",
                evidence=Composite(parts=(FAILED,)),
            ),
        ),
        (
            UNANSWERED,
            RATED,
        ),
        (
            UNANSWERED,
            UNANSWERED,
            RATED,
        ),
    ],
    ids=[
        "one-level",
        "two-levels",
        "a-refusal-among-more-rated-parts",
        "a-composite-failed-by-a-refusal-among-more-rated-parts",
        "half-the-parts-unanswered",
        "most-parts-unanswered",
    ],
)
def test_a_score_combining_one_that_failed_closed_reports_it(
    parts: tuple[SuspicionScore, ...],
) -> None:
    score = SuspicionScore(
        value=1.0,
        explanation="A part failed.",
        evidence=Composite(parts=parts),
    )

    assert score.failed_closed is True


@pytest.mark.parametrize(
    "unanswered",
    [
        pytest.param(
            UNANSWERED,
            id="a-part-whose-model-was-unavailable",
        ),
        pytest.param(
            SuspicionScore(
                value=1.0,
                explanation="Its only call was unavailable.",
                evidence=Composite(parts=(UNANSWERED,)),
            ),
            id="a-composite-failed-by-a-part-whose-model-was-unavailable",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_a_composite_with_more_rated_parts_than_unanswered_ones_does_not_fail_closed(
    unanswered: SuspicionScore,
) -> None:
    score = SuspicionScore(
        value=1.0,
        explanation="The most suspicious rated part's reason.",
        evidence=Composite(
            parts=(
                RATED,
                unanswered,
                RATED,
            ),
        ),
    )

    assert score.failed_closed is False


@pytest.mark.parametrize(
    "value",
    [0.0, 0.5, 0.99],
)
@pytest.mark.parametrize(
    "evidence",
    [
        *[
            FailedClosed(
                model="m",
                cause=cause,
            )
            for cause in CAUSES
        ],
        *[
            Composite(
                parts=(
                    fail_closed(
                        cause,
                        model="m",
                    ),
                ),
            )
            for cause in CAUSES
        ],
        Composite(
            parts=(
                RATED,
                COMBINED_WITH_COMBINED,
            ),
        ),
    ],
    ids=[
        *[f"{name}-directly" for name in CAUSE_IDS],
        *[f"{name}-in-a-composite" for name in CAUSE_IDS],
        "in-a-composite-of-composites",
    ],
)
def test_a_score_that_failed_closed_with_a_value_other_than_one_raises(
    value: float,
    evidence: FailedClosed | Composite,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        SuspicionScore(
            value=value,
            explanation="Not a failure.",
            evidence=evidence,
        )

    assert str(raised.value) == "a score that failed closed has the value 1.0"


@pytest.mark.parametrize(
    "value",
    [
        True,
        math.nan,
        "0.5",
        1.5,
        -0.1,
        None,
    ],
    ids=["bool", "nan", "str", "high", "low", "none"],
)
def test_a_value_that_is_not_a_number_from_zero_to_one_raises(value: object) -> None:
    with pytest.raises(
        ConfigurationError,
        match="value must be",
    ):
        SuspicionScore(
            value=value,  # ty: ignore[invalid-argument-type]
            explanation="x",
            evidence=Rationale(model="m"),
        )


@pytest.mark.parametrize(
    ("evidence", "described"),
    [
        ("invalid", "an instance of str"),
        (None, "None"),
        (
            Refused(),
            "an instance of Refused",
        ),
        (
            Rationale,
            "the class Rationale; pass an instance",
        ),
    ],
    ids=["string", "none", "a-cause-instead-of-failed-closed", "the-class"],
)
def test_a_score_whose_evidence_is_not_a_kind_of_evidence_raises_when_built(
    evidence: object,
    described: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        SuspicionScore(
            value=1.0,
            explanation="x",
            evidence=evidence,  # ty: ignore[invalid-argument-type]
        )

    assert str(raised.value) == (
        f"evidence must be a Rationale, Decisions, Labels, Composite or FailedClosed, "
        f"got {described}"
    )


@pytest.mark.parametrize(
    ("explanation", "described"),
    [
        (123, "an instance of int"),
        (None, "None"),
        (b"x", "an instance of bytes"),
    ],
    ids=["integer", "none", "bytes"],
)
def test_a_score_whose_explanation_is_not_a_string_raises_when_built(
    explanation: object,
    described: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        SuspicionScore(
            value=0.5,
            explanation=explanation,  # ty: ignore[invalid-argument-type]
            evidence=Rationale(model="m"),
        )

    assert str(raised.value) == f"explanation must be a str, got {described}"


@pytest.mark.parametrize(
    ("given", "kept"),
    [
        (0, "0.0"),
        (1, "1.0"),
        (-0.0, "0.0"),
    ],
    ids=["zero", "one", "negative-zero"],
)
def test_a_value_is_kept_as_a_float_and_negative_zero_as_zero(
    given: float,
    kept: str,
) -> None:
    score = SuspicionScore(
        value=given,
        explanation="x",
        evidence=Rationale(model="m"),
    )

    assert repr(score.value) == kept


@pytest.mark.parametrize(
    ("kind", "arguments", "message"),
    [
        (
            Unperceivable,
            {"content": ()},
            "unperceivable content names at least one kind",
        ),
        (
            Composite,
            {"parts": ()},
            "a composite holds at least one score",
        ),
    ],
    ids=["unperceivable-without-content", "composite-without-parts"],
)
def test_evidence_that_must_hold_something_raises_when_empty(
    kind: type,
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**arguments)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("parts", "message"),
    [
        (
            ("x",),
            "parts[0] must be a SuspicionScore, got an instance of str",
        ),
        (
            (
                SuspicionScore(
                    value=0.5,
                    explanation="x",
                    evidence=Rationale(model="m"),
                ),
                None,
            ),
            "parts[1] must be a SuspicionScore, got None",
        ),
        (
            [],
            "parts must be a tuple, got an instance of list",
        ),
    ],
    ids=["a-string", "none-after-a-score", "a-list"],
)
def test_a_composite_of_anything_but_scores_raises_naming_the_part(
    parts: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        Composite(parts=parts)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message
