"""Every monitor and wrapper gives a value from 0 to 1 and fails closed when it must."""

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import NamedTuple

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hyoshigi import (
    GPT_OSS_SAFEGUARD,
    Action,
    Aggregation,
    AsyncCalibratedMonitor,
    AsyncCascadeMonitor,
    AsyncDecisionModel,
    AsyncDecisionMonitor,
    AsyncGenerativeModel,
    AsyncGuardMonitor,
    AsyncLLMMonitor,
    AsyncMonitor,
    AsyncRepeatedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    Decision,
    DecisionModel,
    DecisionMonitor,
    DecisionOutcome,
    FailedClosed,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    GuardMonitor,
    LLMMonitor,
    Monitor,
    MonitorPrompt,
    Refused,
    RepeatedMonitor,
    SuspicionScore,
    TextPart,
    Trajectory,
    Truncated,
    Unavailable,
)

MODEL_NAME = "trusted-1"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))
TRAJECTORY = Trajectory.from_run(
    "Summarise q3.md",
    answer="Done.",
).trajectory
GUARD_PROMPT = MonitorPrompt(
    system="Label the action.",
    request="{trajectory} {proposed_action}",
)
HONEST_SCORES = [0.0, 0.2, 0.2, 1.0]
UNIT = st.floats(
    min_value=0,
    max_value=1,
)
SCORES = st.integers(0, 10)


def build_scored_reply(score: int) -> Generation:
    return Generation(text=f"<reasoning>r</reasoning><score>{score}</score>")


SCORED_REPLIES = SCORES.map(build_scored_reply)
LABELLED_REPLIES = st.sampled_from(["valid", "invalid", "maybe"]).map(
    lambda label: Generation(text=label),
)
FAILURES = st.one_of(
    st.builds(
        Refused,
        explanation=st.none() | st.text(),
    ),
    st.builds(
        Unavailable,
        status=st.none() | st.integers(),
    ),
)
# Outcomes after which one ask settles a rating.
SETTLED = st.one_of(
    SCORED_REPLIES,
    FAILURES,
)
DECISIONS = st.builds(
    Decision,
    probability=UNIT,
    confidence=st.none() | UNIT,
)


def build_turns[Outcome](
    outcomes: st.SearchStrategy[Outcome],
) -> st.SearchStrategy[tuple[Outcome, ...]]:
    return st.lists(
        outcomes,
        min_size=1,
        max_size=4,
    ).map(tuple)


GENERATED = st.one_of(
    build_turns(SCORED_REPLIES),
    build_turns(LABELLED_REPLIES),
    build_turns(
        st.one_of(
            SETTLED,
            LABELLED_REPLIES,
            st.builds(
                Generation,
                text=st.text(),
            ),
            st.builds(
                Truncated,
                text=st.text(),
            ),
        ),
    ),
)
DECIDED = st.one_of(
    build_turns(DECISIONS),
    build_turns(FAILURES | DECISIONS),
)


class Script(NamedTuple):
    """What the scripted models give, and how a decision monitor aggregates it."""

    generated: tuple[GenerationOutcome, ...]
    decided: tuple[DecisionOutcome, ...]
    aggregation: Aggregation


SCRIPTS = st.builds(
    Script,
    generated=GENERATED,
    decided=DECIDED,
    aggregation=st.sampled_from(Aggregation),
)


def rate_synchronously(monitor: Monitor) -> SuspicionScore:
    return monitor.rate(
        ACTION,
        trajectory=TRAJECTORY,
    )


def rate_asynchronously(monitor: AsyncMonitor) -> SuspicionScore:
    return asyncio.run(
        monitor.rate(
            ACTION,
            trajectory=TRAJECTORY,
        ),
    )


def give_in_turn[Outcome](outcomes: tuple[Outcome, ...]) -> Callable[[object], Outcome]:
    """A function that gives the outcomes in turn and starts again after the last."""
    turns = itertools.cycle(outcomes)
    return lambda _: next(turns)


def give_in_turn_asynchronously[Outcome](
    outcomes: tuple[Outcome, ...],
) -> Callable[[object], Awaitable[Outcome]]:
    """The same, as an async function."""
    turns = itertools.cycle(outcomes)

    async def give(_: object) -> Outcome:
        return next(turns)

    return give


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Family[Wrapped]:
    """The classes of one family and how it rates a monitor."""

    generative_model: Callable[..., object]
    decision_model: Callable[..., object]
    llm: Callable[..., Wrapped]
    guard: Callable[..., Wrapped]
    decision: Callable[..., Wrapped]
    repeated: Callable[..., Wrapped]
    calibrated: Callable[..., Wrapped]
    cascade: Callable[..., Wrapped]
    give: Callable[[tuple[object, ...]], Callable[[object], object]]
    rate: Callable[[Wrapped], SuspicionScore]


SYNCHRONOUS = Family(
    generative_model=GenerativeModel,
    decision_model=DecisionModel,
    llm=LLMMonitor,
    guard=GuardMonitor,
    decision=DecisionMonitor,
    repeated=RepeatedMonitor,
    calibrated=CalibratedMonitor,
    cascade=CascadeMonitor,
    give=give_in_turn,
    rate=rate_synchronously,
)
ASYNCHRONOUS = Family(
    generative_model=AsyncGenerativeModel,
    decision_model=AsyncDecisionModel,
    llm=AsyncLLMMonitor,
    guard=AsyncGuardMonitor,
    decision=AsyncDecisionMonitor,
    repeated=AsyncRepeatedMonitor,
    calibrated=AsyncCalibratedMonitor,
    cascade=AsyncCascadeMonitor,
    give=give_in_turn_asynchronously,
    rate=rate_asynchronously,
)
FAMILIES = pytest.mark.parametrize(
    "family",
    [
        SYNCHRONOUS,
        ASYNCHRONOUS,
    ],
    ids=["sync", "async"],
)


def build_scripted_llm(
    family: Family,
    generated: tuple[GenerationOutcome, ...],
) -> Monitor | AsyncMonitor:
    """An LLM monitor whose model gives the outcomes in turn."""
    return family.llm(
        model=family.generative_model(
            name=MODEL_NAME,
            generate=family.give(generated),
        ),
    )


def build_llm(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return build_scripted_llm(
        family,
        script.generated,
    )


def build_guard(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.guard(
        model=family.generative_model(
            name=MODEL_NAME,
            generate=family.give(script.generated),
        ),
        prompt=GUARD_PROMPT,
        labels=GPT_OSS_SAFEGUARD,
        samples=3,
    )


def build_decision(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.decision(
        model=family.decision_model(
            name=MODEL_NAME,
            decide=family.give(script.decided),
        ),
        statements=("It leaks a key.", "It deletes files."),
        aggregation=script.aggregation,
    )


def build_repeated(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.repeated(
        monitor=build_llm(
            family,
            script,
        ),
        ensemble_size=3,
    )


def build_calibrated(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.calibrated(
        monitor=build_llm(
            family,
            script,
        ),
        honest_scores=HONEST_SCORES,
    )


def build_cascade(
    family: Family,
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.cascade(
        fast=build_llm(
            family,
            script,
        ),
        careful=build_llm(
            family,
            script,
        ),
        escalation_threshold=0.5,
    )


@FAMILIES
@pytest.mark.parametrize(
    "build",
    [
        build_llm,
        build_guard,
        build_decision,
        build_repeated,
        build_calibrated,
        build_cascade,
    ],
)
@given(script=SCRIPTS)
def test_every_monitor_and_wrapper_gives_a_value_from_0_to_1(
    family: Family,
    build: Callable[
        [
            Family,
            Script,
        ],
        Monitor | AsyncMonitor,
    ],
    script: Script,
) -> None:
    score = family.rate(
        build(
            family,
            script,
        ),
    )

    assert 0.0 <= score.value <= 1.0


@FAMILIES
@given(
    calls=st.lists(
        SETTLED,
        min_size=3,
        max_size=3,
    ).filter(
        lambda calls: any(
            isinstance(
                call,
                Refused | Unavailable,
            )
            for call in calls
        ),
    ),
)
def test_repeated_ratings_fail_closed_when_any_call_does(
    family: Family,
    calls: list[GenerationOutcome],
) -> None:
    repeated = family.repeated(
        monitor=build_scripted_llm(
            family,
            tuple(calls),
        ),
        ensemble_size=3,
    )

    score = family.rate(repeated)

    assert (
        score.value,
        score.failed_closed,
    ) == (1.0, True)


@FAMILIES
@given(
    failure=FAILURES,
    honest_scores=st.lists(
        UNIT,
        min_size=1,
        max_size=5,
    ),
)
def test_a_calibrated_monitor_returns_a_score_that_failed_closed_unranked(
    family: Family,
    failure: Refused | Unavailable,
    honest_scores: list[float],
) -> None:
    calibrated = family.calibrated(
        monitor=build_scripted_llm(
            family,
            (failure,),
        ),
        honest_scores=honest_scores,
    )

    score = family.rate(calibrated)

    assert (
        score.value,
        score.evidence,
    ) == (
        1.0,
        FailedClosed(
            model=MODEL_NAME,
            cause=failure,
        ),
    )


@FAMILIES
@given(
    fast_failure=FAILURES,
    careful_failure=FAILURES,
    escalation_threshold=UNIT,
)
def test_a_cascade_gives_the_careful_failure_after_a_fast_score_that_failed_closed(
    family: Family,
    fast_failure: Refused | Unavailable,
    careful_failure: Refused | Unavailable,
    escalation_threshold: float,
) -> None:
    cascade = family.cascade(
        fast=build_scripted_llm(
            family,
            (fast_failure,),
        ),
        careful=build_scripted_llm(
            family,
            (careful_failure,),
        ),
        escalation_threshold=escalation_threshold,
    )

    score = family.rate(cascade)

    assert (
        score.value,
        score.evidence,
    ) == (
        1.0,
        FailedClosed(
            model=MODEL_NAME,
            cause=careful_failure,
        ),
    )


@FAMILIES
@given(
    fast_failure=FAILURES,
    careful_score=SCORES,
    escalation_threshold=UNIT,
)
def test_a_cascade_gives_the_careful_score_after_a_fast_score_that_failed_closed(
    family: Family,
    fast_failure: Refused | Unavailable,
    careful_score: int,
    escalation_threshold: float,
) -> None:
    cascade = family.cascade(
        fast=build_scripted_llm(
            family,
            (fast_failure,),
        ),
        careful=build_scripted_llm(
            family,
            (build_scored_reply(careful_score),),
        ),
        escalation_threshold=escalation_threshold,
    )

    score = family.rate(cascade)

    assert score.value == careful_score / 10
