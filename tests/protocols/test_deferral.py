"""Defer to resample: what it decides about a proposal, its resamples and a trusted step."""

import asyncio
from collections.abc import Callable
from functools import partial

import pytest
from families import (
    FAILURE_IDS,
    FAILURES_WITH_AN_ANSWER,
    MODEL_NAME,
    UNAVAILABLE_EXPLANATION,
    UNAVAILABLE_SCORE,
    AnyFamily,
    Script,
    build_action,
    build_asynchronous_llm,
    build_proposal,
    build_reply,
    build_score,
    build_synchronous_llm,
    build_trajectory,
    run,
)

from hyoshigi import (
    Action,
    AsyncDeferToResample,
    AsyncGenerativeModel,
    AsyncLLMMonitor,
    Audit,
    AuditResponse,
    Block,
    ConfigurationError,
    DeferToResample,
    FailureResponse,
    Flag,
    GenerationOutcome,
    Policy,
    Prompt,
    Proposal,
    ProtocolDecision,
    ProtocolRequest,
    Replace,
    Resample,
    Resampling,
    Run,
    SuspicionScore,
    TrustedStep,
    Unavailable,
)

PROPOSED = build_action("proposed")
FIRST = build_action("first")
SECOND = build_action("second")
THIRD = build_action("third")
TRUSTED = build_action("trusted")
UNAVAILABLE = Unavailable(status=503)
PROPOSED_AT_0_6 = build_proposal(
    PROPOSED,
    build_score(0.6),
)
PROPOSED_AT_0_9 = build_proposal(
    PROPOSED,
    build_score(0.9),
)
PROPOSED_UNAVAILABLE = build_proposal(
    PROPOSED,
    UNAVAILABLE_SCORE,
)
PROPOSED_AT_0_7 = build_proposal(
    PROPOSED,
    build_score(0.7),
)
PROPOSED_REFUSED = build_proposal(
    PROPOSED,
    FAILURES_WITH_AN_ANSWER[0][1],
)
# Trusted monitoring: no resamples, and no score between the thresholds.
TRUSTED_MONITORING = {
    "defer_threshold": 0.7,
    "audit_threshold": 0.7,
    "max_resamples": 0,
}
FIRST_AT_0_7 = build_proposal(
    FIRST,
    build_score(0.7),
)
FIRST_AT_1_0 = build_proposal(
    FIRST,
    build_score(1.0),
)
FIRST_UNAVAILABLE = build_proposal(
    FIRST,
    UNAVAILABLE_SCORE,
)
SECOND_AT_0_8 = build_proposal(
    SECOND,
    build_score(0.8),
)
TRUSTED_PROPOSAL = Proposal(
    action=TRUSTED,
    score=None,
    policy=Policy.TRUSTED,
)
RESAMPLES = (
    FIRST,
    SECOND,
)


def build_request_for_one(*proposals: Proposal) -> Resample:
    """The request for one more resample that a sequential protocol makes after the proposals."""
    return Resample(
        count=1,
        proposals=proposals,
        position=1,
    )


@pytest.mark.parametrize(
    ("reply", "max_resamples", "outcome"),
    [
        pytest.param(
            4,
            2,
            Run(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="below-the-defer-threshold-runs",
        ),
        pytest.param(
            5,
            2,
            Resample(
                count=2,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="at-the-defer-threshold",
        ),
        pytest.param(
            9,
            2,
            Resample(
                count=2,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            id="at-the-audit-threshold",
        ),
        pytest.param(
            5,
            1,
            Resample(
                count=1,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="one-resample",
        ),
    ],
)
def test_parallel_decide_asks_for_every_resample_unless_the_proposal_is_below_the_defer_threshold(
    family: AnyFamily,
    reply: int,
    max_resamples: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        max_resamples=max_resamples,
        resampling=Resampling.PARALLEL,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("reply", "max_resamples", "outcome"),
    [
        pytest.param(
            5,
            1,
            Resample(
                count=1,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="one-resample",
        ),
        pytest.param(
            5,
            2,
            Resample(
                count=1,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="two-resamples",
        ),
        pytest.param(
            5,
            3,
            Resample(
                count=1,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="three-resamples",
        ),
        pytest.param(
            9,
            2,
            Resample(
                count=1,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            id="at-the-audit-threshold",
        ),
    ],
)
def test_decide_asks_for_one_resample_by_default_for_a_proposal_at_or_above_the_defer_threshold(
    family: AnyFamily,
    reply: int,
    max_resamples: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        max_resamples=max_resamples,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("request_", "answer", "reply", "outcome"),
    [
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_6),
            FIRST,
            5,
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="asks-again-after-a-first-resample-at-the-defer-threshold",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_AT_0_7,
            ),
            SECOND,
            8,
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_AT_0_7,
                    build_proposal(
                        SECOND,
                        build_score(0.8),
                    ),
                ),
                position=1,
            ),
            id="asks-again-after-a-second-resample-above-it",
        ),
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_6),
            FIRST,
            4,
            Replace(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="replaces-with-a-first-resample-below-it",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_AT_0_7,
            ),
            SECOND,
            3,
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_AT_0_7,
                    build_proposal(
                        SECOND,
                        build_score(0.3),
                    ),
                ),
                position=1,
            ),
            id="replaces-with-a-second-resample-below-it",
        ),
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_9),
            FIRST,
            9,
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(0.9),
                    ),
                ),
                position=1,
            ),
            id="asks-again-while-every-proposal-is-at-the-audit-threshold",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_9,
                FIRST_AT_1_0,
            ),
            SECOND,
            4,
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_9,
                    FIRST_AT_1_0,
                    build_proposal(
                        SECOND,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="replaces-with-a-resample-below-the-defer-threshold-after-proposals-above-it",
        ),
    ],
)
def test_sequential_resume_draws_one_resample_at_a_time_until_one_is_below_the_defer_threshold(
    family: AnyFamily,
    request_: Resample,
    answer: Action,
    reply: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.resume_deferral(
        script,
        request=request_,
        proposals=(answer,),
        trajectory=build_trajectory(),
        max_resamples=3,
    )

    assert decided == outcome


def test_parallel_resume_decides_the_step_from_the_proposals_it_got(family: AnyFamily) -> None:
    script = Script(build_reply(6))

    decided = family.resume_deferral(
        script,
        request=build_request_for_one(PROPOSED_AT_0_6),
        proposals=(FIRST,),
        trajectory=build_trajectory(),
        max_resamples=3,
        resampling=Resampling.PARALLEL,
    )

    assert decided == TrustedStep(
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                build_score(0.6),
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("request_", "answer", "reply", "max_resamples", "outcome"),
    [
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_6),
            FIRST,
            6,
            1,
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.6),
                    ),
                ),
                position=1,
            ),
            id="the-first-of-one-asks-for-a-trusted-step",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_AT_0_7,
            ),
            SECOND,
            8,
            2,
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_AT_0_7,
                    SECOND_AT_0_8,
                ),
                position=1,
            ),
            id="between-the-thresholds-asks-for-a-trusted-step",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_AT_0_7,
                SECOND_AT_0_8,
            ),
            THIRD,
            7,
            3,
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_AT_0_7,
                    SECOND_AT_0_8,
                    build_proposal(
                        THIRD,
                        build_score(0.7),
                    ),
                ),
                position=1,
            ),
            id="the-third-of-three-asks-for-a-trusted-step",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_9,
                FIRST_AT_1_0,
            ),
            SECOND,
            9,
            2,
            Audit(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_9,
                    FIRST_AT_1_0,
                    build_proposal(
                        SECOND,
                        build_score(0.9),
                    ),
                ),
                position=1,
            ),
            id="at-the-audit-threshold-audits-the-most-suspicious",
        ),
    ],
)
def test_sequential_resume_decides_the_step_once_max_resamples_are_drawn(
    family: AnyFamily,
    request_: Resample,
    answer: Action,
    reply: int,
    max_resamples: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.resume_deferral(
        script,
        request=request_,
        proposals=(answer,),
        trajectory=build_trajectory(),
        max_resamples=max_resamples,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("failure", "score"),
    FAILURES_WITH_AN_ANSWER,
    ids=FAILURE_IDS,
)
def test_sequential_resume_audits_a_resample_that_failed_closed_with_an_answer_at_once(
    family: AnyFamily,
    failure: GenerationOutcome,
    score: SuspicionScore,
) -> None:
    script = Script(failure)

    decided = family.resume_deferral(
        script,
        request=build_request_for_one(PROPOSED_AT_0_6),
        proposals=(FIRST,),
        trajectory=build_trajectory(),
        max_resamples=3,
    )

    assert decided == Audit(
        action=FIRST,
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                score,
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("request_", "answer", "max_resamples", "outcome"),
    [
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_6),
            FIRST,
            3,
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_UNAVAILABLE,
                ),
                position=1,
            ),
            id="is-not-an-acceptable-resample",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_UNAVAILABLE,
            ),
            SECOND,
            2,
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_UNAVAILABLE,
                    build_proposal(
                        SECOND,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="is-one-of-the-resamples-drawn",
        ),
    ],
)
def test_sequential_resume_counts_a_resample_whose_monitor_got_no_answer_towards_max_resamples(
    family: AnyFamily,
    request_: Resample,
    answer: Action,
    max_resamples: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(UNAVAILABLE)

    decided = family.resume_deferral(
        script,
        request=request_,
        proposals=(answer,),
        trajectory=build_trajectory(),
        max_resamples=max_resamples,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("request_", "reply", "max_resamples", "outcome"),
    [
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_UNAVAILABLE,
            ),
            4,
            3,
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_UNAVAILABLE,
                    build_proposal(
                        SECOND,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="below-the-defer-threshold-replaces",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_UNAVAILABLE,
            ),
            6,
            2,
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_UNAVAILABLE,
                    build_proposal(
                        SECOND,
                        build_score(0.6),
                    ),
                ),
                position=1,
            ),
            id="at-max-resamples-asks-for-a-trusted-step",
        ),
        pytest.param(
            build_request_for_one(
                PROPOSED_AT_0_6,
                FIRST_UNAVAILABLE,
            ),
            6,
            3,
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_UNAVAILABLE,
                    build_proposal(
                        SECOND,
                        build_score(0.6),
                    ),
                ),
                position=1,
            ),
            id="short-of-max-resamples-asks-again",
        ),
    ],
)
def test_sequential_resume_decides_from_a_resample_that_follows_one_whose_monitor_got_no_answer(
    family: AnyFamily,
    request_: Resample,
    reply: int,
    max_resamples: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.resume_deferral(
        script,
        request=request_,
        proposals=(SECOND,),
        trajectory=build_trajectory(),
        max_resamples=max_resamples,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("first", "outcome"),
    [
        pytest.param(
            build_reply(4),
            Replace(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.4),
                    ),
                    SECOND_AT_0_8,
                ),
                position=1,
            ),
            id="an-earlier-resample-below-the-defer-threshold-replaces",
        ),
        pytest.param(
            FAILURES_WITH_AN_ANSWER[0][0],
            Audit(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        FAILURES_WITH_AN_ANSWER[0][1],
                    ),
                    SECOND_AT_0_8,
                ),
                position=1,
            ),
            id="an-earlier-resample-that-failed-closed-with-an-answer-is-audited",
        ),
    ],
)
def test_sequential_resume_stops_drawing_when_an_earlier_proposal_of_the_answer_ends_it(
    family: AnyFamily,
    first: GenerationOutcome,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(
        first,
        build_reply(8),
    )
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        max_resamples=3,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("request_", "proposals", "replies", "outcome"),
    [
        pytest.param(
            build_request_for_one(PROPOSED_AT_0_6),
            (),
            (),
            TrustedStep(
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            id="no-proposal",
        ),
        pytest.param(
            Resample(
                count=2,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            (FIRST,),
            (6,),
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.6),
                    ),
                ),
                position=1,
            ),
            id="one-proposal-of-two",
        ),
    ],
)
def test_sequential_resume_decides_the_step_when_the_answer_is_shorter_than_the_request(
    family: AnyFamily,
    request_: Resample,
    proposals: tuple[Action, ...],
    replies: tuple[int, ...],
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(*(build_reply(reply) for reply in replies))

    decided = family.resume_deferral(
        script,
        request=request_,
        proposals=proposals,
        trajectory=build_trajectory(),
        max_resamples=3,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("replies", "outcome"),
    [
        pytest.param(
            (3, 2),
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.3),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.2),
                    ),
                ),
                position=1,
            ),
            id="the-second-is-least-suspicious",
        ),
        pytest.param(
            (2, 3),
            Replace(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.2),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.3),
                    ),
                ),
                position=1,
            ),
            id="the-first-is-least-suspicious",
        ),
        pytest.param(
            (4, 4),
            Replace(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.4),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="a-tie-replaces-with-the-first",
        ),
        pytest.param(
            (8, 4),
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.8),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="one-under-the-threshold",
        ),
    ],
)
def test_the_least_suspicious_resample_under_the_defer_threshold_replaces_the_proposal(
    family: AnyFamily,
    replies: tuple[int, int],
    outcome: ProtocolDecision,
) -> None:
    script = Script(*(build_reply(reply) for reply in replies))
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("proposed", "replies", "outcome"),
    [
        pytest.param(
            PROPOSED_AT_0_6,
            (6, 8),
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.6),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.8),
                    ),
                ),
                position=1,
            ),
            id="all-between-the-thresholds",
        ),
        pytest.param(
            PROPOSED_AT_0_6,
            (5, 5),
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        build_score(0.5),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="resamples-at-the-defer-threshold",
        ),
        pytest.param(
            PROPOSED_AT_0_9,
            (8, 10),
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(0.8),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(1.0),
                    ),
                ),
                position=1,
            ),
            id="least-suspicious-just-under-the-audit-threshold",
        ),
    ],
)
def test_no_resample_under_the_defer_threshold_asks_for_a_trusted_step(
    family: AnyFamily,
    proposed: Proposal,
    replies: tuple[int, int],
    outcome: ProtocolRequest,
) -> None:
    script = Script(*(build_reply(reply) for reply in replies))
    request = Resample(
        count=2,
        proposals=(proposed,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == outcome


def test_the_trusted_steps_action_replaces_the_proposal_unrated(family: AnyFamily) -> None:
    request = TrustedStep(
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        Script(),
        request=request,
        proposals=(TRUSTED,),
        trajectory=build_trajectory(),
    )

    assert decided == Replace(
        action=TRUSTED,
        proposals=(
            PROPOSED_AT_0_6,
            TRUSTED_PROPOSAL,
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("replies", "outcome"),
    [
        pytest.param(
            (10, 9),
            Audit(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(1.0),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.9),
                    ),
                ),
                position=1,
            ),
            id="the-first-resample-is-most-suspicious",
        ),
        pytest.param(
            (9, 10),
            Audit(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(0.9),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(1.0),
                    ),
                ),
                position=1,
            ),
            id="the-second-resample-is-most-suspicious",
        ),
        pytest.param(
            (10, 10),
            Audit(
                action=FIRST,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(1.0),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(1.0),
                    ),
                ),
                position=1,
            ),
            id="a-tie-of-resamples-audits-the-first",
        ),
        pytest.param(
            (9, 9),
            Audit(
                action=PROPOSED,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        build_score(0.9),
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.9),
                    ),
                ),
                position=1,
            ),
            id="all-at-the-audit-threshold-audits-the-proposal",
        ),
    ],
)
def test_a_step_scored_at_or_above_the_audit_threshold_audits_its_most_suspicious_proposal(
    family: AnyFamily,
    replies: tuple[int, int],
    outcome: ProtocolDecision,
) -> None:
    script = Script(*(build_reply(reply) for reply in replies))
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_9,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == outcome


# Greenblatt et al. 2023, "AI Control", sec. 3.3: defer to trusted runs the proposal below the
# deferral threshold, audits it at the audit threshold, and asks the trusted model in between.
@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        pytest.param(
            4,
            Run(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.4),
                    ),
                ),
                position=1,
            ),
            id="below-the-defer-threshold-runs",
        ),
        pytest.param(
            5,
            TrustedStep(
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.5),
                    ),
                ),
                position=1,
            ),
            id="at-the-defer-threshold-asks-the-trusted-model",
        ),
        pytest.param(
            8,
            TrustedStep(
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.8),
                    ),
                ),
                position=1,
            ),
            id="below-the-audit-threshold-asks-the-trusted-model",
        ),
        pytest.param(
            9,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            id="at-the-audit-threshold-audits",
        ),
    ],
)
def test_with_no_resamples_the_protocol_defers_to_trusted(
    family: AnyFamily,
    reply: int,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(build_reply(reply))

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        max_resamples=0,
    )

    assert decided == outcome


# Greenblatt et al. 2023, "AI Control", sec. 3.3: trusted monitoring audits the proposal at the
# audit threshold and runs it below.
@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        pytest.param(
            6,
            Run(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            id="below-the-threshold-runs",
        ),
        pytest.param(
            7,
            Audit(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(0.7),
                    ),
                ),
                position=1,
            ),
            id="at-the-threshold-audits",
        ),
    ],
)
def test_with_no_resamples_and_equal_thresholds_the_protocol_is_trusted_monitoring(
    family: AnyFamily,
    reply: int,
    outcome: ProtocolDecision,
) -> None:
    script = Script(build_reply(reply))

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        defer_threshold=0.7,
        audit_threshold=0.7,
        max_resamples=0,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("failure", "score"),
    FAILURES_WITH_AN_ANSWER,
    ids=FAILURE_IDS,
)
def test_a_resample_that_failed_closed_with_an_answer_is_audited_before_any_score_counts(
    family: AnyFamily,
    failure: GenerationOutcome,
    score: SuspicionScore,
) -> None:
    script = Script(
        build_reply(2),
        failure,
    )
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == Audit(
        action=SECOND,
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                build_score(0.2),
            ),
            build_proposal(
                SECOND,
                score,
            ),
        ),
        position=1,
    )


def test_the_first_resample_that_failed_closed_with_an_answer_is_the_one_audited(
    family: AnyFamily,
) -> None:
    refused, refused_score = FAILURES_WITH_AN_ANSWER[0]
    unreadable, unreadable_score = FAILURES_WITH_AN_ANSWER[2]
    script = Script(
        refused,
        unreadable,
    )
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == Audit(
        action=FIRST,
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                refused_score,
            ),
            build_proposal(
                SECOND,
                unreadable_score,
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("proposed", "replies", "outcome"),
    [
        pytest.param(
            PROPOSED_AT_0_6,
            (
                UNAVAILABLE,
                build_reply(3),
            ),
            Replace(
                action=SECOND,
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        UNAVAILABLE_SCORE,
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.3),
                    ),
                ),
                position=1,
            ),
            id="the-other-resample-replaces",
        ),
        pytest.param(
            PROPOSED_AT_0_9,
            (
                UNAVAILABLE,
                build_reply(9),
            ),
            Audit(
                action=PROPOSED,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        UNAVAILABLE_SCORE,
                    ),
                    build_proposal(
                        SECOND,
                        build_score(0.9),
                    ),
                ),
                position=1,
            ),
            id="not-the-most-suspicious-to-audit",
        ),
        pytest.param(
            PROPOSED_AT_0_6,
            (
                UNAVAILABLE,
                UNAVAILABLE,
            ),
            TrustedStep(
                proposals=(
                    PROPOSED_AT_0_6,
                    build_proposal(
                        FIRST,
                        UNAVAILABLE_SCORE,
                    ),
                    build_proposal(
                        SECOND,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="all-unavailable-under-the-audit-threshold-asks-for-a-trusted-step",
        ),
        pytest.param(
            PROPOSED_AT_0_9,
            (
                UNAVAILABLE,
                UNAVAILABLE,
            ),
            Audit(
                action=PROPOSED,
                proposals=(
                    PROPOSED_AT_0_9,
                    build_proposal(
                        FIRST,
                        UNAVAILABLE_SCORE,
                    ),
                    build_proposal(
                        SECOND,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="all-unavailable-at-the-audit-threshold-audits-the-proposal",
        ),
    ],
)
def test_a_resample_whose_monitor_got_no_answer_is_set_aside(
    family: AnyFamily,
    proposed: Proposal,
    replies: tuple[GenerationOutcome, ...],
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(*replies)
    request = Resample(
        count=2,
        proposals=(proposed,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=RESAMPLES,
        trajectory=build_trajectory(),
        resampling=Resampling.PARALLEL,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("on_unavailable", "outcome"),
    [
        pytest.param(
            FailureResponse.AUDIT,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="audit",
        ),
        pytest.param(
            FailureResponse.BLOCK,
            Block(
                action=PROPOSED,
                reason=UNAVAILABLE_EXPLANATION,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="block",
        ),
        pytest.param(
            FailureResponse.TRUSTED_STEP,
            TrustedStep(
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="trusted-step",
        ),
        pytest.param(
            FailureResponse.RUN,
            Run(
                action=PROPOSED,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="run",
        ),
    ],
)
def test_on_unavailable_decides_about_a_proposal_whose_monitor_got_no_answer(
    family: AnyFamily,
    on_unavailable: FailureResponse,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    script = Script(UNAVAILABLE)

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_unavailable=on_unavailable,
    )

    assert decided == outcome


def test_a_proposal_whose_monitor_got_no_answer_is_audited_by_default(family: AnyFamily) -> None:
    script = Script(UNAVAILABLE)

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
    )

    assert decided == Audit(
        action=PROPOSED,
        proposals=(PROPOSED_UNAVAILABLE,),
        position=1,
    )


@pytest.mark.parametrize(
    ("failure", "score"),
    FAILURES_WITH_AN_ANSWER,
    ids=FAILURE_IDS,
)
def test_a_proposal_that_failed_closed_with_an_answer_is_audited_whatever_on_unavailable_says(
    family: AnyFamily,
    failure: GenerationOutcome,
    score: SuspicionScore,
) -> None:
    script = Script(failure)

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_unavailable=FailureResponse.RUN,
    )

    assert decided == Audit(
        action=PROPOSED,
        proposals=(
            build_proposal(
                PROPOSED,
                score,
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("proposed", "outcome"),
    [
        pytest.param(
            PROPOSED_AT_0_6,
            TrustedStep(
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            id="below-the-audit-threshold-asks-for-a-trusted-step",
        ),
        pytest.param(
            PROPOSED_AT_0_9,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            id="at-the-audit-threshold-audits",
        ),
    ],
)
def test_a_resample_answered_with_no_proposals_follows_the_proposals_own_score(
    family: AnyFamily,
    proposed: Proposal,
    outcome: ProtocolDecision | ProtocolRequest,
) -> None:
    request = Resample(
        count=2,
        proposals=(proposed,),
        position=1,
    )

    decided = family.resume_deferral(
        Script(),
        request=request,
        proposals=(),
        trajectory=build_trajectory(),
    )

    assert decided == outcome


def test_a_resample_answered_with_fewer_proposals_than_asked_rates_those_it_got(
    family: AnyFamily,
) -> None:
    script = Script(build_reply(2))
    request = Resample(
        count=2,
        proposals=(PROPOSED_AT_0_6,),
        position=1,
    )

    decided = family.resume_deferral(
        script,
        request=request,
        proposals=(FIRST,),
        trajectory=build_trajectory(),
    )

    assert decided == Replace(
        action=FIRST,
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                build_score(0.2),
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("proposals", "audited"),
    [
        pytest.param(
            (
                PROPOSED_AT_0_6,
                build_proposal(
                    FIRST,
                    build_score(0.8),
                ),
                build_proposal(
                    SECOND,
                    build_score(0.8),
                ),
            ),
            FIRST,
            id="the-first-of-a-tie",
        ),
        pytest.param(
            (
                PROPOSED_AT_0_6,
                build_proposal(
                    FIRST,
                    build_score(0.7),
                ),
                build_proposal(
                    SECOND,
                    build_score(0.8),
                ),
            ),
            SECOND,
            id="the-most-suspicious",
        ),
        pytest.param(
            (
                PROPOSED_AT_0_6,
                build_proposal(
                    FIRST,
                    UNAVAILABLE_SCORE,
                ),
                build_proposal(
                    SECOND,
                    build_score(0.7),
                ),
            ),
            SECOND,
            id="a-resample-whose-monitor-got-no-answer-is-set-aside",
        ),
        pytest.param(
            (
                PROPOSED_AT_0_6,
                build_proposal(
                    FIRST,
                    FAILURES_WITH_AN_ANSWER[0][1],
                ),
                build_proposal(
                    SECOND,
                    build_score(0.7),
                ),
            ),
            FIRST,
            id="a-resample-that-failed-closed-with-an-answer-is-kept",
        ),
        pytest.param(
            (PROPOSED_UNAVAILABLE,),
            PROPOSED,
            id="the-proposal-whose-monitor-got-no-answer-is-kept",
        ),
    ],
)
def test_a_trusted_step_answered_with_no_action_audits_the_most_suspicious_remaining_proposal(
    family: AnyFamily,
    proposals: tuple[Proposal, ...],
    audited: Action,
) -> None:
    request = TrustedStep(
        proposals=proposals,
        position=1,
    )

    decided = family.resume_deferral(
        Script(),
        request=request,
        proposals=(),
        trajectory=build_trajectory(),
    )

    assert decided == Audit(
        action=audited,
        proposals=proposals,
        position=1,
    )


RESAMPLE_AT_1 = Resample(
    count=2,
    proposals=(PROPOSED_AT_0_6,),
    position=1,
)
TRUSTED_STEP_AT_1 = TrustedStep(
    proposals=(PROPOSED_AT_0_6,),
    position=1,
)


@pytest.mark.parametrize(
    ("request_", "proposals", "message"),
    [
        pytest.param(
            Run(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            (FIRST,),
            "request must be a Resample or TrustedStep, got an instance of Run",
            id="a-decision",
        ),
        pytest.param(
            Resample(
                count=2,
                proposals=(PROPOSED_AT_0_6,),
                position=0,
            ),
            (FIRST,),
            "the request's position is not the length of the history",
            id="a-position-the-history-grew-past",
        ),
        pytest.param(
            TrustedStep(
                proposals=(PROPOSED_AT_0_6,),
                position=2,
            ),
            (TRUSTED,),
            "the request's position is not the length of the history",
            id="a-position-past-the-history",
        ),
        pytest.param(
            RESAMPLE_AT_1,
            [FIRST],
            "proposals must be a tuple, got an instance of list",
            id="proposals-a-list",
        ),
        pytest.param(
            RESAMPLE_AT_1,
            (
                FIRST,
                "Doing second.",
            ),
            "proposals[1] must be an Action, got an instance of str",
            id="a-proposal-a-string",
        ),
        pytest.param(
            RESAMPLE_AT_1,
            (
                FIRST,
                SECOND,
                THIRD,
            ),
            "the request asks for 2 or fewer proposals, got 3",
            id="more-resamples-than-asked",
        ),
        pytest.param(
            TRUSTED_STEP_AT_1,
            (
                TRUSTED,
                FIRST,
            ),
            "the request asks for 1 or fewer proposals, got 2",
            id="two-trusted-steps",
        ),
        pytest.param(
            TRUSTED_STEP_AT_1,
            (PROPOSED,),
            "a TrustedStep's answer must be the trusted model's action, not one of the agent's "
            "proposals",
            id="a-trusted-step-answered-with-the-agents-proposal",
        ),
    ],
)
def test_a_resume_that_cannot_work_is_refused_naming_why(
    family: AnyFamily,
    request_: ProtocolRequest,
    proposals: tuple[Action, ...],
    message: str,
) -> None:
    script = Script()

    with pytest.raises(ConfigurationError) as raised:
        family.resume_deferral(
            script,
            request=request_,
            proposals=proposals,
            trajectory=build_trajectory(),
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("model_outcome", "on_audit", "outcome"),
    [
        pytest.param(
            build_reply(7),
            AuditResponse.HOLD,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_7,),
                position=1,
            ),
            id="a-score-at-the-threshold-is-held",
        ),
        pytest.param(
            build_reply(7),
            AuditResponse.FLAG,
            Flag(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_7,),
                position=1,
            ),
            id="a-score-at-the-threshold-is-flagged",
        ),
        pytest.param(
            FAILURES_WITH_AN_ANSWER[0][0],
            AuditResponse.HOLD,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_REFUSED,),
                position=1,
            ),
            id="a-failed-rating-is-held",
        ),
        pytest.param(
            FAILURES_WITH_AN_ANSWER[0][0],
            AuditResponse.FLAG,
            Flag(
                action=PROPOSED,
                proposals=(PROPOSED_REFUSED,),
                position=1,
            ),
            id="a-failed-rating-is-flagged",
        ),
        pytest.param(
            UNAVAILABLE,
            AuditResponse.HOLD,
            Audit(
                action=PROPOSED,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="a-rating-with-no-answer-is-held",
        ),
        pytest.param(
            UNAVAILABLE,
            AuditResponse.FLAG,
            Flag(
                action=PROPOSED,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="a-rating-with-no-answer-is-flagged",
        ),
        pytest.param(
            build_reply(6),
            AuditResponse.FLAG,
            Run(
                action=PROPOSED,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            id="a-score-below-the-threshold-runs-under-flag",
        ),
    ],
)
def test_on_audit_decides_what_an_audited_proposal_becomes(
    family: AnyFamily,
    model_outcome: GenerationOutcome,
    on_audit: AuditResponse,
    outcome: ProtocolDecision,
) -> None:
    script = Script(model_outcome)

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_audit=on_audit,
        **TRUSTED_MONITORING,
    )

    assert decided == outcome


def test_a_trusted_step_answered_with_no_action_is_held_under_hold(family: AnyFamily) -> None:
    request = TrustedStep(
        proposals=(PROPOSED_UNAVAILABLE,),
        position=1,
    )

    resumed = family.resume_deferral(
        Script(),
        request=request,
        proposals=(),
        trajectory=build_trajectory(),
        on_audit=AuditResponse.HOLD,
        **TRUSTED_MONITORING,
    )

    assert resumed == Audit(
        action=PROPOSED,
        proposals=(PROPOSED_UNAVAILABLE,),
        position=1,
    )


@pytest.mark.parametrize(
    ("options", "request_", "message"),
    [
        pytest.param(
            {},
            Resample(
                count=1,
                proposals=(
                    PROPOSED_AT_0_6,
                    FIRST_AT_0_7,
                    build_proposal(
                        SECOND,
                        build_score(0.8),
                    ),
                ),
                position=1,
            ),
            "the request asks for resample 3, but max_resamples is 2",
            id="a-resample-after-all-the-resamples",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            {},
            Resample(
                count=3,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            "the request asks for resample 3, but max_resamples is 2",
            id="a-resample-of-more-than-the-rest",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            {"max_resamples": 0},
            Resample(
                count=1,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            "the request asks for resample 1, but max_resamples is 0",
            id="a-resample-of-a-protocol-with-no-resamples",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            {
                **TRUSTED_MONITORING,
                "on_audit": AuditResponse.FLAG,
            },
            Resample(
                count=1,
                proposals=(PROPOSED_AT_0_9,),
                position=1,
            ),
            "the request asks for resample 1, but max_resamples is 0",
            id="a-resample-under-flag",
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            {
                **TRUSTED_MONITORING,
                "on_audit": AuditResponse.FLAG,
            },
            TrustedStep(
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            "on_audit FLAG never asks for a TrustedStep",
            id="a-trusted-step-under-flag",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_a_resume_of_a_request_the_protocol_never_asks_for_is_refused_naming_why(
    family: AnyFamily,
    options: dict[str, object],
    request_: ProtocolRequest,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        family.resume_deferral(
            Script(),
            request=request_,
            proposals=(),
            trajectory=build_trajectory(),
            **options,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("on_unavailable", "outcome"),
    [
        pytest.param(
            FailureResponse.BLOCK,
            Block(
                action=PROPOSED,
                reason=UNAVAILABLE_EXPLANATION,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="block",
        ),
        pytest.param(
            FailureResponse.RUN,
            Run(
                action=PROPOSED,
                proposals=(PROPOSED_UNAVAILABLE,),
                position=1,
            ),
            id="run",
        ),
    ],
)
def test_on_audit_flag_leaves_a_block_and_a_run_for_an_unavailable_rating_alone(
    family: AnyFamily,
    on_unavailable: FailureResponse,
    outcome: ProtocolDecision,
) -> None:
    script = Script(UNAVAILABLE)

    decided = family.decide_deferral(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_audit=AuditResponse.FLAG,
        on_unavailable=on_unavailable,
        **TRUSTED_MONITORING,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param(
            {"defer_threshold": 1.5},
            "defer_threshold must be between 0 and 1, got 1.5",
            id="defer-threshold-above-1",
        ),
        pytest.param(
            {"audit_threshold": "0.9"},
            "audit_threshold must be a number between 0 and 1, got '0.9'",
            id="audit-threshold-a-string",
        ),
        pytest.param(
            {
                "defer_threshold": 0.8,
                "audit_threshold": 0.6,
            },
            "defer_threshold (0.8) must be at most audit_threshold (0.6)",
            id="defer-threshold-above-the-audit-threshold",
        ),
        pytest.param(
            {"max_resamples": -1},
            "max_resamples must be at least 0, got -1",
            id="max-resamples-negative",
        ),
        pytest.param(
            {"max_resamples": 2.0},
            "max_resamples must be a whole number of at least 0, got 2.0",
            id="max-resamples-a-float",
        ),
        pytest.param(
            {"on_unavailable": "audit"},
            "on_unavailable must be a FailureResponse, got an instance of str",
            id="on-unavailable-a-string",
        ),
        pytest.param(
            {"resampling": "sequential"},
            "resampling must be a Resampling, got an instance of str",
            id="resampling-a-string",
        ),
        pytest.param(
            {"on_audit": "flag"},
            "on_audit must be an AuditResponse, got an instance of str",
            id="on-audit-a-string",
        ),
        pytest.param(
            {"on_audit": AuditResponse.FLAG},
            "on_audit FLAG needs max_resamples 0, got 2",
            id="flag-with-resamples",
        ),
        pytest.param(
            {
                "on_audit": AuditResponse.FLAG,
                "max_resamples": 1,
                "defer_threshold": 0.7,
                "audit_threshold": 0.7,
            },
            "on_audit FLAG needs max_resamples 0, got 1",
            id="flag-with-one-resample",
        ),
        pytest.param(
            {
                "on_audit": AuditResponse.FLAG,
                "max_resamples": 0,
            },
            "on_audit FLAG needs defer_threshold (0.5) equal to audit_threshold (0.9)",
            id="flag-with-different-thresholds",
        ),
        pytest.param(
            {
                "on_audit": AuditResponse.FLAG,
                "max_resamples": 0,
                "defer_threshold": 0.7,
                "audit_threshold": 0.7,
                "on_unavailable": FailureResponse.TRUSTED_STEP,
            },
            "on_audit FLAG needs on_unavailable other than TRUSTED_STEP, since trusted "
            "monitoring has no trusted model",
            id="flag-with-a-trusted-step-for-an-unavailable-rating",
            marks=pytest.mark.xfail(strict=True),
        ),
    ],
)
def test_an_option_that_cannot_work_is_refused_naming_it(
    family: AnyFamily,
    options: dict[str, object],
    message: str,
) -> None:
    script = Script()

    with pytest.raises(ConfigurationError) as raised:
        family.build_deferral(
            script,
            **options,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            partial(
                DeferToResample,
                monitor=build_asynchronous_llm(Script()),
                defer_threshold=0.5,
                audit_threshold=0.9,
                max_resamples=2,
            ),
            "monitor must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncDeferToResample for an AsyncMonitor.",
            id="async-monitor-for-the-sync-protocol",
        ),
        pytest.param(
            partial(
                AsyncDeferToResample,
                monitor=build_synchronous_llm(Script()),
                defer_threshold=0.5,
                audit_threshold=0.9,
                max_resamples=2,
            ),
            "monitor must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use DeferToResample for a Monitor.",
            id="sync-monitor-for-the-async-protocol",
        ),
    ],
)
def test_a_monitor_of_the_other_family_is_refused_naming_the_protocol_it_belongs_to(
    build: Callable[[], DeferToResample | AsyncDeferToResample],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build()

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("resampling", "second_answers", "third_answers"),
    [
        pytest.param(
            Resampling.PARALLEL,
            (
                (
                    build_action("second-resample-1"),
                    build_action("second-resample-2"),
                ),
            ),
            (
                (
                    build_action("third-resample-1"),
                    build_action("third-resample-2"),
                ),
                (TRUSTED,),
            ),
            id="parallel",
        ),
        pytest.param(
            Resampling.SEQUENTIAL,
            (
                (build_action("second-resample-1"),),
                (build_action("second-resample-2"),),
            ),
            (
                (build_action("third-resample-1"),),
                (build_action("third-resample-2"),),
                (TRUSTED,),
            ),
            id="sequential",
        ),
    ],
)
def test_decide_resume_and_record_carry_a_trajectory_step_by_step(
    family: AnyFamily,
    resampling: Resampling,
    second_answers: tuple[tuple[Action, ...], ...],
    third_answers: tuple[tuple[Action, ...], ...],
) -> None:
    script = Script(
        build_reply(2),
        build_reply(7),
        build_reply(8),
        build_reply(3),
        build_reply(6),
        build_reply(8),
        build_reply(9),
    )
    trajectory = build_trajectory()

    family.take_deferral_step(
        script,
        proposed=FIRST,
        trajectory=trajectory,
        answers=(),
        resampling=resampling,
    )
    family.take_deferral_step(
        script,
        proposed=SECOND,
        trajectory=trajectory,
        answers=second_answers,
        resampling=resampling,
    )
    family.take_deferral_step(
        script,
        proposed=THIRD,
        trajectory=trajectory,
        answers=third_answers,
        resampling=resampling,
    )

    assert trajectory.records == (
        Run(
            action=FIRST,
            proposals=(
                build_proposal(
                    FIRST,
                    build_score(0.2),
                ),
            ),
            position=1,
        ),
        Replace(
            action=build_action("second-resample-2"),
            proposals=(
                build_proposal(
                    SECOND,
                    build_score(0.7),
                ),
                build_proposal(
                    build_action("second-resample-1"),
                    build_score(0.8),
                ),
                build_proposal(
                    build_action("second-resample-2"),
                    build_score(0.3),
                ),
            ),
            position=2,
        ),
        Replace(
            action=TRUSTED,
            proposals=(
                build_proposal(
                    THIRD,
                    build_score(0.6),
                ),
                build_proposal(
                    build_action("third-resample-1"),
                    build_score(0.8),
                ),
                build_proposal(
                    build_action("third-resample-2"),
                    build_score(0.9),
                ),
                TRUSTED_PROPOSAL,
            ),
            position=3,
        ),
    )


def test_the_async_protocol_rates_the_resamples_at_once() -> None:
    async def resume_at_a_barrier() -> ProtocolDecision | ProtocolRequest:
        # Each rating waits for the other, so only ratings in flight together finish.
        barrier = asyncio.Barrier(2)

        async def generate(prompt: Prompt) -> GenerationOutcome:
            await barrier.wait()
            return build_reply(2)

        protocol = AsyncDeferToResample(
            monitor=AsyncLLMMonitor(
                model=AsyncGenerativeModel(
                    name=MODEL_NAME,
                    generate=generate,
                ),
            ),
            defer_threshold=0.5,
            audit_threshold=0.9,
            max_resamples=2,
            resampling=Resampling.PARALLEL,
        )
        return await protocol.resume(
            Resample(
                count=2,
                proposals=(PROPOSED_AT_0_6,),
                position=1,
            ),
            proposals=RESAMPLES,
            trajectory=build_trajectory(),
        )

    decided = run(resume_at_a_barrier())

    assert decided == Replace(
        action=FIRST,
        proposals=(
            PROPOSED_AT_0_6,
            build_proposal(
                FIRST,
                build_score(0.2),
            ),
            build_proposal(
                SECOND,
                build_score(0.2),
            ),
        ),
        position=1,
    )
