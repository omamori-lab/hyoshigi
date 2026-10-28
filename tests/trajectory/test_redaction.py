"""Redaction at entry and in storage: what a trajectory keeps, dumps and loads in place of its
content, and what it leaves as it is.
"""

import itertools
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from hyoshigi import (
    JSON,
    Action,
    Audit,
    Author,
    Block,
    Composite,
    ConfigurationError,
    Description,
    FailedClosed,
    FormatError,
    Halt,
    MediaPart,
    Observation,
    OpaquePart,
    Policy,
    Proposal,
    ProtocolDecision,
    Rationale,
    Redaction,
    Refused,
    Replace,
    Run,
    Step,
    SuspicionScore,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
    Truncated,
)
from hyoshigi.trajectory.history import Event
from hyoshigi.trajectory.trajectory import Rating

# Each digit masked, as a redaction of phone or card numbers does; media blurred to new bytes.
MASKED = Redaction(
    text=partial(
        re.sub,
        "[0-9]",
        "#",
    ),
    media=partial(
        replace,
        data=b"blurred",
    ),
)
# Each digit masked; media replaced with a text placeholder.
TO_TEXT = Redaction(
    text=partial(
        re.sub,
        "[0-9]",
        "#",
    ),
    media=lambda part: TextPart(text="image 1"),
)
# Each digit masked; media kept as it is, so the text in text media shows its redaction.
TEXT_MASKED = Redaction(
    text=partial(
        re.sub,
        "[0-9]",
        "#",
    ),
    media=lambda part: part,
)
TASK = Observation(
    author=Author.USER,
    parts=(TextPart(text="Call Ana on 555-0100."),),
)
MASKED_TASK = Observation(
    author=Author.USER,
    parts=(TextPart(text="Call Ana on ###-####."),),
)
PNG = MediaPart(
    media_type="image/png",
    data=b"\x89PNG",
)
# An action with digits in every field; the ids and the tool name are not content.
DIALLING = Action(
    id="a1",
    thought=(
        TextPart(text="Dial 555-0100 first."),
        OpaquePart(kind="redacted_thinking_2"),
    ),
    parts=(
        TextPart(text="Dialling 555-0100."),
        MediaPart(
            media_type="image/png",
            data=b"\x89PNG",
            description=Description(
                text="Card 4111",
                author=Author.AGENT,
            ),
        ),
    ),
    tool_calls=(
        ToolCall(
            id="call_1",
            name="dial_v2",
            arguments='{"number": "555-0100"}',
        ),
    ),
)
MASKED_DIALLING = Action(
    id="a1",
    thought=(
        TextPart(text="Dial ###-#### first."),
        OpaquePart(kind="redacted_thinking_2"),
    ),
    parts=(
        TextPart(text="Dialling ###-####."),
        MediaPart(
            media_type="image/png",
            data=b"blurred",
            description=Description(
                text="Card ####",
                author=Author.AGENT,
            ),
        ),
    ),
    tool_calls=(
        ToolCall(
            id="call_1",
            name="dial_v2",
            arguments='{"number": "###-####"}',
        ),
    ),
)
# An action that delegates, and the score of it.
ASKING = Action(
    id="a2",
    parts=(TextPart(text="Asking who owns 555-0100."),),
    tool_calls=(
        ToolCall(
            id="call_2",
            name="task",
            arguments='{"prompt": "Find who owns 555-0100."}',
        ),
    ),
)
MASKED_ASKING = Action(
    id="a2",
    parts=(TextPart(text="Asking who owns ###-####."),),
    tool_calls=(
        ToolCall(
            id="call_2",
            name="task",
            arguments='{"prompt": "Find who owns ###-####."}',
        ),
    ),
)
TRUSTED_STEP = Action(
    id="t1",
    parts=(TextPart(text="I will not call 555-0100."),),
)
MASKED_TRUSTED_STEP = Action(
    id="t1",
    parts=(TextPart(text="I will not call ###-####."),),
)
CALM = SuspicionScore(
    value=0.2,
    explanation="It looks up 555-0100.",
    evidence=Rationale(model="trusted-1"),
)
MASKED_CALM = SuspicionScore(
    value=0.2,
    explanation="It looks up ###-####.",
    evidence=Rationale(model="trusted-1"),
)
PROPOSED = Proposal(
    action=ASKING,
    score=CALM,
    policy=Policy.UNTRUSTED,
)
MASKED_PROPOSED = Proposal(
    action=MASKED_ASKING,
    score=MASKED_CALM,
    policy=Policy.UNTRUSTED,
)
REASON = "It dials 555-0100."
MASKED_REASON = "It dials ###-####."
TRUNCATED_EXPLANATION = (
    "The monitor's model stopped before it gave a score, so the action is treated as suspicious."
)
REFUSED_EXPLANATION = (
    "The monitor's model refused to rate the action, so it is treated as suspicious."
)
STORED_ACTION_FIELDS: dict[str, JSON] = {
    "kind": "action",
    "id": "a2",
    "thought": [],
}
STORED_SCORE_FIELDS: dict[str, JSON] = {
    "value": 0.2,
    "evidence": {
        "kind": "rationale",
        "model": "trusted-1",
        "hidden_reasoning": [],
    },
}
STORED_SUBAGENT_FIELDS: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "ratings": [],
    "records": [],
    "subagents": {},
}
# The trajectory build_stored_trajectory keeps, as dump writes it.
STORED: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [
        {
            "kind": "observation",
            "author": "user",
            "source": None,
            "parts": [
                {
                    "kind": "text",
                    "text": "Call Ana on 555-0100.",
                },
            ],
        },
        {
            **STORED_ACTION_FIELDS,
            "parts": [
                {
                    "kind": "text",
                    "text": "Asking who owns 555-0100.",
                },
            ],
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "task",
                    "arguments": '{"prompt": "Find who owns 555-0100."}',
                },
            ],
        },
    ],
    "ratings": [
        {
            "action_id": "a2",
            "score": {
                **STORED_SCORE_FIELDS,
                "explanation": "It looks up 555-0100.",
            },
        },
    ],
    "records": [
        {
            "kind": "run",
            "proposal": 0,
            "proposals": [
                {
                    "action": {
                        **STORED_ACTION_FIELDS,
                        "parts": [
                            {
                                "kind": "text",
                                "text": "Asking who owns 555-0100.",
                            },
                        ],
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "name": "task",
                                "arguments": '{"prompt": "Find who owns 555-0100."}',
                            },
                        ],
                    },
                    "score": {
                        **STORED_SCORE_FIELDS,
                        "explanation": "It looks up 555-0100.",
                    },
                    "policy": "untrusted",
                },
            ],
            "position": 1,
            "sequence": 0,
        },
    ],
    "subagents": {
        "call_2": {
            **STORED_SUBAGENT_FIELDS,
            "history": [
                {
                    "kind": "observation",
                    "author": "delegator",
                    "source": None,
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Find who owns 555-0100.",
                        },
                    ],
                },
            ],
        },
    },
}
# The same trajectory with each digit of its content masked.
MASKED_STORED: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 3,
    "history": [
        {
            "kind": "observation",
            "author": "user",
            "source": None,
            "parts": [
                {
                    "kind": "text",
                    "text": "Call Ana on ###-####.",
                },
            ],
        },
        {
            **STORED_ACTION_FIELDS,
            "parts": [
                {
                    "kind": "text",
                    "text": "Asking who owns ###-####.",
                },
            ],
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "task",
                    "arguments": '{"prompt": "Find who owns ###-####."}',
                },
            ],
        },
    ],
    "ratings": [
        {
            "action_id": "a2",
            "score": {
                **STORED_SCORE_FIELDS,
                "explanation": "It looks up ###-####.",
            },
        },
    ],
    "records": [
        {
            "kind": "run",
            "proposal": 0,
            "proposals": [
                {
                    "action": {
                        **STORED_ACTION_FIELDS,
                        "parts": [
                            {
                                "kind": "text",
                                "text": "Asking who owns ###-####.",
                            },
                        ],
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "name": "task",
                                "arguments": '{"prompt": "Find who owns ###-####."}',
                            },
                        ],
                    },
                    "score": {
                        **STORED_SCORE_FIELDS,
                        "explanation": "It looks up ###-####.",
                    },
                    "policy": "untrusted",
                },
            ],
            "position": 1,
            "sequence": 0,
        },
    ],
    "subagents": {
        "call_2": {
            **STORED_SUBAGENT_FIELDS,
            "history": [
                {
                    "kind": "observation",
                    "author": "delegator",
                    "source": None,
                    "parts": [
                        {
                            "kind": "text",
                            "text": "Find who owns ###-####.",
                        },
                    ],
                },
            ],
        },
    },
}


def build_pseudonyms() -> Redaction:
    """A redaction that gives each text it is called on a new pseudonym, numbered in turn, so it
    gives two calls on one text two pseudonyms.
    """
    numbers = itertools.count(1)
    return Redaction(
        text=lambda text: f"person {next(numbers)}",
        media=lambda part: part,
    )


def build_stored_trajectory() -> Trajectory:
    """The task, a run of ASKING scored CALM, and the subagent ASKING delegated, unredacted."""
    trajectory = Trajectory()
    trajectory.observe(TASK)
    trajectory.record(
        Run(
            action=ASKING,
            proposals=(PROPOSED,),
            position=1,
        ),
    )
    trajectory.delegate(
        "Find who owns 555-0100.",
        call_id="call_2",
    )
    return trajectory


@pytest.fixture
def masking() -> Trajectory:
    """A trajectory that masks digits at entry, holding the task."""
    trajectory = Trajectory(redaction=MASKED)
    trajectory.observe(TASK)
    return trajectory


@pytest.mark.parametrize(
    ("functions", "message"),
    [
        (
            {
                "text": "mask",
                "media": partial(
                    replace,
                    data=b"",
                ),
            },
            "text must be callable, got an instance of str",
        ),
        (
            {
                "text": str,
                "media": None,
            },
            "media must be callable, got None",
        ),
    ],
    ids=["text", "media"],
)
def test_a_redaction_refuses_a_function_that_is_not_callable_naming_it(
    functions: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        Redaction(**functions)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("redaction", "message"),
    [
        (
            Redaction(
                text=str.encode,  # ty: ignore[invalid-argument-type]
                media=lambda part: part,
            ),
            "text must return a str, got an instance of bytes",
        ),
        (
            Redaction(
                text=str,
                media=lambda part: "an image",  # ty: ignore[invalid-argument-type]
            ),
            "media must return a TextPart, MediaPart or OpaquePart, got an instance of str",
        ),
    ],
    ids=["text", "media"],
)
def test_a_redaction_whose_function_returns_the_wrong_type_raises_naming_it(
    redaction: Redaction,
    message: str,
) -> None:
    trajectory = Trajectory(redaction=redaction)

    with pytest.raises(ConfigurationError) as raised:
        trajectory.act(DIALLING)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    "redact",
    [
        lambda redaction: Trajectory(redaction=redaction),
        lambda redaction: Trajectory().dump(redaction=redaction),
        lambda redaction: Trajectory.load(
            STORED,
            redaction=redaction,
        ),
    ],
    ids=["trajectory", "dump", "load"],
)
def test_a_redaction_that_is_not_a_redaction_is_refused_naming_it(
    redact: Callable[[object], object],
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        redact(MASKED.text)

    assert (
        str(raised.value) == "redaction must be a Redaction, got an instance of functools.partial"
    )


def test_an_action_is_kept_redacted_with_its_ids_and_tool_names(masking: Trajectory) -> None:
    masking.act(DIALLING)

    assert masking.history[-1] == MASKED_DIALLING


@pytest.mark.parametrize(
    ("observation", "kept"),
    [
        (
            Observation(
                author=Author.SCAFFOLD,
                source="memo 7",
                parts=(TextPart(text="Ana's number is 555-0100."),),
            ),
            Observation(
                author=Author.SCAFFOLD,
                source="memo #",
                parts=(TextPart(text="Ana's number is ###-####."),),
            ),
        ),
        (
            ToolObservation(
                call_id="call_1",
                is_error=True,
                parts=(TextPart(text="No line at 555-0100."),),
            ),
            ToolObservation(
                call_id="call_1",
                is_error=True,
                parts=(TextPart(text="No line at ###-####."),),
            ),
        ),
    ],
    ids=["observation-with-a-source", "tool-observation"],
)
def test_an_observation_is_kept_redacted_with_its_author_and_call(
    masking: Trajectory,
    observation: Observation,
    kept: Observation,
) -> None:
    masking.observe(observation)

    assert masking.history[-1] == kept


def test_media_a_redaction_turns_into_text_is_kept_with_that_text_redacted() -> None:
    trajectory = Trajectory(redaction=TO_TEXT)

    trajectory.observe(
        ToolObservation(
            call_id="call_1",
            parts=(PNG,),
        ),
    )

    assert trajectory.history == (
        ToolObservation(
            call_id="call_1",
            parts=(TextPart(text="image #"),),
        ),
    )


@pytest.mark.parametrize(
    ("score", "kept"),
    [
        pytest.param(
            CALM,
            MASKED_CALM,
            id="rationale",
        ),
        pytest.param(
            SuspicionScore(
                value=1.0,
                explanation=TRUNCATED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Truncated(text="<reasoning>It dials 555-0100"),
                ),
            ),
            SuspicionScore(
                value=1.0,
                explanation=TRUNCATED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Truncated(text="<reasoning>It dials ###-####"),
                ),
            ),
            id="truncated",
        ),
        pytest.param(
            SuspicionScore(
                value=1.0,
                explanation=REFUSED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Refused(explanation="Rating 555-0100 is not allowed."),
                ),
            ),
            SuspicionScore(
                value=1.0,
                explanation=REFUSED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Refused(explanation="Rating ###-#### is not allowed."),
                ),
            ),
            id="refused",
        ),
        pytest.param(
            SuspicionScore(
                value=1.0,
                explanation=REFUSED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Refused(),
                ),
            ),
            SuspicionScore(
                value=1.0,
                explanation=REFUSED_EXPLANATION,
                evidence=FailedClosed(
                    model="trusted-1",
                    cause=Refused(),
                ),
            ),
            id="refused-without-explanation",
        ),
        pytest.param(
            SuspicionScore(
                value=0.2,
                explanation="It looks up 555-0100.",
                evidence=Composite(parts=(CALM,)),
            ),
            SuspicionScore(
                value=0.2,
                explanation="It looks up ###-####.",
                evidence=Composite(parts=(MASKED_CALM,)),
            ),
            id="composite",
        ),
    ],
)
def test_an_attached_score_is_kept_redacted(
    masking: Trajectory,
    score: SuspicionScore,
    kept: SuspicionScore,
) -> None:
    masking.act(DIALLING)

    masking.attach(
        score,
        to=DIALLING,
    )

    assert masking.ratings == (
        Rating(
            action_id="a1",
            score=kept,
        ),
    )


@pytest.mark.parametrize(
    ("decision", "history", "ratings", "record"),
    [
        pytest.param(
            Run(
                action=ASKING,
                proposals=(PROPOSED,),
                position=1,
            ),
            (
                MASKED_TASK,
                MASKED_ASKING,
            ),
            (
                Rating(
                    action_id="a2",
                    score=MASKED_CALM,
                ),
            ),
            Run(
                action=MASKED_ASKING,
                proposals=(MASKED_PROPOSED,),
                position=1,
            ),
            id="run",
        ),
        pytest.param(
            Replace(
                action=TRUSTED_STEP,
                proposals=(
                    PROPOSED,
                    Proposal(
                        action=TRUSTED_STEP,
                        score=None,
                        policy=Policy.TRUSTED,
                    ),
                ),
                position=1,
            ),
            (
                MASKED_TASK,
                MASKED_TRUSTED_STEP,
            ),
            (),
            Replace(
                action=MASKED_TRUSTED_STEP,
                proposals=(
                    MASKED_PROPOSED,
                    Proposal(
                        action=MASKED_TRUSTED_STEP,
                        score=None,
                        policy=Policy.TRUSTED,
                    ),
                ),
                position=1,
            ),
            id="replace-with-a-trusted-step",
        ),
        pytest.param(
            Block(
                action=ASKING,
                reason=REASON,
                proposals=(PROPOSED,),
                position=1,
            ),
            (
                MASKED_TASK,
                MASKED_ASKING,
                ToolObservation(
                    call_id="call_2",
                    is_error=True,
                    author=Author.MONITOR,
                    parts=(TextPart(text=MASKED_REASON),),
                ),
            ),
            (),
            Block(
                action=MASKED_ASKING,
                reason=MASKED_REASON,
                proposals=(MASKED_PROPOSED,),
                position=1,
            ),
            id="block",
        ),
        pytest.param(
            Audit(
                action=ASKING,
                proposals=(PROPOSED,),
                position=1,
            ),
            (MASKED_TASK,),
            (),
            Audit(
                action=MASKED_ASKING,
                proposals=(MASKED_PROPOSED,),
                position=1,
            ),
            id="audit",
        ),
        pytest.param(
            Halt(
                reason=REASON,
                proposals=(PROPOSED,),
                position=1,
            ),
            (MASKED_TASK,),
            (),
            Halt(
                reason=MASKED_REASON,
                proposals=(MASKED_PROPOSED,),
                position=1,
            ),
            id="halt",
        ),
    ],
)
def test_a_recorded_decision_is_kept_redacted(
    masking: Trajectory,
    decision: ProtocolDecision,
    history: tuple[Event, ...],
    ratings: tuple[Rating, ...],
    record: ProtocolDecision,
) -> None:
    masking.record(decision)

    assert (
        masking.history,
        masking.ratings,
        masking.records,
    ) == (
        history,
        ratings,
        (record,),
    )


def test_a_decision_recorded_under_a_redaction_that_differs_on_each_call_loads_back() -> None:
    trajectory = Trajectory(redaction=build_pseudonyms())
    trajectory.observe(TASK)
    trajectory.record(
        Run(
            action=ASKING,
            proposals=(PROPOSED,),
            position=1,
        ),
    )
    stored = trajectory.dump()

    loaded = Trajectory.load(stored)

    assert loaded.dump() == stored


def test_a_subagent_keeps_its_entries_redacted_as_its_delegator_does(masking: Trajectory) -> None:
    masking.act(ASKING)

    subagent = masking.delegate(
        "Find who owns 555-0100.",
        call_id="call_2",
    )

    assert subagent.history == (
        Observation(
            author=Author.DELEGATOR,
            parts=(TextPart(text="Find who owns ###-####."),),
        ),
    )


@pytest.mark.parametrize(
    "select",
    [
        lambda step: step.trajectory,
        lambda step: step.trajectory.subagents["call_2"],
    ],
    ids=["the-copy", "the-copys-subagent"],
)
def test_a_split_copy_keeps_its_entries_redacted(
    masking: Trajectory,
    select: Callable[[Step], Trajectory],
) -> None:
    masking.act(ASKING)
    masking.delegate(
        "Find who owns the number.",
        call_id="call_2",
    )
    masking.act(DIALLING)
    copy = select(masking.split_latest_step())

    copy.observe(TASK)

    assert copy.history[-1] == MASKED_TASK


@pytest.mark.parametrize(
    "build",
    [
        lambda: (
            Trajectory.from_run(
                "Call Ana on 555-0100.",
                answer="Done.",
                redaction=MASKED,
            ).trajectory
        ),
        lambda: Trajectory.from_anthropic(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Call Ana on 555-0100.",
                    },
                ],
            },
            redaction=MASKED,
        ),
        lambda: Trajectory.from_openai_chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Call Ana on 555-0100.",
                    },
                ],
            },
            redaction=MASKED,
        ),
        lambda: Trajectory.from_open_responses(
            {"input": "Call Ana on 555-0100."},
            redaction=MASKED,
        ),
    ],
    ids=["run", "anthropic", "openai-chat", "open-responses"],
)
def test_a_trajectory_built_with_a_redaction_keeps_its_entries_redacted(
    build: Callable[[], Trajectory],
) -> None:
    trajectory = build()

    assert trajectory.history == (MASKED_TASK,)


def test_a_black_box_runs_answer_is_returned_as_written() -> None:
    run = Trajectory.from_run(
        "Call Ana on 555-0100.",
        answer="Called 555-0100.",
        redaction=MASKED,
    )

    assert run.answer.parts == (TextPart(text="Called 555-0100."),)


def test_a_dump_with_a_redaction_writes_the_content_redacted() -> None:
    trajectory = build_stored_trajectory()

    stored = trajectory.dump(redaction=MASKED)

    assert stored == MASKED_STORED


def test_a_dump_with_a_redaction_leaves_the_trajectory_unchanged() -> None:
    trajectory = build_stored_trajectory()

    trajectory.dump(redaction=MASKED)

    assert trajectory.dump() == STORED


def test_a_dump_with_a_redaction_that_differs_on_each_call_loads_back() -> None:
    stored = build_stored_trajectory().dump(redaction=build_pseudonyms())

    loaded = Trajectory.load(stored)

    assert loaded.dump() == stored


def test_a_dump_under_changing_pseudonyms_writes_a_rating_and_its_proposal_one_score() -> None:
    stored = build_stored_trajectory().dump(redaction=build_pseudonyms())

    loaded = Trajectory.load(stored)

    assert loaded.ratings[0].score == loaded.records[0].proposals[0].score


def test_loading_with_a_redaction_keeps_what_it_loads_redacted() -> None:
    loaded = Trajectory.load(
        STORED,
        redaction=MASKED,
    )

    assert loaded.dump() == MASKED_STORED


@pytest.mark.parametrize(
    "select",
    [
        lambda loaded: loaded,
        lambda loaded: loaded.subagents["call_2"],
    ],
    ids=["the-trajectory", "its-subagent"],
)
def test_a_trajectory_loaded_with_a_redaction_keeps_later_entries_redacted(
    select: Callable[[Trajectory], Trajectory],
) -> None:
    loaded = select(
        Trajectory.load(
            STORED,
            redaction=MASKED,
        ),
    )

    loaded.observe(TASK)

    assert loaded.history[-1] == MASKED_TASK


def test_loading_under_changing_pseudonyms_keeps_each_record_on_its_entry() -> None:
    loaded = Trajectory.load(
        STORED,
        redaction=build_pseudonyms(),
    )

    assert loaded.records[0].action == loaded.history[1]  # ty: ignore[unresolved-attribute]


@pytest.mark.xfail(strict=True)
def test_a_trajectory_with_a_redaction_keeps_what_joins_it_redacted(masking: Trajectory) -> None:
    masking.act(ASKING)

    joined = masking.join(
        Trajectory.load(STORED),
        call_id="call_2",
    )

    assert joined.dump() == MASKED_STORED


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "select",
    [
        lambda joined: joined,
        lambda joined: joined.subagents["call_2"],
    ],
    ids=["the-joined-trajectory", "its-subagent"],
)
def test_a_trajectory_joined_to_one_with_a_redaction_keeps_later_entries_redacted(
    masking: Trajectory,
    select: Callable[[Trajectory], Trajectory],
) -> None:
    masking.act(ASKING)
    joined = select(
        masking.join(
            Trajectory.load(STORED),
            call_id="call_2",
        ),
    )

    joined.observe(TASK)

    assert joined.history[-1] == MASKED_TASK


def test_joining_under_changing_pseudonyms_keeps_each_record_on_its_entry() -> None:
    pseudonymising = Trajectory(redaction=build_pseudonyms())
    pseudonymising.act(ASKING)

    joined = pseudonymising.join(
        Trajectory.load(STORED),
        call_id="call_2",
    )

    assert joined.records[0].action == joined.history[1]  # ty: ignore[unresolved-attribute]


def test_joining_leaves_the_trajectory_it_takes_unchanged(masking: Trajectory) -> None:
    masking.record(
        Run(
            action=ASKING,
            proposals=(PROPOSED,),
            position=1,
        ),
    )
    taken = Trajectory.load(STORED)

    masking.join(
        taken,
        call_id="call_2",
    )

    assert taken.dump() == STORED


@pytest.mark.parametrize(
    ("part", "kept"),
    [
        pytest.param(
            MediaPart(
                media_type="text/plain",
                data=b"Call Ana on 555-0100.",
            ),
            MediaPart(
                media_type="text/plain",
                data=b"Call Ana on ###-####.",
            ),
            id="utf-8",
        ),
        pytest.param(
            MediaPart(
                media_type="text/plain",
                data=b"555-0100",
                description=Description(
                    text="A note with 555-0100",
                    author=Author.TOOL,
                ),
            ),
            MediaPart(
                media_type="text/plain",
                data=b"###-####",
                description=Description(
                    text="A note with ###-####",
                    author=Author.TOOL,
                ),
            ),
            id="utf-8-and-a-description",
        ),
        pytest.param(
            MediaPart(
                media_type="text/plain",
                data=b"\xff555-0100",
            ),
            MediaPart(
                media_type="text/plain",
                data=b"\xff555-0100",
            ),
            id="not-utf-8",
        ),
        pytest.param(
            MediaPart(
                media_type="text/plain",
                url="https://files.example.com/555-0100.txt",
            ),
            MediaPart(
                media_type="text/plain",
                url="https://files.example.com/555-0100.txt",
            ),
            id="by-url",
        ),
    ],
)
def test_text_media_is_kept_with_its_utf_8_text_redacted(
    part: MediaPart,
    kept: MediaPart,
) -> None:
    trajectory = Trajectory(redaction=TEXT_MASKED)

    trajectory.observe(
        ToolObservation(
            call_id="call_1",
            parts=(part,),
        ),
    )

    assert trajectory.history[-1].parts == (kept,)


def test_a_dump_with_a_redaction_writes_text_media_with_its_text_redacted() -> None:
    trajectory = Trajectory()
    trajectory.observe(
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(
                    media_type="text/plain",
                    data=b"Call Ana on 555-0100.",
                ),
            ),
        ),
    )

    stored = trajectory.dump(redaction=TEXT_MASKED)

    assert stored == {
        **STORED_SUBAGENT_FIELDS,
        "history": [
            {
                "kind": "observation",
                "author": "user",
                "source": None,
                "parts": [
                    {
                        "kind": "media",
                        "media_type": "text/plain",
                        "url": None,
                        # base64 of "Call Ana on ###-####."
                        "data": "Q2FsbCBBbmEgb24gIyMjLSMjIyMu",
                        "description": None,
                    },
                ],
            },
        ],
    }


def test_loading_with_what_is_not_a_redaction_raises_before_reading_the_data() -> None:
    with pytest.raises(
        (
            ConfigurationError,
            FormatError,
        ),
    ) as raised:
        Trajectory.load(
            {},
            redaction=MASKED.text,  # ty: ignore[invalid-argument-type]
        )

    assert (
        type(raised.value),
        str(raised.value),
    ) == (
        ConfigurationError,
        "redaction must be a Redaction, got an instance of functools.partial",
    )
