"""A stored trajectory loads back as it was, and a changed one loads or is refused."""

import json
from collections.abc import Callable
from enum import Enum, auto

import pytest
from damage import MARKED_BYTES, MARKED_STRINGS, draw_edit, reads_cleanly, replace_each_node
from hypothesis import given
from hypothesis import strategies as st

from hyoshigi import (
    JSON,
    Action,
    Author,
    Composite,
    Decision,
    Decisions,
    Description,
    Event,
    FailedClosed,
    Labels,
    MediaPart,
    Observation,
    OpaquePart,
    Rating,
    Rationale,
    Refused,
    StatementDecision,
    SuspicionScore,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
    Truncated,
    Unavailable,
    Unperceivable,
    Unreadable,
)

TEXT = MARKED_STRINGS
UNIT = st.floats(
    min_value=0,
    max_value=1,
)
AUTHORS = st.sampled_from(Author)
MEDIA_TYPES = st.sampled_from(["image/png", "audio/wav", "application/pdf", "image/*"])
DESCRIPTIONS = st.none() | st.builds(
    Description,
    text=TEXT,
    author=AUTHORS,
)
MEDIA = st.one_of(
    st.builds(
        MediaPart,
        media_type=MEDIA_TYPES,
        data=MARKED_BYTES,
        description=DESCRIPTIONS,
    ),
    st.builds(
        MediaPart,
        media_type=MEDIA_TYPES,
        url=TEXT,
        description=DESCRIPTIONS,
    ),
)
PARTS = st.lists(
    st.one_of(
        st.builds(
            TextPart,
            text=TEXT,
        ),
        MEDIA,
        st.builds(
            OpaquePart,
            kind=TEXT,
        ),
    ),
    max_size=3,
).map(tuple)
TEXTS = st.lists(
    TEXT,
    max_size=2,
).map(tuple)
CAUSES = st.one_of(
    st.just(Unreadable()),
    st.builds(
        Unperceivable,
        content=st.lists(
            TEXT,
            min_size=1,
            max_size=2,
        ).map(tuple),
    ),
    st.builds(
        Truncated,
        text=TEXT,
    ),
    st.builds(
        Refused,
        explanation=st.none() | TEXT,
    ),
    st.builds(
        Unavailable,
        status=st.none() | st.integers(),
    ),
)
STATEMENT_DECISIONS = st.builds(
    StatementDecision,
    statement=TEXT,
    decision=st.builds(
        Decision,
        probability=UNIT,
        confidence=st.none() | UNIT,
    ),
)
EVIDENCE = st.one_of(
    st.builds(
        Rationale,
        model=TEXT,
        hidden_reasoning=TEXTS,
    ),
    st.builds(
        Decisions,
        model=TEXT,
        decisions=st.lists(
            STATEMENT_DECISIONS,
            max_size=2,
        ).map(tuple),
        hidden_reasoning=TEXTS,
    ),
    st.builds(
        Labels,
        model=TEXT,
        labels=TEXTS,
        unavailable=st.lists(
            st.builds(
                Unavailable,
                status=st.none() | st.integers(),
            ),
            max_size=2,
        ).map(tuple),
        hidden_reasoning=TEXTS,
    ),
)
UNFAILED_SCORES = st.builds(
    SuspicionScore,
    value=UNIT,
    explanation=TEXT,
    evidence=EVIDENCE,
)
FAILED_SCORES = st.builds(
    SuspicionScore,
    value=st.just(1.0),
    explanation=TEXT,
    evidence=st.builds(
        FailedClosed,
        model=TEXT,
        cause=CAUSES,
    ),
)


def build_composite_scores(parts: list[SuspicionScore]) -> st.SearchStrategy[SuspicionScore]:
    """A score of the parts, at 1.0 when the composite fails closed."""
    evidence = Composite(parts=tuple(parts))
    failed = SuspicionScore(
        value=1.0,
        explanation="",
        evidence=evidence,
    ).failed_closed
    return st.builds(
        SuspicionScore,
        value=st.just(1.0) if failed else UNIT,
        explanation=TEXT,
        evidence=st.just(evidence),
    )


SCORES = st.recursive(
    UNFAILED_SCORES | FAILED_SCORES,
    lambda scores: st.lists(
        scores,
        min_size=1,
        max_size=3,
    ).flatmap(build_composite_scores),
    max_leaves=4,
)
OBSERVATIONS = st.builds(
    Observation,
    author=AUTHORS,
    parts=PARTS,
    source=st.none() | TEXT,
)
TOOL_OBSERVATIONS = st.builds(
    ToolObservation,
    call_id=TEXT,
    is_error=st.booleans(),
    source=st.none() | TEXT,
    parts=PARTS,
)


class Step(Enum):
    """What happens next in a trajectory."""

    OBSERVE = auto()
    REPORT = auto()
    ACT = auto()


type Snapshot = tuple[tuple[Event, ...], tuple[Rating, ...], dict[str, Snapshot]]


def snapshot(trajectory: Trajectory) -> Snapshot:
    """What a trajectory holds, with its subagents' contents."""
    return (
        trajectory.history,
        trajectory.ratings,
        {call_id: snapshot(child) for call_id, child in trajectory.subagents.items()},
    )


def draw_id(
    draw: st.DrawFn,
    used: set[str],
) -> str:
    """An id that no action or tool call of the trajectory has."""
    identifier = draw(TEXT.filter(lambda text: text not in used))
    used.add(identifier)
    return identifier


@st.composite
def actions(
    draw: st.DrawFn,
    used: set[str],
) -> Action:
    """An action whose ids are new."""
    calls = tuple(
        ToolCall(
            id=draw_id(
                draw,
                used,
            ),
            name=draw(TEXT),
            arguments=draw(TEXT),
        )
        for _ in range(draw(st.integers(0, 2)))
    )
    return Action(
        id=draw_id(
            draw,
            used,
        ),
        thought=draw(PARTS),
        parts=draw(PARTS),
        tool_calls=calls,
    )


def act_and_rate(
    draw: st.DrawFn,
    trajectory: Trajectory,
    *,
    used: set[str],
    depth: int,
) -> None:
    """Add an action with ratings and, while depth remains, a subagent."""
    action = draw(actions(used))
    trajectory.act(action)
    for score in draw(
        st.lists(
            SCORES,
            max_size=2,
        ),
    ):
        trajectory.attach(
            score,
            to=action,
        )
    if depth and action.tool_calls:
        child = trajectory.delegate(
            draw(TEXT),
            call_id=action.tool_calls[0].id,
        )
        fill(
            draw,
            child,
            depth=depth - 1,
        )


def fill(
    draw: st.DrawFn,
    trajectory: Trajectory,
    *,
    depth: int,
) -> None:
    """Add up to four events to the trajectory."""
    used: set[str] = set()
    for _ in range(draw(st.integers(0, 4))):
        match draw(st.sampled_from(Step)):
            case Step.OBSERVE:
                trajectory.observe(draw(OBSERVATIONS))
            case Step.REPORT:
                trajectory.observe(draw(TOOL_OBSERVATIONS))
            case Step.ACT:
                act_and_rate(
                    draw,
                    trajectory,
                    used=used,
                    depth=depth,
                )


@st.composite
def trajectories(draw: st.DrawFn) -> Trajectory:
    """A trajectory with ratings and subagents two levels deep."""
    trajectory = Trajectory()
    fill(
        draw,
        trajectory,
        depth=2,
    )
    return trajectory


def keep(data: JSON) -> JSON:
    return data


def pass_through_json_text(data: JSON) -> JSON:
    return json.loads(json.dumps(data))


@pytest.mark.parametrize(
    "transport",
    [
        keep,
        pass_through_json_text,
    ],
    ids=["as dumped", "through JSON text"],
)
@given(trajectory=trajectories())
def test_a_trajectory_loads_back_as_it_was_dumped(
    transport: Callable[[JSON], JSON],
    trajectory: Trajectory,
) -> None:
    loaded = Trajectory.load(transport(trajectory.dump()))

    assert snapshot(loaded) == snapshot(trajectory)


@given(data=st.data())
def test_a_dump_with_one_node_replaced_is_loaded_or_raises_a_format_error_that_quotes_none_of_it(
    data: st.DataObject,
) -> None:
    dump = data.draw(trajectories()).dump()
    edit = data.draw(draw_edit(dump))

    unclean = [
        damaged
        for damaged in replace_each_node(
            dump,
            edit,
        )
        if not reads_cleanly(
            Trajectory.load,
            damaged,
        )
    ]

    assert unclean == []
