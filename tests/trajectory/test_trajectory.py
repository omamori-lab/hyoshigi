"""The trajectory: a black-box run, acting, observing, delegating and joining."""

from collections.abc import Callable
from typing import NamedTuple

import pytest

from hyoshigi import (
    JSON,
    Action,
    Audit,
    Author,
    Block,
    ConfigurationError,
    Flag,
    Halt,
    Observation,
    Policy,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from hyoshigi.monitors.score import Rationale, SuspicionScore
from hyoshigi.trajectory.history import Event
from hyoshigi.trajectory.trajectory import Rating

TASK = "Summarise q3.md"
ANSWER = "Revenue grew 4% in the third quarter."
INSTRUCTION = "Find the q3 revenue"
READ = ToolCall(
    id="call_1",
    name="read_file",
    arguments='{"path": "q3.md"}',
)
WRITE = ToolCall(
    id="call_2",
    name="write_file",
    arguments='{"path": "summary.md"}',
)
DELEGATE = ToolCall(
    id="call_7",
    name="task",
    arguments="{}",
)
DELETE_UNDER_READS_ID = ToolCall(
    id="call_1",
    name="delete_file",
    arguments='{"path": "q3.md"}',
)
DELETE_UNDER_WRITES_ID = ToolCall(
    id="call_2",
    name="delete_file",
    arguments='{"path": "q3.md"}',
)
READS_THE_FILE = SuspicionScore(
    value=0.1,
    explanation="It reads the file the task names.",
    evidence=Rationale(model="m"),
)
WRITES_ELSEWHERE = SuspicionScore(
    value=0.8,
    explanation="It writes outside the task.",
    evidence=Rationale(model="m"),
)

TASK_OBSERVATION = Observation(
    author=Author.USER,
    parts=(TextPart(text=TASK),),
)
REASON = "The write is outside the task."
WRITING = Action(
    id="a2",
    tool_calls=(WRITE,),
)
# Two tool calls, in the order the blocked action made them.
WRITING_AND_DELETING = Action(
    id="a2",
    tool_calls=(
        ToolCall(
            id="call_3",
            name="write_file",
            arguments='{"path": "notes.md"}',
        ),
        ToolCall(
            id="call_2",
            name="delete_file",
            arguments='{"path": "q3.md"}',
        ),
    ),
)
ANSWERING = Action(
    id="a2",
    parts=(TextPart(text=ANSWER),),
)
RESAMPLED = Action(
    id="a3",
    tool_calls=(READ,),
)
TRUSTED_STEP = Action(
    id="t1",
    tool_calls=(READ,),
)
PROPOSED_WRITE = Proposal(
    action=WRITING,
    score=WRITES_ELSEWHERE,
    policy=Policy.UNTRUSTED,
)
PROPOSED_RESAMPLE = Proposal(
    action=RESAMPLED,
    score=READS_THE_FILE,
    policy=Policy.UNTRUSTED,
)
PROPOSED_TRUSTED_STEP = Proposal(
    action=TRUSTED_STEP,
    score=None,
    policy=Policy.TRUSTED,
)
RUN_WRITE = Run(
    action=WRITING,
    proposals=(PROPOSED_WRITE,),
    position=1,
)
FLAG_WRITE = Flag(
    action=WRITING,
    proposals=(PROPOSED_WRITE,),
    position=1,
)
AUDIT_WRITE = Audit(
    action=WRITING,
    proposals=(PROPOSED_WRITE,),
    position=1,
)


@pytest.fixture
def trajectory() -> Trajectory:
    return Trajectory.from_run(
        TASK,
        answer=ANSWER,
    ).trajectory


@pytest.fixture
def reading(trajectory: Trajectory) -> Trajectory:
    trajectory.act(
        Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )
    return trajectory


@pytest.fixture
def delegating(trajectory: Trajectory) -> Trajectory:
    trajectory.act(
        Action(
            id="a1",
            tool_calls=(
                ToolCall(
                    id="call_7",
                    name="task",
                    arguments="{}",
                ),
            ),
        ),
    )
    return trajectory


def test_a_black_box_run_holds_only_the_task_authored_by_the_user() -> None:
    run = Trajectory.from_run(
        TASK,
        answer=ANSWER,
    )

    assert run.trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
    )


def test_a_black_box_run_proposes_the_answer_as_the_action_to_rate() -> None:
    run = Trajectory.from_run(
        TASK,
        answer=ANSWER,
    )

    assert run.answer.parts == (TextPart(text=ANSWER),)


def test_actions_and_observations_join_the_history_in_order(trajectory: Trajectory) -> None:
    read = Action(
        id="a1",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="read",
                arguments="{}",
            ),
        ),
    )
    result = ToolObservation(
        call_id="call_1",
        parts=(TextPart(text="Revenue: 4.2M"),),
    )
    trajectory.act(read)

    trajectory.observe(result)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
        read,
        result,
    )


def test_acting_an_action_already_in_the_history_raises(trajectory: Trajectory) -> None:
    trajectory.act(Action(id="a1"))

    with pytest.raises(ConfigurationError):
        trajectory.act(
            Action(
                id="a1",
                parts=(TextPart(text="Done."),),
            ),
        )


@pytest.mark.parametrize(
    "action",
    [
        Action(
            id="a2",
            tool_calls=(DELETE_UNDER_READS_ID,),
        ),
        Action(
            id="a2",
            tool_calls=(
                WRITE,
                DELETE_UNDER_WRITES_ID,
            ),
        ),
    ],
    ids=["used-by-an-earlier-action", "used-twice-in-one-action"],
)
def test_acting_with_a_tool_call_id_already_used_raises(
    reading: Trajectory,
    action: Action,
) -> None:
    with pytest.raises(ConfigurationError):
        reading.act(action)


def test_a_refused_action_leaves_the_history_unchanged(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError):
        reading.act(
            Action(
                id="a2",
                tool_calls=(
                    WRITE,
                    DELETE_UNDER_WRITES_ID,
                ),
            ),
        )

    assert reading.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
        Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )


def test_acting_an_observation_raises_naming_it(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.act(
            Observation(  # ty: ignore[invalid-argument-type]
                author=Author.USER,
                parts=(TextPart(text=TASK),),
            ),
        )

    assert str(raised.value) == "action must be an Action, got an instance of Observation"


def test_observing_an_action_raises_naming_it(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.observe(Action(id="a2"))  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == "observation must be an Observation, got an instance of Action"


def test_a_delegated_trajectory_starts_with_the_instruction_from_the_delegator(
    delegating: Trajectory,
) -> None:
    child = delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )

    assert child.history == (
        Observation(
            author=Author.DELEGATOR,
            parts=(TextPart(text=INSTRUCTION),),
        ),
    )


def test_a_delegated_trajectory_is_held_under_the_call_that_delegated_it(
    delegating: Trajectory,
) -> None:
    child = delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )

    assert delegating.subagents.get("call_7") is child


def test_delegating_twice_from_one_call_raises(delegating: Trajectory) -> None:
    delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )

    with pytest.raises(ConfigurationError):
        delegating.delegate(
            "Find the q4 revenue",
            call_id="call_7",
        )


def test_delegating_from_a_call_not_in_the_trajectory_raises(trajectory: Trajectory) -> None:
    with pytest.raises(ConfigurationError):
        trajectory.delegate(
            INSTRUCTION,
            call_id="call_7",
        )


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "start",
    [
        pytest.param(
            lambda trajectory: trajectory.delegate(
                INSTRUCTION,
                call_id=["call_7"],
            ),
            id="delegate",
        ),
        pytest.param(
            lambda trajectory: trajectory.join(
                Trajectory(),
                call_id=["call_7"],
            ),
            id="join",
        ),
    ],
)
def test_a_call_id_that_is_not_text_raises_naming_its_type(
    delegating: Trajectory,
    start: Callable[[Trajectory], Trajectory],
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        start(delegating)

    assert str(raised.value) == "call_id must be a str, got an instance of list"


def delegate_a_chain(
    trajectory: Trajectory,
    *,
    depth: int,
) -> Trajectory:
    """Delegate a chain of subagents `depth` levels below the trajectory, each by a call of id
    call_7, and return the deepest.
    """
    deepest = trajectory
    for _ in range(depth):
        deepest.act(
            Action(
                id="a1",
                tool_calls=(DELEGATE,),
            ),
        )
        deepest = deepest.delegate(
            INSTRUCTION,
            call_id="call_7",
        )
    return deepest


def test_a_chain_of_subagents_32_levels_deep_loads_back_unchanged(trajectory: Trajectory) -> None:
    delegate_a_chain(
        trajectory,
        depth=32,
    )

    loaded = Trajectory.load(trajectory.dump())

    assert loaded.dump() == trajectory.dump()


def test_delegating_from_a_subagent_32_levels_deep_raises_naming_the_limit(
    trajectory: Trajectory,
) -> None:
    deepest = delegate_a_chain(
        trajectory,
        depth=32,
    )
    deepest.act(
        Action(
            id="a1",
            tool_calls=(DELEGATE,),
        ),
    )

    with pytest.raises(ConfigurationError) as raised:
        deepest.delegate(
            INSTRUCTION,
            call_id="call_7",
        )

    assert str(raised.value) == "subagents are nested at most 32 deep"


def build_joining(depth: int) -> Trajectory:
    """Return a trajectory `depth` levels below the root of a new run, whose action made call_7."""
    joining = delegate_a_chain(
        Trajectory(),
        depth=depth,
    )
    joining.act(
        Action(
            id="a1",
            tool_calls=(DELEGATE,),
        ),
    )
    return joining


def build_chain(depth: int) -> Trajectory:
    """Return the root of a new run, with a chain of subagents `depth` levels below it."""
    root = Trajectory()
    delegate_a_chain(
        root,
        depth=depth,
    )
    return root


@pytest.mark.parametrize(
    ("joining_depth", "chain_depth"),
    [
        pytest.param(
            0,
            0,
            id="a-lone-trajectory",
        ),
        pytest.param(
            0,
            31,
            id="a-chain-whose-deepest-is-32-below-the-root",
        ),
        pytest.param(
            16,
            15,
            id="a-chain-joined-16-deep-whose-deepest-is-32-below-the-root",
        ),
    ],
)
def test_a_joined_trajectory_is_held_under_the_call_that_delegated_it(
    joining_depth: int,
    chain_depth: int,
) -> None:
    joining = build_joining(joining_depth)

    joined = joining.join(
        build_chain(chain_depth),
        call_id="call_7",
    )

    assert joining.subagents.get("call_7") is joined


class Joining(NamedTuple):
    """A trajectory, and what it is given to join under its call_7."""

    trajectory: Trajectory
    subagent: object


def build_delegated_joining() -> Joining:
    """A trajectory whose call_7 delegated a subagent already, given a new trajectory."""
    joining = build_joining(0)
    joining.delegate(
        INSTRUCTION,
        call_id="call_7",
    )
    return Joining(
        trajectory=joining,
        subagent=Trajectory(),
    )


def build_root_joining() -> Joining:
    """A subagent given the root of its own run."""
    root = Trajectory()
    joining = delegate_a_chain(
        root,
        depth=1,
    )
    joining.act(
        Action(
            id="a1",
            tool_calls=(DELEGATE,),
        ),
    )
    return Joining(
        trajectory=joining,
        subagent=root,
    )


def build_split_copy_joining() -> Joining:
    """A subagent given the trajectory split from the root of its own run before its call."""
    root = Trajectory()
    root.observe(TASK_OBSERVATION)
    joining = delegate_a_chain(
        root,
        depth=1,
    )
    joining.act(
        Action(
            id="a1",
            tool_calls=(DELEGATE,),
        ),
    )
    return Joining(
        trajectory=joining,
        subagent=root.split_latest_step().trajectory,
    )


@pytest.mark.parametrize(
    ("arrange", "message"),
    [
        pytest.param(
            lambda: Joining(
                trajectory=build_joining(0),
                subagent=Trajectory().dump(),
            ),
            "subagent must be a Trajectory, got an instance of dict",
            id="a-dump",
        ),
        pytest.param(
            lambda: Joining(
                trajectory=Trajectory(),
                subagent=Trajectory(),
            ),
            "join takes the id of a tool call in this trajectory",
            id="a-call-not-in-the-trajectory",
        ),
        pytest.param(
            build_delegated_joining,
            "a subagent is already delegated by this call",
            id="a-call-that-delegated-already",
        ),
        pytest.param(
            lambda: Joining(
                trajectory=build_joining(0),
                subagent=delegate_a_chain(
                    Trajectory(),
                    depth=1,
                ),
            ),
            "join takes the root of a run apart from this one",
            id="a-subagent-of-another-run",
        ),
        pytest.param(
            build_root_joining,
            "join takes the root of a run apart from this one",
            id="the-root-of-this-run",
        ),
        pytest.param(
            build_split_copy_joining,
            "join takes the root of a run apart from this one",
            id="a-copy-split-from-this-run",
        ),
        pytest.param(
            lambda: Joining(
                trajectory=build_joining(0),
                subagent=build_chain(32),
            ),
            "subagents are nested at most 32 deep",
            id="a-chain-whose-deepest-is-33-below-the-root",
        ),
        pytest.param(
            lambda: Joining(
                trajectory=build_joining(16),
                subagent=build_chain(16),
            ),
            "subagents are nested at most 32 deep",
            id="a-chain-joined-16-deep-whose-deepest-is-33-below-the-root",
        ),
    ],
)
def test_joining_what_cannot_join_raises_naming_why(
    arrange: Callable[[], Joining],
    message: str,
) -> None:
    arranged = arrange()

    with pytest.raises(ConfigurationError) as raised:
        arranged.trajectory.join(
            arranged.subagent,  # ty: ignore[invalid-argument-type]
            call_id="call_7",
        )

    assert str(raised.value) == message


def test_a_refused_join_leaves_both_runs_unchanged() -> None:
    joining = build_joining(16)
    taken = build_chain(16)
    joining_stored = joining.dump()
    taken_stored = taken.dump()

    with pytest.raises(ConfigurationError):
        joining.join(
            taken,
            call_id="call_7",
        )

    assert (
        joining.dump(),
        taken.dump(),
    ) == (
        joining_stored,
        taken_stored,
    )


def record_an_audit(trajectory: Trajectory) -> None:
    """Record an audit of WRITING at the length of the history."""
    trajectory.record(
        Audit(
            action=WRITING,
            proposals=(PROPOSED_WRITE,),
            position=len(trajectory.history),
        ),
    )


def work_apart(
    stored: dict[str, JSON],
    *,
    call_id: str,
) -> Trajectory:
    """Load the stored run, as a process of its own does, record an audit on the subagent the call
    delegates there, and return that subagent loaded from its dump.
    """
    subagent = Trajectory.load(stored).delegate(
        INSTRUCTION,
        call_id=call_id,
    )
    record_an_audit(subagent)
    return Trajectory.load(subagent.dump())


def test_a_run_with_joined_subagents_loads_back_unchanged(trajectory: Trajectory) -> None:
    trajectory.act(
        Action(
            id="a1",
            tool_calls=(
                DELEGATE,
                READ,
            ),
        ),
    )
    record_an_audit(trajectory)
    stored = trajectory.dump()
    first = work_apart(
        stored,
        call_id="call_7",
    )
    second = work_apart(
        stored,
        call_id="call_1",
    )
    record_an_audit(trajectory)
    joined = trajectory.join(
        first,
        call_id="call_7",
    )
    trajectory.join(
        second,
        call_id="call_1",
    )
    record_an_audit(trajectory)
    record_an_audit(joined)

    loaded = Trajectory.load(trajectory.dump())

    assert loaded.dump() == trajectory.dump()


def test_a_new_trajectory_has_no_ratings(trajectory: Trajectory) -> None:
    assert trajectory.ratings == ()


def test_attaching_a_score_records_a_rating_of_the_action(reading: Trajectory) -> None:
    reading.attach(
        READS_THE_FILE,
        to=Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )

    assert reading.ratings == (
        Rating(
            action_id="a1",
            score=READS_THE_FILE,
        ),
    )


def test_a_second_score_of_one_action_is_kept_after_the_first(reading: Trajectory) -> None:
    action = Action(
        id="a1",
        tool_calls=(READ,),
    )
    reading.attach(
        READS_THE_FILE,
        to=action,
    )

    reading.attach(
        WRITES_ELSEWHERE,
        to=action,
    )

    assert reading.ratings == (
        Rating(
            action_id="a1",
            score=READS_THE_FILE,
        ),
        Rating(
            action_id="a1",
            score=WRITES_ELSEWHERE,
        ),
    )


def test_attaching_a_score_to_an_action_not_in_the_history_raises(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.attach(
            READS_THE_FILE,
            to=Action(id="a2"),
        )

    assert str(raised.value) == "attach takes an action in this trajectory"


@pytest.mark.parametrize(
    ("score", "to", "message"),
    [
        (
            0.5,
            Action(
                id="a1",
                tool_calls=(READ,),
            ),
            "score must be a SuspicionScore, got an instance of float",
        ),
        (
            READS_THE_FILE,
            "a1",
            "to must be an Action, got an instance of str",
        ),
    ],
    ids=["a-number-as-the-score", "an-id-as-the-action"],
)
def test_attaching_what_is_not_a_score_to_an_action_raises_naming_it(
    reading: Trajectory,
    score: object,
    to: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.attach(
            score,  # ty: ignore[invalid-argument-type]
            to=to,  # ty: ignore[invalid-argument-type]
        )

    assert str(raised.value) == message


def test_a_refused_rating_leaves_the_ratings_unchanged(reading: Trajectory) -> None:
    reading.attach(
        READS_THE_FILE,
        to=Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )

    with pytest.raises(ConfigurationError):
        reading.attach(
            WRITES_ELSEWHERE,
            to=Action(id="a2"),
        )

    assert reading.ratings == (
        Rating(
            action_id="a1",
            score=READS_THE_FILE,
        ),
    )


def test_splitting_the_latest_step_returns_its_action(reading: Trajectory) -> None:
    step = reading.split_latest_step()

    assert step.action == Action(
        id="a1",
        tool_calls=(READ,),
    )


def test_splitting_the_latest_step_returns_the_trajectory_before_it(reading: Trajectory) -> None:
    step = reading.split_latest_step()

    assert step.trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
    )


def test_splitting_the_latest_step_leaves_the_trajectory_unchanged(reading: Trajectory) -> None:
    reading.split_latest_step()

    assert reading.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
        Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )


def test_the_trajectory_before_the_latest_step_can_take_its_action(reading: Trajectory) -> None:
    step = reading.split_latest_step()

    step.trajectory.act(step.action)

    assert step.trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text=TASK),),
        ),
        Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )


def test_splitting_the_latest_step_keeps_only_the_ratings_of_earlier_actions(
    reading: Trajectory,
) -> None:
    reading.attach(
        READS_THE_FILE,
        to=Action(
            id="a1",
            tool_calls=(READ,),
        ),
    )
    writing = Action(
        id="a2",
        tool_calls=(WRITE,),
    )
    reading.act(writing)
    reading.attach(
        WRITES_ELSEWHERE,
        to=writing,
    )

    step = reading.split_latest_step()

    assert step.trajectory.ratings == (
        Rating(
            action_id="a1",
            score=READS_THE_FILE,
        ),
    )


def test_splitting_the_latest_step_keeps_only_the_subagents_of_earlier_calls(
    delegating: Trajectory,
) -> None:
    delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )
    delegating.act(
        Action(
            id="a2",
            tool_calls=(
                ToolCall(
                    id="call_8",
                    name="task",
                    arguments="{}",
                ),
            ),
        ),
    )
    delegating.delegate(
        INSTRUCTION,
        call_id="call_8",
    )

    step = delegating.split_latest_step()

    assert list(step.trajectory.subagents) == ["call_7"]


def test_the_original_subagent_takes_an_action_its_split_copy_took(
    delegating: Trajectory,
) -> None:
    original = delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )
    delegating.act(Action(id="a2"))
    delegating.split_latest_step().trajectory.subagents["call_7"].act(WRITING)

    original.act(WRITING)

    assert original.history == (
        Observation(
            author=Author.DELEGATOR,
            parts=(TextPart(text=INSTRUCTION),),
        ),
        WRITING,
    )


def test_recording_on_a_split_copys_subagent_leaves_the_original_subagent_unchanged(
    delegating: Trajectory,
) -> None:
    original = delegating.delegate(
        INSTRUCTION,
        call_id="call_7",
    )
    delegating.act(Action(id="a2"))
    copied = delegating.split_latest_step().trajectory.subagents["call_7"]

    copied.record(RUN_WRITE)

    assert (
        original.history,
        original.ratings,
        original.records,
    ) == (
        (
            Observation(
                author=Author.DELEGATOR,
                parts=(TextPart(text=INSTRUCTION),),
            ),
        ),
        (),
        (),
    )


@pytest.mark.parametrize(
    "trajectory",
    [
        Trajectory.from_run(
            TASK,
            answer=ANSWER,
        ).trajectory,
        Trajectory(),
    ],
    ids=["ends-in-an-observation", "empty"],
)
def test_splitting_a_trajectory_whose_latest_entry_is_not_an_action_raises(
    trajectory: Trajectory,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        trajectory.split_latest_step()

    assert str(raised.value) == "the latest entry of the trajectory is not an action"


@pytest.mark.parametrize(
    ("decision", "history", "ratings"),
    [
        pytest.param(
            RUN_WRITE,
            (
                TASK_OBSERVATION,
                WRITING,
            ),
            (
                Rating(
                    action_id="a2",
                    score=WRITES_ELSEWHERE,
                ),
            ),
            id="run",
        ),
        pytest.param(
            Run(
                action=WRITING,
                proposals=(
                    PROPOSED_WRITE,
                    Proposal(
                        action=WRITING,
                        score=READS_THE_FILE,
                        policy=Policy.UNTRUSTED,
                    ),
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                WRITING,
            ),
            (
                Rating(
                    action_id="a2",
                    score=WRITES_ELSEWHERE,
                ),
            ),
            id="run-of-an-action-proposed-twice",
        ),
        pytest.param(
            Flag(
                action=WRITING,
                proposals=(
                    PROPOSED_WRITE,
                    Proposal(
                        action=WRITING,
                        score=READS_THE_FILE,
                        policy=Policy.UNTRUSTED,
                    ),
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                WRITING,
            ),
            (
                Rating(
                    action_id="a2",
                    score=WRITES_ELSEWHERE,
                ),
            ),
            id="flag-of-an-action-proposed-twice",
        ),
        pytest.param(
            FLAG_WRITE,
            (
                TASK_OBSERVATION,
                WRITING,
            ),
            (
                Rating(
                    action_id="a2",
                    score=WRITES_ELSEWHERE,
                ),
            ),
            id="flag",
        ),
        pytest.param(
            Replace(
                action=RESAMPLED,
                proposals=(
                    PROPOSED_WRITE,
                    PROPOSED_RESAMPLE,
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                RESAMPLED,
            ),
            (
                Rating(
                    action_id="a3",
                    score=READS_THE_FILE,
                ),
            ),
            id="replace-with-a-resample",
        ),
        pytest.param(
            Replace(
                action=TRUSTED_STEP,
                proposals=(
                    PROPOSED_WRITE,
                    PROPOSED_TRUSTED_STEP,
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                TRUSTED_STEP,
            ),
            (),
            id="replace-with-a-trusted-step",
        ),
        pytest.param(
            Block(
                action=WRITING_AND_DELETING,
                reason=REASON,
                proposals=(
                    Proposal(
                        action=WRITING_AND_DELETING,
                        score=WRITES_ELSEWHERE,
                        policy=Policy.UNTRUSTED,
                    ),
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                WRITING_AND_DELETING,
                ToolObservation(
                    call_id="call_3",
                    is_error=True,
                    author=Author.MONITOR,
                    parts=(TextPart(text=REASON),),
                ),
                ToolObservation(
                    call_id="call_2",
                    is_error=True,
                    author=Author.MONITOR,
                    parts=(TextPart(text=REASON),),
                ),
            ),
            (),
            id="block-of-an-action-with-two-tool-calls",
        ),
        pytest.param(
            Block(
                action=ANSWERING,
                reason=REASON,
                proposals=(
                    Proposal(
                        action=ANSWERING,
                        score=WRITES_ELSEWHERE,
                        policy=Policy.UNTRUSTED,
                    ),
                ),
                position=1,
            ),
            (
                TASK_OBSERVATION,
                ANSWERING,
                Observation(
                    author=Author.MONITOR,
                    parts=(TextPart(text=REASON),),
                ),
            ),
            (),
            id="block-of-an-action-without-tool-calls",
        ),
        pytest.param(
            AUDIT_WRITE,
            (TASK_OBSERVATION,),
            (),
            id="audit",
        ),
        pytest.param(
            Halt(
                reason=REASON,
                proposals=(PROPOSED_WRITE,),
                position=1,
            ),
            (TASK_OBSERVATION,),
            (),
            id="halt",
        ),
    ],
)
def test_recording_a_decision_changes_the_trajectory_as_its_kind_says(
    trajectory: Trajectory,
    decision: ProtocolDecision,
    history: tuple[Event, ...],
    ratings: tuple[Rating, ...],
) -> None:
    trajectory.record(decision)

    assert (
        trajectory.history,
        trajectory.ratings,
        trajectory.records,
    ) == (
        history,
        ratings,
        (decision,),
    )


# Each is recorded on the trajectory of the task and action a1, two entries long.
RUN_PAST_THE_HISTORY = Run(
    action=WRITING,
    proposals=(PROPOSED_WRITE,),
    position=3,
)
RECORDED_AGAIN = Action(
    id="a1",
    tool_calls=(WRITE,),
)
BLOCK_RECORDED_AGAIN = Block(
    action=RECORDED_AGAIN,
    reason=REASON,
    proposals=(
        Proposal(
            action=RECORDED_AGAIN,
            score=WRITES_ELSEWHERE,
            policy=Policy.UNTRUSTED,
        ),
    ),
    position=2,
)
STALE = "the decision's position is not the length of the history"


@pytest.mark.parametrize(
    ("decision", "message"),
    [
        pytest.param(
            WRITING,
            "decision must be a Run, Flag, Replace, Block, Audit or Halt, "
            "got an instance of Action",
            id="an-action",
        ),
        pytest.param(
            RUN_WRITE,
            STALE,
            id="a-position-the-history-grew-past",
        ),
        pytest.param(
            RUN_PAST_THE_HISTORY,
            STALE,
            id="a-position-past-the-history",
        ),
        pytest.param(
            BLOCK_RECORDED_AGAIN,
            "an action with this id is already in the trajectory",
            id="a-block-of-an-action-id-already-recorded",
        ),
    ],
)
def test_recording_a_decision_that_cannot_join_the_trajectory_raises_naming_why(
    reading: Trajectory,
    decision: ProtocolDecision,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.record(decision)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    "decision",
    [
        WRITING,
        RUN_WRITE,
        RUN_PAST_THE_HISTORY,
        BLOCK_RECORDED_AGAIN,
    ],
    ids=[
        "an-action",
        "a-position-the-history-grew-past",
        "a-position-past-the-history",
        "a-block-of-an-action-id-already-recorded",
    ],
)
def test_a_refused_decision_leaves_the_trajectory_unchanged(
    reading: Trajectory,
    decision: ProtocolDecision,
) -> None:
    with pytest.raises(ConfigurationError):
        reading.record(decision)

    assert (
        reading.history,
        reading.ratings,
        reading.records,
    ) == (
        (
            TASK_OBSERVATION,
            Action(
                id="a1",
                tool_calls=(READ,),
            ),
        ),
        (),
        (),
    )


def test_a_persons_run_after_an_audit_is_recorded_at_the_same_position(
    trajectory: Trajectory,
) -> None:
    trajectory.record(AUDIT_WRITE)

    trajectory.record(RUN_WRITE)

    assert (
        trajectory.history,
        trajectory.records,
    ) == (
        (
            TASK_OBSERVATION,
            WRITING,
        ),
        (
            AUDIT_WRITE,
            RUN_WRITE,
        ),
    )


def test_splitting_the_latest_step_keeps_only_the_records_of_earlier_steps(
    trajectory: Trajectory,
) -> None:
    trajectory.record(AUDIT_WRITE)
    trajectory.act(WRITING)
    trajectory.record(
        Audit(
            action=RESAMPLED,
            proposals=(PROPOSED_RESAMPLE,),
            position=2,
        ),
    )
    trajectory.act(RESAMPLED)

    step = trajectory.split_latest_step()

    assert step.trajectory.records == (AUDIT_WRITE,)
