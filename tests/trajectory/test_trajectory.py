"""The trajectory: a black-box run, acting, observing and delegating."""

import pytest

from hyoshigi import (
    Action,
    Author,
    ConfigurationError,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from hyoshigi.monitors.score import Rationale, SuspicionScore
from hyoshigi.trajectory.trajectory import Rating

TASK = "Summarise q3.md"
ANSWER = "Revenue grew 4% in the third quarter."
INSTRUCTION = "Find the q3 revenue"
READ = ToolCall(id="call_1", name="read_file", arguments='{"path": "q3.md"}')
WRITE = ToolCall(id="call_2", name="write_file", arguments='{"path": "summary.md"}')
DELETE_UNDER_READS_ID = ToolCall(id="call_1", name="delete_file", arguments='{"path": "q3.md"}')
DELETE_UNDER_WRITES_ID = ToolCall(id="call_2", name="delete_file", arguments='{"path": "q3.md"}')
READS_THE_FILE = SuspicionScore(
    value=0.1, explanation="It reads the file the task names.", evidence=Rationale(model="m")
)
WRITES_ELSEWHERE = SuspicionScore(
    value=0.8, explanation="It writes outside the task.", evidence=Rationale(model="m")
)


@pytest.fixture
def trajectory() -> Trajectory:
    return Trajectory.from_run(TASK, answer=ANSWER).trajectory


@pytest.fixture
def reading(trajectory: Trajectory) -> Trajectory:
    trajectory.act(Action(id="a1", tool_calls=(READ,)))
    return trajectory


@pytest.fixture
def delegating(trajectory: Trajectory) -> Trajectory:
    trajectory.act(
        Action(id="a1", tool_calls=(ToolCall(id="call_7", name="task", arguments="{}"),))
    )
    return trajectory


def test_a_black_box_run_holds_only_the_task_authored_by_the_user() -> None:
    run = Trajectory.from_run(TASK, answer=ANSWER)

    assert run.trajectory.history == (
        Observation(author=Author.USER, parts=(TextPart(text=TASK),)),
    )


def test_a_black_box_run_proposes_the_answer_as_the_action_to_rate() -> None:
    run = Trajectory.from_run(TASK, answer=ANSWER)

    assert run.answer.parts == (TextPart(text=ANSWER),)


def test_actions_and_observations_join_the_history_in_order(trajectory: Trajectory) -> None:
    read = Action(id="a1", tool_calls=(ToolCall(id="call_1", name="read", arguments="{}"),))
    result = ToolObservation(call_id="call_1", parts=(TextPart(text="Revenue: 4.2M"),))
    trajectory.act(read)

    trajectory.observe(result)

    assert trajectory.history == (
        Observation(author=Author.USER, parts=(TextPart(text=TASK),)),
        read,
        result,
    )


def test_acting_an_action_already_in_the_history_raises(trajectory: Trajectory) -> None:
    trajectory.act(Action(id="a1"))

    with pytest.raises(ConfigurationError):
        trajectory.act(Action(id="a1", parts=(TextPart(text="Done."),)))


@pytest.mark.parametrize(
    "action",
    [
        Action(id="a2", tool_calls=(DELETE_UNDER_READS_ID,)),
        Action(id="a2", tool_calls=(WRITE, DELETE_UNDER_WRITES_ID)),
    ],
    ids=["used-by-an-earlier-action", "used-twice-in-one-action"],
)
def test_acting_with_a_tool_call_id_already_used_raises(
    reading: Trajectory, action: Action
) -> None:
    with pytest.raises(ConfigurationError):
        reading.act(action)


def test_a_refused_action_leaves_the_history_unchanged(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError):
        reading.act(Action(id="a2", tool_calls=(WRITE, DELETE_UNDER_WRITES_ID)))

    assert reading.history == (
        Observation(author=Author.USER, parts=(TextPart(text=TASK),)),
        Action(id="a1", tool_calls=(READ,)),
    )


def test_a_delegated_trajectory_starts_with_the_instruction_from_the_delegator(
    delegating: Trajectory,
) -> None:
    child = delegating.delegate(INSTRUCTION, call_id="call_7")

    assert child.history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text=INSTRUCTION),)),
    )


def test_a_delegated_trajectory_is_held_under_the_call_that_delegated_it(
    delegating: Trajectory,
) -> None:
    child = delegating.delegate(INSTRUCTION, call_id="call_7")

    assert delegating.subagents.get("call_7") is child


def test_delegating_twice_from_one_call_raises(delegating: Trajectory) -> None:
    delegating.delegate(INSTRUCTION, call_id="call_7")

    with pytest.raises(ConfigurationError):
        delegating.delegate("Find the q4 revenue", call_id="call_7")


def test_delegating_from_a_call_not_in_the_trajectory_raises(trajectory: Trajectory) -> None:
    with pytest.raises(ConfigurationError):
        trajectory.delegate(INSTRUCTION, call_id="call_7")


def test_a_new_trajectory_has_no_ratings(trajectory: Trajectory) -> None:
    assert trajectory.ratings == ()


def test_attaching_a_score_records_a_rating_of_the_action(reading: Trajectory) -> None:
    reading.attach(READS_THE_FILE, to=Action(id="a1", tool_calls=(READ,)))

    assert reading.ratings == (Rating(action_id="a1", score=READS_THE_FILE),)


def test_a_second_score_of_one_action_is_kept_after_the_first(reading: Trajectory) -> None:
    action = Action(id="a1", tool_calls=(READ,))
    reading.attach(READS_THE_FILE, to=action)

    reading.attach(WRITES_ELSEWHERE, to=action)

    assert reading.ratings == (
        Rating(action_id="a1", score=READS_THE_FILE),
        Rating(action_id="a1", score=WRITES_ELSEWHERE),
    )


def test_attaching_a_score_to_an_action_not_in_the_history_raises(reading: Trajectory) -> None:
    with pytest.raises(ConfigurationError) as raised:
        reading.attach(READS_THE_FILE, to=Action(id="a2"))

    assert str(raised.value) == "attach takes an action in this trajectory"


def test_a_refused_rating_leaves_the_ratings_unchanged(reading: Trajectory) -> None:
    reading.attach(READS_THE_FILE, to=Action(id="a1", tool_calls=(READ,)))

    with pytest.raises(ConfigurationError):
        reading.attach(WRITES_ELSEWHERE, to=Action(id="a2"))

    assert reading.ratings == (Rating(action_id="a1", score=READS_THE_FILE),)
