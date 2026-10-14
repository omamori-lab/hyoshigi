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

TASK = "Summarise q3.md"
ANSWER = "Revenue grew 4% in the third quarter."
INSTRUCTION = "Find the q3 revenue"


@pytest.fixture
def trajectory() -> Trajectory:
    return Trajectory.from_run(TASK, answer=ANSWER).trajectory


@pytest.mark.xfail(strict=True)
def test_a_black_box_run_holds_only_the_task_authored_by_the_user() -> None:
    run = Trajectory.from_run(TASK, answer=ANSWER)

    assert run.trajectory.history == (
        Observation(author=Author.USER, parts=(TextPart(text=TASK),)),
    )


@pytest.mark.xfail(strict=True)
def test_a_black_box_run_proposes_the_answer_as_the_action_to_rate() -> None:
    run = Trajectory.from_run(TASK, answer=ANSWER)

    assert run.answer.parts == (TextPart(text=ANSWER),)


@pytest.mark.xfail(strict=True)
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


@pytest.mark.xfail(strict=True)
def test_acting_an_action_already_in_the_history_raises(trajectory: Trajectory) -> None:
    trajectory.act(Action(id="a1"))

    with pytest.raises(ConfigurationError):
        trajectory.act(Action(id="a1"))


@pytest.mark.xfail(strict=True)
def test_a_delegated_trajectory_starts_with_the_instruction_from_the_delegator(
    trajectory: Trajectory,
) -> None:
    child = trajectory.delegate(INSTRUCTION, call_id="call_7")

    assert child.history == (
        Observation(author=Author.DELEGATOR, parts=(TextPart(text=INSTRUCTION),)),
    )


@pytest.mark.xfail(strict=True)
def test_a_delegated_trajectory_is_held_under_the_call_that_delegated_it(
    trajectory: Trajectory,
) -> None:
    child = trajectory.delegate(INSTRUCTION, call_id="call_7")

    assert trajectory.subagents.get("call_7") is child


@pytest.mark.xfail(strict=True)
def test_delegating_twice_from_one_call_raises(trajectory: Trajectory) -> None:
    trajectory.delegate(INSTRUCTION, call_id="call_7")

    with pytest.raises(ConfigurationError):
        trajectory.delegate("Find the q4 revenue", call_id="call_7")
