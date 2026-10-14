"""Actions and observations: what each one is given without being told."""

import pytest

from hyoshigi import Action, Author, TextPart, ToolObservation


@pytest.mark.xfail(strict=True)
def test_a_tool_observation_is_authored_by_the_tool() -> None:
    observation = ToolObservation(call_id="call_1", parts=(TextPart(text="Revenue: 4.2M"),))

    assert observation.author is Author.TOOL


@pytest.mark.xfail(strict=True)
def test_actions_built_without_an_id_get_distinct_ids() -> None:
    first, second = Action(), Action()

    assert first.id != second.id
