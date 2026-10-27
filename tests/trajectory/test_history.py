"""Actions and observations: what each one is given without being told, and the fields that
cannot work.
"""

import pytest

from hyoshigi import (
    Action,
    Author,
    ConfigurationError,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
)

TASK = TextPart(text="Summarise q3.md")
READ = ToolCall(
    id="call_1",
    name="read_file",
    arguments='{"path": "q3.md"}',
)


def test_a_tool_observation_is_authored_by_the_tool() -> None:
    observation = ToolObservation(
        call_id="call_1",
        parts=(TextPart(text="Revenue: 4.2M"),),
    )

    assert observation.author is Author.TOOL


def test_actions_built_without_an_id_get_distinct_ids() -> None:
    first, second = Action(), Action()

    assert first.id != second.id


@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        pytest.param(
            Action,
            {"id": 1},
            "id must be a str, got an instance of int",
            id="action-id-a-number",
        ),
        pytest.param(
            Action,
            {"thought": [TASK]},
            "thought must be a tuple, got an instance of list",
            id="thought-a-list",
        ),
        pytest.param(
            Action,
            {
                "parts": (
                    TASK,
                    "Summarise q3.md",
                ),
            },
            "parts[1] must be a TextPart, MediaPart or OpaquePart, got an instance of str",
            id="part-a-string",
        ),
        pytest.param(
            Action,
            {"tool_calls": READ},
            "tool_calls must be a tuple, got an instance of ToolCall",
            id="tool-calls-a-call",
        ),
        pytest.param(
            Action,
            {"tool_calls": ({"id": "call_1"},)},
            "tool_calls[0] must be a ToolCall, got an instance of dict",
            id="tool-call-a-dict",
        ),
        pytest.param(
            ToolCall,
            {
                "id": 1,
                "name": "read_file",
                "arguments": "{}",
            },
            "id must be a str, got an instance of int",
            id="call-id-a-number",
        ),
        pytest.param(
            ToolCall,
            {
                "id": "call_1",
                "name": None,
                "arguments": "{}",
            },
            "name must be a str, got None",
            id="name-none",
        ),
        pytest.param(
            ToolCall,
            {
                "id": "call_1",
                "name": "read_file",
                "arguments": {"path": "q3.md"},
            },
            "arguments must be a str, got an instance of dict",
            id="arguments-a-dict",
        ),
        pytest.param(
            Observation,
            {
                "author": "user",
                "parts": (TASK,),
            },
            "author must be an Author, got an instance of str",
            id="author-a-string",
        ),
        pytest.param(
            Observation,
            {
                "author": Author.USER,
                "parts": "Summarise q3.md",
            },
            "parts must be a tuple, got an instance of str",
            id="parts-a-string",
        ),
        pytest.param(
            Observation,
            {
                "author": Author.USER,
                "parts": (TASK,),
                "source": 1,
            },
            "source must be a str, got an instance of int",
            id="source-a-number",
        ),
        pytest.param(
            ToolObservation,
            {
                "call_id": "call_1",
                "author": "monitor",
                "parts": (TASK,),
            },
            "author must be an Author, got an instance of str",
            id="tool-observation-author-a-string",
        ),
        pytest.param(
            ToolObservation,
            {
                "call_id": 1,
                "parts": (TASK,),
            },
            "call_id must be a str, got an instance of int",
            id="call-id-of-the-result-a-number",
        ),
        pytest.param(
            ToolObservation,
            {
                "call_id": "call_1",
                "is_error": "yes",
                "parts": (TASK,),
            },
            "is_error must be a bool, got an instance of str",
            id="is-error-a-string",
        ),
    ],
)
def test_a_field_of_the_wrong_type_raises_naming_it(
    kind: type,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**fields)

    assert str(raised.value) == message
