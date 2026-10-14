"""Dumping a trajectory to JSON-compatible data, and loading it back."""

import pytest

from hyoshigi import (
    JSON,
    Action,
    Author,
    Description,
    FormatError,
    MediaPart,
    Observation,
    OpaquePart,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)

TASK = Observation(author=Author.USER, parts=(TextPart(text="Plot q3.csv"),))
READ_FILE = Action(
    id="a1",
    thought=(TextPart(text="I should read the file first."), OpaquePart(kind="redacted_thinking")),
    parts=(TextPart(text="Reading q3.csv."),),
    tool_calls=(ToolCall(id="call_1", name="read_file", arguments='{"path": "q3.csv"}'),),
)
CHART = ToolObservation(
    call_id="call_1",
    parts=(
        MediaPart(
            media_type="image/png",
            data=b"\x89PNG",
            description=Description(text="A bar chart of q3 revenue", author=Author.TOOL),
        ),
    ),
)
MEMORY = Observation(
    author=Author.SCAFFOLD,
    source="memory",
    parts=(TextPart(text="The user prefers charts."),),
)

# The stored format, written out by hand: base64 of the bytes 89 50 4E 47 is "iVBORw==".
STORED: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "history": [
        {
            "kind": "observation",
            "author": "user",
            "source": None,
            "parts": [{"kind": "text", "text": "Plot q3.csv"}],
        },
        {
            "kind": "action",
            "id": "a1",
            "thought": [
                {"kind": "text", "text": "I should read the file first."},
                {"kind": "opaque", "opaque_kind": "redacted_thinking"},
            ],
            "parts": [{"kind": "text", "text": "Reading q3.csv."}],
            "tool_calls": [
                {"id": "call_1", "name": "read_file", "arguments": '{"path": "q3.csv"}'}
            ],
        },
        {
            "kind": "tool_observation",
            "call_id": "call_1",
            "is_error": False,
            "source": None,
            "parts": [
                {
                    "kind": "media",
                    "media_type": "image/png",
                    "url": None,
                    "data": "iVBORw==",
                    "description": {"text": "A bar chart of q3 revenue", "author": "tool"},
                },
            ],
        },
        {
            "kind": "observation",
            "author": "scaffold",
            "source": "memory",
            "parts": [{"kind": "text", "text": "The user prefers charts."}],
        },
    ],
    "subagents": {},
}

DELEGATING: dict[str, JSON] = {
    "format": "hyoshigi.trajectory",
    "version": 1,
    "history": [
        {
            "kind": "action",
            "id": "a1",
            "thought": [],
            "parts": [],
            "tool_calls": [{"id": "call_7", "name": "task", "arguments": "{}"}],
        },
        {
            "kind": "tool_observation",
            "call_id": "call_7",
            "is_error": True,
            "source": "subagent",
            "parts": [{"kind": "text", "text": "The subagent stopped: no access to q3.csv."}],
        },
    ],
    "subagents": {
        "call_7": {
            "format": "hyoshigi.trajectory",
            "version": 1,
            "history": [
                {
                    "kind": "observation",
                    "author": "delegator",
                    "source": None,
                    "parts": [{"kind": "text", "text": "Find the q3 revenue"}],
                },
                {
                    "kind": "action",
                    "id": "b1",
                    "thought": [],
                    "parts": [
                        {
                            "kind": "media",
                            "media_type": "image/png",
                            "url": "https://example.com/q3.png",
                            "data": None,
                            "description": None,
                        }
                    ],
                    "tool_calls": [],
                },
            ],
            "subagents": {},
        },
    },
}

ENVELOPE: dict[str, JSON] = {"format": "hyoshigi.trajectory", "version": 1, "subagents": {}}
ACTION: dict[str, JSON] = {
    "kind": "action",
    "id": "a1",
    "thought": [],
    "parts": [],
    "tool_calls": [],
}
OBSERVATION: dict[str, JSON] = {
    "kind": "observation",
    "author": "user",
    "source": None,
    "parts": [],
}
TOOL_OBSERVATION: dict[str, JSON] = {
    "kind": "tool_observation",
    "call_id": "call_1",
    "is_error": False,
    "source": None,
    "parts": [],
}
TEXT: dict[str, JSON] = {"kind": "text", "text": "Plot q3.csv"}
MEDIA: dict[str, JSON] = {
    "kind": "media",
    "media_type": "image/png",
    "url": None,
    "data": "iVBORw==",
    "description": None,
}
AUDIO: dict[str, JSON] = {"kind": "audio"}
TASK_CALL: dict[str, JSON] = {"id": "call_7", "name": "task", "arguments": "{}"}


@pytest.fixture
def trajectory() -> Trajectory:
    trajectory = Trajectory()
    trajectory.observe(TASK)
    trajectory.act(READ_FILE)
    trajectory.observe(CHART)
    trajectory.observe(MEMORY)
    return trajectory


@pytest.mark.xfail(strict=True)
def test_a_trajectory_dumps_to_the_stored_format(trajectory: Trajectory) -> None:
    stored = trajectory.dump()

    assert stored == STORED


@pytest.mark.xfail(strict=True, raises=FormatError)
def test_loading_the_stored_format_restores_the_history() -> None:
    loaded = Trajectory.load(STORED)

    assert loaded.history == (TASK, READ_FILE, CHART, MEMORY)


def test_a_trajectory_with_a_subagent_loads_back_unchanged() -> None:
    loaded = Trajectory.load(DELEGATING)

    assert loaded.dump() == DELEGATING


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ([], "$"),
        ({**ENVELOPE, "version": 2, "history": []}, "$"),
        ({**ENVELOPE, "version": True, "history": []}, "$"),
        ({**ENVELOPE, "history": [{"kind": "decision"}]}, "$.history[0]"),
        pytest.param(
            {**ENVELOPE, "history": [ACTION, {**ACTION, "parts": [TEXT]}]},
            "$.history[1]",
        ),
        pytest.param(
            {
                **ENVELOPE,
                "history": [
                    {**ACTION, "tool_calls": [TASK_CALL]},
                    {**ACTION, "id": "a2", "tool_calls": [{**TASK_CALL, "name": "review"}]},
                ],
            },
            "$.history[1]",
        ),
        ({**ENVELOPE, "history": [{**ACTION, "thought": [AUDIO]}]}, "$.history[0].thought[0]"),
        ({**ENVELOPE, "history": [{**ACTION, "parts": [AUDIO]}]}, "$.history[0].parts[0]"),
        (
            {**ENVELOPE, "history": [{**ACTION, "tool_calls": [{"id": "c1"}]}]},
            "$.history[0].tool_calls[0]",
        ),
        ({**ENVELOPE, "history": [{**OBSERVATION, "author": "admin"}]}, "$.history[0].author"),
        (
            {**ENVELOPE, "history": [{**OBSERVATION, "parts": [TEXT, AUDIO]}]},
            "$.history[0].parts[1]",
        ),
        (
            {**ENVELOPE, "history": [{**TOOL_OBSERVATION, "parts": [AUDIO]}]},
            "$.history[0].parts[0]",
        ),
        (
            {**ENVELOPE, "history": [{**OBSERVATION, "parts": [{**MEDIA, "data": "iVBO*Rw=="}]}]},
            "$.history[0].parts[0].data",
        ),
        (
            {**ENVELOPE, "history": [{**OBSERVATION, "parts": [{**MEDIA, "media_type": "png"}]}]},
            "$.history[0].parts[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {**OBSERVATION, "parts": [{**MEDIA, "description": {"text": "A chart"}}]}
                ],
            },
            "$.history[0].parts[0].description",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        **OBSERVATION,
                        "parts": [{**MEDIA, "description": {"text": "A chart", "author": "admin"}}],
                    }
                ],
            },
            "$.history[0].parts[0].description.author",
        ),
        (
            {
                **ENVELOPE,
                "history": [{**ACTION, "tool_calls": [TASK_CALL]}],
                "subagents": {"call_7": {**ENVELOPE, "history": [AUDIO]}},
            },
            "$.subagents[0].history[0]",
        ),
        pytest.param(
            {**ENVELOPE, "history": [], "subagents": {"call_7": {**ENVELOPE, "history": []}}},
            "$.subagents[0]",
        ),
    ],
    ids=[
        "not-a-mapping",
        "unknown-version",
        "boolean-version",
        "unknown-kind",
        "repeated-action-id",
        "repeated-tool-call-id",
        "malformed-thought",
        "malformed-action-part",
        "tool-call-without-arguments",
        "unknown-author",
        "malformed-observation-part",
        "malformed-tool-observation-part",
        "data-not-base64",
        "media-type-not-iana",
        "description-without-author",
        "description-with-unknown-author",
        "malformed-subagent",
        "subagent-without-its-call",
    ],
)
def test_loading_a_malformed_payload_raises_format_error_at_its_path(
    payload: JSON, path: str
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert raised.value.path == path


def test_a_format_error_names_its_path_in_its_message() -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load({**ENVELOPE, "history": [AUDIO]})

    assert str(raised.value).startswith("$.history[0]: ")


def test_a_format_error_never_quotes_the_payload() -> None:
    payload: JSON = {**ENVELOPE, "history": [{**OBSERVATION, "author": "s3cret-0f3c"}]}

    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert "s3cret-0f3c" not in str(raised.value)
