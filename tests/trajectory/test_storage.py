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
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)

TASK = Observation(author=Author.USER, parts=(TextPart(text="Plot q3.csv"),))
READ_FILE = Action(
    id="a1",
    thought=(TextPart(text="I should read the file first."),),
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
            "thought": [{"kind": "text", "text": "I should read the file first."}],
            "parts": [],
            "tool_calls": [
                {"id": "call_1", "name": "read_file", "arguments": '{"path": "q3.csv"}'}
            ],
        },
        {
            "kind": "tool_observation",
            "call_id": "call_1",
            "is_error": False,
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
            ],
            "subagents": {},
        },
    },
}

ENVELOPE: dict[str, JSON] = {"format": "hyoshigi.trajectory", "version": 1, "subagents": {}}
TEXT: dict[str, JSON] = {"kind": "text", "text": "Plot q3.csv"}


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


@pytest.mark.xfail(strict=True)
def test_loading_the_stored_format_restores_the_history() -> None:
    loaded = Trajectory.load(STORED)

    assert loaded.history == (TASK, READ_FILE, CHART, MEMORY)


@pytest.mark.xfail(strict=True)
def test_a_trajectory_with_a_subagent_loads_back_unchanged() -> None:
    loaded = Trajectory.load(DELEGATING)

    assert loaded.dump() == DELEGATING


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ([], "$"),
        ({**ENVELOPE, "version": 2, "history": []}, "$"),
        ({**ENVELOPE, "version": True, "history": []}, "$"),
        ({**ENVELOPE, "history": [{"kind": "decision"}]}, "$.history[0]"),
        (
            {
                **ENVELOPE,
                "history": [
                    {"kind": "observation", "author": "admin", "source": None, "parts": []}
                ],
            },
            "$.history[0].author",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        "kind": "action",
                        "id": "a1",
                        "thought": [],
                        "parts": [],
                        "tool_calls": [{"id": "c1"}],
                    }
                ],
            },
            "$.history[0].tool_calls[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        "kind": "observation",
                        "author": "user",
                        "source": None,
                        "parts": [
                            TEXT,
                            {
                                "kind": "media",
                                "media_type": "image/png",
                                "url": None,
                                "data": "not base64!",
                                "description": None,
                            },
                        ],
                    }
                ],
            },
            "$.history[0].parts[1].data",
        ),
        (
            {
                **ENVELOPE,
                "history": [
                    {
                        "kind": "observation",
                        "author": "user",
                        "source": None,
                        "parts": [
                            {
                                "kind": "media",
                                "media_type": "png",
                                "url": "https://example.com/a",
                                "data": None,
                                "description": None,
                            }
                        ],
                    }
                ],
            },
            "$.history[0].parts[0]",
        ),
        (
            {
                **ENVELOPE,
                "history": [],
                "subagents": {
                    "call_7": {
                        "format": "hyoshigi.trajectory",
                        "version": 1,
                        "history": [{"kind": "decision"}],
                        "subagents": {},
                    }
                },
            },
            "$.subagents[0].history[0]",
        ),
    ],
    ids=[
        "not-a-mapping",
        "unknown-version",
        "boolean-version",
        "unknown-kind",
        "unknown-author",
        "tool-call-without-arguments",
        "data-not-base64",
        "media-type-not-iana",
        "malformed-subagent",
    ],
)
def test_loading_a_malformed_payload_raises_format_error_at_its_path(
    payload: JSON, path: str
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert raised.value.path == path


@pytest.mark.xfail(strict=True)
def test_a_format_error_never_quotes_the_payload() -> None:
    payload = {
        **ENVELOPE,
        "history": [{"kind": "observation", "author": "s3cret-0f3c", "source": None, "parts": []}],
    }

    with pytest.raises(FormatError) as raised:
        Trajectory.load(payload)

    assert "s3cret-0f3c" not in str(raised.value)
