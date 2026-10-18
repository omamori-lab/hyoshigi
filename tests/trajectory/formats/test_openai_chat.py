"""The OpenAI chat completions format, with OpenRouter's reasoning fields."""

import pytest

from hyoshigi import (
    JSON,
    Action,
    Author,
    FormatError,
    MediaPart,
    Observation,
    OpaquePart,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
    marked,
)

# https://github.com/openai/openai-openapi, openapi.yaml, the chat completions examples
# "Default", "Image input" and "Functions": their request bodies and response messages.
DEFAULT_REQUEST: JSON = {
    "model": "gpt-6-astra",
    "messages": [
        {
            "role": "developer",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Hello!",
        },
    ],
}
DEFAULT_MESSAGE: JSON = {
    "role": "assistant",
    "content": "Hello! How can I assist you today?",
    "refusal": None,
    "annotations": [],
}
BOARDWALK = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)
IMAGE_INPUT_REQUEST: JSON = {
    "model": "gpt-6-astra",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": BOARDWALK},
                },
            ],
        },
    ],
    "max_tokens": 300,
}
WEATHER_QUESTION: JSON = {
    "role": "user",
    "content": "What is the weather like in Boston today?",
}
FOLLOW_UP: JSON = {
    "role": "user",
    "content": "And tomorrow?",
}
SCHEDULER_NUDGE: JSON = {
    "role": "user",
    "content": "It's 9am: run the daily report",
}
FUNCTIONS_MESSAGE: JSON = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": '{\n"location": "Boston, MA"\n}',
            },
        },
    ],
}
GET_CURRENT_WEATHER = ToolCall(
    id="call_abc123",
    name="get_current_weather",
    arguments='{\n"location": "Boston, MA"\n}',
)

# https://openrouter.ai/docs/use-cases/reasoning-tokens: the non-streaming response's
# message, and the encrypted reasoning detail.
OPENROUTER_MESSAGE: JSON = {
    "role": "assistant",
    "content": "Based on my analysis, I recommend the following approach...",
    "reasoning_details": [
        {
            "type": "reasoning.summary",
            "summary": "Analyzed the problem by breaking it into components",
            "id": "reasoning-summary-1",
            "format": "anthropic-claude-v1",
            "index": 0,
        },
        {
            "type": "reasoning.text",
            "text": "Let me work through this systematically:\n1. First consideration...\n"
            "2. Second consideration...",
            "signature": None,
            "id": "reasoning-text-1",
            "format": "anthropic-claude-v1",
            "index": 1,
        },
    ],
}
ENCRYPTED_REASONING: JSON = {
    "type": "reasoning.encrypted",
    "data": "eyJlbmNyeXB0ZWQiOiJ0cnVlIiwiY29udGVudCI6IltSRURBQ1RFRF0ifQ==",
    "id": "reasoning-encrypted-1",
    "format": "anthropic-claude-v1",
    "index": 1,
}


def build_user_request(*parts: JSON) -> JSON:
    return {
        "messages": [
            {
                "role": "user",
                "content": list(parts),
            },
        ],
    }


def build_assistant_request(message: JSON) -> JSON:
    return {"messages": [message]}


def test_the_published_default_request_becomes_the_developer_s_then_the_user_s_message() -> None:
    trajectory = Trajectory.from_openai_chat(DEFAULT_REQUEST)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            source="developer",
            parts=(TextPart(text="You are a helpful assistant."),),
        ),
        Observation(
            author=Author.USER,
            parts=(TextPart(text="Hello!"),),
        ),
    )


def test_the_published_image_input_becomes_text_and_an_image_by_url() -> None:
    trajectory = Trajectory.from_openai_chat(IMAGE_INPUT_REQUEST)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(
                TextPart(text="What is in this image?"),
                MediaPart(
                    media_type="image/*",
                    url=BOARDWALK,
                ),
            ),
        ),
    )


def test_the_published_default_response_becomes_the_action_to_rate() -> None:
    action = Action.from_openai_chat(DEFAULT_MESSAGE)

    assert (
        action.thought,
        action.parts,
        action.tool_calls,
        type(action.id),
    ) == (
        (),
        (TextPart(text="Hello! How can I assist you today?"),),
        (),
        str,
    )


def test_the_published_function_call_keeps_its_arguments_as_emitted() -> None:
    action = Action.from_openai_chat(FUNCTIONS_MESSAGE)

    assert (
        action.parts,
        action.tool_calls,
    ) == (
        (),
        (GET_CURRENT_WEATHER,),
    )


def test_a_function_round_trip_becomes_the_action_then_the_tool_s_observation() -> None:
    request: JSON = {
        "messages": [
            WEATHER_QUESTION,
            FUNCTIONS_MESSAGE,
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "22 degrees, sunny",
            },
        ],
    }

    trajectory = Trajectory.from_openai_chat(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text="What is the weather like in Boston today?"),),
        ),
        Action(
            id="$.messages[1]",
            tool_calls=(GET_CURRENT_WEATHER,),
        ),
        ToolObservation(
            call_id="call_abc123",
            parts=(TextPart(text="22 degrees, sunny"),),
        ),
    )


def test_openrouter_s_reasoning_details_become_the_thought() -> None:
    action = Action.from_openai_chat(OPENROUTER_MESSAGE)

    assert (
        action.thought,
        action.parts,
    ) == (
        (
            TextPart(text="Analyzed the problem by breaking it into components"),
            TextPart(
                text="Let me work through this systematically:\n1. First consideration...\n"
                "2. Second consideration...",
            ),
        ),
        (TextPart(text="Based on my analysis, I recommend the following approach..."),),
    )


@pytest.mark.parametrize(
    ("message", "thought"),
    [
        (
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning": "The user wants a summary.",
            },
            (TextPart(text="The user wants a summary."),),
        ),
        (
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning": "A summary.",
                "reasoning_details": [ENCRYPTED_REASONING],
            },
            (OpaquePart(kind="reasoning.encrypted"),),
        ),
        (
            {
                "role": "assistant",
                "content": "Done.",
                "reasoning": None,
            },
            (),
        ),
    ],
    ids=["reasoning", "details-over-reasoning", "null-reasoning"],
)
def test_openrouter_s_reasoning_becomes_the_thought(
    message: JSON,
    thought: tuple,
) -> None:
    action = Action.from_openai_chat(message)

    assert (
        action.thought,
        action.parts,
    ) == (
        thought,
        (TextPart(text="Done."),),
    )


def test_marked_authorship_makes_only_the_marked_message_the_user_s() -> None:
    request: JSON = {
        "messages": [
            WEATHER_QUESTION,
            FOLLOW_UP,
        ],
    }

    trajectory = Trajectory.from_openai_chat(
        request,
        authorship=marked({1}),
    )

    assert trajectory.history == (
        Observation(
            author=Author.UNKNOWN,
            parts=(TextPart(text="What is the weather like in Boston today?"),),
        ),
        Observation(
            author=Author.USER,
            parts=(TextPart(text="And tomorrow?"),),
        ),
    )


def attribute_nudges_to_the_scheduler(
    index: int,
    *,
    message: JSON,
) -> Author:
    return Author.SCAFFOLD if message is SCHEDULER_NUDGE else Author.USER


def test_a_harness_s_own_authorship_reads_each_user_message() -> None:
    request: JSON = {"messages": [SCHEDULER_NUDGE]}

    trajectory = Trajectory.from_openai_chat(
        request,
        authorship=attribute_nudges_to_the_scheduler,
    )

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            parts=(TextPart(text="It's 9am: run the daily report"),),
        ),
    )


def test_an_assistant_s_text_parts_are_its_parts() -> None:
    action = Action.from_openai_chat(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Sunny,",
                },
                {
                    "type": "text",
                    "text": " 22 degrees.",
                },
            ],
        },
    )

    assert action.parts == (
        TextPart(text="Sunny,"),
        TextPart(text=" 22 degrees."),
    )


def test_a_refusal_and_an_unknown_part_are_what_the_assistant_said() -> None:
    action = Action.from_openai_chat(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "refusal",
                    "refusal": "I can't help with that.",
                },
                {
                    "type": "output_audio",
                    "id": "audio_1",
                },
            ],
            "refusal": "I can't help with that.",
            "audio": {"id": "audio_abc123"},
        },
    )

    assert action.parts == (
        TextPart(text="I can't help with that."),
        OpaquePart(kind="output_audio"),
        TextPart(text="I can't help with that."),
        OpaquePart(kind="audio"),
    )


def test_a_custom_tool_call_and_a_deprecated_function_call_become_tool_calls() -> None:
    request = build_assistant_request(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "custom",
                    "custom": {
                        "name": "grep",
                        "input": "q3",
                    },
                },
                {
                    "id": "call_2",
                    "type": "computer_use",
                    "action": {},
                },
            ],
            "function_call": {
                "name": "get_current_weather",
                "arguments": "{}",
            },
        },
    )

    trajectory = Trajectory.from_openai_chat(request)

    assert trajectory.history == (
        Action(
            id="$.messages[0]",
            parts=(OpaquePart(kind="computer_use"),),
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="grep",
                    arguments="q3",
                ),
                ToolCall(
                    id="$.messages[0].function_call",
                    name="get_current_weather",
                    arguments="{}",
                ),
            ),
        ),
    )


# RFC 2397: a data URL carries its media type and its bytes, here in base64 or percent-encoded.
def test_media_parts_become_media_or_opaque_parts() -> None:
    request = build_user_request(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw=="},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:text/plain;charset=UTF-8,15%20degrees"},
        },
        {
            "type": "input_audio",
            "input_audio": {
                "data": "UklGRg==",
                "format": "wav",
            },
        },
        {
            "type": "input_audio",
            "input_audio": {
                "data": "SUQz",
                "format": "mp3",
            },
        },
        {
            "type": "file",
            "file": {
                "filename": "q3.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjcK",
            },
        },
        {
            "type": "file",
            "file": {"file_id": "file-abc123"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "data:,15%20degrees"},
        },
        {
            "type": "file",
            "file": {"file_data": "JVBERi0xLjcK"},
        },
        {
            "type": "input_video",
            "video_url": "https://example.com/a.mp4",
        },
    )

    trajectory = Trajectory.from_openai_chat(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(
                    media_type="image/png",
                    data=b"\x89PNG",
                ),
                MediaPart(
                    media_type="text/plain",
                    data=b"15 degrees",
                ),
                MediaPart(
                    media_type="audio/wav",
                    data=b"RIFF",
                ),
                MediaPart(
                    media_type="audio/mpeg",
                    data=b"ID3",
                ),
                MediaPart(
                    media_type="application/pdf",
                    data=b"%PDF-1.7\n",
                ),
                OpaquePart(kind="file"),
                MediaPart(
                    media_type="text/plain",
                    data=b"15 degrees",
                ),
                OpaquePart(kind="file"),
                OpaquePart(kind="input_video"),
            ),
        ),
    )


def test_text_parts_and_the_deprecated_function_message_keep_their_authors() -> None:
    request: JSON = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer in French.",
                    },
                ],
            },
            {
                "role": "function",
                "name": "get_current_weather",
                "content": "22 degrees",
            },
            {
                "role": "function",
                "name": "get_current_weather",
                "content": None,
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {
                        "type": "text",
                        "text": "ok",
                    },
                ],
            },
        ],
    }

    trajectory = Trajectory.from_openai_chat(request)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            source="system",
            parts=(TextPart(text="Answer in French."),),
        ),
        Observation(
            author=Author.TOOL,
            source="get_current_weather",
            parts=(TextPart(text="22 degrees"),),
        ),
        Observation(
            author=Author.TOOL,
            source="get_current_weather",
            parts=(),
        ),
        ToolObservation(
            call_id="call_1",
            parts=(TextPart(text="ok"),),
        ),
    )


@pytest.mark.parametrize(
    ("request_body", "path"),
    [
        (
            [],
            "$",
        ),
        (
            {"messages": "Hello"},
            "$",
        ),
        (
            {
                "messages": [
                    {
                        "role": "critic",
                        "content": "Hi",
                    },
                ],
            },
            "$.messages[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": 42,
                    },
                ],
            },
            "$.messages[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": "ok",
                    },
                ],
            },
            "$.messages[0]",
        ),
        (
            build_user_request({"type": "text"}),
            "$.messages[0].content[0]",
        ),
        (
            build_user_request({"text": "Hi"}),
            "$.messages[0].content[0]",
        ),
        (
            build_user_request(
                {
                    "type": "image_url",
                    "image_url": {},
                },
            ),
            "$.messages[0].content[0]",
        ),
        (
            build_user_request(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,*"},
                },
            ),
            "$.messages[0].content[0].image_url.url",
        ),
        (
            build_user_request(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "SUQz",
                        "format": "flac",
                    },
                },
            ),
            "$.messages[0].content[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "image_url"}],
                    },
                ],
            },
            "$.messages[0].content[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "tool_calls": "call_1",
                },
            ),
            "$.messages[0].tool_calls",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f"},
                        },
                    ],
                },
            ),
            "$.messages[0].tool_calls[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "reasoning_details": "thinking",
                },
            ),
            "$.messages[0].reasoning_details",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "reasoning_details": [{"type": "reasoning.text"}],
                },
            ),
            "$.messages[0].reasoning_details[0]",
        ),
        (
            {
                "messages": [
                    FUNCTIONS_MESSAGE,
                    FUNCTIONS_MESSAGE,
                ],
            },
            "$.messages[1]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "content": 42,
                },
            ),
            "$.messages[0].content",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "content": [{"type": "refusal"}],
                },
            ),
            "$.messages[0].content[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "content": [{"text": "Hi"}],
                },
            ),
            "$.messages[0].content[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "refusal": 42,
                },
            ),
            "$.messages[0].refusal",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "audio": "a1",
                },
            ),
            "$.messages[0].audio",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "reasoning": 42,
                },
            ),
            "$.messages[0].reasoning",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "reasoning_details": [{"summary": "s"}],
                },
            ),
            "$.messages[0].reasoning_details[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "reasoning_details": [{"type": "reasoning.summary"}],
                },
            ),
            "$.messages[0].reasoning_details[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1"}],
                },
            ),
            "$.messages[0].tool_calls[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "custom",
                            "custom": {"name": "grep"},
                        },
                    ],
                },
            ),
            "$.messages[0].tool_calls[0]",
        ),
        (
            build_assistant_request(
                {
                    "role": "assistant",
                    "function_call": {"name": "f"},
                },
            ),
            "$.messages[0].function_call",
        ),
        (
            build_user_request(
                {
                    "type": "file",
                    "file": "q3.pdf",
                },
            ),
            "$.messages[0].content[0]",
        ),
        (
            build_user_request(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "*",
                        "format": "wav",
                    },
                },
            ),
            "$.messages[0].content[0].input_audio.data",
        ),
        (
            build_user_request(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:png;base64,iVBORw=="},
                },
            ),
            "$.messages[0].content[0].image_url.url",
        ),
        (
            build_user_request(
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64,*"},
                },
            ),
            "$.messages[0].content[0].file.file_data",
        ),
        (
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "c1",
                        "content": [{"type": "image_url"}],
                    },
                ],
            },
            "$.messages[0].content[0]",
        ),
    ],
    ids=[
        "not-a-mapping",
        "messages-not-a-list",
        "unknown-role",
        "user-content-not-text-or-parts",
        "tool-message-without-call-id",
        "text-part-without-text",
        "part-without-type",
        "image-url-without-url",
        "data-url-not-base64",
        "audio-format-not-wav-or-mp3",
        "system-part-not-text",
        "tool-calls-not-a-list",
        "function-call-without-arguments",
        "reasoning-details-not-a-list",
        "reasoning-text-without-text",
        "repeated-tool-call-id",
        "assistant-content-not-text-or-parts",
        "refusal-part-without-refusal",
        "assistant-part-without-type",
        "refusal-not-text",
        "audio-not-an-object",
        "reasoning-not-text",
        "reasoning-detail-without-type",
        "reasoning-summary-without-summary",
        "tool-call-without-type",
        "custom-tool-call-without-input",
        "function-call-without-arguments-deprecated",
        "file-part-not-an-object",
        "audio-data-not-base64",
        "data-url-media-type-not-iana",
        "file-data-url-not-base64",
        "tool-content-part-not-text",
    ],
)
def test_a_malformed_request_raises_format_error_at_its_path(
    request_body: JSON,
    path: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.from_openai_chat(request_body)

    assert raised.value.path == path


@pytest.mark.parametrize(
    ("message", "path"),
    [
        (
            WEATHER_QUESTION,
            "$",
        ),
        (
            {
                "role": "assistant",
                "content": [{"type": "text"}],
            },
            "$.content[0]",
        ),
    ],
    ids=["not-the-assistant-s", "malformed-part"],
)
def test_a_malformed_response_raises_format_error_at_its_path(
    message: JSON,
    path: str,
) -> None:
    with pytest.raises(FormatError) as raised:
        Action.from_openai_chat(message)

    assert raised.value.path == path
