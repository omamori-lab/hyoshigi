"""Anthropic's Messages format: requests into trajectories, responses into actions to rate."""

import pytest

from hyoshigi import (
    JSON,
    Action,
    Author,
    Event,
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

# https://platform.claude.com/docs/en/api/messages, "Example": the published request body.
REFERENCE_REQUEST: JSON = {
    "max_tokens": 1024,
    "messages": [{"content": "Hello, world", "role": "user"}],
    "model": "claude-opus-5-5",
    "stream": False,
    "system": [{"text": "Today's date is 2024-06-01.", "type": "text"}],
    "temperature": 1,
    "thinking": {"type": "adaptive"},
    "tools": [
        {
            "input_schema": {
                "type": "object",
                "properties": {"location": "bar", "unit": "bar"},
                "required": ["location"],
            },
            "name": "name",
        }
    ],
    "top_k": 5,
    "top_p": 0.7,
}

# https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
TOOL_USE_RESPONSE: JSON = {
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "I'll help you check the current weather and time in San Francisco.",
        },
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_weather",
            "input": {"location": "San Francisco, CA"},
        },
    ],
}

# https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls, "Example of
# tool result with images". The published data ends in "...", so here it stops at the bytes its
# prefix encodes: FF D8 FF E0 00 10 "JFIF".
TOOL_RESULT_WITH_IMAGE: JSON = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
            "content": [
                {"type": "text", "text": "15 degrees"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQSkZJRg==",
                    },
                },
            ],
        }
    ],
}

# The same page, "Example of empty tool result".
EMPTY_TOOL_RESULT: JSON = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9"}],
}

# https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool, the example
# response structure, without its comments.
WEB_SEARCH_CONTENT: JSON = [
    {"type": "text", "text": "I'll search for when Claude Shannon was born."},
    {
        "type": "server_tool_use",
        "id": "srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
        "name": "web_search",
        "input": {"query": "claude shannon birth date"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                "title": "Claude Shannon - Wikipedia",
                "encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5LTQ1NG"
                "UtODgxNC1kNjNjNTk1ZWI3Y...",
                "page_age": "April 30, 2025",
            }
        ],
    },
    {"text": "Based on the search results, ", "type": "text"},
    {
        "text": "Claude Shannon was born on April 30, 1916, in Petoskey, Michigan",
        "type": "text",
        "citations": [
            {
                "type": "web_search_result_location",
                "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                "title": "Claude Shannon - Wikipedia",
                "encrypted_index": "Eo8BCioIAhgBIiQyYjQ0OWJmZi1lNm..",
                "cited_text": "Claude Elwood Shannon (April 30, 1916 \u2013 February 24, 2001) "
                "was an "
                "American mathematician, electrical engineer, computer scientist, "
                "cryptographer and i...",
            }
        ],
    },
]
WEB_SEARCH_RESPONSE: JSON = {
    "role": "assistant",
    "content": WEB_SEARCH_CONTENT,
    "id": "msg_a930390d3a",
    "usage": {
        "input_tokens": 6039,
        "output_tokens": 931,
        "server_tool_use": {"web_search_requests": 1},
    },
    "stop_reason": "end_turn",
}

WEATHER_QUESTION: JSON = {"role": "user", "content": "What's the weather in San Francisco?"}
SHANNON_QUESTION: JSON = {"role": "user", "content": "When was Claude Shannon born?"}
SCHEDULER_NUDGE: JSON = {"role": "user", "content": "It's 9am: run the daily report"}
WEATHER_CALL_ID = "toolu_01A09q90qw90lq917835lq9"
MODEL_CARD = "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf"

# https://platform.claude.com/docs/en/api/messages, the messages parameter's example with
# several conversational turns.
CONVERSATION: JSON = {
    "messages": [
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
        {"role": "user", "content": "Can you explain LLMs in plain English?"},
    ]
}

# https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls: "Example of
# successful tool result", "Example of tool result with documents", a tool execution error, and
# the text blocks of its examples on where text goes.
SUCCESSFUL_TOOL_RESULT: JSON = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": WEATHER_CALL_ID, "content": "15 degrees"}],
}
TOOL_RESULT_WITH_DOCUMENT: JSON = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": WEATHER_CALL_ID,
            "content": [
                {"type": "text", "text": "The weather is"},
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "15 degrees"},
                },
            ],
        }
    ],
}
TOOL_ERROR: JSON = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": WEATHER_CALL_ID,
            "content": "ConnectionError: the weather service API is not available (HTTP 500)",
            "is_error": True,
        }
    ],
}
NEXT_QUESTION: JSON = {"type": "text", "text": "What should I do next?"}
HERE_ARE_THE_RESULTS: JSON = {"type": "text", "text": "Here are the results:"}

GET_WEATHER = ToolCall(
    id="toolu_01A09q90qw90lq917835lq9",
    name="get_weather",
    arguments='{"location": "San Francisco, CA"}',
)
WEB_SEARCH = ToolCall(
    id="srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
    name="web_search",
    arguments='{"query": "claude shannon birth date"}',
)
# The search results' content, as json.dumps writes it.
SEARCH_RESULTS = (
    '[{"type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Claude_Shannon", '
    '"title": "Claude Shannon - Wikipedia", '
    '"encrypted_content": "EqgfCioIARgBIiQ3YTAwMjY1Mi1mZjM5LTQ1NGUtODgxNC1kNjNjNTk1ZWI3Y...", '
    '"page_age": "April 30, 2025"}]'
)


SOURCE = "$.messages[0].content[0].source"


def build_user_request(*blocks: JSON) -> JSON:
    return {"messages": [{"role": "user", "content": list(blocks)}]}


def build_tool_result(content: JSON) -> JSON:
    return {"type": "tool_result", "tool_use_id": WEATHER_CALL_ID, "content": content}


def build_image(source: JSON) -> JSON:
    return {"type": "image", "source": source}


def build_document(source: JSON) -> JSON:
    return {"type": "document", "source": source}


def test_the_published_request_becomes_the_system_prompt_then_the_user_s_message() -> None:
    trajectory = Trajectory.from_anthropic(REFERENCE_REQUEST)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            source="system",
            parts=(TextPart(text="Today's date is 2024-06-01."),),
        ),
        Observation(author=Author.USER, parts=(TextPart(text="Hello, world"),)),
    )


def test_a_tool_round_trip_becomes_the_action_then_its_tool_observation() -> None:
    request: JSON = {"messages": [WEATHER_QUESTION, TOOL_USE_RESPONSE, TOOL_RESULT_WITH_IMAGE]}

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER, parts=(TextPart(text="What's the weather in San Francisco?"),)
        ),
        Action(
            id="$.messages[1].content[0]",
            parts=(
                TextPart(text="I'll help you check the current weather and time in San Francisco."),
            ),
            tool_calls=(GET_WEATHER,),
        ),
        ToolObservation(
            call_id="toolu_01A09q90qw90lq917835lq9",
            parts=(
                TextPart(text="15 degrees"),
                MediaPart(media_type="image/jpeg", data=b"\xff\xd8\xff\xe0\x00\x10JFIF"),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("tool_result", "expected"),
    [
        (
            SUCCESSFUL_TOOL_RESULT,
            ToolObservation(call_id=WEATHER_CALL_ID, parts=(TextPart(text="15 degrees"),)),
        ),
        (
            TOOL_RESULT_WITH_DOCUMENT,
            ToolObservation(
                call_id=WEATHER_CALL_ID,
                parts=(TextPart(text="The weather is"), TextPart(text="15 degrees")),
            ),
        ),
        (
            TOOL_ERROR,
            ToolObservation(
                call_id=WEATHER_CALL_ID,
                is_error=True,
                parts=(
                    TextPart(
                        text="ConnectionError: the weather service API is not available (HTTP 500)"
                    ),
                ),
            ),
        ),
        (EMPTY_TOOL_RESULT, ToolObservation(call_id=WEATHER_CALL_ID, parts=())),
        (
            {"role": "user", "content": [build_tool_result(None)]},
            ToolObservation(call_id=WEATHER_CALL_ID, parts=()),
        ),
    ],
    ids=["successful", "with-a-document", "error", "empty", "null-content"],
)
def test_a_published_tool_result_becomes_the_tool_s_observation(
    tool_result: JSON, expected: ToolObservation
) -> None:
    request: JSON = {"messages": [WEATHER_QUESTION, TOOL_USE_RESPONSE, tool_result]}

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history[-1:] == (expected,)


# The same page: text after the tool results, as the API requires, and text before them, which
# the API refuses with a 400 error; either way the events keep the blocks' order.
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            [{"type": "tool_result", "tool_use_id": WEATHER_CALL_ID}, NEXT_QUESTION],
            (
                ToolObservation(call_id=WEATHER_CALL_ID, parts=()),
                Observation(author=Author.USER, parts=(TextPart(text="What should I do next?"),)),
            ),
        ),
        (
            [HERE_ARE_THE_RESULTS, {"type": "tool_result", "tool_use_id": WEATHER_CALL_ID}],
            (
                Observation(author=Author.USER, parts=(TextPart(text="Here are the results:"),)),
                ToolObservation(call_id=WEATHER_CALL_ID, parts=()),
            ),
        ),
    ],
    ids=["results-then-text", "text-then-results"],
)
def test_tool_results_and_text_keep_their_order(content: JSON, expected: tuple[Event, ...]) -> None:
    request: JSON = {
        "messages": [WEATHER_QUESTION, TOOL_USE_RESPONSE, {"role": "user", "content": content}]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history[2:] == expected


def test_a_server_tool_in_the_history_becomes_its_call_its_result_and_the_text_after() -> None:
    request: JSON = {
        "messages": [SHANNON_QUESTION, {"role": "assistant", "content": WEB_SEARCH_CONTENT}]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Observation(author=Author.USER, parts=(TextPart(text="When was Claude Shannon born?"),)),
        Action(
            id="$.messages[1].content[0]",
            parts=(TextPart(text="I'll search for when Claude Shannon was born."),),
            tool_calls=(WEB_SEARCH,),
        ),
        ToolObservation(
            call_id="srvtoolu_01WYG3ziw53XMcoyKL4XcZmE",
            source="server",
            parts=(TextPart(text=SEARCH_RESULTS),),
        ),
        Action(
            id="$.messages[1].content[3]",
            parts=(
                TextPart(text="Based on the search results, "),
                TextPart(text="Claude Shannon was born on April 30, 1916, in Petoskey, Michigan"),
            ),
        ),
    )


def test_thinking_becomes_the_action_s_thought() -> None:
    request: JSON = {
        "messages": [
            WEATHER_QUESTION,
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need the weather.", "signature": "EqQB"},
                    {"type": "redacted_thinking", "data": "EmwKAhgB"},
                    {"type": "text", "text": "Checking now."},
                ],
            },
        ]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history[-1:] == (
        Action(
            id="$.messages[1].content[0]",
            thought=(
                TextPart(text="I need the weather."),
                OpaquePart(kind="redacted_thinking"),
            ),
            parts=(TextPart(text="Checking now."),),
        ),
    )


# https://platform.claude.com/docs/en/build-with-claude/vision: an image by URL, which names
# no subtype, and a block the builder does not model.
def test_an_image_by_url_and_an_unmodelled_block_keep_their_place() -> None:
    request: JSON = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://platform.claude.com/docs/images/vision-example.jpg",
                        },
                    },
                    {"type": "container_upload", "file_id": "file_011CNha8iCJcU1wXNR6q4V8w"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(
                    media_type="image/*",
                    url="https://platform.claude.com/docs/images/vision-example.jpg",
                ),
                OpaquePart(kind="container_upload"),
                TextPart(text="Describe this image."),
            ),
        ),
    )


def test_marked_authorship_makes_only_the_marked_turns_the_user_s() -> None:
    trajectory = Trajectory.from_anthropic(CONVERSATION, authorship=marked({2}))

    assert trajectory.history == (
        Observation(author=Author.UNKNOWN, parts=(TextPart(text="Hello there."),)),
        Action(
            id="$.messages[1].content",
            parts=(TextPart(text="Hi, I'm Claude. How can I help you?"),),
        ),
        Observation(
            author=Author.USER, parts=(TextPart(text="Can you explain LLMs in plain English?"),)
        ),
    )


def attribute_nudges_to_the_scheduler(index: int, *, message: JSON) -> Author:
    return Author.SCAFFOLD if message is SCHEDULER_NUDGE else Author.USER


def test_a_harness_s_own_authorship_reads_each_user_role_message() -> None:
    request: JSON = {"messages": [SCHEDULER_NUDGE]}

    trajectory = Trajectory.from_anthropic(request, authorship=attribute_nudges_to_the_scheduler)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD, parts=(TextPart(text="It's 9am: run the daily report"),)
        ),
    )


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        (
            "You are a weather assistant.",
            (
                Observation(
                    author=Author.SCAFFOLD,
                    source="system",
                    parts=(TextPart(text="You are a weather assistant."),),
                ),
            ),
        ),
        (None, ()),
    ],
    ids=["text", "null"],
)
def test_a_system_prompt_given_as_text_or_null(system: JSON, expected: tuple[Event, ...]) -> None:
    request: JSON = {"system": system, "messages": []}

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == expected


# https://platform.claude.com/docs/en/build-with-claude/pdf-support: a PDF by URL, and one in
# base64, whose published data is a shell variable, here the bytes "%PDF-1.7" and a newline.
# Files referred to by id are opaque.
def test_documents_become_media_and_files_become_opaque() -> None:
    request: JSON = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "document", "source": {"type": "url", "url": MODEL_CARD}},
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjcK",
                        },
                    },
                    {"type": "image", "source": {"type": "file", "file_id": "file_011"}},
                    {"type": "document", "source": {"type": "file", "file_id": "file_012"}},
                    {"type": "text", "text": "What are the key findings in this document?"},
                ],
            }
        ]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(media_type="application/pdf", url=MODEL_CARD),
                MediaPart(media_type="application/pdf", data=b"%PDF-1.7\n"),
                OpaquePart(kind="image"),
                OpaquePart(kind="document"),
                TextPart(text="What are the key findings in this document?"),
            ),
        ),
    )


# https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool, "Errors": a
# server tool's failure is an error object in the result's content.
@pytest.mark.xfail(strict=True)
def test_a_server_tool_s_error_is_observed_as_an_error() -> None:
    request: JSON = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_a93jad",
                        "name": "web_search",
                        "input": {"query": "q3 revenue"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_a93jad",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    },
                ],
            }
        ]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history[-1:] == (
        ToolObservation(
            call_id="srvtoolu_a93jad",
            source="server",
            is_error=True,
            parts=(
                TextPart(
                    text='{"type": "web_search_tool_result_error", '
                    '"error_code": "max_uses_exceeded"}'
                ),
            ),
        ),
    )


def test_an_assistant_message_without_blocks_adds_no_action() -> None:
    request: JSON = {"messages": [WEATHER_QUESTION, {"role": "assistant", "content": []}]}

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER, parts=(TextPart(text="What's the weather in San Francisco?"),)
        ),
    )


def test_text_outside_ascii_stays_as_written_in_arguments_and_results() -> None:
    request: JSON = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "Zürich"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [{"title": "Zürich"}],
                    },
                ],
            }
        ]
    }

    trajectory = Trajectory.from_anthropic(request)

    assert trajectory.history == (
        Action(
            id="$.messages[0].content[0]",
            tool_calls=(
                ToolCall(id="srvtoolu_1", name="web_search", arguments='{"query": "Zürich"}'),
            ),
        ),
        ToolObservation(
            call_id="srvtoolu_1", source="server", parts=(TextPart(text='[{"title": "Zürich"}]'),)
        ),
    )


def test_a_response_whose_server_tool_already_ran_holds_an_opaque_part_in_its_place() -> None:
    action = Action.from_anthropic(WEB_SEARCH_RESPONSE)

    assert action == Action(
        id="msg_a930390d3a",
        parts=(
            TextPart(text="I'll search for when Claude Shannon was born."),
            OpaquePart(kind="web_search_tool_result"),
            TextPart(text="Based on the search results, "),
            TextPart(text="Claude Shannon was born on April 30, 1916, in Petoskey, Michigan"),
        ),
        tool_calls=(WEB_SEARCH,),
    )


def test_the_published_tool_use_response_becomes_the_action_to_rate() -> None:
    action = Action.from_anthropic(TOOL_USE_RESPONSE)

    assert (action.parts, action.tool_calls, type(action.id)) == (
        (TextPart(text="I'll help you check the current weather and time in San Francisco."),),
        (GET_WEATHER,),
        str,
    )


@pytest.mark.parametrize(
    ("request_body", "path"),
    [
        ([], "$"),
        ({"messages": "Hello"}, "$"),
        ({"system": 42, "messages": []}, "$.system"),
        ({"messages": [{"role": "system", "content": "Hi"}]}, "$.messages[0]"),
        ({"messages": [{"role": "user", "content": 42}]}, "$.messages[0].content"),
        (
            {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
            "$.messages[0].content[0]",
        ),
        ({"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}, "$.messages[0].content[0]"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBO*Rw==",
                                },
                            }
                        ],
                    }
                ]
            },
            "$.messages[0].content[0].source.data",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "is_error": "yes"}
                        ],
                    }
                ]
            },
            "$.messages[0].content[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t1", "name": "read_file"}],
                    }
                ]
            },
            "$.messages[0].content[0]",
        ),
        (
            {"messages": [{"role": "assistant", "content": [{"type": "thinking"}]}]},
            "$.messages[0].content[0]",
        ),
        (
            {"messages": [TOOL_USE_RESPONSE, TOOL_USE_RESPONSE]},
            "$.messages[1].content[0]",
        ),
        ({"messages": [{"role": "assistant", "content": 42}]}, "$.messages[0].content"),
        ({"system": [{"type": "image"}], "messages": []}, "$.system[0]"),
        (
            build_user_request({"type": "tool_result", "content": "15 degrees"}),
            "$.messages[0].content[0]",
        ),
        (build_user_request(build_tool_result(42)), "$.messages[0].content[0].content"),
        (
            build_user_request(build_tool_result([{"text": "Hi"}])),
            "$.messages[0].content[0].content[0]",
        ),
        (build_user_request({"type": "image"}), "$.messages[0].content[0]"),
        (build_user_request({"type": "document"}), "$.messages[0].content[0]"),
        (build_user_request(build_image({"type": "base64", "data": "iVBORw=="})), SOURCE),
        (
            build_user_request(
                build_image({"type": "base64", "media_type": "png", "data": "iVBORw=="})
            ),
            SOURCE,
        ),
        (build_user_request(build_image({"url": "https://example.com/a.png"})), SOURCE),
        (build_user_request(build_document({"type": "url"})), SOURCE),
        (build_user_request(build_document({"data": "15 degrees"})), SOURCE),
        (
            build_user_request(
                build_document({"type": "base64", "media_type": "application/pdf", "data": "*"})
            ),
            f"{SOURCE}.data",
        ),
        (build_user_request(build_image({"type": "url"})), SOURCE),
        (
            build_user_request(
                build_document({"type": "base64", "media_type": "pdf", "data": "JVBERi0xLjcK"})
            ),
            SOURCE,
        ),
        (
            build_user_request(build_document({"type": "base64", "media_type": "application/pdf"})),
            SOURCE,
        ),
        (build_user_request(build_document({"type": "text", "media_type": "text/plain"})), SOURCE),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "server_tool_use", "id": "s1", "name": "web_search"}],
                    }
                ]
            },
            "$.messages[0].content[0]",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "s1",
                                "name": "web_search",
                                "input": {"query": "q3 revenue"},
                            },
                            {"type": "web_search_tool_result", "tool_use_id": "s1"},
                        ],
                    }
                ]
            },
            "$.messages[0].content[1]",
        ),
    ],
    ids=[
        "not-a-mapping",
        "messages-not-a-list",
        "system-not-text",
        "unknown-role",
        "content-not-text-or-blocks",
        "text-block-without-text",
        "block-without-type",
        "image-data-not-base64",
        "is-error-not-a-boolean",
        "tool-use-without-input",
        "thinking-without-text",
        "repeated-tool-use-id",
        "assistant-content-not-text-or-blocks",
        "system-block-not-text",
        "tool-result-without-tool-use-id",
        "tool-result-content-not-text-or-blocks",
        "tool-result-block-without-type",
        "image-without-source",
        "document-without-source",
        "base64-image-without-media-type",
        "base64-image-not-iana",
        "image-source-without-type",
        "url-document-without-url",
        "document-source-without-type",
        "document-data-not-base64",
        "url-image-without-url",
        "base64-document-not-iana",
        "base64-document-without-data",
        "text-document-without-data",
        "server-tool-use-without-input",
        "server-result-without-content",
    ],
)
def test_a_malformed_request_raises_format_error_at_its_path(request_body: JSON, path: str) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.from_anthropic(request_body)

    assert raised.value.path == path


@pytest.mark.parametrize(
    ("message", "path"),
    [
        ({"role": "user", "content": []}, "$"),
        ({"role": "assistant", "content": [{"type": "text"}]}, "$.content[0]"),
    ],
    ids=["not-an-assistant-message", "malformed-block"],
)
def test_a_malformed_response_raises_format_error_at_its_path(message: JSON, path: str) -> None:
    with pytest.raises(FormatError) as raised:
        Action.from_anthropic(message)

    assert raised.value.path == path
