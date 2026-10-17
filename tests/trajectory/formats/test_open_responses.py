"""The Open Responses format."""

import pytest

from hyoshigi import (
    JSON,
    Action,
    Author,
    Authorship,
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

# https://www.openresponses.org/specification/2026-04-24: the examples of a message, a
# function call, reasoning, a provider's item, a WebSocket request and its continuation,
# and the user message of the allowed_tools example.
SPEC_MESSAGE: JSON = {
    "type": "message",
    "id": "msg_01A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Hello! How can I assist you today?"}],
}
SPEC_FUNCTION_CALL: JSON = {
    "type": "function_call",
    "id": "fc_00123xyzabc9876def0gh1ij2klmno345",
    "name": "sendEmail",
    "call_id": "call_987zyx654wvu321",
    "arguments": '{"recipient":"jane.doe@example.com","subject":"Meeting Reminder",'
    '"body":"Don\'t forget about our meeting tomorrow at 10am."}',
}
SEND_EMAIL = ToolCall(
    id="call_987zyx654wvu321",
    name="sendEmail",
    arguments='{"recipient":"jane.doe@example.com","subject":"Meeting Reminder",'
    '"body":"Don\'t forget about our meeting tomorrow at 10am."}',
)
RESTAURANT_SUMMARY = (
    "Determined that the user is asking for restaurant recommendations in a specific city, "
    "then filtered options by distance and rating before selecting three suggestions."
)
RESTAURANT_REASONING = (
    'User asked: "Where should I eat dinner in downtown Seattle tonight?"↵↵'
    "1. Interpreted task as a search for nearby restaurants suitable for dinner.↵"
    "2. Prioritized places within a short walking radius of downtown.↵"
    "3. Filtered candidates by overall rating and number of reviews.↵"
    "4. Preferred options with vegetarian choices and moderate price range.↵"
    "5. Selected three restaurants that best match the inferred preferences."
)
SPEC_REASONING: JSON = {
    "type": "reasoning",
    "status": "completed",
    "summary": [{"type": "summary_text", "text": RESTAURANT_SUMMARY}],
    "content": [{"type": "output_text", "text": RESTAURANT_REASONING}],
    "encrypted_content": None,
}
WEB_SEARCH_CALL: JSON = {
    "id": "ws_0df093a2d268cd7f0068efe79ac9408190b9833ec01e5d05ed",
    "type": "openai:web_search_call",
    "status": "completed",
    "action": {"type": "search", "query": "weather: San Francisco, CA"},
}
WEBSOCKET_REQUEST: JSON = {
    "type": "response.create",
    "model": "gpt-openresponses-1",
    "store": False,
    "input": [{"type": "message", "role": "user", "content": "Find the bug in fizz_buzz()."}],
    "tools": [],
}
WEBSOCKET_CONTINUATION: JSON = {
    "type": "response.create",
    "model": "gpt-openresponses-1",
    "store": False,
    "previous_response_id": "resp_123",
    "input": [
        {"type": "function_call_output", "call_id": "call_123", "output": "tool result"},
        {"type": "message", "role": "user", "content": "Now optimize it."},
    ],
}
SALES_QUESTION: JSON = {
    "role": "user",
    "content": [
        {
            "type": "input_text",
            "text": "Summarize the latest sales data and then draft a follow-up email.",
        }
    ],
}

# https://www.openresponses.org/openapi/2026-04-24/openapi.json: the output of the
# ResponseResource and CompactResource examples.
RESPONSE_OUTPUT: JSON = [
    {
        "type": "message",
        "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "The image depicts a scenic landscape with a wooden boardwalk or pathway "
                "leading through lush, green grass under a blue sky with some clouds. The "
                "setting suggests a peaceful natural area, possibly a park or nature reserve. "
                "There are trees and shrubs in the background.",
                "annotations": [],
            }
        ],
    }
]
COMPACTED_OUTPUT: JSON = [
    {
        "id": "msg_000",
        "type": "message",
        "status": "completed",
        "content": [
            {"type": "input_text", "text": "Create a simple landing page for a dog petting cafe."}
        ],
        "role": "user",
    },
    {"id": "cmp_001", "type": "compaction", "encrypted_content": "gAAAAABpM0Yj-...="},
]

WEATHER_QUESTION: JSON = {"type": "message", "role": "user", "content": "Weather in Paris?"}
FOLLOW_UP: JSON = {"type": "message", "role": "user", "content": "And tomorrow?"}
NUDGE = "It's 9am: run the report"
SCHEDULER_NUDGE: JSON = {"type": "message", "role": "user", "content": NUDGE}
GET_WEATHER: JSON = {
    "type": "function_call",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city": "Paris"}',
}


def build_user_request(*parts: JSON) -> JSON:
    return {"input": [{"type": "message", "role": "user", "content": list(parts)}]}


def build_input_request(*items: JSON) -> JSON:
    return {"input": list(items)}


def attribute_nudges_to_the_scheduler(index: int, *, message: JSON) -> Author:
    return Author.SCAFFOLD if message in (SCHEDULER_NUDGE, NUDGE) else Author.USER


def test_the_published_websocket_request_becomes_the_user_s_message() -> None:
    trajectory = Trajectory.from_open_responses(WEBSOCKET_REQUEST)

    assert trajectory.history == (
        Observation(author=Author.USER, parts=(TextPart(text="Find the bug in fizz_buzz()."),)),
    )


def test_the_published_continuation_marks_the_previous_response_then_reads_its_input() -> None:
    trajectory = Trajectory.from_open_responses(WEBSOCKET_CONTINUATION)

    assert trajectory.history == (
        Observation(
            author=Author.UNKNOWN,
            source="previous_response",
            parts=(OpaquePart(kind="previous_response"),),
        ),
        ToolObservation(call_id="call_123", parts=(TextPart(text="tool result"),)),
        Observation(author=Author.USER, parts=(TextPart(text="Now optimize it."),)),
    )


@pytest.mark.parametrize(
    ("message", "event"),
    [
        (
            SALES_QUESTION,
            Observation(
                author=Author.USER,
                parts=(
                    TextPart(
                        text="Summarize the latest sales data and then draft a follow-up email."
                    ),
                ),
            ),
        ),
        (
            {"role": "assistant", "content": "Done."},
            Action(id="$.input[0]", parts=(TextPart(text="Done."),)),
        ),
    ],
    ids=["published-user-message", "assistant-message"],
)
def test_a_message_without_a_type_is_read_by_its_role(message: JSON, event: Event) -> None:
    trajectory = Trajectory.from_open_responses(build_input_request(message))

    assert trajectory.history == (event,)


def test_the_published_compacted_window_keeps_the_compaction_opaque() -> None:
    trajectory = Trajectory.from_open_responses({"input": COMPACTED_OUTPUT})

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(TextPart(text="Create a simple landing page for a dog petting cafe."),),
        ),
        Observation(
            author=Author.AGENT, source="compaction", parts=(OpaquePart(kind="compaction"),)
        ),
    )


def test_the_published_response_becomes_the_action_to_rate() -> None:
    action = Action.from_open_responses(RESPONSE_OUTPUT)

    assert (action.thought, action.parts, action.tool_calls, type(action.id)) == (
        (),
        (
            TextPart(
                text="The image depicts a scenic landscape with a wooden boardwalk or pathway "
                "leading through lush, green grass under a blue sky with some clouds. The "
                "setting suggests a peaceful natural area, possibly a park or nature reserve. "
                "There are trees and shrubs in the background."
            ),
        ),
        (),
        str,
    )


def test_the_published_message_and_function_call_are_one_action_to_rate() -> None:
    action = Action.from_open_responses([SPEC_MESSAGE, SPEC_FUNCTION_CALL])

    assert (action.parts, action.tool_calls) == (
        (TextPart(text="Hello! How can I assist you today?"),),
        (SEND_EMAIL,),
    )


def test_the_published_reasoning_becomes_the_thought_content_then_summary() -> None:
    action = Action.from_open_responses([SPEC_REASONING])

    assert action.thought == (
        TextPart(text=RESTAURANT_REASONING),
        TextPart(text=RESTAURANT_SUMMARY),
    )


def test_encrypted_reasoning_is_an_opaque_part_of_the_thought() -> None:
    action = Action.from_open_responses(
        [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Checked the forecast."}],
                "encrypted_content": "gAAAAABpM0Yj-...=",
            }
        ]
    )

    assert action.thought == (
        TextPart(text="Checked the forecast."),
        OpaquePart(kind="reasoning"),
    )


@pytest.mark.parametrize(
    ("item", "kind"),
    [
        (WEB_SEARCH_CALL, "openai:web_search_call"),
        (COMPACTED_OUTPUT[1], "compaction"),
        (
            {"type": "function_call_output", "call_id": "call_1", "output": "18 C"},
            "function_call_output",
        ),
    ],
    ids=["provider-item", "compaction", "tool-output"],
)
def test_an_item_the_model_did_not_write_is_an_opaque_part_of_the_action_to_rate(
    item: JSON, kind: str
) -> None:
    action = Action.from_open_responses([item, SPEC_MESSAGE])

    assert action.parts == (
        OpaquePart(kind=kind),
        TextPart(text="Hello! How can I assist you today?"),
    )


def test_consecutive_model_items_are_one_action_named_by_its_first_item() -> None:
    request: JSON = {
        "instructions": "You are a weather assistant.",
        "input": [
            WEATHER_QUESTION,
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "The user wants the weather."}],
            },
            GET_WEATHER,
            {"type": "function_call_output", "call_id": "call_1", "output": "18 C, light rain"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "18 C and light rain."}],
            },
        ],
    }

    trajectory = Trajectory.from_open_responses(request)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            source="instructions",
            parts=(TextPart(text="You are a weather assistant."),),
        ),
        Observation(author=Author.USER, parts=(TextPart(text="Weather in Paris?"),)),
        Action(
            id="$.input[1]",
            thought=(TextPart(text="The user wants the weather."),),
            tool_calls=(ToolCall(id="call_1", name="get_weather", arguments='{"city": "Paris"}'),),
        ),
        ToolObservation(call_id="call_1", parts=(TextPart(text="18 C, light rain"),)),
        Action(id="$.input[4]", parts=(TextPart(text="18 C and light rain."),)),
    )


def test_a_provider_s_item_in_the_history_is_an_unknown_author_s_observation() -> None:
    request = build_input_request(SPEC_REASONING, WEB_SEARCH_CALL, SPEC_MESSAGE)

    trajectory = Trajectory.from_open_responses(request)

    assert trajectory.history == (
        Action(
            id="$.input[0]",
            thought=(TextPart(text=RESTAURANT_REASONING), TextPart(text=RESTAURANT_SUMMARY)),
        ),
        Observation(author=Author.UNKNOWN, parts=(OpaquePart(kind="openai:web_search_call"),)),
        Action(id="$.input[2]", parts=(TextPart(text="Hello! How can I assist you today?"),)),
    )


@pytest.mark.parametrize(
    "reference",
    [
        {"type": "item_reference", "id": "msg_123"},
        {"id": "msg_123"},
        {"type": None, "id": "msg_123"},
    ],
    ids=["typed", "without-type", "null-type"],
)
def test_an_item_reference_is_an_unknown_author_s_observation(reference: JSON) -> None:
    trajectory = Trajectory.from_open_responses(build_input_request(reference))

    assert trajectory.history == (
        Observation(author=Author.UNKNOWN, parts=(OpaquePart(kind="item_reference"),)),
    )


def test_a_request_without_input_is_its_instructions() -> None:
    trajectory = Trajectory.from_open_responses({"instructions": "Answer in French."})

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD,
            source="instructions",
            parts=(TextPart(text="Answer in French."),),
        ),
    )


@pytest.mark.parametrize(
    ("authorship", "text", "event"),
    [
        (marked({0}), "Hello!", Observation(author=Author.USER, parts=(TextPart(text="Hello!"),))),
        (
            attribute_nudges_to_the_scheduler,
            NUDGE,
            Observation(author=Author.SCAFFOLD, parts=(TextPart(text="It's 9am: run the report"),)),
        ),
    ],
    ids=["marked-first", "harness"],
)
def test_a_string_input_is_the_first_message_with_the_authorship_s_author(
    authorship: Authorship, text: str, event: Event
) -> None:
    trajectory = Trajectory.from_open_responses({"input": text}, authorship=authorship)

    assert trajectory.history == (event,)


def test_marked_authorship_makes_only_the_marked_message_the_user_s() -> None:
    request = build_input_request(WEATHER_QUESTION, FOLLOW_UP)

    trajectory = Trajectory.from_open_responses(request, authorship=marked({1}))

    assert trajectory.history == (
        Observation(author=Author.UNKNOWN, parts=(TextPart(text="Weather in Paris?"),)),
        Observation(author=Author.USER, parts=(TextPart(text="And tomorrow?"),)),
    )


def test_a_harness_s_own_authorship_reads_each_user_message() -> None:
    request = build_input_request(SCHEDULER_NUDGE)

    trajectory = Trajectory.from_open_responses(
        request, authorship=attribute_nudges_to_the_scheduler
    )

    assert trajectory.history == (
        Observation(author=Author.SCAFFOLD, parts=(TextPart(text="It's 9am: run the report"),)),
    )


def test_scaffold_messages_and_tool_output_parts_keep_their_authors() -> None:
    request = build_input_request(
        {"type": "message", "role": "system", "content": "Answer in French."},
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": "Be brief."}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "Radar:"},
                {"type": "input_image", "image_url": "https://example.com/radar.png"},
            ],
        },
    )

    trajectory = Trajectory.from_open_responses(request)

    assert trajectory.history == (
        Observation(
            author=Author.SCAFFOLD, source="system", parts=(TextPart(text="Answer in French."),)
        ),
        Observation(
            author=Author.SCAFFOLD, source="developer", parts=(TextPart(text="Be brief."),)
        ),
        ToolObservation(
            call_id="call_1",
            parts=(
                TextPart(text="Radar:"),
                MediaPart(media_type="image/*", url="https://example.com/radar.png"),
            ),
        ),
    )


def test_a_refusal_and_an_unknown_part_are_what_the_assistant_said() -> None:
    action = Action.from_open_responses(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "I can't help with that."},
                    {"type": "output_audio", "id": "audio_1"},
                ],
            }
        ]
    )

    assert action.parts == (
        TextPart(text="I can't help with that."),
        OpaquePart(kind="output_audio"),
    )


# RFC 2397: a data URL carries its media type and its bytes; bare base64 names no type.
def test_media_parts_become_media_or_opaque_parts() -> None:
    request = build_user_request(
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
        {"type": "input_image", "image_url": "https://example.com/radar.png", "detail": "low"},
        {"type": "input_image", "image_url": None},
        {"type": "input_image", "image_url": "iVBORw=="},
        {"type": "input_file", "file_data": "data:application/pdf;base64,JVBERi0xLjcK"},
        {"type": "input_file", "filename": "q3.pdf", "file_data": "JVBERi0xLjcK"},
        {"type": "input_file", "file_url": "https://example.com/q3.pdf"},
        {"type": "input_video", "video_url": "https://example.com/storm.mp4"},
        {"type": "input_video", "video_url": "data:video/mp4;base64,AAAAIGZ0eXA="},
        {"type": "input_video", "video_url": "AAAAIGZ0eXA="},
        {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
    )

    trajectory = Trajectory.from_open_responses(request)

    assert trajectory.history == (
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(media_type="image/png", data=b"\x89PNG"),
                MediaPart(media_type="image/*", url="https://example.com/radar.png"),
                OpaquePart(kind="input_image"),
                OpaquePart(kind="input_image"),
                MediaPart(media_type="application/pdf", data=b"%PDF-1.7\n"),
                OpaquePart(kind="input_file"),
                OpaquePart(kind="input_file"),
                MediaPart(media_type="video/*", url="https://example.com/storm.mp4"),
                MediaPart(media_type="video/mp4", data=b"\x00\x00\x00 ftyp"),
                OpaquePart(kind="input_video"),
                OpaquePart(kind="input_audio"),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("request_body", "path"),
    [
        ([], "$"),
        ({"instructions": 42}, "$.instructions"),
        ({"previous_response_id": 42}, "$.previous_response_id"),
        ({"input": 42}, "$.input"),
        (build_input_request({"content": "Hi"}), "$.input[0]"),
        (
            build_input_request({"type": 42, "id": "msg_1", "role": "user", "content": "Hi"}),
            "$.input[0]",
        ),
        (build_input_request({"type": "message", "content": "Hi"}), "$.input[0]"),
        (build_input_request({"type": "message", "role": "critic", "content": "Hi"}), "$.input[0]"),
        (build_input_request({"type": "message", "role": "user", "content": 42}), "$.input[0]"),
        (build_input_request({"type": "message", "role": "assistant"}), "$.input[0]"),
        (
            build_input_request({"type": "function_call", "call_id": "c1", "name": "f"}),
            "$.input[0]",
        ),
        (build_input_request({"type": "function_call_output", "output": "ok"}), "$.input[0]"),
        (
            build_input_request({"type": "function_call_output", "call_id": "c1", "output": 42}),
            "$.input[0]",
        ),
        (build_input_request(GET_WEATHER, FOLLOW_UP, GET_WEATHER), "$.input[2]"),
        (build_input_request({"type": "reasoning"}), "$.input[0]"),
        (
            build_input_request({"type": "reasoning", "summary": [{"type": "summary_text"}]}),
            "$.input[0].summary[0]",
        ),
        (
            build_input_request({"type": "reasoning", "summary": [], "content": "x"}),
            "$.input[0].content",
        ),
        (
            build_input_request({"type": "reasoning", "summary": [], "encrypted_content": 42}),
            "$.input[0].encrypted_content",
        ),
        (
            build_input_request(
                {"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}
            ),
            "$.input[0].content[0]",
        ),
        (
            build_input_request(
                {"type": "message", "role": "assistant", "content": [{"text": "Hi"}]}
            ),
            "$.input[0].content[0]",
        ),
        (build_user_request({"text": "Hi"}), "$.input[0].content[0]"),
        (build_user_request({"type": "input_text"}), "$.input[0].content[0]"),
        (build_user_request({"type": "input_image", "image_url": 42}), "$.input[0].content[0]"),
        (
            build_user_request({"type": "input_image", "image_url": "data:image/png;base64,*"}),
            "$.input[0].content[0].image_url",
        ),
        (build_user_request({"type": "input_video", "video_url": 42}), "$.input[0].content[0]"),
        (build_user_request({"type": "input_file", "file_data": 42}), "$.input[0].content[0]"),
        (
            build_user_request(
                {"type": "input_file", "file_data": "data:application/pdf;base64,*"}
            ),
            "$.input[0].content[0].file_data",
        ),
        (
            build_input_request({"type": "message", "role": "system", "content": [{"text": "Hi"}]}),
            "$.input[0].content[0]",
        ),
        (
            build_input_request(
                {"type": "function_call_output", "call_id": "c1", "output": [{"text": "ok"}]}
            ),
            "$.input[0].output[0]",
        ),
        (
            build_input_request(
                {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text"}]}
            ),
            "$.input[0].content[0]",
        ),
        (
            build_user_request({"type": "input_video", "video_url": "data:video/mp4;base64,*"}),
            "$.input[0].content[0].video_url",
        ),
    ],
    ids=[
        "not-a-mapping",
        "instructions-not-text",
        "previous-response-id-not-text",
        "input-not-text-or-items",
        "item-without-type-role-or-id",
        "type-not-text",
        "message-without-role",
        "unknown-role",
        "user-content-not-text-or-parts",
        "assistant-message-without-content",
        "function-call-without-arguments",
        "function-call-output-without-call-id",
        "function-call-output-not-text-or-parts",
        "repeated-call-id",
        "reasoning-without-summary",
        "summary-text-without-text",
        "reasoning-content-not-a-list",
        "encrypted-content-not-text",
        "refusal-part-without-refusal",
        "assistant-part-without-type",
        "user-part-without-type",
        "input-text-without-text",
        "image-url-not-text",
        "image-data-url-not-base64",
        "video-url-not-text",
        "file-data-not-text",
        "file-data-url-not-base64",
        "scaffold-part-without-type",
        "tool-output-part-without-type",
        "reasoning-text-without-text",
        "video-data-url-not-base64",
    ],
)
def test_a_malformed_request_raises_format_error_at_its_path(request_body: JSON, path: str) -> None:
    with pytest.raises(FormatError) as raised:
        Trajectory.from_open_responses(request_body)

    assert raised.value.path == path


@pytest.mark.parametrize(
    ("output", "path"),
    [
        ({"output": RESPONSE_OUTPUT}, "$"),
        ([{"role": "assistant", "content": "Hi"}], "$[0]"),
        (
            [{"type": "message", "role": "assistant", "content": [{"type": "output_text"}]}],
            "$[0].content[0]",
        ),
        ([{"type": "message", "content": "Hi"}], "$[0]"),
        ([{"type": "message", "role": "user", "content": "Hi"}], "$[0]"),
        pytest.param(
            [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "f",
                    "arguments": 1,
                    "content": "x",
                }
            ],
            "$[0]",
            marks=pytest.mark.xfail(strict=True, reason="a function call's content reads as text"),
        ),
    ],
    ids=[
        "not-a-list",
        "item-without-type",
        "malformed-part",
        "message-without-role",
        "message-not-the-assistant-s",
        "function-call-with-content",
    ],
)
def test_a_malformed_output_raises_format_error_at_its_path(output: JSON, path: str) -> None:
    with pytest.raises(FormatError) as raised:
        Action.from_open_responses(output)

    assert raised.value.path == path
