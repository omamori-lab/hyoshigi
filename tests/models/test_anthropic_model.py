"""Anthropic's Messages API as a model: the request on the wire and the outcome of each reply."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic, DefaultAsyncHttpxClient, DefaultHttpxClient

from hyoshigi import (
    JSON,
    AsyncGenerativeModel,
    ConfigurationError,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaPart,
    MediaTypes,
    Prompt,
    Refused,
    TextPart,
    Truncated,
    Unavailable,
)

type Respond = Callable[[httpx2.Request], httpx2.Response]

MODEL_ID = "claude-opus-5-5"
MAX_TOKENS = 1024
# The media the API reads: https://platform.claude.com/docs/en/build-with-claude/vision,
# "Supported formats"; pdf-support; and citations, "Plain text documents" (same path).
PERCEIVES = MediaTypes(
    "image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf", "text/plain"
)
SYSTEM = "Today's date is 2024-06-01."

# https://platform.claude.com/docs/en/api/messages, "Example": the published request has
# max_tokens 1024, the system prompt above and the user message "Hello, world", and its
# response is the one below. The published response has stop_reason end_turn with a refusal in
# stop_details; the refusals page says to branch on stop_reason.
REFERENCE_RESPONSE: dict[str, JSON] = {
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "container": {
        "id": "container_011CpZohnwH4vuy7gazohgSP",
        "expires_at": "2019-12-27T18:11:19.117Z",
        "skills": [{"skill_id": "pdf", "type": "anthropic", "version": "latest"}],
    },
    "content": [
        {
            "citations": [
                {
                    "cited_text": "The grass is green. The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "end_char_index": 0,
                    "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
                    "start_char_index": 0,
                    "type": "char_location",
                }
            ],
            "text": "Hi! My name is Claude.",
            "type": "text",
        }
    ],
    "diagnostics": {"cache_miss_reason": {"cache_missed_input_tokens": 0, "type": "model_changed"}},
    "model": "claude-opus-5-5",
    "role": "assistant",
    "stop_details": {
        "category": "cyber",
        "explanation": "This request was declined because it conflicts with "
        "Anthropic's Usage Policy.",
        "type": "refusal",
    },
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "type": "message",
    "usage": {
        "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
        "cache_creation_input_tokens": 2051,
        "cache_read_input_tokens": 2051,
        "inference_geo": "global",
        "input_tokens": 2095,
        "output_tokens": 503,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_fetch_requests": 2, "web_search_requests": 0},
        "service_tier": "standard",
        "speed": "standard",
    },
}

# https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons, "The stop_reason
# field": the example response. The variants below set stop_reason to another of the values the
# page documents (https://platform.claude.com/docs/en/api/messages lists them) and, where a
# value needs it, replace the content with another published example.
STOP_REASONS_RESPONSE: dict[str, JSON] = {
    "id": "msg_01234",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Here's the answer to your question..."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "stop_details": None,
    "usage": {"input_tokens": 100, "output_tokens": 50},
}
ANSWER = "Here's the answer to your question..."

# https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use, "Model
# responses with tools".
TOOL_USE_CONTENT: JSON = [
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
]
TOOL_USE_TEXT = "I'll help you check the current weather and time in San Francisco."

# https://platform.claude.com/docs/en/build-with-claude/thinking: a response's thinking
# block before its text block.
THINKING_CONTENT: JSON = [
    {
        "type": "thinking",
        "thinking": "Let me break this down. The question has two parts, so I'll start with the "
        "simpler one and use its result to constrain the second...",
        "signature": "WaUjzkypQ2mUEVM36O2Txu....",
    },
    {"type": "text", "text": "Based on my analysis..."},
]

# https://platform.claude.com/docs/en/build-with-claude/thinking, "Redacted thinking blocks",
# before the text block above.
REDACTED_THINKING_CONTENT: JSON = [
    {"type": "redacted_thinking", "data": "..."},
    {"type": "text", "text": "Based on my analysis..."},
]

# https://platform.claude.com/docs/en/build-with-claude/citations, "Response structure": one
# sentence split across text blocks, some with citations.
CITED_CONTENT: JSON = [
    {"type": "text", "text": "According to the document, "},
    {
        "type": "text",
        "text": "the grass is green",
        "citations": [
            {
                "type": "char_location",
                "cited_text": "The grass is green.",
                "document_index": 0,
                "document_title": "Example Document",
                "start_char_index": 0,
                "end_char_index": 20,
            }
        ],
    },
    {"type": "text", "text": " and "},
    {
        "type": "text",
        "text": "the sky is blue",
        "citations": [
            {
                "type": "char_location",
                "cited_text": "The sky is blue.",
                "document_index": 0,
                "document_title": "Example Document",
                "start_char_index": 20,
                "end_char_index": 36,
            }
        ],
    },
    {"type": "text", "text": ". Information from page 5 states that "},
    {
        "type": "text",
        "text": "water is essential",
        "citations": [
            {
                "type": "page_location",
                "cited_text": "Water is essential for life.",
                "document_index": 1,
                "document_title": "PDF Document",
                "start_page_number": 5,
                "end_page_number": 6,
            }
        ],
    },
    {"type": "text", "text": ". The custom document mentions "},
    {
        "type": "text",
        "text": "important findings",
        "citations": [
            {
                "type": "content_block_location",
                "cited_text": "These are important findings.",
                "document_index": 2,
                "document_title": "Custom Content Document",
                "start_block_index": 0,
                "end_block_index": 1,
            }
        ],
    },
]
CITED_TEXT = (
    "According to the document, the grass is green and the sky is blue. "
    "Information from page 5 states that water is essential. "
    "The custom document mentions important findings"
)

# A block type the API adds after this adapter: only text blocks are the reply's text.
FUTURE_CONTENT: JSON = [
    {"type": "text", "text": "first "},
    {"type": "a_block_added_later", "text": "unseen "},
    {"type": "text", "text": "last"},
]

# https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback, "What a refusal
# looks like".
REFUSAL_RESPONSE: dict[str, JSON] = {
    "id": "msg_01XFUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_details": {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request was declined because it could enable cyber harm.",
    },
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
REFUSAL_EXPLANATION = "This request was declined because it could enable cyber harm."

# https://platform.claude.com/docs/en/api/errors, "Error shapes", and
# https://platform.claude.com/docs/en/build-with-claude/streaming, "Error events": the
# overloaded_error, which corresponds to HTTP 529.
NOT_FOUND_ERROR: JSON = {
    "type": "error",
    "error": {
        "type": "not_found_error",
        "message": "The requested resource could not be found.",
    },
    "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy",
}
NOT_FOUND_MESSAGE = "The requested resource could not be found."
OVERLOADED_ERROR: JSON = {
    "type": "error",
    "error": {"type": "overloaded_error", "message": "Overloaded"},
}

# https://platform.claude.com/docs/en/build-with-claude/vision, "Base64-encoded image" and
# "URL-based image"; the bytes are JPEG's FF D8 FF E0 00 10 "JFIF", whose base64 is the prefix
# of the page's example data.
VISION_URL = "https://platform.claude.com/docs/images/vision-example.jpg"
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF"
JPEG_BASE64 = "/9j/4AAQSkZJRg=="

# https://platform.claude.com/docs/en/build-with-claude/pdf-support, "Option 1: URL-based PDF
# document"; RFC 4648, section 10: the base64 of "foobar" is "Zm9vYmFy".
MODEL_CARD = "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf"
FOOBAR = b"foobar"
FOOBAR_BASE64 = "Zm9vYmFy"

# https://platform.claude.com/docs/en/build-with-claude/citations, "Plain text documents".
GRASS_AND_SKY = b"The grass is green. The sky is blue."

HELLO = Prompt(system=SYSTEM, content=(TextPart(text="Hello, world"),))


class Wire:
    """The Messages API's end of the connection: replays one response, records each request."""

    def __init__(self, body: JSON, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[JSON] = []

    def respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(self.status, json=self.body)


def fail_to_connect(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


def time_out(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ReadTimeout("timed out", request=request)


# What a proxy in front of the API can answer with status 200: an HTML page, or a JSON reply
# cut off.
def send_html(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200, text="<html>Bad gateway</html>", headers={"content-type": "text/html"}
    )


def send_cut_off_json(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200, text='{"id": "msg_01234", "ty', headers={"content-type": "application/json"}
    )


def send_invalid_utf8(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200, content=b'{"id": "\xff"}', headers={"content-type": "application/json"}
    )


def send_corrupt_gzip(request: httpx2.Request) -> httpx2.Response:
    headers = {"content-type": "application/json", "content-encoding": "gzip"}
    return httpx2.Response(200, content=b"not gzip", headers=headers)


@dataclass(frozen=True, slots=True)
class Driver:
    """A model built by one of the two families, driven to a result either way."""

    name: str
    perceives: MediaTypes
    generate: Callable[[Prompt], GenerationOutcome]


class BuildDriver(Protocol):
    def __call__(
        self,
        respond: Respond,
        *,
        model: str = MODEL_ID,
        max_tokens: int = MAX_TOKENS,
        perceives: MediaTypes = PERCEIVES,
        strict: bool = False,
    ) -> Driver: ...


def build_sync_driver(
    respond: Respond,
    *,
    model: str = MODEL_ID,
    max_tokens: int = MAX_TOKENS,
    perceives: MediaTypes = PERCEIVES,
    strict: bool = False,
) -> Driver:
    client = Anthropic(
        api_key="test-key",
        max_retries=0,
        _strict_response_validation=strict,
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(respond)),
    )
    built = GenerativeModel.from_anthropic(
        client, model=model, perceives=perceives, max_tokens=max_tokens
    )
    return Driver(name=built.name, perceives=built.perceives, generate=built.generate)


def build_async_driver(
    respond: Respond,
    *,
    model: str = MODEL_ID,
    max_tokens: int = MAX_TOKENS,
    perceives: MediaTypes = PERCEIVES,
    strict: bool = False,
) -> Driver:
    client = AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        _strict_response_validation=strict,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond)),
    )
    built = AsyncGenerativeModel.from_anthropic(
        client, model=model, perceives=perceives, max_tokens=max_tokens
    )

    def generate(prompt: Prompt) -> GenerationOutcome:
        return asyncio.run(await_outcome(built.generate(prompt)))

    return Driver(name=built.name, perceives=built.perceives, generate=generate)


async def await_outcome(outcome: Awaitable[GenerationOutcome]) -> GenerationOutcome:
    return await outcome


@pytest.fixture(params=[build_sync_driver, build_async_driver], ids=["sync", "async"])
def build_driver(request: pytest.FixtureRequest) -> BuildDriver:
    return request.param


def build_request(content: list[JSON]) -> JSON:
    return {
        "max_tokens": MAX_TOKENS,
        "model": MODEL_ID,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": content}],
    }


def test_the_model_is_named_for_the_provider_and_the_model_id(build_driver: BuildDriver) -> None:
    model = build_driver(Wire(REFERENCE_RESPONSE).respond)

    assert model.name == "anthropic:claude-opus-5-5"


def test_the_model_perceives_the_media_it_was_given(build_driver: BuildDriver) -> None:
    model = build_driver(Wire(REFERENCE_RESPONSE).respond)

    assert model.perceives == PERCEIVES


@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (("image/*",), "image/*"),
        (("image/svg+xml",), "image/svg+xml"),
        (("application/zip",), "application/zip"),
        (("video/mp4", "image/png", "audio/wav"), "audio/wav"),
    ],
    ids=["image-range", "image-type-outside-the-four", "zip", "the-first-of-those-it-cannot-carry"],
)
@pytest.mark.xfail(strict=True, reason="perceives is not checked when the model is built")
def test_a_model_that_perceives_media_the_api_cannot_carry_raises_naming_it_when_built(
    build_driver: BuildDriver, patterns: tuple[str, ...], named: str
) -> None:
    with pytest.raises(ConfigurationError, match=re.escape(named)):
        build_driver(Wire(REFERENCE_RESPONSE).respond, perceives=MediaTypes(*patterns))


@pytest.mark.parametrize(
    ("model", "max_tokens", "argument"),
    [("", MAX_TOKENS, "model"), (MODEL_ID, 0, "max_tokens"), (MODEL_ID, -1, "max_tokens")],
    ids=["no-model-id", "no-token", "negative"],
)
def test_a_model_without_a_model_id_or_a_token_to_write_raises_naming_the_argument_when_built(
    build_driver: BuildDriver, model: str, max_tokens: int, argument: str
) -> None:
    with pytest.raises(ConfigurationError, match=argument):
        build_driver(Wire(REFERENCE_RESPONSE).respond, model=model, max_tokens=max_tokens)


def test_a_model_may_write_a_single_token(build_driver: BuildDriver) -> None:
    built = build_driver(Wire(REFERENCE_RESPONSE).respond, max_tokens=1)

    assert built.name == "anthropic:claude-opus-5-5"


def test_a_prompt_is_sent_as_the_system_prompt_and_one_user_message(
    build_driver: BuildDriver,
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    model.generate(HELLO)

    assert wire.requests == [build_request([{"type": "text", "text": "Hello, world"}])]


@pytest.mark.parametrize(
    ("parts", "content"),
    [
        (
            (
                MediaPart(media_type="image/jpeg", data=JPEG_BYTES),
                TextPart(text="Describe this image."),
            ),
            [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": JPEG_BASE64},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        ),
        (
            (MediaPart(media_type="IMAGE/JPEG", data=JPEG_BYTES),),
            [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": JPEG_BASE64},
                }
            ],
        ),
        (
            (MediaPart(media_type="image/jpeg", url=VISION_URL),),
            [{"type": "image", "source": {"type": "url", "url": VISION_URL}}],
        ),
        (
            (MediaPart(media_type="image/*", url=VISION_URL),),
            [{"type": "image", "source": {"type": "url", "url": VISION_URL}}],
        ),
        (
            (MediaPart(media_type="application/pdf", data=FOOBAR),),
            [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": FOOBAR_BASE64,
                    },
                }
            ],
        ),
        (
            (MediaPart(media_type="application/pdf", url=MODEL_CARD),),
            [{"type": "document", "source": {"type": "url", "url": MODEL_CARD}}],
        ),
        (
            (MediaPart(media_type="APPLICATION/PDF", url=MODEL_CARD),),
            [{"type": "document", "source": {"type": "url", "url": MODEL_CARD}}],
        ),
        (
            (MediaPart(media_type="text/plain", data=GRASS_AND_SKY),),
            [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "The grass is green. The sky is blue.",
                    },
                }
            ],
        ),
    ],
    ids=[
        "image-bytes-then-text-in-order",
        "media-type-case-is-ignored-for-bytes",
        "image-url",
        "image-range-url",
        "pdf-bytes",
        "pdf-url",
        "media-type-case-is-ignored-for-a-url",
        "plain-text-bytes",
    ],
)
def test_media_is_sent_in_the_content_block_the_api_documents_for_it(
    build_driver: BuildDriver, parts: tuple[MediaPart | TextPart, ...], content: list[JSON]
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    model.generate(Prompt(system=SYSTEM, content=parts))

    assert wire.requests == [build_request(content)]


# https://platform.claude.com/docs/en/build-with-claude/vision, "Supported formats": JPEG, PNG,
# GIF and WebP.
@pytest.mark.parametrize("media_type", ["image/jpeg", "image/png", "image/gif", "image/webp"])
def test_each_image_type_the_api_supports_is_sent_as_base64_with_its_type(
    build_driver: BuildDriver, media_type: str
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    model.generate(Prompt(system=SYSTEM, content=(MediaPart(media_type=media_type, data=FOOBAR),)))

    source: JSON = {"type": "base64", "media_type": media_type, "data": FOOBAR_BASE64}
    assert wire.requests == [build_request([{"type": "image", "source": source}])]


@pytest.mark.parametrize("media_type", ["image/jpeg", "image/png", "image/gif", "image/webp"])
def test_each_image_type_the_api_supports_is_sent_by_url(
    build_driver: BuildDriver, media_type: str
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    model.generate(
        Prompt(system=SYSTEM, content=(MediaPart(media_type=media_type, url=VISION_URL),))
    )

    source: JSON = {"type": "url", "url": VISION_URL}
    assert wire.requests == [build_request([{"type": "image", "source": source}])]


def test_empty_text_is_not_sent(build_driver: BuildDriver) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)
    content = (TextPart(text=""), TextPart(text="Hello, world"), TextPart(text=""))

    model.generate(Prompt(system=SYSTEM, content=content))

    assert wire.requests == [build_request([{"type": "text", "text": "Hello, world"}])]


@pytest.mark.parametrize("content", [(TextPart(text=""),), ()], ids=["empty-text", "no-content"])
def test_a_prompt_without_content_to_send_is_unavailable_before_any_request(
    build_driver: BuildDriver, content: tuple[TextPart, ...]
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    outcome = model.generate(Prompt(system=SYSTEM, content=content))

    assert (outcome, wire.requests) == (Unavailable(status=None), [])


# json.loads gives a lone surrogate for "\ud800" in a tool's output.
@pytest.mark.parametrize(
    "prompt",
    [
        Prompt(system=SYSTEM, content=(TextPart(text="a\ud800b"),)),
        Prompt(system="a\ud800b", content=(TextPart(text="Hello, world"),)),
    ],
    ids=["text", "system"],
)
def test_text_that_cannot_be_encoded_is_unavailable_before_any_request(
    build_driver: BuildDriver, prompt: Prompt
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    outcome = model.generate(prompt)

    assert (outcome, wire.requests) == (Unavailable(status=None), [])


@pytest.mark.parametrize(
    "part",
    [
        MediaPart(media_type="audio/wav", data=FOOBAR),
        MediaPart(media_type="video/mp4", url="https://example.com/clip.mp4"),
        MediaPart(media_type="video/*", url="https://example.com/clip"),
        MediaPart(media_type="image/svg+xml", data=FOOBAR),
        MediaPart(media_type="image/svg+xml", url="https://example.com/figure.svg"),
        MediaPart(media_type="application/zip", data=FOOBAR),
    ],
    ids=["audio", "video-url", "video-range", "svg-bytes", "svg-url", "zip"],
)
def test_a_media_type_the_api_cannot_carry_raises_naming_the_type_before_any_request(
    build_driver: BuildDriver, part: MediaPart
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    with pytest.raises(ConfigurationError, match=re.escape(part.media_type)):
        model.generate(Prompt(system=SYSTEM, content=(TextPart(text="Rate this."), part)))

    assert wire.requests == []


def test_a_media_type_the_api_cannot_carry_raises_after_media_it_cannot_take_in_its_form(
    build_driver: BuildDriver,
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)
    notes = MediaPart(media_type="text/plain", url="https://example.com/notes.txt")
    prompt = Prompt(system=SYSTEM, content=(notes, MediaPart(media_type="audio/wav", data=FOOBAR)))

    with pytest.raises(ConfigurationError, match=re.escape("audio/wav")):
        model.generate(prompt)


@pytest.mark.parametrize(
    "part",
    [
        MediaPart(media_type="text/plain", url="https://example.com/notes.txt"),
        MediaPart(media_type="image/*", data=FOOBAR),
        MediaPart(media_type="text/plain", data=b"\xff\xfe"),
    ],
    ids=["plain-text-url", "image-range-bytes", "plain-text-not-utf-8"],
)
def test_media_in_a_form_the_api_cannot_take_is_unavailable_before_any_request(
    build_driver: BuildDriver, part: MediaPart
) -> None:
    wire = Wire(REFERENCE_RESPONSE)
    model = build_driver(wire.respond)

    outcome = model.generate(Prompt(system=SYSTEM, content=(TextPart(text="Rate this."), part)))

    assert (outcome, wire.requests) == (Unavailable(status=None), [])


@pytest.mark.parametrize(
    ("body", "text"),
    [
        (REFERENCE_RESPONSE, "Hi! My name is Claude."),
        (
            {
                **STOP_REASONS_RESPONSE,
                "stop_reason": "stop_sequence",
                "stop_sequence": "\n\nHuman:",
            },
            ANSWER,
        ),
    ],
    ids=["end-turn-despite-a-refusal-in-stop-details", "stop-sequence"],
)
def test_a_reply_that_stopped_on_its_own_is_a_generation_of_its_text(
    build_driver: BuildDriver, body: JSON, text: str
) -> None:
    model = build_driver(Wire(body).respond)

    outcome = model.generate(HELLO)

    assert outcome == Generation(text=text)


def test_the_text_blocks_of_a_reply_are_joined_without_a_separator(
    build_driver: BuildDriver,
) -> None:
    model = build_driver(Wire({**STOP_REASONS_RESPONSE, "content": CITED_CONTENT}).respond)

    outcome = model.generate(HELLO)

    assert outcome == Generation(text=CITED_TEXT)


@pytest.mark.parametrize(
    ("content", "text"),
    [
        (THINKING_CONTENT, "Based on my analysis..."),
        (REDACTED_THINKING_CONTENT, "Based on my analysis..."),
        (FUTURE_CONTENT, "first last"),
    ],
    ids=["thinking", "redacted-thinking", "block-type-added-later"],
)
def test_blocks_that_are_not_text_are_not_part_of_the_reply(
    build_driver: BuildDriver, content: JSON, text: str
) -> None:
    model = build_driver(Wire({**STOP_REASONS_RESPONSE, "content": content}).respond)

    outcome = model.generate(HELLO)

    assert outcome == Generation(text=text)


@pytest.mark.parametrize(
    "body",
    [
        {**STOP_REASONS_RESPONSE, "stop_reason": "max_tokens"},
        {**STOP_REASONS_RESPONSE, "stop_reason": "pause_turn"},
        {**STOP_REASONS_RESPONSE, "stop_reason": "model_context_window_exceeded"},
        {**STOP_REASONS_RESPONSE, "stop_reason": "a_stop_reason_added_later"},
        {**STOP_REASONS_RESPONSE, "stop_reason": None},
    ],
    ids=["max-tokens", "pause-turn", "context-window", "added-later", "none"],
)
def test_a_reply_that_stopped_for_another_reason_is_truncated_with_its_text_so_far(
    build_driver: BuildDriver, body: JSON
) -> None:
    model = build_driver(Wire(body).respond)

    outcome = model.generate(HELLO)

    assert outcome == Truncated(text=ANSWER)


def test_a_reply_that_stopped_to_use_a_tool_is_truncated_with_the_text_before_the_call(
    build_driver: BuildDriver,
) -> None:
    body = {**STOP_REASONS_RESPONSE, "content": TOOL_USE_CONTENT, "stop_reason": "tool_use"}
    model = build_driver(Wire(body).respond)

    outcome = model.generate(HELLO)

    assert outcome == Truncated(text=TOOL_USE_TEXT)


def test_a_refusal_is_refused_with_the_explanation_the_api_gives(build_driver: BuildDriver) -> None:
    model = build_driver(Wire(REFUSAL_RESPONSE).respond)

    outcome = model.generate(HELLO)

    assert outcome == Refused(explanation=REFUSAL_EXPLANATION)


# https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback: category and
# explanation are both null when the refusal does not map to a named category, and
# stop_details is null for every stop reason other than refusal.
@pytest.mark.parametrize(
    "stop_details",
    [{"type": "refusal", "category": None, "explanation": None}, None],
    ids=["no-explanation", "no-details"],
)
def test_a_refusal_without_an_explanation_is_refused_without_one(
    build_driver: BuildDriver, stop_details: JSON
) -> None:
    body = {**REFUSAL_RESPONSE, "stop_details": stop_details}
    model = build_driver(Wire(body).respond)

    outcome = model.generate(HELLO)

    assert outcome == Refused(explanation=None)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": "Here is a string"}, id="content-is-a-string"
        ),
        pytest.param({"id": "msg_01234", "type": "message", "role": "assistant"}, id="no-content"),
        pytest.param("a string", id="not-a-message"),
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": [{"type": "text"}]}, id="text-block-without-text"
        ),
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": [{"type": "text", "text": 123}]},
            id="text-is-not-a-string",
        ),
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": ["not a block"]},
            id="element-is-not-a-block",
        ),
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": [{"text": "missing type"}]},
            id="block-without-type",
        ),
        pytest.param(
            {**STOP_REASONS_RESPONSE, "content": [{"type": 123, "text": "bad type"}]},
            id="block-type-is-not-a-string",
        ),
        pytest.param({"stop_reason": "end_turn", "content": []}, id="no-type-or-role"),
        pytest.param({**STOP_REASONS_RESPONSE, "type": "error"}, id="type-is-not-message"),
        pytest.param({**STOP_REASONS_RESPONSE, "role": "user"}, id="role-is-not-assistant"),
        pytest.param({**REFUSAL_RESPONSE, "type": "error"}, id="refusal-is-not-a-message"),
    ],
)
def test_a_reply_that_is_not_a_message_is_unavailable_without_a_status(
    build_driver: BuildDriver, body: JSON
) -> None:
    model = build_driver(Wire(body).respond)

    outcome = model.generate(HELLO)

    assert outcome == Unavailable(status=None)


def test_a_reply_the_client_rejects_as_invalid_is_unavailable_without_a_status(
    build_driver: BuildDriver,
) -> None:
    model = build_driver(Wire({"type": "message"}).respond, strict=True)

    outcome = model.generate(HELLO)

    assert outcome == Unavailable(status=None)


# https://platform.claude.com/docs/en/api/errors, "HTTP errors": 409, 413, 429, 500, 504 and
# 529 are documented; 408 and 503 are the statuses the SDK retries and names. 413, a request
# too large, is a status the prompt's content can cause. 301 and 304 are statuses that are not
# a success and not an error of the request.
@pytest.mark.parametrize(
    "status",
    [
        301,
        304,
        408,
        409,
        413,
        429,
        500,
        503,
        504,
        529,
    ],
)
def test_a_status_that_is_not_a_configuration_error_is_unavailable_with_its_status(
    build_driver: BuildDriver, status: int
) -> None:
    wire = Wire(OVERLOADED_ERROR, status=status)
    model = build_driver(wire.respond)

    outcome = model.generate(HELLO)

    assert (outcome, len(wire.requests)) == (Unavailable(status=status), 1)


@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(fail_to_connect, id="connection"),
        pytest.param(time_out, id="timeout"),
        pytest.param(send_html, id="html"),
        pytest.param(send_cut_off_json, id="cut-off-json"),
        pytest.param(send_invalid_utf8, id="invalid-utf-8"),
        pytest.param(send_corrupt_gzip, id="corrupt-gzip"),
    ],
)
def test_a_connection_that_failed_or_a_reply_that_is_not_json_is_unavailable_without_a_status(
    build_driver: BuildDriver, respond: Respond
) -> None:
    model = build_driver(respond)

    outcome = model.generate(HELLO)

    assert outcome == Unavailable(status=None)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_any_other_error_status_raises_naming_the_status(
    build_driver: BuildDriver, status: int
) -> None:
    model = build_driver(Wire(NOT_FOUND_ERROR, status=status).respond)

    with pytest.raises(ConfigurationError, match=str(status)):
        model.generate(HELLO)


def test_the_error_for_a_status_does_not_carry_the_text_the_api_sent(
    build_driver: BuildDriver,
) -> None:
    model = build_driver(Wire(NOT_FOUND_ERROR, status=404).respond)

    with pytest.raises(ConfigurationError) as raised:
        model.generate(HELLO)

    error = raised.value
    assert (NOT_FOUND_MESSAGE in str(error), error.__cause__, error.__context__) == (
        False,
        None,
        None,
    )
