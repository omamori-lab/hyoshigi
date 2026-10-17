"""OpenAI's Responses API as a model: the request the SDK sends and the outcome of each reply."""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx2
import pytest
from openai import AsyncOpenAI, OpenAI

from hyoshigi import (
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

MODEL = "gpt-6-astra"
API_KEY = "stand-in"
TEXT_ONLY = MediaTypes()


@dataclass(frozen=True, slots=True, kw_only=True)
class Exchange:
    """What a model returned for a prompt, and the JSON request bodies the SDK sent."""

    model: GenerativeModel | AsyncGenerativeModel
    outcome: GenerationOutcome
    requests: list[dict[str, object]]


class Ask(Protocol):
    """Gives a prompt to a model on a stand-in for OpenAI's API that serves one reply."""

    def __call__(
        self,
        reply: httpx2.Response | Exception,
        prompt: Prompt = ...,
        *,
        max_output_tokens: int | None = None,
        perceives: MediaTypes = ...,
    ) -> Exchange: ...


PROMPT = Prompt(system="You are a helpful assistant.", content=(TextPart(text="Hello!"),))


def respond(body: object, *, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=body)


def serve(
    reply: httpx2.Response | Exception, requests: list[dict[str, object]]
) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        if isinstance(reply, Exception):
            raise reply
        return reply

    return httpx2.MockTransport(handle)


def generate_synchronously(
    reply: httpx2.Response | Exception,
    prompt: Prompt = PROMPT,
    *,
    max_output_tokens: int | None = None,
    perceives: MediaTypes = TEXT_ONLY,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = OpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.Client(transport=serve(reply, requests)),
    )
    model = GenerativeModel.from_openai(
        client, model=MODEL, max_output_tokens=max_output_tokens, perceives=perceives
    )
    return Exchange(model=model, outcome=model.generate(prompt), requests=requests)


def generate_asynchronously(
    reply: httpx2.Response | Exception,
    prompt: Prompt = PROMPT,
    *,
    max_output_tokens: int | None = None,
    perceives: MediaTypes = TEXT_ONLY,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = AsyncOpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=serve(reply, requests)),
    )
    model = AsyncGenerativeModel.from_openai(
        client, model=MODEL, max_output_tokens=max_output_tokens, perceives=perceives
    )

    async def generate() -> GenerationOutcome:
        return await model.generate(prompt)

    return Exchange(model=model, outcome=asyncio.run(generate()), requests=requests)


@pytest.fixture(params=[generate_synchronously, generate_asynchronously], ids=["sync", "async"])
def ask(request: pytest.FixtureRequest) -> Ask:
    return request.param


def replace(body: Mapping[str, object], **fields: object) -> dict[str, object]:
    return {**body, **fields}


def reject(message: str, *, code: str | None = None) -> dict[str, object]:
    """An error body in the shape of the Error schema; its message is the provider's text."""
    return {
        "error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}
    }


# https://github.com/openai/openai-openapi, openapi.yaml, the create-response examples
# "Image input" and "File input": their request bodies, and the "Image input" response.
BOARDWALK = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
    "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
)
BERKSHIRE_LETTER = "https://www.berkshirehathaway.com/letters/2024ltr.pdf"
IMAGE_PROMPT = Prompt(
    system="You are a helpful assistant.",
    content=(
        TextPart(text="what is in this image?"),
        MediaPart(media_type="image/jpeg", url=BOARDWALK),
    ),
)
# The published request has no instructions or store; CreateResponse has both, and
# InputImageContent requires detail, which defaults to auto. The "File input" request
# carries detail "auto" too, which InputFileContent leaves optional.
IMAGE_REQUEST: dict[str, object] = {
    "model": MODEL,
    "instructions": "You are a helpful assistant.",
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is in this image?"},
                {"type": "input_image", "image_url": BOARDWALK, "detail": "auto"},
            ],
        }
    ],
    "store": False,
}
BOARDWALK_DESCRIPTION = (
    "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through "
    "lush, green grass under a blue sky with some clouds. The setting suggests a peaceful "
    "natural area, possibly a park or nature reserve. There are trees and shrubs in the "
    "background."
)
OUTPUT_TEXT: dict[str, object] = {
    "type": "output_text",
    "text": BOARDWALK_DESCRIPTION,
    "annotations": [],
    "logprobs": [],
}
MESSAGE: dict[str, object] = {
    "type": "message",
    "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
    "status": "completed",
    "role": "assistant",
    "content": [OUTPUT_TEXT],
}
IMAGE_INPUT_RESPONSE: dict[str, object] = {
    "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
    "object": "response",
    "access_programs": None,
    "created_at": 1741476777,
    "status": "completed",
    "completed_at": 1741476778,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [MESSAGE],
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None},
    "store": True,
    "temperature": 1.0,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 328,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 52,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 380,
    },
    "user": None,
    "metadata": {},
}

# The same spec, the "Web search" response: a tool call before the message, with citations.
WEB_SEARCH_RESPONSE: dict[str, object] = {
    "id": "resp_67ccf18ef5fc8190b16dbee19bc54e5f087bb177ab789d5c",
    "object": "response",
    "access_programs": None,
    "created_at": 1741484430,
    "status": "completed",
    "completed_at": 1741484431,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [
        {
            "type": "web_search_call",
            "id": "ws_67ccf18f64008190a39b619f4c8455ef087bb177ab789d5c",
            "status": "completed",
        },
        {
            "type": "message",
            "id": "msg_67ccf190ca3881909d433c50b1f6357e087bb177ab789d5c",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "As of today, March 9, 2025, one notable positive news story...",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "start_index": 442,
                            "end_index": 557,
                            "url": "https://.../?utm_source=chatgpt.com",
                            "title": "...",
                        },
                        {
                            "type": "url_citation",
                            "start_index": 962,
                            "end_index": 1077,
                            "url": "https://.../?utm_source=chatgpt.com",
                            "title": "...",
                        },
                        {
                            "type": "url_citation",
                            "start_index": 1336,
                            "end_index": 1451,
                            "url": "https://.../?utm_source=chatgpt.com",
                            "title": "...",
                        },
                    ],
                    "logprobs": [],
                },
            ],
        },
    ],
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None},
    "store": True,
    "temperature": 1.0,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [
        {
            "type": "web_search_preview",
            "domains": [],
            "search_context_size": "medium",
            "user_location": {
                "type": "approximate",
                "city": None,
                "country": "US",
                "region": None,
                "timezone": None,
            },
        }
    ],
    "top_p": 1.0,
    "truncation": "disabled",
    "usage": {
        "input_tokens": 328,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 356,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 684,
    },
    "user": None,
    "metadata": {},
}

# The same spec, the response objects of the "response.incomplete" and "response.failed"
# event examples; the event's output is empty.
INCOMPLETE_RESPONSE: dict[str, object] = {
    "id": "resp_123",
    "object": "response",
    "access_programs": None,
    "created_at": 1740855869,
    "status": "incomplete",
    "completed_at": None,
    "error": None,
    "incomplete_details": {"reason": "max_output_tokens"},
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [],
    "previous_response_id": None,
    "reasoning_effort": None,
    "store": False,
    "temperature": 1,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": None,
    "user": None,
    "metadata": {},
    "parallel_tool_calls": True,
}
SERVER_ERROR = {"code": "server_error", "message": "The model failed to generate a response."}
FAILED_RESPONSE: dict[str, object] = {
    "id": "resp_123",
    "object": "response",
    "access_programs": None,
    "created_at": 1740855869,
    "status": "failed",
    "completed_at": None,
    "error": SERVER_ERROR,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [],
    "previous_response_id": None,
    "reasoning_effort": None,
    "store": False,
    "temperature": 1,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": None,
    "user": None,
    "metadata": {},
    "parallel_tool_calls": True,
}
# https://developers.openai.com/api/docs/guides/pdf-files: every example of file_data
# carries a filename, such as "draconomicon.pdf".
# RFC 4648, section 10: the base64 of "foob" is "Zm9vYg==", of "fooba" "Zm9vYmE=", of
# "foobar" "Zm9vYmFy". The PNG signature (the PNG specification, section 5.2) is
# 89 50 4E 47 0D 0A 1A 0A.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# The InferenceRateLimited (slowDown) and InferenceServiceUnavailable (serverOverloaded)
# examples of openapi.yaml.
SLOW_DOWN = {
    "error": {
        "message": (
            "Your request rate increased too quickly. Please reduce the request rate and "
            "gradually increase it again."
        ),
        "type": "rate_limit_error",
        "param": None,
        "code": "slow_down",
    }
}
SERVER_OVERLOADED = {
    "error": {
        "message": (
            "The model is temporarily overloaded. Please retry your request after a brief delay."
        ),
        "type": "service_unavailable_error",
        "param": None,
        "code": "server_is_overloaded",
    }
}


@pytest.mark.xfail(strict=True)
def test_a_model_is_named_for_the_provider_and_its_model_id(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE))

    assert exchange.model.name == "openai:gpt-6-astra"


def test_a_model_perceives_the_media_it_was_given(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), perceives=MediaTypes("image/*"))

    assert exchange.model.perceives == MediaTypes("image/*")


@pytest.mark.xfail(strict=True)
def test_a_prompt_is_sent_as_instructions_and_one_user_message(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), IMAGE_PROMPT)

    assert exchange.requests == [IMAGE_REQUEST]


# openapi.yaml, CreateResponse: max_output_tokens is an optional field of the request.
@pytest.mark.parametrize(
    ("max_output_tokens", "expected"),
    [(None, IMAGE_REQUEST), (300, replace(IMAGE_REQUEST, max_output_tokens=300))],
    ids=["no-limit-sends-no-field", "limit-is-sent"],
)
@pytest.mark.xfail(strict=True)
def test_the_output_limit_is_sent_only_when_given(
    ask: Ask, max_output_tokens: int | None, expected: dict[str, object]
) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), IMAGE_PROMPT, max_output_tokens=max_output_tokens)

    assert exchange.requests == [expected]


@pytest.mark.parametrize(
    ("media", "expected"),
    [
        (
            MediaPart(media_type="image/png", data=PNG_SIGNATURE),
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0KGgo=",
                "detail": "auto",
            },
        ),
        (
            MediaPart(media_type="image/*", url=BOARDWALK),
            {"type": "input_image", "image_url": BOARDWALK, "detail": "auto"},
        ),
        (
            MediaPart(media_type="application/pdf", url=BERKSHIRE_LETTER),
            {"type": "input_file", "file_url": BERKSHIRE_LETTER},
        ),
        (
            MediaPart(media_type="application/pdf", data=b"foob"),
            {
                "type": "input_file",
                "filename": "attachment.pdf",
                "file_data": "data:application/pdf;base64,Zm9vYg==",
            },
        ),
        (
            MediaPart(media_type="text/plain", data=b"foobar"),
            {
                "type": "input_file",
                "filename": "attachment.txt",
                "file_data": "data:text/plain;base64,Zm9vYmFy",
            },
        ),
        (
            MediaPart(media_type="application/vnd.example", data=b"fooba"),
            {
                "type": "input_file",
                "filename": "attachment",
                "file_data": "data:application/vnd.example;base64,Zm9vYmE=",
            },
        ),
    ],
    ids=[
        "image-bytes-as-a-data-url",
        "image-range-by-url",
        "file-by-url",
        "pdf-bytes-with-a-filename",
        "text-bytes-with-a-filename",
        "file-of-unknown-extension-with-a-bare-filename",
    ],
)
@pytest.mark.xfail(strict=True)
def test_media_is_sent_as_the_content_part_of_its_kind(
    ask: Ask, media: MediaPart, expected: dict[str, object]
) -> None:
    prompt = Prompt(system="You are a helpful assistant.", content=(media,))

    exchange = ask(respond(IMAGE_INPUT_RESPONSE), prompt)

    assert [request["input"] for request in exchange.requests] == [
        [{"role": "user", "content": [expected]}]
    ]


@pytest.mark.parametrize(
    "media",
    [
        MediaPart(media_type="audio/wav", data=b"foob"),
        MediaPart(media_type="video/mp4", url="https://example.com/clip.mp4"),
        MediaPart(media_type="image/*", data=b"foob"),
    ],
    ids=["audio", "video", "bytes-of-a-media-range"],
)
@pytest.mark.xfail(strict=True)
def test_media_the_api_cannot_carry_raises_naming_its_media_type(
    ask: Ask, media: MediaPart
) -> None:
    prompt = Prompt(system="You are a helpful assistant.", content=(media,))

    with pytest.raises(ConfigurationError, match=re.escape(media.media_type)):
        ask(respond(IMAGE_INPUT_RESPONSE), prompt)


@pytest.mark.xfail(strict=True)
def test_a_completed_reply_is_a_generation_of_its_text(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE))

    assert exchange.outcome == Generation(text=BOARDWALK_DESCRIPTION)


@pytest.mark.xfail(strict=True)
def test_a_reply_after_a_tool_call_is_a_generation_of_the_message_text(ask: Ask) -> None:
    exchange = ask(respond(WEB_SEARCH_RESPONSE))

    assert exchange.outcome == Generation(
        text="As of today, March 9, 2025, one notable positive news story..."
    )


# openai's Response.output_text joins the text parts in order with no separator.
@pytest.mark.xfail(strict=True)
def test_the_text_parts_of_a_reply_are_joined_in_order(ask: Ask) -> None:
    parts = [replace(OUTPUT_TEXT, text="foo"), replace(OUTPUT_TEXT, text="bar")]
    body = replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=parts)])

    exchange = ask(respond(body))

    assert exchange.outcome == Generation(text="foobar")


# openapi.yaml, schema RefusalContent: a content part {"type": "refusal", "refusal": text};
# the text is the "response.refusal.done" event example's.
REFUSAL = {"type": "refusal", "refusal": "final refusal text"}


@pytest.mark.parametrize(
    "content",
    [[REFUSAL], [OUTPUT_TEXT, REFUSAL]],
    ids=["refusal-alone", "refusal-after-text"],
)
@pytest.mark.xfail(strict=True)
def test_a_completed_reply_with_a_refusal_part_is_a_refusal_with_its_text(
    ask: Ask, content: list[dict[str, object]]
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=content)])

    exchange = ask(respond(body))

    assert exchange.outcome == Refused(explanation="final refusal text")


# The "response.incomplete" example has no output yet; the second case gives it the "Image
# input" message, so the text written so far is not empty.
@pytest.mark.parametrize(
    ("body", "text"),
    [
        (INCOMPLETE_RESPONSE, ""),
        (replace(INCOMPLETE_RESPONSE, output=[MESSAGE]), BOARDWALK_DESCRIPTION),
    ],
    ids=["nothing-written", "text-written"],
)
@pytest.mark.xfail(strict=True)
def test_a_reply_cut_off_at_the_output_limit_is_truncated_with_its_text(
    ask: Ask, body: dict[str, object], text: str
) -> None:
    exchange = ask(respond(body))

    assert exchange.outcome == Truncated(text=text)


# openapi.yaml, schema Response: status is one of completed, failed, in_progress,
# cancelled, queued, incomplete; incomplete_details.reason one of max_output_tokens,
# max_messages, content_filter, steered.
@pytest.mark.parametrize(
    ("status", "details"),
    [
        ("incomplete", {"reason": "max_messages"}),
        ("incomplete", {"reason": "steered"}),
        ("incomplete", None),
        ("cancelled", None),
        ("queued", None),
        ("in_progress", None),
        (None, None),
    ],
    ids=[
        "incomplete-max-messages",
        "incomplete-steered",
        "incomplete-without-reason",
        "cancelled",
        "queued",
        "in-progress",
        "without-status",
    ],
)
@pytest.mark.xfail(strict=True)
def test_a_reply_that_is_not_complete_for_another_reason_is_truncated(
    ask: Ask, status: str | None, details: dict[str, str] | None
) -> None:
    body = replace(
        IMAGE_INPUT_RESPONSE, status=status, incomplete_details=details, completed_at=None
    )

    exchange = ask(respond(body))

    assert exchange.outcome == Truncated(text=BOARDWALK_DESCRIPTION)


def test_a_reply_stopped_by_the_content_filter_is_a_refusal_without_an_explanation(
    ask: Ask,
) -> None:
    body = replace(
        INCOMPLETE_RESPONSE, incomplete_details={"reason": "content_filter"}, output=[MESSAGE]
    )

    exchange = ask(respond(body))

    assert exchange.outcome == Refused(explanation=None)


@pytest.mark.parametrize(
    "error",
    [SERVER_ERROR, {"code": "rate_limit_exceeded", "message": "Slow down."}, None],
    ids=["server-error", "rate-limit", "no-error-object"],
)
@pytest.mark.xfail(strict=True)
def test_a_failed_reply_is_unavailable_without_a_status(
    ask: Ask, error: dict[str, str] | None
) -> None:
    body = replace(FAILED_RESPONSE, error=error)

    exchange = ask(respond(body))

    assert exchange.outcome == Unavailable(status=None)


# openapi.yaml, schema ResponseErrorCode: the codes that name a policy.
@pytest.mark.parametrize(
    "code",
    ["bio_policy", "misalignment_policy_violation", "image_content_policy_violation"],
)
def test_a_failed_reply_whose_code_names_a_policy_is_a_refusal_without_an_explanation(
    ask: Ask, code: str
) -> None:
    body = replace(FAILED_RESPONSE, error={"code": code, "message": "Blocked."})

    exchange = ask(respond(body))

    assert exchange.outcome == Refused(explanation=None)


@pytest.mark.parametrize(
    "body",
    [
        [],
        replace(IMAGE_INPUT_RESPONSE, output="nope"),
        replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content="nope")]),
        replace(
            IMAGE_INPUT_RESPONSE,
            output=[replace(MESSAGE, content=[OUTPUT_TEXT, replace(OUTPUT_TEXT, text=5)])],
        ),
        replace(
            IMAGE_INPUT_RESPONSE,
            output=[replace(MESSAGE, content=[replace(REFUSAL, refusal=5)])],
        ),
    ],
    ids=[
        "body-is-not-an-object",
        "output-is-not-a-list",
        "content-is-not-a-list",
        "text-is-not-a-string",
        "refusal-is-not-a-string",
    ],
)
@pytest.mark.xfail(strict=True)
def test_a_reply_with_fields_of_the_wrong_type_is_unavailable_without_a_status(
    ask: Ask, body: object
) -> None:
    exchange = ask(respond(body))

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.xfail(strict=True)
def test_a_reply_that_is_not_json_is_unavailable_without_a_status(ask: Ask) -> None:
    reply = httpx2.Response(200, text="<html>502 Bad Gateway</html>")

    exchange = ask(reply)

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.parametrize(
    "failure",
    [httpx2.ConnectError("connection refused"), httpx2.ReadTimeout("read timed out")],
    ids=["connection-error", "timeout"],
)
@pytest.mark.xfail(strict=True)
def test_a_connection_that_fails_is_unavailable_without_a_status(
    ask: Ask, failure: Exception
) -> None:
    exchange = ask(failure)

    assert exchange.outcome == Unavailable(status=None)


# The InferenceRateLimited and InferenceServiceUnavailable examples answer 429 and 503. The
# other statuses reuse the 503 body: the status decides, not the body.
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (429, SLOW_DOWN),
        (503, SERVER_OVERLOADED),
        (408, SERVER_OVERLOADED),
        (409, SERVER_OVERLOADED),
        (413, SERVER_OVERLOADED),
        (500, SERVER_OVERLOADED),
        (502, SERVER_OVERLOADED),
    ],
)
@pytest.mark.xfail(strict=True)
def test_an_http_error_that_may_pass_is_unavailable_with_its_status(
    ask: Ask, status: int, body: dict[str, object]
) -> None:
    exchange = ask(respond(body, status=status))

    assert exchange.outcome == Unavailable(status=status)


# openapi.yaml, schema SessionTurnErrorCodeResource: context_length_exceeded, "The request
# exceeds the model's context window." The Responses API's own error codes list none.
@pytest.mark.xfail(strict=True)
def test_an_input_that_exceeds_the_context_window_is_unavailable_with_its_status(
    ask: Ask,
) -> None:
    body = reject("The request exceeds the model's context window.", code="context_length_exceeded")

    exchange = ask(respond(body, status=400))

    assert exchange.outcome == Unavailable(status=400)


# openapi.yaml, schema ResponseErrorCode: the codes that say an image cannot be used, other
# than the policy one and image_file_not_found, which names a file the request never refers to.
@pytest.mark.parametrize(
    "code",
    [
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
    ],
)
@pytest.mark.xfail(strict=True)
def test_an_image_the_api_cannot_use_is_unavailable_with_its_status(ask: Ask, code: str) -> None:
    body = reject("The image cannot be used.", code=code)

    exchange = ask(respond(body, status=400))

    assert exchange.outcome == Unavailable(status=400)


# openapi.yaml, schema ResponseErrorCode: image_content_policy_violation.
def test_an_http_error_whose_code_names_a_policy_is_a_refusal_without_an_explanation(
    ask: Ask,
) -> None:
    body = reject("The image was blocked.", code="image_content_policy_violation")

    exchange = ask(respond(body, status=400))

    assert exchange.outcome == Refused(explanation=None)


# https://developers.openai.com/api/docs/guides/error-codes, the API errors: their messages.
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Invalid service_tier argument: The requested service tier is not allowed."),
        (401, "Incorrect API key provided"),
        (403, "Country, region, or territory not supported"),
    ],
)
@pytest.mark.xfail(strict=True)
def test_a_rejected_request_raises_naming_only_its_status(
    ask: Ask, status: int, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask(respond(reject(message), status=status))

    assert str(status) in str(raised.value)
    assert message not in str(raised.value)


@pytest.mark.xfail(strict=True)
def test_a_rejected_request_does_not_carry_the_providers_error(ask: Ask) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask(respond(reject("Incorrect API key provided"), status=401))

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
