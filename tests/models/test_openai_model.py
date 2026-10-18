"""OpenAI's Responses and Decisions APIs as models: the request the SDK sends and the outcome of
each reply.
"""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx2
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types import ReasoningEffort

from hyoshigi import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    ConfigurationError,
    Decision,
    DecisionModel,
    DecisionOutcome,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaPart,
    MediaTypes,
    Prompt,
    Question,
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
    """What a model returned for a prompt or a question, and the JSON request bodies sent."""

    model: GenerativeModel | AsyncGenerativeModel | DecisionModel | AsyncDecisionModel
    outcome: GenerationOutcome | DecisionOutcome
    requests: list[dict[str, object]]


class Ask(Protocol):
    """Gives a prompt to a model on a stand-in for OpenAI's API that serves one reply."""

    def __call__(
        self,
        reply: httpx2.Response | Exception,
        prompt: Prompt = ...,
        *,
        model: str = ...,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = None,
        perceives: MediaTypes = ...,
        strict_validation: bool = False,
    ) -> Exchange: ...


class AskDecision(Protocol):
    """Gives a question to a decision model on a stand-in for OpenAI's API that serves one reply."""

    def __call__(
        self,
        reply: httpx2.Response | Exception,
        question: Question = ...,
        *,
        model: str = ...,
        perceives: MediaTypes = ...,
        strict_validation: bool = False,
    ) -> Exchange: ...


class AskEither(Protocol):
    """Gives its default prompt or question to a model of either kind."""

    def __call__(
        self,
        reply: httpx2.Response | Exception,
        *,
        model: str = ...,
        strict_validation: bool = False,
    ) -> Exchange: ...


PROMPT = Prompt(system="You are a helpful assistant.", content=(TextPart(text="Hello!"),))
DECISION_MODEL = "gpt-6-luna"
QUESTION = Question(statement="The action is safe.", state=(TextPart(text="Run: ls"),))


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
    model: str = MODEL,
    max_output_tokens: int | None = None,
    reasoning_effort: ReasoningEffort = None,
    perceives: MediaTypes = TEXT_ONLY,
    strict_validation: bool = False,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = OpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.Client(transport=serve(reply, requests)),
        _strict_response_validation=strict_validation,
    )
    built = GenerativeModel.from_openai(
        client,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        perceives=perceives,
    )
    return Exchange(model=built, outcome=built.generate(prompt), requests=requests)


def generate_asynchronously(
    reply: httpx2.Response | Exception,
    prompt: Prompt = PROMPT,
    *,
    model: str = MODEL,
    max_output_tokens: int | None = None,
    reasoning_effort: ReasoningEffort = None,
    perceives: MediaTypes = TEXT_ONLY,
    strict_validation: bool = False,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = AsyncOpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=serve(reply, requests)),
        _strict_response_validation=strict_validation,
    )
    built = AsyncGenerativeModel.from_openai(
        client,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        perceives=perceives,
    )

    async def generate() -> GenerationOutcome:
        return await built.generate(prompt)

    return Exchange(model=built, outcome=asyncio.run(generate()), requests=requests)


@pytest.fixture(params=[generate_synchronously, generate_asynchronously], ids=["sync", "async"])
def ask(request: pytest.FixtureRequest) -> Ask:
    return request.param


def decide_synchronously(
    reply: httpx2.Response | Exception,
    question: Question = QUESTION,
    *,
    model: str = DECISION_MODEL,
    perceives: MediaTypes = TEXT_ONLY,
    strict_validation: bool = False,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = OpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.Client(transport=serve(reply, requests)),
        _strict_response_validation=strict_validation,
    )
    built = DecisionModel.from_openai(client, model=model, perceives=perceives)
    return Exchange(model=built, outcome=built.decide(question), requests=requests)


def decide_asynchronously(
    reply: httpx2.Response | Exception,
    question: Question = QUESTION,
    *,
    model: str = DECISION_MODEL,
    perceives: MediaTypes = TEXT_ONLY,
    strict_validation: bool = False,
) -> Exchange:
    requests: list[dict[str, object]] = []
    client = AsyncOpenAI(
        api_key=API_KEY,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=serve(reply, requests)),
        _strict_response_validation=strict_validation,
    )
    built = AsyncDecisionModel.from_openai(client, model=model, perceives=perceives)

    async def decide() -> DecisionOutcome:
        return await built.decide(question)

    return Exchange(model=built, outcome=asyncio.run(decide()), requests=requests)


@pytest.fixture(params=[decide_synchronously, decide_asynchronously], ids=["sync", "async"])
def ask_decision(request: pytest.FixtureRequest) -> AskDecision:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(generate_synchronously, id="generative-sync"),
        pytest.param(generate_asynchronously, id="generative-async"),
        pytest.param(decide_synchronously, id="decision-sync"),
        pytest.param(decide_asynchronously, id="decision-async"),
    ]
)
def ask_either(request: pytest.FixtureRequest) -> AskEither:
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


def test_a_model_is_named_for_the_provider_and_its_model_id(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE))

    assert exchange.model.name == "openai:gpt-6-astra"


@pytest.mark.parametrize(
    "perceives",
    [
        MediaTypes("image/*", "application/*", "text/*"),
        MediaTypes("image/png", "application/pdf", "text/plain"),
    ],
    ids=["ranges", "types-within-the-ranges"],
)
def test_a_model_perceives_the_media_it_was_given(ask: Ask, perceives: MediaTypes) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), perceives=perceives)

    assert exchange.model.perceives == perceives


@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (("audio/*",), "audio/*"),
        (("video/mp4",), "video/mp4"),
        (("video/mp4", "image/png", "audio/wav"), "audio/wav"),
    ],
    ids=["range-of-another-type", "another-type", "the-first-of-those-it-cannot-carry"],
)
def test_a_model_that_perceives_media_the_api_cannot_carry_raises_naming_it_when_built(
    ask: Ask, patterns: tuple[str, ...], named: str
) -> None:
    with pytest.raises(ConfigurationError, match=re.escape(named)):
        ask(respond(IMAGE_INPUT_RESPONSE), perceives=MediaTypes(*patterns))


def test_a_prompt_is_sent_as_instructions_and_one_user_message(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), IMAGE_PROMPT)

    assert exchange.requests == [IMAGE_REQUEST]


# openapi.yaml, CreateResponse: max_output_tokens is an optional field of the request.
@pytest.mark.parametrize(
    ("max_output_tokens", "expected"),
    [(None, IMAGE_REQUEST), (300, replace(IMAGE_REQUEST, max_output_tokens=300))],
    ids=["no-limit-sends-no-field", "limit-is-sent"],
)
def test_the_output_limit_is_sent_only_when_given(
    ask: Ask, max_output_tokens: int | None, expected: dict[str, object]
) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), IMAGE_PROMPT, max_output_tokens=max_output_tokens)

    assert exchange.requests == [expected]


# https://developers.openai.com/api/docs/guides/reasoning, "Get started with reasoning": the
# request carries "reasoning": {"effort": "low"}; "Reasoning effort" lists none among the values.
@pytest.mark.parametrize(
    ("reasoning_effort", "expected"),
    [
        (None, IMAGE_REQUEST),
        ("low", replace(IMAGE_REQUEST, reasoning={"effort": "low"})),
        ("none", replace(IMAGE_REQUEST, reasoning={"effort": "none"})),
    ],
    ids=["no-effort-sends-no-field", "effort-is-sent", "none-is-an-effort-and-is-sent"],
)
def test_the_reasoning_effort_is_sent_only_when_given(
    ask: Ask, reasoning_effort: ReasoningEffort, expected: dict[str, object]
) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), IMAGE_PROMPT, reasoning_effort=reasoning_effort)

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
            MediaPart(media_type="text/plain", url="https://example.com/notes.txt"),
            {"type": "input_file", "file_url": "https://example.com/notes.txt"},
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
        "text-by-url",
        "pdf-bytes-with-a-filename",
        "text-bytes-with-a-filename",
        "file-of-unknown-extension-with-a-bare-filename",
    ],
)
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
    ],
    ids=["audio", "video"],
)
def test_media_the_api_cannot_carry_raises_naming_its_media_type(
    ask: Ask, media: MediaPart
) -> None:
    prompt = Prompt(system="You are a helpful assistant.", content=(media,))

    with pytest.raises(ConfigurationError, match=re.escape(media.media_type)):
        ask(respond(IMAGE_INPUT_RESPONSE), prompt)


# The type is carried by a media range given as bytes: no data URL can name its subtype.
@pytest.mark.parametrize(
    "media",
    [
        MediaPart(media_type="image/*", data=b"foob"),
        MediaPart(media_type="application/*", data=b"foob"),
        MediaPart(media_type="text/*", data=b"foob"),
    ],
    ids=["image-range", "application-range", "text-range"],
)
def test_media_in_a_form_the_api_cannot_take_is_unavailable_without_a_status(
    ask: Ask, media: MediaPart
) -> None:
    prompt = Prompt(system="You are a helpful assistant.", content=(media,))

    exchange = ask(respond(IMAGE_INPUT_RESPONSE), prompt)

    assert exchange.outcome == Unavailable(status=None)


RANGE_BYTES = MediaPart(media_type="image/*", data=b"foob")
AUDIO = MediaPart(media_type="audio/wav", data=b"foob")


@pytest.mark.parametrize(
    "content",
    [
        pytest.param((RANGE_BYTES, AUDIO), id="unusable-form-first"),
        pytest.param((AUDIO, RANGE_BYTES), id="never-carried-first"),
    ],
)
def test_media_the_api_can_never_carry_raises_whatever_the_order_of_the_parts(
    ask: Ask, content: tuple[MediaPart, ...]
) -> None:
    prompt = Prompt(system="You are a helpful assistant.", content=content)

    with pytest.raises(ConfigurationError, match=re.escape("audio/wav")):
        ask(respond(IMAGE_INPUT_RESPONSE), prompt)


# A lone surrogate, as json.loads produces from "a\\ud800b", has no UTF-8 encoding.
@pytest.mark.parametrize(
    "prompt",
    [
        Prompt(system="a\ud800b", content=()),
        Prompt(system="You are a helpful assistant.", content=(TextPart(text="a\ud800b"),)),
    ],
    ids=["instructions", "text-part"],
)
def test_a_prompt_that_cannot_be_encoded_is_unavailable_without_a_status(
    ask: Ask, prompt: Prompt
) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE), prompt)

    assert exchange.outcome == Unavailable(status=None)


def test_a_model_without_a_model_id_raises_when_it_is_built(ask_either: AskEither) -> None:
    with pytest.raises(ConfigurationError):
        ask_either(respond(IMAGE_INPUT_RESPONSE), model="")


def test_a_completed_reply_is_a_generation_of_its_text(ask: Ask) -> None:
    exchange = ask(respond(IMAGE_INPUT_RESPONSE))

    assert exchange.outcome == Generation(text=BOARDWALK_DESCRIPTION)


def test_a_reply_after_a_tool_call_is_a_generation_of_the_message_text(ask: Ask) -> None:
    exchange = ask(respond(WEB_SEARCH_RESPONSE))

    assert exchange.outcome == Generation(
        text="As of today, March 9, 2025, one notable positive news story..."
    )


# https://developers.openai.com/api/docs/guides/reasoning, "Reasoning summaries": the output of a
# request with a reasoning summary, a reasoning item before the assistant's message.
REASONING_ITEM: dict[str, object] = {
    "id": "rs_6876cf02e0bc8192b74af0fb64b715ff06fa2fcced15a5ac",
    "type": "reasoning",
    "summary": [
        {
            "type": "summary_text",
            "text": "**Answering a simple question**\n\nI\u2019m looking at a straightforward "
            "question: the capital of France is Paris. It\u2019s a well-known fact, and I want "
            "to keep it brief and to the point.",
        }
    ],
}
PARIS_MESSAGE: dict[str, object] = {
    "id": "msg_6876cf054f58819284ecc1058131305506fa2fcced15a5ac",
    "type": "message",
    "status": "completed",
    "content": [
        {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": "The capital of France is Paris.",
        }
    ],
    "role": "assistant",
}


def test_a_reply_that_holds_a_reasoning_item_is_a_generation_of_the_message_text(
    ask: Ask,
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[REASONING_ITEM, PARIS_MESSAGE])

    exchange = ask(respond(body))

    assert exchange.outcome == Generation(text="The capital of France is Paris.")


# openai's Response.output_text joins the text parts in order with no separator.
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
def test_a_completed_reply_with_a_refusal_part_is_a_refusal_with_its_text(
    ask: Ask, content: list[dict[str, object]]
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=content)])

    exchange = ask(respond(body))

    assert exchange.outcome == Refused(explanation="final refusal text")


# openapi.yaml, schemas OutputMessage and ReasoningTextContent: output items and content parts
# are told apart by their type field.
REASONING_TEXT = {"type": "reasoning_text", "text": "internal thought"}


@pytest.mark.parametrize(
    ("content", "text"),
    [
        ([REASONING_TEXT], ""),
        ([OUTPUT_TEXT, REASONING_TEXT], BOARDWALK_DESCRIPTION),
        ([REASONING_TEXT, OUTPUT_TEXT], BOARDWALK_DESCRIPTION),
    ],
    ids=[
        "part-of-another-type",
        "known-part-before-another-type",
        "known-part-after-another-type",
    ],
)
def test_a_completed_reply_with_a_part_of_an_unknown_type_is_truncated_with_the_known_text(
    ask: Ask, content: list[dict[str, object]], text: str
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=content)])

    exchange = ask(respond(body))

    assert exchange.outcome == Truncated(text=text)


NOT_A_MESSAGE = {"type": "future_item", "content": [replace(OUTPUT_TEXT, text="from an item")]}


def test_an_output_item_that_is_not_a_message_is_not_read(ask: Ask) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[NOT_A_MESSAGE, MESSAGE])

    exchange = ask(respond(body))

    assert exchange.outcome == Generation(text=BOARDWALK_DESCRIPTION)


@pytest.mark.parametrize(
    "output",
    [[], [NOT_A_MESSAGE]],
    ids=["no-output", "no-message"],
)
def test_a_completed_reply_without_a_message_is_unavailable_without_a_status(
    ask: Ask, output: list[dict[str, object]]
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=output)

    exchange = ask(respond(body))

    assert exchange.outcome == Unavailable(status=None)


# openapi.yaml, schema OutputMessage: a message's status is in_progress, completed or
# incomplete.
@pytest.mark.parametrize(
    "status", ["incomplete", "in_progress", None], ids=["incomplete", "in-progress", "no-status"]
)
def test_a_completed_reply_with_an_unfinished_message_is_truncated_with_its_text(
    ask: Ask, status: str | None
) -> None:
    body = replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, status=status)])

    exchange = ask(respond(body))

    assert exchange.outcome == Truncated(text=BOARDWALK_DESCRIPTION)


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
def test_a_failed_reply_is_unavailable_without_a_status(
    ask: Ask, error: dict[str, str] | None
) -> None:
    body = replace(FAILED_RESPONSE, error=error)

    exchange = ask(respond(body))

    assert exchange.outcome == Unavailable(status=None)


# openapi.yaml, schema ResponseErrorCode: the codes that name a policy; and the code "refusal".
@pytest.mark.parametrize(
    "code",
    [
        "bio_policy",
        "misalignment_policy_violation",
        "image_content_policy_violation",
        "refusal",
    ],
)
def test_a_failed_reply_whose_code_names_a_refusal_is_a_refusal_without_an_explanation(
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
        replace(IMAGE_INPUT_RESPONSE, output=["nope"]),
        replace(IMAGE_INPUT_RESPONSE, output=[MESSAGE, 5]),
        replace(IMAGE_INPUT_RESPONSE, output=[None]),
        replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=["nope"])]),
        replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=[OUTPUT_TEXT, 5])]),
        replace(IMAGE_INPUT_RESPONSE, output=[{"content": [OUTPUT_TEXT]}, MESSAGE]),
        replace(IMAGE_INPUT_RESPONSE, output=[replace(MESSAGE, content=[{"text": "spoofed"}])]),
    ],
    ids=[
        "body-is-not-an-object",
        "output-is-not-a-list",
        "content-is-not-a-list",
        "text-is-not-a-string",
        "refusal-is-not-a-string",
        "output-element-is-a-string",
        "output-element-is-a-number",
        "output-element-is-null",
        "content-element-is-a-string",
        "content-element-is-a-number",
        "output-element-without-a-type",
        "content-element-without-a-type",
    ],
)
def test_a_reply_with_fields_of_the_wrong_type_is_unavailable_without_a_status(
    ask: Ask, body: object
) -> None:
    exchange = ask(respond(body))

    assert exchange.outcome == Unavailable(status=None)


def test_a_reply_that_is_not_json_is_unavailable_without_a_status(ask_either: AskEither) -> None:
    reply = httpx2.Response(200, text="<html>502 Bad Gateway</html>")

    exchange = ask_either(reply)

    assert exchange.outcome == Unavailable(status=None)


# The SDK validates a reply against its models when the caller's client asks it to, and then
# raises its own error for a reply that does not fit.
@pytest.mark.parametrize("body", [[], "nope"], ids=["list", "string"])
def test_a_reply_the_client_rejects_on_validation_is_unavailable_without_a_status(
    ask_either: AskEither, body: object
) -> None:
    exchange = ask_either(respond(body), strict_validation=True)

    assert exchange.outcome == Unavailable(status=None)


def test_a_reply_that_is_not_json_on_validation_is_unavailable_without_a_status(
    ask_either: AskEither,
) -> None:
    reply = httpx2.Response(200, text="<html>502 Bad Gateway</html>")

    exchange = ask_either(reply, strict_validation=True)

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.parametrize(
    "content",
    [b'{"id": ', b"", b'{"id": "\xff"}'],
    ids=["truncated", "empty", "not-utf-8"],
)
def test_a_reply_labelled_json_that_cannot_be_decoded_is_unavailable_without_a_status(
    ask_either: AskEither, content: bytes
) -> None:
    reply = httpx2.Response(200, content=content, headers={"content-type": "application/json"})

    exchange = ask_either(reply)

    assert exchange.outcome == Unavailable(status=None)


def test_a_reply_whose_content_encoding_cannot_be_decoded_is_unavailable_without_a_status(
    ask_either: AskEither,
) -> None:
    headers = {"content-type": "application/json", "content-encoding": "gzip"}
    reply = httpx2.Response(200, headers=headers, stream=httpx2.ByteStream(b"not gzip"))

    exchange = ask_either(reply)

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.parametrize(
    "failure",
    [httpx2.ConnectError("connection refused"), httpx2.ReadTimeout("read timed out")],
    ids=["connection-error", "timeout"],
)
def test_a_connection_that_fails_is_unavailable_without_a_status(
    ask_either: AskEither, failure: Exception
) -> None:
    exchange = ask_either(failure)

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
        (100, SERVER_OVERLOADED),
        (302, SERVER_OVERLOADED),
        (304, SERVER_OVERLOADED),
    ],
)
def test_an_http_error_that_may_pass_is_unavailable_with_its_status(
    ask_either: AskEither, status: int, body: dict[str, object]
) -> None:
    exchange = ask_either(respond(body, status=status))

    assert exchange.outcome == Unavailable(status=status)


# openapi.yaml, schema SessionTurnErrorCodeResource: context_length_exceeded, "The request
# exceeds the model's context window." The Responses API's own error codes list none.
def test_an_input_that_exceeds_the_context_window_is_unavailable_with_its_status(
    ask_either: AskEither,
) -> None:
    body = reject("The request exceeds the model's context window.", code="context_length_exceeded")

    exchange = ask_either(respond(body, status=400))

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
def test_an_image_the_api_cannot_use_is_unavailable_with_its_status(
    ask_either: AskEither, code: str
) -> None:
    body = reject("The image cannot be used.", code=code)

    exchange = ask_either(respond(body, status=400))

    assert exchange.outcome == Unavailable(status=400)


# openapi.yaml, schema ResponseErrorCode: image_content_policy_violation; and the code "refusal".
@pytest.mark.parametrize(
    "code",
    [
        "image_content_policy_violation",
        "refusal",
    ],
)
def test_an_http_error_whose_code_names_a_refusal_is_a_refusal_without_an_explanation(
    ask_either: AskEither, code: str
) -> None:
    body = reject("The image was blocked.", code=code)

    exchange = ask_either(respond(body, status=400))

    assert exchange.outcome == Refused(explanation=None)


# https://developers.openai.com/api/docs/guides/error-codes, the API errors: their messages.
@pytest.mark.parametrize(
    ("status", "message"),
    [
        (400, "Invalid service_tier argument: The requested service tier is not allowed."),
        (401, "Incorrect API key provided"),
        (403, "Country, region, or territory not supported"),
        (404, "Incorrect API key provided"),
        (422, "Incorrect API key provided"),
        (499, "Incorrect API key provided"),
    ],
)
def test_a_rejected_request_raises_naming_only_its_status(
    ask_either: AskEither, status: int, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask_either(respond(reject(message), status=status))

    assert str(status) in str(raised.value)
    assert message not in str(raised.value)


def test_a_rejected_request_does_not_carry_the_providers_error(ask_either: AskEither) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask_either(respond(reject("Incorrect API key provided"), status=401))

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


# https://developers.openai.com/api/reference/resources/decisions/methods/create, "Create a
# decision": the example request and its response. The published question is named "damaged"; the
# one question this adapter asks is not named, since its answer is the only one.
DAMAGED_INPUT = "The package arrived with a broken screen."
DAMAGED_STATEMENT = "Does the customer report a damaged item?"
DAMAGED = Question(statement=DAMAGED_STATEMENT, state=(TextPart(text=DAMAGED_INPUT),))
DAMAGED_REQUEST: dict[str, object] = {
    "model": DECISION_MODEL,
    "input": DAMAGED_INPUT,
    "questions": [{"type": "predicate", "instructions": DAMAGED_STATEMENT}],
}
USAGE: dict[str, object] = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 42,
}
DAMAGED_RESPONSE: dict[str, object] = {
    "model": DECISION_MODEL,
    "answers": [{"type": "predicate", "name": "damaged", "probability": 0.95}],
    "usage": USAGE,
}

# https://developers.openai.com/api/docs/guides/decisions, "Check an image for visible damage": a
# user message of an input_text part and an input_image part holding a base64 data URL, and the
# guide's response excerpt, which it calls illustrative. The data is the PNG signature.
INSPECTION = "Inspect the product in this photo."
DAMAGE_STATEMENT = (
    "Does the product have visible damage, such as a crack, tear, or dent? "
    "Ignore shadows and damage to the packaging."
)
PHOTO = Question(
    statement=DAMAGE_STATEMENT,
    state=(TextPart(text=INSPECTION), MediaPart(media_type="image/png", data=PNG_SIGNATURE)),
)
PHOTO_REQUEST: dict[str, object] = {
    "model": DECISION_MODEL,
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": INSPECTION},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
            ],
        }
    ],
    "questions": [{"type": "predicate", "instructions": DAMAGE_STATEMENT}],
}
PHOTO_RESPONSE: dict[str, object] = replace(
    DAMAGED_RESPONSE,
    answers=[{"type": "predicate", "name": "visible_damage", "probability": 0.92}],
)

# The same guide's "Select from fixed options" and "Score against a rubric": the response
# excerpts, whose answers are of the other two types.
CHOICE_ANSWER: dict[str, object] = {
    "type": "choice",
    "name": "department",
    "choice": "billing",
    "probabilities": [
        {"value": "billing", "probability": 0.95},
        {"value": "technical", "probability": 0.02},
        {"value": "shipping", "probability": 0.01},
        {"value": "other", "probability": 0.02},
    ],
    "confidence": 0.93,
}
SCORE_ANSWER: dict[str, object] = {
    "type": "score",
    "name": "severity",
    "score": 1.1,
    "probabilities": [
        {"value": 0, "label": "Cosmetic", "probability": 0.1},
        {"value": 1, "label": "Workaround available", "probability": 0.7},
        {"value": 2, "label": "Fully blocked", "probability": 0.2},
    ],
    "confidence": 0.55,
}
# The reference's schema Refusal is {"name": string or null, "type": "refusal"}; it has no
# example response of a refusal, so this answer is written from the schema.
REFUSAL_ANSWER: dict[str, object] = {"type": "refusal", "name": None}

# The reference and the SDK's DecisionInputImageParam say an image is "a base64 data URL or a
# publicly accessible HTTP(S) image URL"; the guide says images must be inline base64 data URLs.
# A hosted URL, this one, was accepted live on 2026-10-18 and answered with probability 1.0.
DICE = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"


def test_a_decision_model_is_named_for_the_provider_and_its_model_id(
    ask_decision: AskDecision,
) -> None:
    exchange = ask_decision(respond(DAMAGED_RESPONSE))

    assert exchange.model.name == "openai:gpt-6-luna"


@pytest.mark.parametrize(
    "perceives",
    [
        MediaTypes("image/png", "image/jpeg", "image/webp", "image/gif"),
        MediaTypes("image/png"),
    ],
    ids=["every-type-the-api-reads", "one-of-them"],
)
def test_a_decision_model_perceives_the_images_it_was_given(
    ask_decision: AskDecision, perceives: MediaTypes
) -> None:
    exchange = ask_decision(respond(DAMAGED_RESPONSE), perceives=perceives)

    assert exchange.model.perceives == perceives


# https://developers.openai.com/api/docs/guides/images-vision, "Image input requirements": PNG,
# JPEG, WEBP and non-animated GIF. The Decisions pages list no types, so a range names types
# that the API may not read.
@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (("image/*",), "image/*"),
        (("image/svg+xml",), "image/svg+xml"),
        (("application/pdf",), "application/pdf"),
        (("video/mp4", "image/png", "audio/wav"), "audio/wav"),
    ],
    ids=[
        "image-range",
        "image-type-outside-the-four",
        "pdf",
        "the-first-of-those-it-cannot-carry",
    ],
)
def test_a_decision_model_that_perceives_media_the_api_cannot_read_raises_naming_it_when_built(
    ask_decision: AskDecision, patterns: tuple[str, ...], named: str
) -> None:
    with pytest.raises(ConfigurationError, match=re.escape(named)):
        ask_decision(respond(DAMAGED_RESPONSE), perceives=MediaTypes(*patterns))


def test_text_is_sent_as_the_input_with_one_predicate_question(
    ask_decision: AskDecision,
) -> None:
    exchange = ask_decision(respond(DAMAGED_RESPONSE), DAMAGED)

    assert exchange.requests == [DAMAGED_REQUEST]


# Adjacent text is one run of text, as the monitor builds a state; so no separator is added.
def test_text_parts_are_joined_into_the_input_without_a_separator(
    ask_decision: AskDecision,
) -> None:
    question = Question(
        statement=DAMAGED_STATEMENT,
        state=(TextPart(text="The package arrived "), TextPart(text="with a broken screen.")),
    )

    exchange = ask_decision(respond(DAMAGED_RESPONSE), question)

    assert exchange.requests == [DAMAGED_REQUEST]


def test_a_state_with_an_image_is_sent_as_one_user_message_of_its_parts_in_order(
    ask_decision: AskDecision,
) -> None:
    exchange = ask_decision(respond(PHOTO_RESPONSE), PHOTO)

    assert exchange.requests == [PHOTO_REQUEST]


# RFC 4648, section 10: the base64 of "foobar" is "Zm9vYmFy".
@pytest.mark.parametrize("media_type", ["image/png", "image/jpeg", "image/webp", "image/gif"])
def test_each_image_type_the_api_reads_is_sent_as_a_data_url_of_its_type(
    ask_decision: AskDecision, media_type: str
) -> None:
    media = MediaPart(media_type=media_type, data=b"foobar")
    question = Question(statement=DAMAGE_STATEMENT, state=(media,))

    exchange = ask_decision(respond(PHOTO_RESPONSE), question)

    image_url = f"data:{media_type};base64,Zm9vYmFy"
    assert [request["input"] for request in exchange.requests] == [
        [{"role": "user", "content": [{"type": "input_image", "image_url": image_url}]}]
    ]


@pytest.mark.parametrize("media_type", ["image/png", "image/*"])
def test_an_image_at_a_url_is_sent_as_that_url(ask_decision: AskDecision, media_type: str) -> None:
    media = MediaPart(media_type=media_type, url=DICE)
    question = Question(statement="The image shows dice.", state=(media,))

    exchange = ask_decision(respond(PHOTO_RESPONSE), question)

    assert [request["input"] for request in exchange.requests] == [
        [{"role": "user", "content": [{"type": "input_image", "image_url": DICE}]}]
    ]


# A data URL cannot name the subtype of a range such as image/*.
# A question about image bytes of a range, such as image/*, which no data URL can name.
RANGE_BYTES_QUESTION = Question(statement=DAMAGE_STATEMENT, state=(RANGE_BYTES,))


def test_image_bytes_of_a_range_are_unavailable_without_a_status(ask_decision: AskDecision) -> None:
    exchange = ask_decision(respond(PHOTO_RESPONSE), RANGE_BYTES_QUESTION)

    assert exchange.outcome == Unavailable(status=None)


def test_image_bytes_of_a_range_send_no_request(ask_decision: AskDecision) -> None:
    exchange = ask_decision(respond(PHOTO_RESPONSE), RANGE_BYTES_QUESTION)

    assert exchange.requests == []


@pytest.mark.parametrize(
    "media",
    [
        MediaPart(media_type="application/pdf", data=b"foobar"),
        MediaPart(media_type="text/plain", url="https://example.com/notes.txt"),
        MediaPart(media_type="audio/wav", data=b"foobar"),
    ],
    ids=["pdf", "text-by-url", "audio"],
)
def test_media_that_is_not_an_image_raises_naming_its_media_type(
    ask_decision: AskDecision, media: MediaPart
) -> None:
    question = Question(statement=DAMAGE_STATEMENT, state=(TextPart(text="Look."), media))

    with pytest.raises(ConfigurationError, match=re.escape(media.media_type)):
        ask_decision(respond(PHOTO_RESPONSE), question)


@pytest.mark.parametrize(
    "state",
    [
        pytest.param((RANGE_BYTES, AUDIO), id="unusable-form-first"),
        pytest.param((AUDIO, RANGE_BYTES), id="never-carried-first"),
    ],
)
def test_media_the_api_can_never_read_raises_whatever_the_order_of_the_parts(
    ask_decision: AskDecision, state: tuple[MediaPart, ...]
) -> None:
    question = Question(statement=DAMAGE_STATEMENT, state=state)

    with pytest.raises(ConfigurationError, match=re.escape("audio/wav")):
        ask_decision(respond(PHOTO_RESPONSE), question)


# A lone surrogate, as json.loads produces from "a\\ud800b", has no UTF-8 encoding.
@pytest.mark.parametrize(
    "question",
    [
        Question(statement="a\ud800b", state=(TextPart(text="Run: ls"),)),
        Question(statement="The action is safe.", state=(TextPart(text="a\ud800b"),)),
    ],
    ids=["statement", "text-part"],
)
def test_a_question_that_cannot_be_encoded_is_unavailable_without_a_status(
    ask_decision: AskDecision, question: Question
) -> None:
    exchange = ask_decision(respond(DAMAGED_RESPONSE), question)

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.parametrize(
    ("body", "probability"),
    [(DAMAGED_RESPONSE, 0.95), (PHOTO_RESPONSE, 0.92)],
    ids=["reference-example", "guide-image-example"],
)
def test_a_predicate_answer_is_a_decision_of_its_probability(
    ask_decision: AskDecision, body: dict[str, object], probability: float
) -> None:
    exchange = ask_decision(respond(body))

    assert exchange.outcome == Decision(probability=probability)


# JSON writes a probability of exactly 0 or 1 without a fraction, which reads as an integer.
@pytest.mark.parametrize(("written", "expected"), [(0, 0.0), (1, 1.0), (0.0, 0.0), (1.0, 1.0)])
def test_a_probability_at_either_end_of_its_range_is_a_decision(
    ask_decision: AskDecision, written: float, expected: float
) -> None:
    answer = {"type": "predicate", "name": "damaged", "probability": written}
    body = replace(DAMAGED_RESPONSE, answers=[answer])

    exchange = ask_decision(respond(body))

    assert exchange.outcome == Decision(probability=expected)


def test_a_refusal_answer_is_a_refusal_without_an_explanation(
    ask_decision: AskDecision,
) -> None:
    body = replace(DAMAGED_RESPONSE, answers=[REFUSAL_ANSWER])

    exchange = ask_decision(respond(body))

    assert exchange.outcome == Refused(explanation=None)


@pytest.mark.parametrize("answer", [CHOICE_ANSWER, SCORE_ANSWER], ids=["choice", "score"])
def test_an_answer_of_another_type_is_unavailable_without_a_status(
    ask_decision: AskDecision, answer: dict[str, object]
) -> None:
    body = replace(DAMAGED_RESPONSE, answers=[answer])

    exchange = ask_decision(respond(body))

    assert exchange.outcome == Unavailable(status=None)


PREDICATE: dict[str, object] = {"type": "predicate", "name": "damaged", "probability": 0.95}


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"model": DECISION_MODEL, "usage": USAGE},
        replace(DAMAGED_RESPONSE, answers="nope"),
        replace(DAMAGED_RESPONSE, answers=[]),
        replace(DAMAGED_RESPONSE, answers=[PREDICATE, PREDICATE]),
        replace(DAMAGED_RESPONSE, answers=[REFUSAL_ANSWER, PREDICATE]),
        replace(DAMAGED_RESPONSE, answers=["nope"]),
        replace(DAMAGED_RESPONSE, answers=[None]),
        replace(DAMAGED_RESPONSE, answers=[{"probability": 0.95}]),
        replace(DAMAGED_RESPONSE, answers=[{"type": "predicate", "name": "damaged"}]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability="0.95")]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability=True)]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability=None)]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability=[0.95])]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability=1.5)]),
        replace(DAMAGED_RESPONSE, answers=[replace(PREDICATE, probability=-0.1)]),
    ],
    ids=[
        "body-is-not-an-object",
        "no-answers-field",
        "answers-is-not-a-list",
        "no-answer",
        "two-answers",
        "refusal-and-answer",
        "answer-is-a-string",
        "answer-is-null",
        "answer-without-a-type",
        "predicate-without-a-probability",
        "probability-is-a-string",
        "probability-is-a-bool",
        "probability-is-null",
        "probability-is-a-list",
        "probability-above-one",
        "probability-below-zero",
    ],
)
def test_a_reply_that_is_not_one_answer_to_the_question_is_unavailable_without_a_status(
    ask_decision: AskDecision, body: object
) -> None:
    exchange = ask_decision(respond(body))

    assert exchange.outcome == Unavailable(status=None)


# Python's json module reads the literal NaN, which JSON does not have.
def test_a_probability_that_is_not_a_number_is_unavailable_without_a_status(
    ask_decision: AskDecision,
) -> None:
    content = b'{"answers": [{"type": "predicate", "probability": NaN}]}'
    reply = httpx2.Response(200, content=content, headers={"content-type": "application/json"})

    exchange = ask_decision(reply)

    assert exchange.outcome == Unavailable(status=None)
