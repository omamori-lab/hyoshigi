"""OpenRouter's Responses and Decisions APIs as a monitor's models, on a stand-in for the wire."""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
import pytest
from openrouter import OpenRouter, components

from hyoshigi import (
    JSON,
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
    Rejected,
    TextPart,
    Truncated,
    Unavailable,
)

type Model = GenerativeModel | AsyncGenerativeModel | DecisionModel | AsyncDecisionModel
type Respond = Callable[[httpx.Request], httpx.Response]

MODEL = "openai/gpt-4o"
JEV = "typesafe/jev-1.14"
PROMPT = Prompt(
    system="Rate the action.",
    content=(TextPart(text="Run: ls"),),
)
QUESTION = Question(
    statement="The action is safe.",
    state=(TextPart(text="Run: ls"),),
)

# https://openrouter.ai/docs/api/api-reference/responses/create-a-response: the 200 example, the
# examples of its schemas OpenAIResponsesRefusalContent, IncompleteDetails, OutputReasoningItem
# and OpenResponsesErrorField, and the error examples of each status.
JOKE = "Why did the chicken cross the road? To get to the other side!"
REFUSAL = "I'm sorry, I cannot assist with that request"
REFUSAL_PART: JSON = {
    "refusal": REFUSAL,
    "type": "refusal",
}
REASONING: JSON = {
    "content": [
        {
            "text": "First, we analyze the problem...",
            "type": "reasoning_text",
        },
    ],
    "format": "anthropic-claude-v1",
    "id": "reasoning-123",
    "signature": "EvcBCkgIChABGAIqQKkSDbRuVEQUk9qN1odC098l9SEj...",
    "status": "completed",
    "summary": [
        {
            "text": "Analyzed the problem and found the optimal solution.",
            "type": "summary_text",
        },
    ],
    "type": "reasoning",
}
EVALUATOR_FAILED = "Alignment plugin: the reply could not be evaluated: evaluator_error"
# Not in the schema, which lists output_text and refusal: a content part the SDK cannot classify.
UNLISTED_PART: JSON = {"type": "unlisted"}


def build_text(text: str) -> JSON:
    return {
        "annotations": [],
        "text": text,
        "type": "output_text",
    }


def build_message(*content: JSON) -> JSON:
    return {
        "content": list(content),
        "id": "msg-abc123",
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


COMPLETED: dict[str, JSON] = {
    "completed_at": 1700000010,
    "created_at": 1700000000,
    "error": None,
    "frequency_penalty": None,
    "id": "resp_abc123",
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "metadata": None,
    "model": MODEL,
    "object": "response",
    "output": [build_message(build_text(JOKE))],
    "parallel_tool_calls": True,
    "presence_penalty": None,
    "status": "completed",
    "temperature": None,
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "usage": {
        "input_tokens": 10,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 30,
    },
}


def build_response(**changes: JSON) -> dict[str, JSON]:
    return {
        **COMPLETED,
        **changes,
    }


# https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request: the 200
# example's envelope and answers, the noul answer put under the question's key.
IS_BUG: JSON = {
    "noul": 0.96,
    "type": "noul",
}
TEAM: JSON = {
    "choice": "payments",
    "confidence": 0.75,
    "probabilities": {
        "account": 0,
        "frontend": 0.16,
        "payments": 0.84,
    },
    "type": "choice",
}
ENVELOPE: dict[str, JSON] = {
    "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "usage": {
        "cost": 0.000019992,
        "input_tokens": 476,
        "output_tokens": 70,
    },
}


def build_decisions(answers: JSON) -> dict[str, JSON]:
    return {
        **ENVELOPE,
        "answers": answers,
    }


# https://openrouter.ai/docs/guides/community/multimodal-decisions: the quickstart's request, a
# title and a PNG with a noul question, and its answer.
STICKER_TITLE = "Listing title: Red square sticker"
STICKER_STATEMENT = "The photo shows the item in the title."
MATCHES_TITLE: JSON = {
    "noul": 0.8607,
    "type": "noul",
}
PNG = MediaPart(
    media_type="image/png",
    data=b"\x89PNG",
)
PNG_URL = "data:image/png;base64,iVBORw=="

# The error examples of create-a-response (the Responses API reference), which has all of these
# statuses but 409, and of submit-a-decisions-request (the Decisions API reference), which lacks
# 408 and 422 too. 409 has no example in either: it takes the shape of 408's.
ERROR_MESSAGES = {
    400: "Invalid request parameters",
    401: "Missing Authentication header",
    402: "Insufficient credits. Add more using https://openrouter.ai/credits",
    403: "Only management keys can perform this operation",
    404: "Resource not found",
    408: "Operation timed out. Please try again later.",
    409: "Operation timed out. Please try again later.",
    413: "Request payload too large",
    422: "Invalid argument",
    429: "Rate limit exceeded",
    500: "Internal Server Error",
    502: "Provider returned error",
    503: "Service temporarily unavailable",
    524: "Request timed out. Please try again later.",
    529: "Provider returned error",
}


def build_error(
    status: int,
    **fields: JSON,
) -> JSON:
    return {
        "error": {
            "code": status,
            "message": ERROR_MESSAGES[status],
        },
        **fields,
    }


# https://openrouter.ai/docs/api_reference/errors-and-debugging, "Provider errors": the error type
# is in error.metadata, beside the integer code and the message of the published envelope.
def build_typed_error(
    status: int,
    error_type: str,
) -> JSON:
    metadata: JSON = {"error_type": error_type}
    return {
        "error": {
            "code": status,
            "message": ERROR_MESSAGES[status],
            "metadata": metadata,
        },
    }


# The same page, "Anthropic Messages": the error type is in the error object, beside its type and
# message, in an envelope that has a type and a request id.
def build_skin_error(error_type: str) -> JSON:
    error: JSON = {
        "type": "invalid_request_error",
        "message": "Blocked",
        "error_type": error_type,
    }
    return {
        "type": "error",
        "error": error,
        "request_id": None,
    }


# https://openrouter.ai/docs/api_reference/responses/error-handling: an error body is
# {"error": {"code", "message"}, "metadata"} and its code is a string.
def build_coded_error(code: str) -> JSON:
    return {
        "error": {
            "code": code,
            "message": "Detailed error description",
        },
        "metadata": None,
    }


# Observed on 2026-10-19 (issue #30): the 400 for a prompt over the endpoint's context window,
# and the 400 for an unknown model id. Both have the number 400 as the error code and no error
# type, so only the message tells them apart.
CONTEXT_OVERFLOW_ERROR: JSON = {
    "error": {
        "message": (
            "This endpoint's maximum context length is 131072 tokens. However, you requested "
            "about 240016 tokens (240000 of text input, 16 in the output). Please reduce the "
            "length of either one, or use the context-compression plugin to compress your "
            "prompt automatically."
        ),
        "code": 400,
        "metadata": {"provider_name": None},
    },
}


def build_unknown_model_error(model: str) -> JSON:
    return {
        "error": {
            "message": f"{model} is not a valid model ID",
            "code": 400,
        },
    }


class Wire:
    """The stand-in for OpenRouter on the wire: it records each request and answers as told."""

    def __init__(
        self,
        respond: Respond,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.respond = respond

    def handle(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def build_client(self) -> OpenRouter:
        return OpenRouter(
            api_key="test-key",
            client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            async_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
            retry_config=None,
        )

    def read_bodies(
        self,
        *keys: str,
    ) -> list[JSON]:
        bodies = [json.loads(request.content) for request in self.requests]
        return [{key: body[key] for key in keys if key in body} for body in bodies]

    def read_paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


def build_reply(
    status: int,
    body: JSON,
) -> Respond:
    return lambda request: httpx.Response(
        status,
        json=body,
    )


# Constructed replies, not published ones: a gateway page, and a JSON body cut off mid-array.
def build_page(status: int) -> Respond:
    return lambda request: httpx.Response(
        status,
        text="<html>Gateway page</html>",
    )


def build_truncation() -> Respond:
    return lambda request: httpx.Response(
        200,
        content=b'{"id": "resp_abc123", "output": [',
        headers={"content-type": "application/json"},
    )


# Constructed too: a 200 labelled as an event stream, and one whose body is not the gzip it says.
def build_event_stream() -> Respond:
    return lambda request: httpx.Response(
        200,
        text="data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def build_bad_encoding() -> Respond:
    return lambda request: httpx.Response(
        200,
        content=b"not-gzip",
        headers={
            "content-type": "application/json",
            "content-encoding": "gzip",
        },
    )


def build_failure(error: Exception) -> Respond:
    def fail(request: httpx.Request) -> httpx.Response:
        raise error

    return fail


class SilentClient(httpx.Client):
    """A client that returns no response, the case the SDK reports as NoResponseError."""

    def send(
        self,
        request: httpx.Request,
        **options: object,
    ) -> httpx.Response:
        return cast(
            httpx.Response,
            None,
        )


class SilentAsyncClient(httpx.AsyncClient):
    """The async client that returns no response."""

    async def send(
        self,
        request: httpx.Request,
        **options: object,
    ) -> httpx.Response:
        return cast(
            httpx.Response,
            None,
        )


def build_silent_client() -> OpenRouter:
    return OpenRouter(
        api_key="test-key",
        client=SilentClient(),
        async_client=SilentAsyncClient(),
        retry_config=None,
    )


async def await_outcome[Outcome](awaitable: Awaitable[Outcome]) -> Outcome:
    return await awaitable


def generate_synchronously(
    client: OpenRouter,
    prompt: Prompt = PROMPT,
    max_output_tokens: int | None = None,
    reasoning_effort: components.ReasoningEffort | None = None,
) -> GenerationOutcome:
    model = GenerativeModel.from_openrouter(
        client,
        model=MODEL,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    return model.generate(prompt)


def generate_asynchronously(
    client: OpenRouter,
    prompt: Prompt = PROMPT,
    max_output_tokens: int | None = None,
    reasoning_effort: components.ReasoningEffort | None = None,
) -> GenerationOutcome:
    model = AsyncGenerativeModel.from_openrouter(
        client,
        model=MODEL,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    return asyncio.run(await_outcome(model.generate(prompt)))


def decide_synchronously(
    client: OpenRouter,
    question: Question = QUESTION,
) -> DecisionOutcome:
    return DecisionModel.from_openrouter(
        client,
        model=JEV,
    ).decide(question)


def decide_asynchronously(
    client: OpenRouter,
    question: Question = QUESTION,
) -> DecisionOutcome:
    model = AsyncDecisionModel.from_openrouter(
        client,
        model=JEV,
    )
    return asyncio.run(await_outcome(model.decide(question)))


type Ask = Callable[[OpenRouter], GenerationOutcome | DecisionOutcome]


@pytest.fixture(
    params=[
        generate_synchronously,
        generate_asynchronously,
    ],
    ids=["sync", "async"],
)
def generate(request: pytest.FixtureRequest) -> Callable[..., GenerationOutcome]:
    return request.param


@pytest.fixture(
    params=[
        decide_synchronously,
        decide_asynchronously,
    ],
    ids=["sync", "async"],
)
def decide(request: pytest.FixtureRequest) -> Callable[..., DecisionOutcome]:
    return request.param


@pytest.fixture(
    params=[
        generate_synchronously,
        generate_asynchronously,
        decide_synchronously,
        decide_asynchronously,
    ],
    ids=["responses-sync", "responses-async", "decisions-sync", "decisions-async"],
)
def ask(request: pytest.FixtureRequest) -> Ask:
    return request.param


@pytest.mark.parametrize(
    ("build", "name", "perceives"),
    [
        (
            lambda client, perceives: GenerativeModel.from_openrouter(
                client,
                model=MODEL,
                perceives=perceives,
            ),
            f"openrouter:{MODEL}",
            MediaTypes("image/*", "application/pdf", "text/plain"),
        ),
        (
            lambda client, perceives: AsyncGenerativeModel.from_openrouter(
                client,
                model=MODEL,
                perceives=perceives,
            ),
            f"openrouter:{MODEL}",
            MediaTypes("image/*", "application/pdf", "text/plain"),
        ),
        (
            lambda client, perceives: DecisionModel.from_openrouter(
                client,
                model=JEV,
                perceives=perceives,
            ),
            f"openrouter:{JEV}",
            MediaTypes("image/png", "image/jpeg", "image/webp"),
        ),
        (
            lambda client, perceives: AsyncDecisionModel.from_openrouter(
                client,
                model=JEV,
                perceives=perceives,
            ),
            f"openrouter:{JEV}",
            MediaTypes("image/png", "image/jpeg", "image/webp"),
        ),
    ],
    ids=["generative", "async-generative", "decision", "async-decision"],
)
def test_a_model_is_named_for_the_provider_and_the_model_id(
    build: Callable[
        [
            OpenRouter,
            MediaTypes,
        ],
        Model,
    ],
    name: str,
    perceives: MediaTypes,
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()

    model = build(
        client,
        perceives,
    )

    assert (
        model.name,
        model.perceives,
    ) == (
        name,
        perceives,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda client, perceives: GenerativeModel.from_openrouter(
            client,
            model=MODEL,
            perceives=perceives,
        ),
        lambda client, perceives: AsyncGenerativeModel.from_openrouter(
            client,
            model=MODEL,
            perceives=perceives,
        ),
    ],
    ids=["generative", "async-generative"],
)
@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (
            ("audio/*",),
            "audio/*",
        ),
        (
            ("video/mp4",),
            "video/mp4",
        ),
        (
            ("video/mp4", "image/png", "audio/wav"),
            "audio/wav",
        ),
    ],
    ids=["range-of-another-type", "another-type", "the-first-of-those-it-cannot-carry"],
)
def test_a_responses_model_that_perceives_media_the_api_cannot_carry_raises_naming_it_when_built(
    build: Callable[
        [
            OpenRouter,
            MediaTypes,
        ],
        Model,
    ],
    patterns: tuple[str, ...],
    named: str,
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()

    with pytest.raises(
        ConfigurationError,
        match=re.escape(named),
    ):
        build(
            client,
            MediaTypes(*patterns),
        )


# https://openrouter.ai/docs/guides/community/multimodal-decisions: the Decisions API reads png,
# jpeg and webp images; a range such as image/* names types it cannot read.
@pytest.mark.parametrize(
    "build",
    [
        lambda client, perceives: DecisionModel.from_openrouter(
            client,
            model=JEV,
            perceives=perceives,
        ),
        lambda client, perceives: AsyncDecisionModel.from_openrouter(
            client,
            model=JEV,
            perceives=perceives,
        ),
    ],
    ids=["decision", "async-decision"],
)
@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (
            ("image/*",),
            "image/*",
        ),
        (
            ("image/gif",),
            "image/gif",
        ),
        (
            ("application/pdf",),
            "application/pdf",
        ),
        (
            ("video/mp4", "image/png", "audio/wav"),
            "audio/wav",
        ),
    ],
    ids=[
        "image-range",
        "image-type-outside-the-three",
        "pdf",
        "the-first-of-those-it-cannot-carry",
    ],
)
def test_a_decision_model_that_perceives_media_the_api_cannot_read_raises_naming_it_when_built(
    build: Callable[
        [
            OpenRouter,
            MediaTypes,
        ],
        Model,
    ],
    patterns: tuple[str, ...],
    named: str,
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()

    with pytest.raises(
        ConfigurationError,
        match=re.escape(named),
    ):
        build(
            client,
            MediaTypes(*patterns),
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda client: GenerativeModel.from_openrouter(
            client,
            model="",
        ),
        lambda client: AsyncGenerativeModel.from_openrouter(
            client,
            model="",
        ),
        lambda client: DecisionModel.from_openrouter(
            client,
            model="",
        ),
        lambda client: AsyncDecisionModel.from_openrouter(
            client,
            model="",
        ),
    ],
    ids=["generative", "async-generative", "decision", "async-decision"],
)
def test_a_model_without_a_model_id_raises(build: Callable[[OpenRouter], Model]) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()

    with pytest.raises(ConfigurationError):
        build(client)


def test_the_published_completed_response_is_a_generation(
    generate: Callable[..., GenerationOutcome],
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Generation(text=JOKE)


def test_the_prompt_becomes_instructions_and_one_user_message(
    generate: Callable[..., GenerationOutcome],
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )

    generate(wire.build_client())

    # The user message is the structured input of https://openrouter.ai/docs/api_reference/responses/basic-usage
    # and the endpoint is its /api/v1/responses; the API is stateless, so store is always false.
    assert (
        wire.read_paths(),
        wire.read_bodies("model", "instructions", "input", "store"),
    ) == (
        ["/api/v1/responses"],
        [
            {
                "model": MODEL,
                "instructions": "Rate the action.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Run: ls",
                            },
                        ],
                    },
                ],
                "store": False,
            },
        ],
    )


@pytest.mark.parametrize(
    ("max_output_tokens", "sent"),
    [
        (
            None,
            [{}],
        ),
        (
            256,
            [{"max_output_tokens": 256}],
        ),
    ],
)
def test_the_output_limit_is_sent_only_when_given(
    generate: Callable[..., GenerationOutcome],
    max_output_tokens: int | None,
    sent: list[JSON],
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )

    generate(
        wire.build_client(),
        max_output_tokens=max_output_tokens,
    )

    assert wire.read_bodies("max_output_tokens") == sent


# https://openrouter.ai/docs/api_reference/responses/reasoning: the request carries "reasoning":
# {"effort": "high"}; "Reasoning Effort Levels" lists minimal, low, medium and high.
@pytest.mark.parametrize(
    ("reasoning_effort", "sent"),
    [
        (
            None,
            [{}],
        ),
        (
            "high",
            [{"reasoning": {"effort": "high"}}],
        ),
    ],
    ids=["no-effort-sends-no-field", "effort-is-sent"],
)
def test_the_reasoning_effort_is_sent_only_when_given(
    generate: Callable[..., GenerationOutcome],
    reasoning_effort: components.ReasoningEffort | None,
    sent: list[JSON],
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )

    generate(
        wire.build_client(),
        reasoning_effort=reasoning_effort,
    )

    assert wire.read_bodies("reasoning") == sent


# A lone surrogate is what json.loads makes of "\ud800", as in a tool's output.
def test_a_lone_surrogate_in_the_prompt_reaches_the_wire_and_the_reply_is_read(
    generate: Callable[..., GenerationOutcome],
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )
    prompt = Prompt(
        system="Rate the action.",
        content=(TextPart(text="a\ud800b"),),
    )

    outcome = generate(
        wire.build_client(),
        prompt,
    )

    message: JSON = {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "a\ud800b",
            },
        ],
    }
    assert (
        outcome,
        wire.read_bodies("input"),
    ) == (
        Generation(text=JOKE),
        [{"input": [message]}],
    )


def test_a_lone_surrogate_in_the_state_reaches_the_wire_and_the_reply_is_read(
    decide: Callable[..., DecisionOutcome],
) -> None:
    wire = Wire(
        build_reply(
            200,
            build_decisions({"statement": IS_BUG}),
        ),
    )
    question = Question(
        statement="The action is safe.",
        state=(TextPart(text="a\ud800b"),),
    )

    outcome = decide(
        wire.build_client(),
        question,
    )

    assert (
        outcome,
        wire.read_bodies("state"),
    ) == (
        Decision(probability=0.96),
        [{"state": ["a\ud800b"]}],
    )


# The wire parts are the schemas InputImage (its example, https://example.com/image.jpg, detail
# auto) and InputFile (file_url, file_data, filename) of
# https://openrouter.ai/docs/api/api-reference/responses/create-a-response; the PDF URL is
# the one of https://openrouter.ai/docs/guides/overview/multimodal/pdfs. The base64 and the
# filename extensions follow RFC 4648 and the standard library's table of media types.
@pytest.mark.parametrize(
    ("part", "content"),
    [
        (
            MediaPart(
                media_type="image/*",
                url="https://example.com/image.jpg",
            ),
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": "https://example.com/image.jpg",
            },
        ),
        (
            PNG,
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": PNG_URL,
            },
        ),
        (
            MediaPart(
                media_type="IMAGE/PNG",
                url="https://example.com/image.png",
            ),
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": "https://example.com/image.png",
            },
        ),
        (
            MediaPart(
                media_type="application/pdf",
                url="https://bitcoin.org/bitcoin.pdf",
            ),
            {
                "type": "input_file",
                "file_url": "https://bitcoin.org/bitcoin.pdf",
            },
        ),
        (
            MediaPart(
                media_type="application/pdf",
                data=b"%PDF-1.7",
            ),
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "attachment.pdf",
            },
        ),
        (
            MediaPart(
                media_type="text/plain",
                data=b"ls",
            ),
            {
                "type": "input_file",
                "file_data": "data:text/plain;base64,bHM=",
                "filename": "attachment.txt",
            },
        ),
        (
            MediaPart(
                media_type="application/x-unlisted",
                data=b"ls",
            ),
            {
                "type": "input_file",
                "file_data": "data:application/x-unlisted;base64,bHM=",
                "filename": "attachment",
            },
        ),
    ],
    ids=[
        "image-by-url",
        "image-bytes",
        "uppercase-type",
        "file-by-url",
        "pdf-bytes",
        "text-bytes",
        "unknown-type-bytes",
    ],
)
def test_media_becomes_the_responses_api_s_image_and_file_content(
    generate: Callable[..., GenerationOutcome],
    part: MediaPart,
    content: JSON,
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )
    prompt = Prompt(
        system="Rate the action.",
        content=(
            TextPart(text="See:"),
            part,
        ),
    )

    generate(
        wire.build_client(),
        prompt,
    )

    assert wire.read_bodies("input") == [
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "See:",
                        },
                        content,
                    ],
                },
            ],
        },
    ]


@pytest.mark.parametrize(
    "part",
    [
        MediaPart(
            media_type="audio/mpeg",
            data=b"ID3",
        ),
        MediaPart(
            media_type="video/mp4",
            url="https://example.com/clip.mp4",
        ),
    ],
    ids=["audio", "video"],
)
def test_media_the_responses_api_cannot_carry_raises_naming_its_type(
    generate: Callable[..., GenerationOutcome],
    part: MediaPart,
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()
    prompt = Prompt(
        system="Rate the action.",
        content=(part,),
    )

    with pytest.raises(
        ConfigurationError,
        match=re.escape(part.media_type),
    ):
        generate(
            client,
            prompt,
        )


# A range such as image/* is a type the API carries, but its bytes have no type for a data URL.
RANGE_BYTES = MediaPart(
    media_type="image/*",
    data=b"\x89PNG",
)
AUDIO = MediaPart(
    media_type="audio/mpeg",
    data=b"ID3",
)


@pytest.mark.parametrize(
    "part",
    [
        RANGE_BYTES,
        MediaPart(
            media_type="application/*",
            data=b"%PDF-1.7",
        ),
    ],
    ids=["image-range", "file-range"],
)
def test_bytes_of_a_media_range_are_unavailable_without_a_status(
    generate: Callable[..., GenerationOutcome],
    part: MediaPart,
) -> None:
    wire = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    )
    prompt = Prompt(
        system="Rate the action.",
        content=(part,),
    )

    outcome = generate(
        wire.build_client(),
        prompt,
    )

    assert (
        outcome,
        wire.requests,
    ) == (
        Unavailable(),
        [],
    )


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            (
                RANGE_BYTES,
                AUDIO,
            ),
            id="range-first",
        ),
        pytest.param(
            (
                AUDIO,
                RANGE_BYTES,
            ),
            id="range-last",
        ),
    ],
)
def test_a_type_the_responses_api_cannot_carry_raises_whatever_the_order(
    generate: Callable[..., GenerationOutcome],
    content: tuple[MediaPart, ...],
) -> None:
    client = Wire(
        build_reply(
            200,
            COMPLETED,
        ),
    ).build_client()
    prompt = Prompt(
        system="Rate the action.",
        content=content,
    )

    with pytest.raises(
        ConfigurationError,
        match="audio/mpeg",
    ):
        generate(
            client,
            prompt,
        )


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (
            build_response(output=[build_message(REFUSAL_PART)]),
            Refused(explanation=REFUSAL),
        ),
        (
            build_response(
                output=[
                    build_message(
                        build_text("Why did the chicken "),
                        build_text("cross the road? "),
                    ),
                    build_message(build_text("To get to the other side!")),
                ],
            ),
            Generation(text=JOKE),
        ),
        (
            build_response(
                output=[
                    REASONING,
                    build_message(build_text(JOKE)),
                ],
            ),
            Generation(text=JOKE),
        ),
        (
            build_response(
                output=[
                    {"type": "unlisted"},
                    build_message(build_text(JOKE)),
                ],
            ),
            Generation(text=JOKE),
        ),
        (
            build_response(
                output=[
                    build_message(
                        build_text(JOKE),
                        {"type": "unlisted"},
                    ),
                ],
            ),
            Truncated(text=JOKE),
        ),
        (
            build_response(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            Truncated(text=JOKE),
        ),
        (
            build_response(status="incomplete"),
            Truncated(text=JOKE),
        ),
        (
            build_response(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
                output=[build_message(REFUSAL_PART)],
            ),
            Truncated(text=""),
        ),
        (
            build_response(status="cancelled"),
            Truncated(text=JOKE),
        ),
        (
            build_response(status="in_progress"),
            Truncated(text=JOKE),
        ),
        (
            build_response(
                status="failed",
                error={
                    "code": "server_error",
                    "message": EVALUATOR_FAILED,
                },
            ),
            Unavailable(),
        ),
    ],
    ids=[
        "refusal-part",
        "text-in-several-parts-and-messages",
        "reasoning-before-the-message",
        "item-of-an-unlisted-type",
        "part-of-an-unlisted-type",
        "incomplete-at-the-output-limit",
        "incomplete-without-a-reason",
        "incomplete-with-a-refusal-part",
        "cancelled",
        "in-progress",
        "failed",
    ],
)
def test_a_response_becomes_the_outcome_its_status_gives(
    generate: Callable[..., GenerationOutcome],
    response: JSON,
    outcome: GenerationOutcome,
) -> None:
    client = Wire(
        build_reply(
            200,
            response,
        ),
    ).build_client()

    generated = generate(client)

    assert generated == outcome


# Constructed from the published envelope: a message, or a part of one, that names a documented
# type but breaks its schema, and replies with no message to read.
CRACKED_MESSAGE: JSON = {
    "type": "message",
    "id": "msg-abc123",
    "role": "assistant",
    "status": "completed",
    "content": "not-a-list",
}


@pytest.mark.parametrize(
    "response",
    [
        build_response(output=[CRACKED_MESSAGE]),
        build_response(output=[build_message({"refusal": "I refuse"})]),
        build_response(
            output=[
                build_message(
                    {
                        "type": "output_text",
                        "text": 123,
                    },
                ),
            ],
        ),
        build_response(
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
            output=[CRACKED_MESSAGE],
        ),
    ],
    ids=[
        "message-without-a-list-of-parts",
        "part-without-a-type",
        "text-of-the-wrong-type",
        "incomplete-with-a-cracked-message",
    ],
)
def test_a_response_with_an_item_or_part_the_sdk_could_not_read_is_unavailable(
    generate: Callable[..., GenerationOutcome],
    response: JSON,
) -> None:
    client = Wire(
        build_reply(
            200,
            response,
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Unavailable()


@pytest.mark.parametrize(
    "response",
    [
        build_response(output=[]),
        build_response(output=[REASONING]),
    ],
    ids=[
        "no-output",
        "no-message",
    ],
)
def test_a_completed_response_without_a_message_is_an_empty_generation(
    generate: Callable[..., GenerationOutcome],
    response: JSON,
) -> None:
    client = Wire(
        build_reply(
            200,
            response,
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Generation(text="")


@pytest.mark.parametrize(
    "part",
    [
        {"type": "refusal"},
        {
            "refusal": None,
            "type": "refusal",
        },
    ],
    ids=["no-text", "null-text"],
)
def test_a_refusal_part_without_its_text_is_a_refusal_without_an_explanation(
    generate: Callable[..., GenerationOutcome],
    part: JSON,
) -> None:
    client = Wire(
        build_reply(
            200,
            build_response(output=[build_message(part)]),
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Refused()


# content_filter is the other reason of the schema IncompleteDetails; the error codes are those
# of OpenResponsesErrorField, and the error types those of
# https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes
@pytest.mark.parametrize(
    "response",
    [
        build_response(
            status="incomplete",
            incomplete_details={"reason": "content_filter"},
        ),
        build_response(
            status="failed",
            error={
                "code": "image_content_policy_violation",
                "message": EVALUATOR_FAILED,
            },
        ),
        build_response(
            status="failed",
            error={
                "code": "server_error",
                "message": EVALUATOR_FAILED,
            },
            error_type="content_policy_violation",
        ),
        build_response(
            status="failed",
            error={
                "code": "invalid_prompt",
                "message": EVALUATOR_FAILED,
            },
            error_type="refusal",
        ),
    ],
    ids=["content-filter", "policy-code", "policy-error-type", "refusal-error-type"],
)
def test_a_response_the_provider_blocks_is_refused(
    generate: Callable[..., GenerationOutcome],
    response: JSON,
) -> None:
    client = Wire(
        build_reply(
            200,
            response,
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Refused()


# The same page, "Responses API": a context overflow takes the code invalid_prompt and keeps
# its error type at the top level of the failed reply.
@pytest.mark.xfail(strict=True)
def test_a_response_whose_error_type_says_the_input_cannot_be_used_is_rejected(
    generate: Callable[..., GenerationOutcome],
) -> None:
    client = Wire(
        build_reply(
            200,
            build_response(
                status="failed",
                error={
                    "code": "invalid_prompt",
                    "message": "Detailed error description",
                },
                error_type="context_length_exceeded",
            ),
        ),
    ).build_client()

    outcome = generate(client)

    assert outcome == Rejected()


@pytest.mark.parametrize(
    ("answer", "probability"),
    [
        (
            IS_BUG,
            0.96,
        ),
        (
            MATCHES_TITLE,
            0.8607,
        ),
        (
            {
                "noul": 0.0,
                "type": "noul",
            },
            0.0,
        ),
        (
            {
                "noul": 1.0,
                "type": "noul",
            },
            1.0,
        ),
    ],
    ids=["published", "published-with-an-image", "lower-bound", "upper-bound"],
)
def test_the_noul_answer_is_the_probability_the_statement_holds(
    decide: Callable[..., DecisionOutcome],
    answer: JSON,
    probability: float,
) -> None:
    client = Wire(
        build_reply(
            200,
            build_decisions({"statement": answer}),
        ),
    ).build_client()

    outcome = decide(client)

    assert outcome == Decision(
        probability=probability,
        confidence=None,
    )


def test_the_question_becomes_one_noul_question_about_the_state(
    decide: Callable[..., DecisionOutcome],
) -> None:
    wire = Wire(
        build_reply(
            200,
            build_decisions({"statement": IS_BUG}),
        ),
    )

    decide(wire.build_client())

    # A noul question is {"type", "instructions"} and the state an array of strings, in the
    # Decisions reference's request example and the multimodal guide; the path is the reference's.
    assert (
        wire.read_paths(),
        wire.read_bodies("model", "state", "questions"),
    ) == (
        ["/api/alpha/decisions"],
        [
            {
                "model": JEV,
                "state": ["Run: ls"],
                "questions": {
                    "statement": {
                        "type": "noul",
                        "instructions": "The action is safe.",
                    },
                },
            },
        ],
    )


@pytest.mark.parametrize(
    ("part", "item"),
    [
        (
            PNG,
            {
                "type": "image_url",
                "image_url": {"url": PNG_URL},
            },
        ),
        (
            MediaPart(
                media_type="image/jpeg",
                data=b"\x89PNG",
            ),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,iVBORw=="},
            },
        ),
        (
            MediaPart(
                media_type="image/webp",
                data=b"\x89PNG",
            ),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/webp;base64,iVBORw=="},
            },
        ),
    ],
    ids=["png-bytes", "jpeg-bytes", "webp-bytes"],
)
def test_an_image_becomes_an_image_url_object_in_the_state(
    decide: Callable[..., DecisionOutcome],
    part: MediaPart,
    item: JSON,
) -> None:
    wire = Wire(
        build_reply(
            200,
            build_decisions({"statement": MATCHES_TITLE}),
        ),
    )
    question = Question(
        statement=STICKER_STATEMENT,
        state=(
            TextPart(text=STICKER_TITLE),
            part,
        ),
    )

    decide(
        wire.build_client(),
        question,
    )

    assert wire.read_bodies("state", "questions") == [
        {
            "state": [
                STICKER_TITLE,
                item,
            ],
            "questions": {
                "statement": {
                    "type": "noul",
                    "instructions": STICKER_STATEMENT,
                },
            },
        },
    ]


@pytest.mark.parametrize(
    "part",
    [
        MediaPart(
            media_type="audio/wav",
            data=b"RIFF",
        ),
        MediaPart(
            media_type="image/gif",
            data=b"GIF8",
        ),
    ],
    ids=["audio", "image-type-outside-the-three"],
)
def test_media_the_decisions_api_cannot_carry_raises_naming_its_type(
    decide: Callable[..., DecisionOutcome],
    part: MediaPart,
) -> None:
    client = Wire(
        build_reply(
            200,
            build_decisions({"statement": IS_BUG}),
        ),
    ).build_client()
    question = Question(
        statement=STICKER_STATEMENT,
        state=(part,),
    )

    with pytest.raises(
        ConfigurationError,
        match=re.escape(part.media_type),
    ):
        decide(
            client,
            question,
        )


# https://openrouter.ai/docs/guides/community/multimodal-decisions: an image is a base64 data URL
# in the state, and "remote http(s) URLs are not fetched". A MediaPart URL is a URL even if it
# is a data URL; the bytes of a range such as image/* have no type to put in one.
@pytest.mark.parametrize(
    "part",
    [
        MediaPart(
            media_type="image/png",
            url="https://example.com/image.png",
        ),
        MediaPart(
            media_type="image/png",
            url=PNG_URL,
        ),
        MediaPart(
            media_type="image/*",
            url="https://example.com/image.png",
        ),
        RANGE_BYTES,
    ],
    ids=["image-by-url", "image-by-data-url", "range-by-url", "range-bytes"],
)
def test_an_image_in_a_form_the_decisions_api_cannot_take_is_unavailable_without_a_status(
    decide: Callable[..., DecisionOutcome],
    part: MediaPart,
) -> None:
    wire = Wire(
        build_reply(
            200,
            build_decisions({"statement": IS_BUG}),
        ),
    )
    question = Question(
        statement=STICKER_STATEMENT,
        state=(
            TextPart(text=STICKER_TITLE),
            part,
        ),
    )

    outcome = decide(
        wire.build_client(),
        question,
    )

    assert (
        outcome,
        wire.requests,
    ) == (
        Unavailable(),
        [],
    )


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(
            (
                MediaPart(
                    media_type="image/png",
                    url="https://example.com/image.png",
                ),
                AUDIO,
            ),
            id="image-by-url-first",
        ),
        pytest.param(
            (
                AUDIO,
                MediaPart(
                    media_type="image/png",
                    url="https://example.com/image.png",
                ),
            ),
            id="image-by-url-last",
        ),
    ],
)
def test_a_type_the_decisions_api_cannot_carry_raises_whatever_the_order(
    decide: Callable[..., DecisionOutcome],
    state: tuple[MediaPart, ...],
) -> None:
    client = Wire(
        build_reply(
            200,
            build_decisions({"statement": IS_BUG}),
        ),
    ).build_client()
    question = Question(
        statement=STICKER_STATEMENT,
        state=state,
    )

    with pytest.raises(
        ConfigurationError,
        match="audio/mpeg",
    ):
        decide(
            client,
            question,
        )


# The published answers of another question and of another type (TEAM, a choice answer), and
# answers constructed to be outside the schema: an unlisted type and probabilities outside [0, 1].
@pytest.mark.parametrize(
    "answers",
    [
        {"is_bug": IS_BUG},
        {"statement": TEAM},
        {"statement": {"type": "unlisted"}},
        {
            "statement": {
                "noul": 1.2,
                "type": "noul",
            },
        },
        {
            "statement": {
                "noul": -0.1,
                "type": "noul",
            },
        },
        {
            "statement": {
                "noul": "high",
                "type": "noul",
            },
        },
    ],
    ids=[
        "no-answer-to-the-question",
        "answer-of-another-type",
        "unknown-type",
        "above-one",
        "below-zero",
        "noul-of-the-wrong-type",
    ],
)
def test_an_answer_that_is_not_a_probability_is_unavailable_without_a_status(
    decide: Callable[..., DecisionOutcome],
    answers: JSON,
) -> None:
    client = Wire(
        build_reply(
            200,
            build_decisions(answers),
        ),
    ).build_client()

    outcome = decide(client)

    assert outcome == Unavailable()


@pytest.mark.parametrize(
    "status",
    [408, 409, 429, 500, 502, 503, 524, 529],
)
def test_a_status_that_means_try_later_is_unavailable_with_that_status(
    ask: Ask,
    status: int,
) -> None:
    client = Wire(
        build_reply(
            status,
            build_error(status),
        ),
    ).build_client()

    outcome = ask(client)

    assert outcome == Unavailable(status=status)


def test_a_gateway_page_is_judged_by_its_status(ask: Ask) -> None:
    client = Wire(build_page(502)).build_client()

    outcome = ask(client)

    assert outcome == Unavailable(status=502)


# The envelopes are the published error examples; invalid_request and permission_denied are error
# types of https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes, put
# in the envelope's error_type field, where that page puts the Responses API's error type.
@pytest.mark.parametrize(
    ("status", "fields"),
    [
        (
            400,
            {},
        ),
        (
            401,
            {},
        ),
        (
            402,
            {},
        ),
        (
            403,
            {},
        ),
        (
            404,
            {},
        ),
        (
            422,
            {},
        ),
        (
            400,
            {"error_type": "invalid_request"},
        ),
        (
            403,
            {"error_type": "permission_denied"},
        ),
    ],
    ids=["400", "401", "402", "403", "404", "422", "invalid-request", "permission-denied"],
)
def test_any_other_rejection_raises_naming_only_the_status(
    ask: Ask,
    status: int,
    fields: dict[str, JSON],
) -> None:
    client = Wire(
        build_reply(
            status,
            build_error(
                status,
                **fields,
            ),
        ),
    ).build_client()

    with pytest.raises(ConfigurationError) as raised:
        ask(client)

    # The provider's message can echo the prompt: it is in neither the message nor the chain of
    # exceptions, which holds the provider's error as its cause or its context.
    message = str(raised.value)
    assert (
        str(status) in message,
        ERROR_MESSAGES[status] in message,
        raised.value.__cause__,
        raised.value.__context__,
    ) == (True, False, None, None)


# The second id holds the overflow wording, which counts only at the start of a message.
@pytest.mark.parametrize(
    "model",
    [
        "openai/no-such-model-xyz",
        "openai/maximum context length",
    ],
)
def test_an_unknown_model_id_raises_naming_only_the_status(
    ask: Ask,
    model: str,
) -> None:
    client = Wire(
        build_reply(
            400,
            build_unknown_model_error(model),
        ),
    ).build_client()

    with pytest.raises(
        ConfigurationError,
        match="400",
    ):
        ask(client)


def test_a_rejection_with_a_gateway_page_raises_naming_only_the_status(ask: Ask) -> None:
    client = Wire(build_page(403)).build_client()

    with pytest.raises(
        ConfigurationError,
        match="403",
    ):
        ask(client)


# image_content_policy_violation is the code of a 403 in the table of
# https://openrouter.ai/docs/api_reference/responses/error-handling; content_policy_violation
# and refusal are error types of the 403 rows of
# https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes, put where that
# page puts the error type: in error.metadata ("Provider errors"), in the error object of the
# Anthropic skin, and, composed here with the published 403 envelope, at the top level, where
# it puts the error type of a failed Responses reply.
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            build_coded_error("image_content_policy_violation"),
            id="policy-code",
        ),
        pytest.param(
            build_error(
                403,
                error_type="content_policy_violation",
            ),
            id="top-policy",
        ),
        pytest.param(
            build_error(
                403,
                error_type="refusal",
            ),
            id="top-refusal",
        ),
        pytest.param(
            build_typed_error(403, "content_policy_violation"),
            id="metadata-policy",
        ),
        pytest.param(
            build_typed_error(403, "refusal"),
            id="metadata-refusal",
        ),
        pytest.param(
            build_skin_error("content_policy_violation"),
            id="skin-policy",
        ),
    ],
)
def test_a_request_the_provider_blocks_is_refused_without_an_explanation(
    ask: Ask,
    body: JSON,
) -> None:
    client = Wire(
        build_reply(
            403,
            body,
        ),
    ).build_client()

    outcome = ask(client)

    assert outcome == Refused()


# Each error type with the HTTP status that the typed-error-codes tables of
# https://openrouter.ai/docs/api_reference/errors-and-debugging give it, in error.metadata; the
# image codes of the schema OpenResponsesErrorField, with the status 400 (the schema gives none),
# as the string code of the error body of
# https://openrouter.ai/docs/api_reference/responses/error-handling. The Decisions reference
# documents only integer codes, so no page shows these on its replies.
TYPED_REJECTIONS = [
    (400, "context_length_exceeded"),
    (400, "string_too_long"),
    (400, "invalid_image"),
    (400, "image_too_large"),
    (400, "image_too_small"),
    (400, "unsupported_image_format"),
    (400, "image_download_failed"),
    (404, "image_not_found"),
]
CODED_REJECTIONS = [
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_parse_error",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
]


@pytest.mark.parametrize(
    ("status", "body"),
    [
        *(
            pytest.param(
                status,
                build_typed_error(
                    status,
                    error_type,
                ),
                id=error_type,
            )
            for status, error_type in TYPED_REJECTIONS
        ),
        *(
            pytest.param(
                400,
                build_coded_error(code),
                id=code,
            )
            for code in CODED_REJECTIONS
        ),
        pytest.param(
            400,
            build_skin_error("context_length_exceeded"),
            id="skin",
        ),
        pytest.param(
            413,
            build_typed_error(413, "payload_too_large"),
            id="payload_too_large",
        ),
        pytest.param(
            400,
            build_error(
                400,
                error_type="context_length_exceeded",
            ),
            id="top-level",
        ),
        pytest.param(
            400,
            CONTEXT_OVERFLOW_ERROR,
            id="context-length-message",
        ),
        pytest.param(
            413,
            build_error(413),
            id="payload-too-large-status",
        ),
    ],
)
@pytest.mark.xfail(strict=True)
def test_input_the_provider_rejects_is_rejected_with_the_status(
    ask: Ask,
    status: int,
    body: JSON,
) -> None:
    client = Wire(
        build_reply(
            status,
            body,
        ),
    ).build_client()

    outcome = ask(client)

    assert outcome == Rejected(status=status)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
    ids=["connection", "timeout", "disconnect"],
)
def test_a_connection_failure_is_unavailable_without_a_status(
    ask: Ask,
    error: Exception,
) -> None:
    client = Wire(build_failure(error)).build_client()

    outcome = ask(client)

    assert outcome == Unavailable()


def test_no_response_is_unavailable_without_a_status(ask: Ask) -> None:
    client = build_silent_client()

    outcome = ask(client)

    assert outcome == Unavailable()


# Constructed, not published: an empty object, a body that names a refusal but is no reply, a
# gateway page, a body cut off mid-array, a body that is not the gzip it says, a success status
# with no body, and an event stream (which only the Responses API can send).
@pytest.mark.parametrize(
    "respond",
    [
        pytest.param(
            build_reply(
                200,
                {},
            ),
            id="json-of-no-known-shape",
        ),
        pytest.param(
            build_reply(
                200,
                {"error_type": "refusal"},
            ),
            id="refusal-in-no-reply",
        ),
        pytest.param(
            build_page(200),
            id="gateway-page",
        ),
        pytest.param(
            build_truncation(),
            id="truncated-json",
        ),
        pytest.param(
            build_bad_encoding(),
            id="bad-content-encoding",
        ),
        pytest.param(
            lambda request: httpx.Response(204),
            id="no-content",
        ),
    ],
)
def test_a_reply_that_cannot_be_read_is_unavailable_without_a_status(
    ask: Ask,
    respond: Respond,
) -> None:
    client = Wire(respond).build_client()

    outcome = ask(client)

    assert outcome == Unavailable()


def test_an_event_stream_is_unavailable_without_a_status(
    generate: Callable[..., GenerationOutcome],
) -> None:
    client = Wire(build_event_stream()).build_client()

    outcome = generate(client)

    assert outcome == Unavailable()


@pytest.mark.parametrize(
    "status",
    [300, 302],
)
def test_a_status_between_success_and_error_is_unavailable_with_that_status(
    ask: Ask,
    status: int,
) -> None:
    client = Wire(
        lambda request: httpx.Response(
            status,
            headers={"location": "https://openrouter.ai/"},
        ),
    ).build_client()

    outcome = ask(client)

    assert outcome == Unavailable(status=status)
