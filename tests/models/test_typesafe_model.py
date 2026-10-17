"""TypeSafe's decision models, tested through the TypeSafe SDK over a mock HTTP transport."""

import asyncio
import base64
import json
from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient

from hyoshigi import (
    JSON,
    AsyncDecisionModel,
    ConfigurationError,
    Decision,
    DecisionModel,
    DecisionOutcome,
    MediaPart,
    MediaTypes,
    Question,
    TextPart,
    Unavailable,
)

type Handler = Callable[[httpx2.Request], httpx2.Response]
type ModelBuilder = Callable[..., DecisionModel | AsyncDecisionModel]

TEXT_ONLY = MediaTypes()

# https://api.typesafe.ai/openapi.json, POST /v1/systemone, the "Noul", "Choice" and "Box"
# examples, and components.schemas.HTTPValidationError.
BILLING_STATE = (
    "I was charged twice for my subscription this month. Please refund the extra charge."
)
BILLING_STATEMENT = "Is this message about billing?"
BILLING_QUESTION = Question(
    statement=BILLING_STATEMENT,
    state=(TextPart(text=BILLING_STATE),),
)
NOUL_RESPONSE: JSON = {
    "model": "jev-latest",
    "answers": {"statement": {"type": "noul", "noul": 0.98}},
    "usage": {"input_tokens": 120, "output_tokens": 12},
}
CHOICE_RESPONSE: JSON = {
    "model": "jev-latest",
    "answers": {
        "statement": {
            "type": "choice",
            "choice": "calm",
            "confidence": 0.7,
            "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05},
        }
    },
    "usage": {"input_tokens": 120, "output_tokens": 12},
}
RED_RECTANGLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAHElEQVR4nGP4TyJgoKcGBgbs"
    "aFQDfTUQB4aDBgDAen2fpitsVAAAAABJRU5ErkJggg=="
)
# RFC 2046 / JFIF SOI marker recognized by typesafe_sdk._core.assets.CONTENT_TYPES.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF"


def build_sync_model(
    handler: Handler, *, model: str = "jev-latest", perceives: MediaTypes = TEXT_ONLY
) -> DecisionModel:
    client = TypeSafeClient(
        api_key="ts-test",
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    )
    return DecisionModel.from_typesafe(client, model=model, perceives=perceives)


def build_async_model(
    handler: Handler, *, model: str = "jev-latest", perceives: MediaTypes = TEXT_ONLY
) -> AsyncDecisionModel:
    client = AsyncTypeSafeClient(
        api_key="ts-test",
        retry=RetryPolicy(max_retries=0),
        transport=httpx2.MockTransport(handler),
    )
    return AsyncDecisionModel.from_typesafe(client, model=model, perceives=perceives)


def run_decision(
    model: DecisionModel | AsyncDecisionModel, *, question: Question
) -> DecisionOutcome:
    match model:
        case DecisionModel(decide=decide):
            return decide(question)
        case AsyncDecisionModel(decide=decide):

            async def call() -> DecisionOutcome:
                return await decide(question)

            return asyncio.run(call())


def return_noul_response(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=NOUL_RESPONSE)


def parse_multipart(request: httpx2.Request) -> tuple[JSON, str, bytes]:
    header = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = BytesParser(policy=default).parsebytes(header + request.content)
    request_part, asset_part = list(message.iter_parts())
    return (
        json.loads(request_part.get_content()),
        asset_part.get_content_type(),
        asset_part.get_content(),
    )


BUILDERS = pytest.mark.parametrize(
    "build_model", [build_sync_model, build_async_model], ids=["sync", "async"]
)


@pytest.mark.xfail(strict=True)
@BUILDERS
def test_the_model_carries_its_typesafe_name_with_its_media_types(
    build_model: ModelBuilder,
) -> None:
    perceives = MediaTypes("image/png", "image/jpeg")

    model = build_model(return_noul_response, model="jev-1.14.0", perceives=perceives)

    assert (model.name, model.perceives) == ("typesafe:jev-1.14.0", perceives)


@BUILDERS
def test_an_empty_model_name_raises_configuration_error(build_model: ModelBuilder) -> None:
    with pytest.raises(ConfigurationError):
        build_model(return_noul_response, model="")


@pytest.mark.xfail(strict=True)
@BUILDERS
def test_a_text_question_is_sent_as_a_noul_question_about_the_state(
    build_model: ModelBuilder,
) -> None:
    captured: list[tuple[str, str, JSON]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append((request.method, str(request.url), json.loads(request.content)))
        return httpx2.Response(200, json=NOUL_RESPONSE)

    model = build_model(handler, model="jev-latest")

    outcome = run_decision(model, question=BILLING_QUESTION)

    assert (captured, outcome) == (
        [
            (
                "POST",
                "https://api.typesafe.ai/v1/systemone",
                {
                    "model": "jev-latest",
                    "state": [BILLING_STATE],
                    "questions": {
                        "statement": {
                            "type": "noul",
                            "instructions": BILLING_STATEMENT,
                        }
                    },
                },
            )
        ],
        Decision(probability=0.98),
    )


# https://docs.typesafe.ai/sdk/python/usage/basic.md, "Assets in structured objects":
# an Image in the state list is uploaded as a multipart asset part referenced from state.
@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize(
    ("media_type", "image_bytes", "part_content_type", "reference"),
    [
        (
            "image/png",
            RED_RECTANGLE_PNG,
            "image/png",
            "<image:4a3bbdcf5f5d4e59868b344a9e1f9a84>",
        ),
        (
            "image/jpeg",
            JPEG_BYTES,
            "image/jpeg",
            "<image:50979a7b55574b0dba7a7062707d2837>",
        ),
        (
            "IMAGE/PNG",
            RED_RECTANGLE_PNG,
            "image/png",
            "<image:4a3bbdcf5f5d4e59868b344a9e1f9a84>",
        ),
        (
            "image/*",
            RED_RECTANGLE_PNG,
            "image/png",
            "<image:4a3bbdcf5f5d4e59868b344a9e1f9a84>",
        ),
    ],
    ids=["png", "jpeg", "uppercase-png", "image-range"],
)
def test_an_image_part_with_bytes_is_uploaded_as_an_image_asset(
    build_model: ModelBuilder,
    media_type: str,
    image_bytes: bytes,
    part_content_type: str,
    reference: str,
) -> None:
    captured: list[tuple[JSON, str, bytes]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(parse_multipart(request))
        return httpx2.Response(200, json=NOUL_RESPONSE)

    question = Question(
        statement="Is the item damaged?",
        state=(
            TextPart(text="The customer returned this item:"),
            MediaPart(media_type=media_type, data=image_bytes),
        ),
    )
    model = build_model(handler, model="jev-latest", perceives=MediaTypes("image/*"))

    outcome = run_decision(model, question=question)

    assert (captured, outcome) == (
        [
            (
                {
                    "model": "jev-latest",
                    "state": ["The customer returned this item:", reference],
                    "questions": {
                        "statement": {
                            "type": "noul",
                            "instructions": "Is the item damaged?",
                        }
                    },
                    "assets": {reference: {"type": "image", "part": "asset-0"}},
                },
                part_content_type,
                image_bytes,
            )
        ],
        Decision(probability=0.98),
    )


@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize(
    "part",
    [
        MediaPart(media_type="audio/wav", data=b"RIFF"),
        MediaPart(media_type="application/pdf", data=b"%PDF-1.7\n"),
        MediaPart(media_type="image/gif", data=b"GIF89a"),
        MediaPart(media_type="image/webp", url="https://example.com/photo.webp"),
    ],
    ids=["audio", "pdf", "gif", "webp-by-url"],
)
def test_a_media_type_typesafe_cannot_carry_raises_configuration_error(
    build_model: ModelBuilder, part: MediaPart
) -> None:
    question = Question(statement=BILLING_STATEMENT, state=(part,))
    model = build_model(return_noul_response, perceives=MediaTypes("*/*"))

    with pytest.raises(ConfigurationError, match=part.media_type):
        run_decision(model, question=question)


@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize(
    "part",
    [
        MediaPart(media_type="image/png", url="https://example.com/photo.png"),
        MediaPart(media_type="image/jpeg", url="https://example.com/photo.jpg"),
        MediaPart(media_type="image/*", url="https://example.com/photo.png"),
        MediaPart(media_type="image/png", data=b""),
    ],
    ids=["png-by-url", "jpeg-by-url", "range-by-url", "empty-image-bytes"],
)
def test_an_image_without_usable_bytes_returns_unavailable(
    build_model: ModelBuilder, part: MediaPart
) -> None:
    question = Question(statement=BILLING_STATEMENT, state=(part,))
    model = build_model(return_noul_response, perceives=MediaTypes("image/*"))

    outcome = run_decision(model, question=question)

    assert outcome == Unavailable(status=None)


# https://docs.typesafe.ai/api.md, "Errors": 413 Payload Too Large, 429 Too Many Requests,
# 529 Overloaded, plus HTTP 408, 409 and 5xx.
@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize("status", [408, 409, 413, 429, 500, 502, 503, 529])
def test_a_transient_or_oversized_http_failure_returns_unavailable_with_its_status(
    build_model: ModelBuilder, status: int
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": "unavailable"})

    model = build_model(handler)

    outcome = run_decision(model, question=BILLING_QUESTION)

    assert outcome == Unavailable(status=status)


@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize(
    "error",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadTimeout("read timed out"),
    ],
    ids=["connection-error", "timeout"],
)
def test_a_connection_or_timeout_failure_returns_unavailable_without_a_status(
    build_model: ModelBuilder, error: httpx2.RequestError
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise error

    model = build_model(handler)

    outcome = run_decision(model, question=BILLING_QUESTION)

    assert outcome == Unavailable(status=None)


@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize("status", [400, 401, 403, 404, 415, 422])
def test_a_client_http_error_raises_configuration_error_naming_only_the_status(
    build_model: ModelBuilder, status: int
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={
                "detail": [
                    {"loc": ["body", "state"], "msg": "secret-prompt-text", "type": "missing"}
                ]
            },
        )

    model = build_model(handler)

    with pytest.raises(ConfigurationError) as raised:
        run_decision(model, question=BILLING_QUESTION)

    assert (str(status) in str(raised.value), "secret-prompt-text" in str(raised.value)) == (
        True,
        False,
    )


@pytest.mark.xfail(strict=True)
@BUILDERS
@pytest.mark.parametrize(
    "response_body",
    [
        {"model": "jev-latest"},
        {"model": "jev-latest", "answers": {}, "usage": {"input_tokens": 120, "output_tokens": 12}},
        {
            "model": "jev-latest",
            "answers": {"billing": {"type": "noul", "noul": 0.98}},
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
        CHOICE_RESPONSE,
        {
            "model": "jev-latest",
            "answers": {"statement": {"type": "future"}},
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
        {
            "model": "jev-latest",
            "answers": {"statement": {"type": "noul", "noul": 1.2}},
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
        {
            "model": "jev-latest",
            "answers": {"statement": {"type": "noul", "noul": -0.1}},
            "usage": {"input_tokens": 120, "output_tokens": 12},
        },
    ],
    ids=[
        "missing-usage",
        "empty-answers",
        "wrong-answer-key",
        "choice-answer",
        "unknown-answer-type",
        "probability-above-one",
        "probability-below-zero",
    ],
)
def test_an_unreadable_reply_returns_unavailable(
    build_model: ModelBuilder, response_body: JSON
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=response_body)

    model = build_model(handler)

    outcome = run_decision(model, question=BILLING_QUESTION)

    assert outcome == Unavailable(status=None)
