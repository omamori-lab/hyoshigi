"""OpenAI's Responses and Decisions APIs as a monitor's models, through the caller's client.

The generative model is the Responses API, https://platform.openai.com/docs/api-reference/responses.
The decision model is the Decisions API,
https://developers.openai.com/api/reference/resources/decisions/methods/create.
"""

import json
from typing import TypedDict, assert_never, cast

try:
    from openai import APIError, APIStatusError, AsyncOpenAI, Omit, OpenAI, omit
    from openai.types import DecisionInputMessageParam, DecisionInputPartUnionParam, ReasoningEffort
    from openai.types.responses import ResponseInputMessageContentListParam, ResponseInputParam
    from openai.types.shared_params import Reasoning
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenAI models need the extra: pip install 'hyoshigi[openai]'",
    ) from error

from hyoshigi.errors import ConfigurationError
from hyoshigi.models import responses
from hyoshigi.models.model import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    PromptPart,
    Question,
    Refused,
    Unavailable,
    check_client,
    check_name,
    check_perceives,
    collect,
)
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import MediaPart, TextPart

type Failure = APIError | UnicodeError
type DecisionInput = str | list[DecisionInputMessageParam]

# https://developers.openai.com/api/docs/guides/images-vision, "Image input requirements": PNG,
# JPEG, WEBP and non-animated GIF. The Decisions pages list no image types.
DECISION_IMAGES = MediaTypes("image/png", "image/jpeg", "image/webp", "image/gif")

# What a call to the client raises when it gets no usable reply: the SDK's own errors, and the
# error of encoding a prompt text that has no UTF-8 form, such as a lone surrogate.
_FAILURES = (
    APIError,
    UnicodeError,
)
# Statuses that are no answer rather than a mistake: request timeout, conflict, payload too
# large and too many requests.
_UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 429})
# The error codes that blame what the prompt holds, which an agent can shape, not the request:
# an input beyond the context window (https://github.com/openai/openai-openapi, openapi.yaml,
# schema SessionTurnErrorCodeResource) and an image that cannot be used (the same file, schema
# ResponseErrorCode, less image_content_policy_violation, a policy refusal, and
# image_file_not_found, which no request built here can cause).
_UNUSABLE_INPUT_CODES = frozenset(
    {
        "context_length_exceeded",
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
    },
)


class RequestArguments(TypedDict):
    """The keyword arguments of one call to the client's responses.create."""

    model: str
    instructions: str
    input: ResponseInputParam
    max_output_tokens: int | Omit
    reasoning: Reasoning | Omit
    store: bool


def build_arguments(
    prompt: Prompt,
    *,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: ReasoningEffort,
) -> RequestArguments | None:
    """The prompt as one user message under the instructions, with nothing stored; None if a
    part is in a form the API cannot take. The reasoning effort is sent only when given.
    """
    content = responses.build_content(prompt)
    if content is None:
        return None
    # The SDK's part types name every optional key, which no type outside the SDK can, so the
    # type checker cannot see that the shared parts are of them.
    parts = cast(
        ResponseInputMessageContentListParam,
        content,
    )
    return {
        "model": model,
        "instructions": prompt.system,
        "input": [
            {
                "role": "user",
                "content": parts,
            },
        ],
        "max_output_tokens": omit if max_output_tokens is None else max_output_tokens,
        "reasoning": omit if reasoning_effort is None else {"effort": reasoning_effort},
        "store": False,
    }


def load_body(body: bytes) -> JSON:
    """The JSON of a reply's raw body; None, which no reply is, if it is not JSON."""
    try:
        return json.loads(body)
    except ValueError:
        return None


def build_decision_part(part: PromptPart) -> DecisionInputPartUnionParam | None:
    """A state part as an input text or an input image; None for an image in a form the API
    cannot take. Raise ConfigurationError for media that is not an image the API reads.

    Bytes are sent as a base64 data URL and a URL as given, the two forms of an image in the API
    reference, https://developers.openai.com/api/reference/resources/decisions/methods/create.
    """
    match part:
        case TextPart(text=text):
            return {
                "type": "input_text",
                "text": text,
            }
        case MediaPart(media_type=media_type) if (
            DECISION_IMAGES.covers(media_type) or media_type.lower() == "image/*"
        ):
            image_url = responses.locate(part)
            return (
                None
                if image_url is None
                else {
                    "type": "input_image",
                    "image_url": image_url,
                }
            )
        case MediaPart(media_type=media_type):
            raise ConfigurationError(
                f"OpenAI's Decisions API reads png, jpeg, webp and gif images, not {media_type}",
            )
        case _:
            assert_never(part)


def build_decision_input(state: tuple[PromptPart, ...]) -> DecisionInput | None:
    """The state as a string if it is text alone, else as one user message of its parts in
    order; None if a part is in a form the API cannot take.

    Adjacent text is one run of text, as in a state a monitor builds, so text parts are joined
    without a separator. Every part is built before the result is decided, so media the API
    can never read raises whatever parts come before it.
    """
    texts = [
        part.text
        for part in state
        if isinstance(
            part,
            TextPart,
        )
    ]
    if len(texts) == len(state):
        return "".join(texts)
    parts = collect([build_decision_part(part) for part in state])
    return (
        None
        if parts is None
        else [
            {
                "role": "user",
                "content": parts,
            },
        ]
    )


def read_decision(reply: JSON) -> DecisionOutcome:
    """The outcome of a Decisions reply's JSON body, which answers the one question it was asked.

    A predicate answer is a decision of its probability; one that is not a number from 0 to 1 is
    unavailable. A refusal answer is a refusal, which carries no explanation. Any other reply,
    including an answer of another type, is unavailable.
    """
    match reply:
        case {
            "answers": [
                {
                    "type": "predicate",
                    "probability": int() | float() as probability,
                },
            ],
        }:
            try:
                return Decision(probability=probability)
            except ConfigurationError:
                return Unavailable()
        case {"answers": [{"type": "refusal"}]}:
            return Refused()
        case _:
            return Unavailable()


def read_failure(error: Failure) -> Refused | Unavailable:
    """The outcome of a request that failed after the client's retries.

    An error code that is a refusal or names a policy is a refusal. A 4xx status other than 408,
    409, 413 and 429, whose code does not blame the prompt's content, raises ConfigurationError
    naming only the status. Any other status is unavailable with it, and any other failure, a
    connection error, a timeout, a body that cannot be decoded or a prompt that cannot be
    encoded, is unavailable.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    match error:
        case APIStatusError(code=str(code)) if responses.names_refusal(code):
            return Refused()
        case APIStatusError(
            status_code=status,
            code=code,
        ) if (
            400 <= status < 500
            and status not in _UNAVAILABLE_STATUSES
            and code not in _UNUSABLE_INPUT_CODES
        ):
            raise ConfigurationError(f"OpenAI rejected the request with HTTP status {status}")
        case APIStatusError(status_code=status):
            return Unavailable(status=status)
    return Unavailable()


def build_model(
    client: OpenAI,
    *,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: ReasoningEffort,
    perceives: MediaTypes,
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    Each request sets store to false, so the Responses API stores no response for later
    retrieval. The reply is read from its raw body. Raise ConfigurationError if the client is
    not an OpenAI, the model id is empty or perceives names media the API cannot carry.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    check_client(
        client,
        expected=OpenAI,
        twin=AsyncOpenAI,
        twin_constructor="AsyncGenerativeModel.from_openai",
    )
    check_name(model)
    check_perceives(
        perceives,
        carries=responses.CARRIED_MEDIA,
    )

    def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(
            prompt,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        if arguments is None:
            return Unavailable()
        try:
            reply = client.responses.with_raw_response.create(**arguments)
        except _FAILURES as caught:
            error = caught
        else:
            return responses.read_reply(load_body(reply.http_response.content))
        return read_failure(error)

    return GenerativeModel(
        name=f"openai:{model}",
        generate=generate,
        perceives=perceives,
    )


def build_async_model(
    client: AsyncOpenAI,
    *,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: ReasoningEffort,
    perceives: MediaTypes,
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's async client.

    Each request sets store to false, so the Responses API stores no response for later
    retrieval. The reply is read from its raw body. Raise ConfigurationError if the client is
    not an AsyncOpenAI, the model id is empty or perceives names media the API cannot carry.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    check_client(
        client,
        expected=AsyncOpenAI,
        twin=OpenAI,
        twin_constructor="GenerativeModel.from_openai",
    )
    check_name(model)
    check_perceives(
        perceives,
        carries=responses.CARRIED_MEDIA,
    )

    async def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(
            prompt,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        if arguments is None:
            return Unavailable()
        try:
            reply = await client.responses.with_raw_response.create(**arguments)
        except _FAILURES as caught:
            error = caught
        else:
            return responses.read_reply(load_body(reply.http_response.content))
        return read_failure(error)

    return AsyncGenerativeModel(
        name=f"openai:{model}",
        generate=generate,
        perceives=perceives,
    )


def build_decision_model(
    client: OpenAI,
    *,
    model: str,
    perceives: MediaTypes,
) -> DecisionModel:
    """A decision model on the Decisions API, through the caller's client.

    The question is one predicate question, whose instructions are the statement; the
    probability of its answer is the decision's. The reply is read from its raw body. Raise
    ConfigurationError if the client is not an OpenAI, the model id is empty or perceives
    names media the API cannot read.
    """
    check_client(
        client,
        expected=OpenAI,
        twin=AsyncOpenAI,
        twin_constructor="AsyncDecisionModel.from_openai",
    )
    check_name(model)
    check_perceives(
        perceives,
        carries=DECISION_IMAGES,
    )

    def decide(question: Question) -> DecisionOutcome:
        state = build_decision_input(question.state)
        if state is None:
            return Unavailable()
        try:
            reply = client.decisions.with_raw_response.create(
                model=model,
                input=state,
                questions=[
                    {
                        "type": "predicate",
                        "instructions": question.statement,
                    },
                ],
            )
        except _FAILURES as caught:
            error = caught
        else:
            return read_decision(load_body(reply.http_response.content))
        return read_failure(error)

    return DecisionModel(
        name=f"openai:{model}",
        decide=decide,
        perceives=perceives,
    )


def build_async_decision_model(
    client: AsyncOpenAI,
    *,
    model: str,
    perceives: MediaTypes,
) -> AsyncDecisionModel:
    """A decision model on the Decisions API, through the caller's async client.

    The question is one predicate question, whose instructions are the statement; the
    probability of its answer is the decision's. The reply is read from its raw body. Raise
    ConfigurationError if the client is not an AsyncOpenAI, the model id is empty or perceives
    names media the API cannot read.
    """
    check_client(
        client,
        expected=AsyncOpenAI,
        twin=OpenAI,
        twin_constructor="DecisionModel.from_openai",
    )
    check_name(model)
    check_perceives(
        perceives,
        carries=DECISION_IMAGES,
    )

    async def decide(question: Question) -> DecisionOutcome:
        state = build_decision_input(question.state)
        if state is None:
            return Unavailable()
        try:
            reply = await client.decisions.with_raw_response.create(
                model=model,
                input=state,
                questions=[
                    {
                        "type": "predicate",
                        "instructions": question.statement,
                    },
                ],
            )
        except _FAILURES as caught:
            error = caught
        else:
            return read_decision(load_body(reply.http_response.content))
        return read_failure(error)

    return AsyncDecisionModel(
        name=f"openai:{model}",
        decide=decide,
        perceives=perceives,
    )
