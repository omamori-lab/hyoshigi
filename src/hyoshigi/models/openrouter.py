"""OpenRouter's Responses and Decisions APIs as a monitor's models, through the caller's client.

The generative model is the Responses API, https://openrouter.ai/docs/api_reference/responses/overview.
The decision model is the Decisions API, https://openrouter.ai/docs/guides/community/jev.
"""

import json
from typing import Literal, TypedDict, assert_never

try:
    import httpx
    from openrouter import OpenRouter, components, errors
    from openrouter.types import UNSET
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenRouter models need the extra: pip install 'hyoshigi[openrouter]'",
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
    check_name,
    check_perceives,
    collect,
)
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import MediaPart, TextPart

# The name of the one question a decision model is asked, and so of its answer.
QUESTION_NAME = "statement"

# https://openrouter.ai/docs/guides/community/multimodal-decisions: the Decisions API reads an
# image only as a base64 data URL of one of these types.
DECISION_IMAGES = MediaTypes("image/png", "image/jpeg", "image/webp")

# https://openrouter.ai/docs/api_reference/errors-and-debugging: a timeout, a conflict, a
# request body too large and a rate limit mean no answer now.
UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 429})

# The codes of input the provider rejects, which an agent's own trajectory can cause: the error
# types of https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes, and
# the codes of an unusable image in the schema OpenResponsesErrorField of
# https://openrouter.ai/docs/api/api-reference/responses/create-a-response
REJECTED_INPUT = frozenset(
    {
        "context_length_exceeded",
        "string_too_long",
        "payload_too_large",
        "invalid_image",
        "image_too_large",
        "image_too_small",
        "unsupported_image_format",
        "image_download_failed",
        "image_not_found",
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
    },
)

# On 2026-10-19 (issue #30), the 400 for a prompt over the endpoint's context window had the
# number 400 as its code and no error type. Only its message says what went wrong.
CONTEXT_LENGTH_PHRASE = "maximum context length"

# Every failure the SDK raises: the HTTP library's request errors (connection, timeout, decoding),
# the SDK's errors for a status or an unreadable reply, and its error for no response.
FAILURES = (
    httpx.RequestError,
    errors.OpenRouterError,
    errors.NoResponseError,
)
type Failure = httpx.RequestError | errors.OpenRouterError | errors.NoResponseError


class ImageUrl(TypedDict):
    url: str


class ImageObject(TypedDict):
    """An image in a Decisions request's state, as in OpenRouter's multimodal Decisions guide."""

    type: Literal["image_url"]
    image_url: ImageUrl


def build_input(prompt: Prompt) -> components.InputsUnionTypedDict | None:
    """The prompt's content as the one user message of a Responses API request; None if a part
    is in a form the API cannot take.
    """
    content = responses.build_content(prompt)
    return (
        None
        if content is None
        else [
            {
                "type": "message",
                "role": "user",
                "content": content,
            },
        ]
    )


def build_image(part: MediaPart) -> ImageObject | None:
    """An image as an image object; None unless it is bytes of a concrete type, since the
    Decisions API reads data URLs and does not fetch a URL, even one that is a data URL.
    """
    data_url = responses.encode(part)
    return (
        None
        if data_url is None
        else {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
    )


def build_state_item(part: PromptPart) -> str | ImageObject | None:
    """A question's state part as a Decisions API state item: a string or an image object;
    None if the part is an image the API can read but not in this form.
    """
    match part:
        case TextPart(text=text):
            return text
        case MediaPart(media_type=media_type) if (
            DECISION_IMAGES.covers(media_type) or media_type.lower() == "image/*"
        ):
            return build_image(part)
        case MediaPart(media_type=media_type):
            raise ConfigurationError(
                f"OpenRouter's Decisions API reads png, jpeg and webp images, not {media_type}",
            )
        case _:
            assert_never(part)


def read_overflow(payload: JSON) -> tuple[str, ...]:
    """The code context_length_exceeded if the error's message holds the provider's wording for
    a prompt over the context window, else no code. The message is never kept, because it can
    echo the prompt.
    """
    match payload:
        case {"error": {"message": str(message)}} if CONTEXT_LENGTH_PHRASE in message:
            return ("context_length_exceeded",)
        case _:
            return ()


def read_codes(body: str) -> tuple[str, ...]:
    """The machine-readable codes of an HTTP error body: its error code and its error type; a
    body that is not JSON has none.

    The error type is where https://openrouter.ai/docs/api_reference/errors-and-debugging puts
    it: in the error's metadata, in the error itself (the Anthropic skin) and at the top level
    (a failed Responses reply). The code context_length_exceeded is added when the error's
    message holds the provider's wording for a prompt over the context window.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return ()
    codes: list[str] = []
    match payload:
        case {"error": {"code": str(code)}}:
            codes.append(code)
    codes.extend(read_overflow(payload))
    match payload:
        case {"error": {"error_type": str(error_type)}}:
            codes.append(error_type)
    match payload:
        case {"error": {"metadata": {"error_type": str(error_type)}}}:
            codes.append(error_type)
    match payload:
        case {"error_type": str(error_type)}:
            codes.append(error_type)
    return tuple(codes)


def read_failure(
    status: int,
    *,
    codes: tuple[str, ...],
) -> Refused | Unavailable:
    """The outcome of an HTTP failure with this status and these error codes.

    A refusal is refused. No answer now (a status below 400, above 499 or one to ask again
    later), or input the provider rejects, is unavailable. Any other rejection is misuse: it
    raises ConfigurationError naming only the status.
    """
    if any(
        map(
            responses.names_refusal,
            codes,
        ),
    ):
        return Refused()
    if (
        not 400 <= status < 500
        or status in UNAVAILABLE_STATUSES
        or not REJECTED_INPUT.isdisjoint(codes)
    ):
        return Unavailable(status=status)
    raise ConfigurationError(f"OpenRouter answered HTTP {status}")


def read_error(error: Failure) -> Refused | Unavailable:
    """The outcome of an exception the SDK raised.

    A success status with a reply the SDK could not read has no status to report. This is
    called outside the handler of the exception, so that the exception, whose message can
    echo the prompt, is not in the chain of a ConfigurationError raised for a rejection.
    """
    match error:
        case httpx.RequestError() | errors.NoResponseError():
            return Unavailable()
        case errors.OpenRouterError(status_code=status) if 200 <= status < 300:
            return Unavailable()
        case errors.OpenRouterError(
            status_code=status,
            body=body,
        ):
            return read_failure(
                status,
                codes=read_codes(body),
            )
        case _:
            assert_never(error)


def read_reply(reply: object) -> GenerationOutcome:
    """The outcome of what the SDK returned: a result, or an event stream, which is not one.

    The SDK gives no raw body for a result, so the result is read from its dump, which does
    not warn of a value the SDK kept unrecognized, such as a status it does not list.
    """
    match reply:
        case components.OpenResponsesResult():
            return responses.read_reply(restore(reply.model_dump(warnings=False)))
        case _:
            return Unavailable()


def restore(value: JSON) -> JSON:
    """The SDK's dump of a reply with each variant the SDK did not recognize, or could not
    validate, put back as the payload it kept.
    """
    match value:
        case {
            "is_unknown": True,
            "raw": dict() as raw,
        }:
            return raw
        case dict():
            return {key: restore(item) for key, item in value.items()}
        case list():
            return [restore(item) for item in value]
        case _:
            return value


def read_decision(reply: components.DecisionsResponse) -> DecisionOutcome:
    """The probability a Decisions API reply gives the question; any other answer is unavailable."""
    match reply.answers.get(QUESTION_NAME):
        case components.DecisionsNoulAnswer(noul=probability) if 0 <= probability <= 1:
            return Decision(probability=probability)
        case _:
            return Unavailable()


def build_model(
    client: OpenRouter,
    *,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: components.ReasoningEffort | None,
    perceives: MediaTypes,
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    The API is stateless and the SDK sends store false, so no stored response holds the
    monitored agent's content. The reasoning effort is sent only when given. Raise
    ConfigurationError if the model id is empty or perceives names media the API cannot carry.
    """
    check_name(model)
    check_perceives(
        perceives,
        carries=responses.CARRIED_MEDIA,
    )
    limit = UNSET if max_output_tokens is None else max_output_tokens
    reasoning = (
        UNSET if reasoning_effort is None else components.ReasoningConfig(effort=reasoning_effort)
    )

    def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_input(prompt)
        if request is None:
            return Unavailable()
        try:
            reply = client.responses.send(
                model=model,
                instructions=prompt.system,
                input=request,
                max_output_tokens=limit,
                reasoning=reasoning,
            )
        except FAILURES as caught:
            error = caught
        else:
            return read_reply(reply)
        return read_error(error)

    return GenerativeModel(
        name=f"openrouter:{model}",
        generate=generate,
        perceives=perceives,
    )


def build_async_model(
    client: OpenRouter,
    *,
    model: str,
    max_output_tokens: int | None,
    reasoning_effort: components.ReasoningEffort | None,
    perceives: MediaTypes,
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's client, awaited."""
    check_name(model)
    check_perceives(
        perceives,
        carries=responses.CARRIED_MEDIA,
    )
    limit = UNSET if max_output_tokens is None else max_output_tokens
    reasoning = (
        UNSET if reasoning_effort is None else components.ReasoningConfig(effort=reasoning_effort)
    )

    async def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_input(prompt)
        if request is None:
            return Unavailable()
        try:
            reply = await client.responses.send_async(
                model=model,
                instructions=prompt.system,
                input=request,
                max_output_tokens=limit,
                reasoning=reasoning,
            )
        except FAILURES as caught:
            error = caught
        else:
            return read_reply(reply)
        return read_error(error)

    return AsyncGenerativeModel(
        name=f"openrouter:{model}",
        generate=generate,
        perceives=perceives,
    )


def build_decision_model(
    client: OpenRouter,
    *,
    model: str,
    perceives: MediaTypes,
) -> DecisionModel:
    """A decision model on the Decisions API, through the caller's client.

    The question is one noul question, whose instructions are the statement; the probability
    of its answer is the decision's. Raise ConfigurationError if the model id is empty or
    perceives names media the API cannot read.
    """
    check_name(model)
    check_perceives(
        perceives,
        carries=DECISION_IMAGES,
    )

    def decide(question: Question) -> DecisionOutcome:
        state = collect([build_state_item(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            reply = client.alpha.decisions.create(
                model=model,
                state=state,
                questions={
                    QUESTION_NAME: {
                        "type": "noul",
                        "instructions": question.statement,
                    },
                },
            )
        except FAILURES as caught:
            error = caught
        else:
            return read_decision(reply)
        return read_error(error)

    return DecisionModel(
        name=f"openrouter:{model}",
        decide=decide,
        perceives=perceives,
    )


def build_async_decision_model(
    client: OpenRouter,
    *,
    model: str,
    perceives: MediaTypes,
) -> AsyncDecisionModel:
    """A decision model on the Decisions API, through the caller's client, awaited."""
    check_name(model)
    check_perceives(
        perceives,
        carries=DECISION_IMAGES,
    )

    async def decide(question: Question) -> DecisionOutcome:
        state = collect([build_state_item(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            reply = await client.alpha.decisions.create_async(
                model=model,
                state=state,
                questions={
                    QUESTION_NAME: {
                        "type": "noul",
                        "instructions": question.statement,
                    },
                },
            )
        except FAILURES as caught:
            error = caught
        else:
            return read_decision(reply)
        return read_error(error)

    return AsyncDecisionModel(
        name=f"openrouter:{model}",
        decide=decide,
        perceives=perceives,
    )
