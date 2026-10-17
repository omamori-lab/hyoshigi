"""OpenAI's Responses API as a monitor's model, through the caller's client.

The API is OpenAI's Responses API reference, https://platform.openai.com/docs/api-reference/responses.
"""

import json
from typing import TypedDict, cast

try:
    from openai import APIError, APIStatusError, AsyncOpenAI, Omit, OpenAI, omit
    from openai.types.responses import ResponseInputMessageContentListParam, ResponseInputParam
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenAI models need the extra: pip install 'hyoshigi[openai]'"
    ) from error

from hyoshigi.errors import ConfigurationError
from hyoshigi.models import responses
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Refused,
    Unavailable,
    check_name,
    check_perceives,
)

type Failure = APIError | UnicodeError

# What a call to the client raises when it gets no usable reply: the SDK's own errors, and the
# error of encoding a prompt text that has no UTF-8 form, such as a lone surrogate.
_FAILURES = (APIError, UnicodeError)
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
    }
)


class RequestArguments(TypedDict):
    """The keyword arguments of one call to the client's responses.create."""

    model: str
    instructions: str
    input: ResponseInputParam
    max_output_tokens: int | Omit
    store: bool


def build_arguments(
    prompt: Prompt, *, model: str, max_output_tokens: int | None
) -> RequestArguments | None:
    """The prompt as one user message under the instructions, with nothing stored; None if a
    part is in a form the API cannot take.
    """
    content = responses.build_content(prompt)
    if content is None:
        return None
    # The SDK's part types name every optional key, which no type outside the SDK can, so the
    # type checker cannot see that the shared parts are of them.
    parts = cast(ResponseInputMessageContentListParam, content)
    return {
        "model": model,
        "instructions": prompt.system,
        "input": [{"role": "user", "content": parts}],
        "max_output_tokens": omit if max_output_tokens is None else max_output_tokens,
        "store": False,
    }


def read_body(body: bytes) -> GenerationOutcome:
    """The outcome of a reply's raw body; a body that is not JSON is unavailable."""
    try:
        reply = json.loads(body)
    except ValueError:
        return Unavailable()
    return responses.read_reply(reply)


def read_failure(error: Failure) -> Refused | Unavailable:
    """The outcome of a request that failed after the client's retries.

    An error code that is a refusal or names a policy is a refusal. A 4xx status that is not
    one the SDK retries, and whose code does not blame the prompt's content, rejects the
    request: it raises ConfigurationError naming only the status. This is called outside the
    handler of the provider's error, so the provider's text, which can echo the prompt, is in
    neither the raised error's cause nor its context. Any other failure, a connection error, a
    timeout, a body that cannot be decoded, a prompt that cannot be encoded, or another status,
    is unavailable.
    """
    match error:
        case APIStatusError(code=str(code)) if responses.names_refusal(code):
            return Refused()
        case APIStatusError(status_code=status, code=code) if (
            400 <= status < 500
            and status not in _UNAVAILABLE_STATUSES
            and code not in _UNUSABLE_INPUT_CODES
        ):
            raise ConfigurationError(f"OpenAI rejected the request with HTTP status {status}")
        case APIStatusError(status_code=status):
            return Unavailable(status=status)
    return Unavailable()


def build_model(
    client: OpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    The reply is read from its raw body. Raise ConfigurationError if the model id is empty or
    perceives names media the API cannot carry.
    """
    check_name(model)
    check_perceives(perceives, carries=responses.CARRIED_MEDIA)

    def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        if arguments is None:
            return Unavailable()
        try:
            reply = client.responses.with_raw_response.create(**arguments)
        except _FAILURES as caught:
            error = caught
        else:
            return read_body(reply.http_response.content)
        return read_failure(error)

    return GenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: AsyncOpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's async client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    The reply is read from its raw body. Raise ConfigurationError if the model id is empty or
    perceives names media the API cannot carry.
    """
    check_name(model)
    check_perceives(perceives, carries=responses.CARRIED_MEDIA)

    async def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        if arguments is None:
            return Unavailable()
        try:
            reply = await client.responses.with_raw_response.create(**arguments)
        except _FAILURES as caught:
            error = caught
        else:
            return read_body(reply.http_response.content)
        return read_failure(error)

    return AsyncGenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)
