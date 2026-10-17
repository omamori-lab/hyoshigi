"""OpenAI's Responses API as a monitor's model, through the caller's client.

The API is OpenAI's Responses API reference, https://platform.openai.com/docs/api-reference/responses.
"""

import mimetypes
from base64 import b64encode
from json import JSONDecodeError
from typing import TypedDict, assert_never

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI, Omit, OpenAI, omit
    from openai.types.responses import (
        Response,
        ResponseInputContentParam,
        ResponseInputFileParam,
        ResponseInputImageParam,
        ResponseInputParam,
        ResponseOutputItem,
        ResponseOutputMessage,
        ResponseOutputRefusal,
        ResponseOutputText,
    )
    from openai.types.responses.response import IncompleteDetails
    from openai.types.responses.response_error import ResponseError
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenAI models need the extra: pip install 'hyoshigi[openai]'"
    ) from error

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    PromptPart,
    Refused,
    Truncated,
    Unavailable,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

type Content = ResponseOutputText | ResponseOutputRefusal
type Failure = APIConnectionError | APIStatusError | JSONDecodeError | UnicodeDecodeError

# What a call to the client raises when it gets no usable reply. The SDK decodes a body
# labelled JSON itself, and lets the decoding errors through.
_FAILURES = (APIConnectionError, APIStatusError, JSONDecodeError, UnicodeDecodeError)
# The built-in table only: the platform's mime.types files differ between machines.
_EXTENSIONS = mimetypes.MimeTypes()
# Statuses that are no answer rather than a mistake: request timeout, conflict, payload too
# large, too many requests; and every 5xx.
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
) -> RequestArguments:
    """The prompt as one user message under the instructions, with nothing stored."""
    return {
        "model": model,
        "instructions": prompt.system,
        "input": [{"role": "user", "content": [build_part(part) for part in prompt.content]}],
        "max_output_tokens": omit if max_output_tokens is None else max_output_tokens,
        "store": False,
    }


def build_part(part: PromptPart) -> ResponseInputContentParam:
    """A text part as input text, a media part as an image or a file."""
    match part:
        case TextPart(text=text):
            return {"type": "input_text", "text": text}
        case MediaPart():
            return build_media(part)
        case _:
            assert_never(part)


def build_media(part: MediaPart) -> ResponseInputImageParam | ResponseInputFileParam:
    """Images become input images, application and text media input files; raise
    ConfigurationError for any other media.
    """
    match part.media_type.partition("/")[0].lower():
        case "image":
            return {"type": "input_image", "image_url": locate(part), "detail": "auto"}
        case "application" | "text":
            return build_file(part)
        case _:
            raise ConfigurationError(f"the Responses API takes no {part.media_type} media")


def build_file(part: MediaPart) -> ResponseInputFileParam:
    """A file by URL, or its bytes as a data URL under a filename.

    The API takes file_data only with a filename, so the file is named for its media type,
    as attachment.pdf. Its examples all carry a filename, such as draconomicon.pdf in
    https://developers.openai.com/api/docs/guides/pdf-files.
    """
    match part:
        case MediaPart(url=str(url)):
            return {"type": "input_file", "file_url": url}
        case _:
            extension = _EXTENSIONS.guess_extension(part.media_type) or ""
            return {
                "type": "input_file",
                "filename": f"attachment{extension}",
                "file_data": locate(part),
            }


def locate(part: MediaPart) -> str:
    """The media's URL, or its bytes as a base64 data URL; raise ConfigurationError for the
    bytes of a media range, which names no media type for the URL.
    """
    match part:
        case MediaPart(url=str(url)):
            return url
        case MediaPart(data=bytes(data)) if not part.media_type.endswith("/*"):
            return f"data:{part.media_type};base64,{b64encode(data).decode()}"
        case _:
            raise ConfigurationError(f"bytes need a media type, not the range {part.media_type}")


def read_response(response: Response) -> GenerationOutcome:
    """The outcome of a reply by its status; a reply nothing can read is unavailable."""
    content = read_content(response)
    match response:
        case Response(status="failed", error=ResponseError(code=str(code))) if "policy" in code:
            return Refused()
        case Response(
            status="incomplete", incomplete_details=IncompleteDetails(reason="content_filter")
        ):
            return Refused()
        case Response(status="completed") if content is not None:
            return read_completed(content)
        case Response(status="failed"):
            return Unavailable()
        case Response() if content is not None:
            return Truncated(text=join_text(content))
        case _:
            return Unavailable()


def read_completed(content: list[Content]) -> Generation | Refused:
    """A complete reply: its first refusal if it has one, else its text."""
    refusal = next(
        (part.refusal for part in content if isinstance(part, ResponseOutputRefusal)), None
    )
    if refusal is not None:
        return Refused(explanation=refusal)
    return Generation(text=join_text(content))


def join_text(content: list[Content]) -> str:
    """The text parts joined in order with no separator, as the SDK's output_text does."""
    return "".join(part.text for part in content if isinstance(part, ResponseOutputText))


def read_content(response: Response) -> list[Content] | None:
    """The text and refusal parts of the output's messages, in order; None if the output, or
    a message in it, is not what the API documents.
    """
    match response:
        case Response(output=list(items)):
            content: list[Content] = []
            for item in items:
                parts = read_message(item)
                if parts is None:
                    return None
                content.extend(parts)
            return content
    return None


def read_message(item: ResponseOutputItem) -> list[Content] | None:
    """The text and refusal parts of a message, none for another output item; None if the
    message's content, or a part in it, is not what the API documents.
    """
    match item:
        case ResponseOutputMessage(content=list(parts)):
            known = [
                part
                for part in parts
                if isinstance(part, ResponseOutputText | ResponseOutputRefusal)
            ]
            return known if all(map(is_readable, known)) else None
        case ResponseOutputMessage():
            return None
        case _:
            return []


def is_readable(part: Content) -> bool:
    """Whether a text part holds text, or a refusal part its refusal, as a string."""
    match part:
        case ResponseOutputText(text=str()) | ResponseOutputRefusal(refusal=str()):
            return True
    return False


def read_failure(error: Failure) -> Refused | Unavailable:
    """The outcome of a request that failed after the client's retries.

    A connection error or a timeout, a body that cannot be decoded, a status that may pass,
    and a prompt the API cannot take are unavailable; an error code that names a policy is
    a refusal. Any other status raises ConfigurationError naming it, without the provider's
    text, which can echo the prompt.
    """
    match error:
        case APIConnectionError() | JSONDecodeError() | UnicodeDecodeError():
            return Unavailable()
        case APIStatusError(code=str(code)) if "policy" in code:
            return Refused()
        case APIStatusError(status_code=status, code=code) if (
            code in _UNUSABLE_INPUT_CODES or status in _UNAVAILABLE_STATUSES or status >= 500
        ):
            return Unavailable(status=status)
        case _:
            raise ConfigurationError(
                f"OpenAI rejected the request with HTTP status {error.status_code}"
            ) from None


def build_model(
    client: OpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    """

    def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        try:
            response = client.responses.create(**arguments)
        except _FAILURES as error:
            return read_failure(error)
        return read_response(response)

    return GenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: AsyncOpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's async client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    """

    async def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        try:
            response = await client.responses.create(**arguments)
        except _FAILURES as error:
            return read_failure(error)
        return read_response(response)

    return AsyncGenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)
