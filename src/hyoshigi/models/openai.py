"""OpenAI's Responses API as a monitor's model, through the caller's client.

The API is OpenAI's Responses API reference, https://platform.openai.com/docs/api-reference/responses.
"""

import mimetypes
from base64 import b64encode
from dataclasses import dataclass
from json import JSONDecodeError
from typing import TypedDict, TypeGuard, assert_never

try:
    from openai import (
        APIError,
        APIStatusError,
        AsyncOpenAI,
        BaseModel,
        Omit,
        OpenAI,
        omit,
    )
    from openai.types.responses import (
        Response,
        ResponseInputContentParam,
        ResponseInputFileParam,
        ResponseInputImageParam,
        ResponseInputParam,
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
    check_name,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

type Content = ResponseOutputText | ResponseOutputRefusal
type Failure = APIError | JSONDecodeError | UnicodeError

# What a call to the client raises when it gets no usable reply: the SDK's own errors, the
# errors of decoding a body it labels JSON, which it lets through, and of encoding a prompt
# text that has no UTF-8 form, such as a lone surrogate.
_FAILURES = (APIError, JSONDecodeError, UnicodeError)
# The built-in table only: the platform's mime.types files differ between machines.
_EXTENSIONS = mimetypes.MimeTypes()
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Rejection:
    """The API turned the request down for a mistake in it, with the HTTP status."""

    status: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Written:
    """What the messages of a reply wrote.

    The text of their output text parts, the first refusal, the number of messages, and
    whether a message was unfinished or held a part of a type that is not read.
    """

    text: str
    refusal: str | None
    messages: int
    cut_off: bool


def build_arguments(
    prompt: Prompt, *, model: str, max_output_tokens: int | None
) -> RequestArguments | None:
    """The prompt as one user message under the instructions, with nothing stored.

    Every part is translated before the result is decided, so media the API can never carry
    raises whatever parts come before it; None if a part is in a form the API cannot take.
    """
    translated = [build_part(part) for part in prompt.content]
    content = [part for part in translated if part is not None]
    if len(content) < len(translated):
        return None
    return {
        "model": model,
        "instructions": prompt.system,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": omit if max_output_tokens is None else max_output_tokens,
        "store": False,
    }


def build_part(part: PromptPart) -> ResponseInputContentParam | None:
    """A text part as input text, a media part as an image or a file, None if it is in a form
    the API cannot take.
    """
    match part:
        case TextPart(text=text):
            return {"type": "input_text", "text": text}
        case MediaPart():
            return build_media(part)
        case _:
            assert_never(part)


def build_media(part: MediaPart) -> ResponseInputImageParam | ResponseInputFileParam | None:
    """Images become input images, application and text media input files.

    Raise ConfigurationError for any other media; None for the bytes of a media range, such
    as image/*, which no data URL can name.
    """
    match part.media_type.partition("/")[0].lower(), locate(part):
        case "image" | "application" | "text", None:
            return None
        case "image", str(image_url):
            return {"type": "input_image", "image_url": image_url, "detail": "auto"}
        case "application" | "text", str(file_url) if part.url is not None:
            return {"type": "input_file", "file_url": file_url}
        case "application" | "text", str(file_data):
            return build_file_data(part, file_data=file_data)
        case _:
            raise ConfigurationError(f"the Responses API takes no {part.media_type} media")


def build_file_data(part: MediaPart, *, file_data: str) -> ResponseInputFileParam:
    """A file's bytes as a data URL under a filename named for its media type.

    The API takes file_data only with a filename, as attachment.pdf. Its examples all carry
    one, such as draconomicon.pdf in https://developers.openai.com/api/docs/guides/pdf-files.
    """
    extension = _EXTENSIONS.guess_extension(part.media_type) or ""
    return {"type": "input_file", "filename": f"attachment{extension}", "file_data": file_data}


def locate(part: MediaPart) -> str | None:
    """The media's URL, or its bytes as a base64 data URL; None for the bytes of a media
    range, which names no media type for the URL.
    """
    match part:
        case MediaPart(url=str(url)):
            return url
        case MediaPart(data=bytes(data)) if not part.media_type.endswith("/*"):
            return f"data:{part.media_type};base64,{b64encode(data).decode()}"
    return None


def read_response(response: Response) -> GenerationOutcome:
    """The outcome of a reply by its status; a reply nothing can read is unavailable."""
    written = read_output(response)
    match response:
        case Response(status="failed", error=ResponseError(code=str(code))) if is_refusal(code):
            return Refused()
        case Response(
            status="incomplete", incomplete_details=IncompleteDetails(reason="content_filter")
        ):
            return Refused()
        case Response(status="failed"):
            return Unavailable()
        case Response(status="completed") if written is not None:
            return read_completed(written)
        case Response() if written is not None:
            return Truncated(text=written.text)
        case _:
            return Unavailable()


def read_completed(written: Written) -> GenerationOutcome:
    """A complete reply: a refusal if it holds one, nothing readable if it holds no message,
    the text so far if a message was unfinished or a part was not read, else the text.
    """
    if written.refusal is not None:
        return Refused(explanation=written.refusal)
    if written.messages == 0:
        return Unavailable()
    if written.cut_off:
        return Truncated(text=written.text)
    return Generation(text=written.text)


def is_refusal(code: str) -> bool:
    """Whether an error code is a refusal or names a policy."""
    return code == "refusal" or "policy" in code


def read_output(response: Response) -> Written | None:
    """What the output's messages wrote; None if the output, or a message or a part in it
    that is read, is not what the API documents.
    """
    messages = list_messages(response)
    if messages is None:
        return None
    parts = [part for message in messages for part in message.content]
    known = [part for part in parts if is_known(part)]
    if not all(map(is_readable, known)):
        return None
    return Written(
        text=join_text(known),
        refusal=find_refusal(known),
        messages=len(messages),
        cut_off=len(known) < len(parts)
        or any(message.status != "completed" for message in messages),
    )


def join_text(content: list[Content]) -> str:
    """The text of the output text parts, in order, with no separator, as the SDK's
    output_text does.
    """
    return "".join(part.text for part in content if isinstance(part, ResponseOutputText))


def find_refusal(content: list[Content]) -> str | None:
    """The first refusal among the parts, if there is one."""
    return next((part.refusal for part in content if isinstance(part, ResponseOutputRefusal)), None)


def list_messages(response: Response) -> list[ResponseOutputMessage] | None:
    """The output's messages in order, skipping its items of other types; None unless the
    output is a list of objects and each message's content is too.

    The SDK builds an object it cannot place as the first variant of a union, so the type
    field, not the class, tells what an item is.
    """
    match response:
        case Response(output=list(items)) if is_list_of_objects(items):
            messages = [item for item in items if is_message(item)]
            if all(is_list_of_objects(message.content) for message in messages):
                return messages
    return None


def is_list_of_objects(value: object) -> bool:
    """Whether a value is a list of objects, as the SDK leaves any other element."""
    return isinstance(value, list) and all(isinstance(element, BaseModel) for element in value)


def is_message(item: object) -> TypeGuard[ResponseOutputMessage]:
    """Whether an output item is a message by its type."""
    match item:
        case ResponseOutputMessage(type="message"):
            return True
    return False


def is_known(part: Content) -> bool:
    """Whether a content part is output text or a refusal by its type."""
    match part:
        case ResponseOutputText(type="output_text") | ResponseOutputRefusal(type="refusal"):
            return True
    return False


def is_readable(part: Content) -> bool:
    """Whether a text part holds text, or a refusal part its refusal, as a string."""
    match part:
        case ResponseOutputText(text=str()) | ResponseOutputRefusal(refusal=str()):
            return True
    return False


def read_failure(error: Failure) -> Refused | Unavailable | Rejection:
    """The outcome of a request that failed after the client's retries.

    An error code that is a refusal or names a policy is a refusal. A 4xx status that is not
    one the SDK retries, and whose code does not blame the prompt's content, is a rejection of
    the request. Any other failure, a connection error, a timeout, a body that cannot be
    decoded or validated, a prompt that cannot be encoded, or another status, is unavailable.
    """
    match error:
        case APIStatusError(code=str(code)) if is_refusal(code):
            return Refused()
        case APIStatusError(status_code=status, code=code) if (
            400 <= status < 500
            and status not in _UNAVAILABLE_STATUSES
            and code not in _UNUSABLE_INPUT_CODES
        ):
            return Rejection(status=status)
        case APIStatusError(status_code=status):
            return Unavailable(status=status)
    return Unavailable()


def settle(failure: Refused | Unavailable | Rejection) -> Refused | Unavailable:
    """The failure as an outcome; a rejection raises ConfigurationError naming only its status.

    The error is raised outside the handler of the provider's error, so the provider's text,
    which can echo the prompt, is in neither its cause nor its context.
    """
    match failure:
        case Rejection(status=status):
            raise ConfigurationError(f"OpenAI rejected the request with HTTP status {status}")
        case _:
            return failure


def build_model(
    client: OpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    Raise ConfigurationError if the model id is empty.
    """
    check_name(model)

    def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        if arguments is None:
            return Unavailable()
        try:
            response = client.responses.create(**arguments)
        except _FAILURES as error:
            failure = read_failure(error)
        else:
            return read_response(response)
        return settle(failure)

    return GenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: AsyncOpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's async client.

    Each request sets store to false, so OpenAI does not retain the monitored agent's content.
    Raise ConfigurationError if the model id is empty.
    """
    check_name(model)

    async def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = build_arguments(prompt, model=model, max_output_tokens=max_output_tokens)
        if arguments is None:
            return Unavailable()
        try:
            response = await client.responses.create(**arguments)
        except _FAILURES as error:
            failure = read_failure(error)
        else:
            return read_response(response)
        return settle(failure)

    return AsyncGenerativeModel(name=f"openai:{model}", generate=generate, perceives=perceives)
