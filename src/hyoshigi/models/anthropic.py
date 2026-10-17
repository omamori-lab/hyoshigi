"""Anthropic's Messages API as a monitor's model, through the caller's client.

The API is Anthropic's Messages API reference, https://platform.claude.com/docs/en/api/messages.
"""

from base64 import b64encode
from json import JSONDecodeError

try:
    from anthropic import Anthropic, APIConnectionError, APIStatusError, AsyncAnthropic
    from anthropic.types import (
        ContentBlock,
        DocumentBlockParam,
        ImageBlockParam,
        Message,
        RefusalStopDetails,
        TextBlock,
        TextBlockParam,
    )
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Anthropic models need the extra: pip install 'hyoshigi[anthropic]'"
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

type Block = TextBlockParam | ImageBlockParam | DocumentBlockParam

# The statuses the SDK retries other than 5xx, and 413: the prompt's content makes a request
# too large, so the error fails closed instead of raising.
_UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 429})


def build_model(
    client: Anthropic, *, model: str, max_tokens: int, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Messages API, through the caller's client.

    A max_tokens the SDK will not send without streaming raises the SDK's own ValueError.
    """

    def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_request(prompt, model=model, max_tokens=max_tokens)
        if request is None:
            return Unavailable()
        try:
            message = client.messages.create(**request)
        except (APIConnectionError, APIStatusError, JSONDecodeError) as error:
            return read_failure(error)
        return read_message(message)

    return GenerativeModel(name=f"anthropic:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: AsyncAnthropic, *, model: str, max_tokens: int, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Messages API, through the caller's async client.

    A max_tokens the SDK will not send without streaming raises the SDK's own ValueError.
    """

    async def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_request(prompt, model=model, max_tokens=max_tokens)
        if request is None:
            return Unavailable()
        try:
            message = await client.messages.create(**request)
        except (APIConnectionError, APIStatusError, JSONDecodeError) as error:
            return read_failure(error)
        return read_message(message)

    return AsyncGenerativeModel(name=f"anthropic:{model}", generate=generate, perceives=perceives)


def build_request(
    prompt: Prompt, *, model: str, max_tokens: int
) -> MessageCreateParamsNonStreaming | None:
    """The prompt as one user message and a system prompt; None if a part cannot be sent."""
    translated = [translate_part(part) for part in prompt.content]
    blocks = [block for block in translated if block is not None]
    if len(blocks) < len(translated):
        return None
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": prompt.system,
        "messages": [{"role": "user", "content": blocks}],
    }


def translate_part(part: PromptPart) -> Block | None:
    """A prompt part as a content block; None if the API cannot take it in its form."""
    match part:
        case TextPart(text=text):
            return {"type": "text", "text": text}
        case MediaPart(url=str(url)):
            return translate_url(part.media_type.lower(), url=url)
        case MediaPart(data=bytes(data)):
            return translate_data(part.media_type.lower(), data=data)


def translate_url(media_type: str, *, url: str) -> Block | None:
    """Media at a URL as an image or a PDF document; plain text is not taken by URL."""
    match media_type:
        case "image/*" | "image/jpeg" | "image/png" | "image/gif" | "image/webp":
            return {"type": "image", "source": {"type": "url", "url": url}}
        case "application/pdf":
            return {"type": "document", "source": {"type": "url", "url": url}}
        case "text/plain":
            return None
        case _:
            raise ConfigurationError(f"the Messages API cannot carry {media_type} media")


def translate_data(media_type: str, *, data: bytes) -> Block | None:
    """Media bytes as an image, a PDF document or a text document; an image range has no type."""
    encoded = b64encode(data).decode("ascii")
    match media_type:
        case "image/jpeg" | "image/png" | "image/gif" | "image/webp" as image_type:
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": image_type, "data": encoded},
            }
        case "application/pdf":
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
            }
        case "text/plain":
            return translate_text(data)
        case "image/*":
            return None
        case _:
            raise ConfigurationError(f"the Messages API cannot carry {media_type} media")


def translate_text(data: bytes) -> DocumentBlockParam | None:
    """UTF-8 bytes as a text document; None if they are not UTF-8."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
    }


def read_message(message: Message) -> GenerationOutcome:
    """The outcome of a reply: only a turn that ended on its own is complete.

    Any other stop reason, including one the API adds later, is truncated. A reply that is
    not a message with readable content is unavailable.
    """
    match message:
        case Message(
            stop_reason="refusal",
            stop_details=RefusalStopDetails(explanation=str(explanation)),
        ):
            return Refused(explanation=explanation)
        case Message(stop_reason="refusal"):
            return Refused()
        case Message(stop_reason=stop_reason, content=list(blocks)):
            return read_content(stop_reason, blocks=blocks)
        case _:
            return Unavailable()


def read_content(stop_reason: str | None, *, blocks: list[ContentBlock]) -> GenerationOutcome:
    """The text of a reply's blocks as a generation if the turn ended on its own, else cut off."""
    match (stop_reason, read_text(blocks)):
        case (_, None):
            return Unavailable()
        case ("end_turn" | "stop_sequence", str(text)):
            return Generation(text=text)
        case (_, str(text)):
            return Truncated(text=text)


def read_text(blocks: list[ContentBlock]) -> str | None:
    """The text blocks of a reply, joined; None if one has no text.

    Thinking and tool use are not part of a reply's text. The SDK reads a block of a type it
    does not know as a text block of that type.
    """
    texts: list[str] = []
    for block in blocks:
        match block:
            case TextBlock(type="text", text=str(text)):
                texts.append(text)
            case TextBlock(type="text"):
                return None
    return "".join(texts)


def read_failure(error: APIConnectionError | APIStatusError | JSONDecodeError) -> Unavailable:
    """Unavailable for a connection failure, a reply that is not JSON and a status a retry or a
    smaller prompt can clear; ConfigurationError for any other status.

    The error names the HTTP status only: the API's text can quote the prompt.
    """
    match error:
        case APIStatusError(status_code=status) if status in _UNAVAILABLE_STATUSES or status >= 500:
            return Unavailable(status=status)
        case APIStatusError(status_code=status):
            raise ConfigurationError(f"the Messages API answered HTTP {status}") from None
        case _:
            return Unavailable(status=None)
