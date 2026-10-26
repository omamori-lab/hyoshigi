"""Anthropic's Messages API as a monitor's model, through the caller's client.

The API is Anthropic's Messages API reference, https://platform.claude.com/docs/en/api/messages.
"""

from base64 import b64encode

try:
    from anthropic import Anthropic, APIError, APIStatusError, AsyncAnthropic, BaseModel
    from anthropic.types import (
        ContentBlock,
        DocumentBlockParam,
        ImageBlockParam,
        Message,
        RefusalStopDetails,
        TextBlock,
        TextBlockParam,
        ThinkingConfigParam,
    )
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Anthropic models need the extra: pip install 'hyoshigi[anthropic]'",
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
    Rejected,
    Truncated,
    Unavailable,
    check_client,
    check_name,
    check_perceives,
    collect,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

type Block = TextBlockParam | ImageBlockParam | DocumentBlockParam
# What the client raises instead of a reply: the SDK's errors, and ValueErrors: a body that is
# not JSON or not UTF-8, text it cannot encode, and the SDK refusing the request's arguments.
type Failure = APIError | ValueError
_FAILURES = (
    APIError,
    ValueError,
)

# The client errors a retry can clear, which the SDK retries.
_UNAVAILABLE_STATUSES = frozenset({408, 409, 429})

# The start of the message of the 400 for an input over the context window: Anthropic's "Context
# windows" guide, section "Context window overflow behavior".
_PROMPT_TOO_LONG = "prompt is too long"

# The media translate_url and translate_data send: the image types, PDF and plain text documents
# of https://platform.claude.com/docs/en/build-with-claude/vision, pdf-support and citations.
CARRIED_MEDIA = MediaTypes(
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "text/plain",
)


def build_model(
    client: Anthropic,
    *,
    model: str,
    max_tokens: int,
    thinking: ThinkingConfigParam | None,
    perceives: MediaTypes,
) -> GenerativeModel:
    """A model on the Messages API, through the caller's client.

    The thinking configuration is sent only when given. Raise ConfigurationError if the client
    is not an Anthropic, the model id is empty, max_tokens is below 1 or perceives names media
    the API cannot carry.
    Generating raises ConfigurationError if the SDK refuses the request's arguments before
    sending them, as it does for a max_tokens that may take longer than ten minutes without
    streaming.
    """
    check_client(
        client,
        expected=Anthropic,
        twin=AsyncAnthropic,
        twin_constructor="AsyncGenerativeModel.from_anthropic",
    )
    check_arguments(
        model=model,
        max_tokens=max_tokens,
        perceives=perceives,
    )

    def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_request(
            prompt,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        if request is None:
            return Unavailable()
        try:
            reply: Message | Failure = client.messages.create(**request)
        except _FAILURES as error:
            reply = error
        return read_reply(reply)

    return GenerativeModel(
        name=f"anthropic:{model}",
        generate=generate,
        perceives=perceives,
    )


def build_async_model(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    thinking: ThinkingConfigParam | None,
    perceives: MediaTypes,
) -> AsyncGenerativeModel:
    """A model on the Messages API, through the caller's async client.

    The thinking configuration is sent only when given. Raise ConfigurationError if the client
    is not an AsyncAnthropic, the model id is empty, max_tokens is below 1 or perceives names media
    the API cannot carry.
    Generating raises ConfigurationError if the SDK refuses the request's arguments before
    sending them, as it does for a max_tokens that may take longer than ten minutes without
    streaming.
    """
    check_client(
        client,
        expected=AsyncAnthropic,
        twin=Anthropic,
        twin_constructor="GenerativeModel.from_anthropic",
    )
    check_arguments(
        model=model,
        max_tokens=max_tokens,
        perceives=perceives,
    )

    async def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_request(
            prompt,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        if request is None:
            return Unavailable()
        try:
            reply: Message | Failure = await client.messages.create(**request)
        except _FAILURES as error:
            reply = error
        return read_reply(reply)

    return AsyncGenerativeModel(
        name=f"anthropic:{model}",
        generate=generate,
        perceives=perceives,
    )


def check_arguments(
    *,
    model: str,
    max_tokens: int,
    perceives: MediaTypes,
) -> None:
    """Raise ConfigurationError unless there is a model id, room for a token and perceives
    names only media the API carries.
    """
    check_name(model)
    if max_tokens < 1:
        raise ConfigurationError(f"max_tokens must be at least 1, not {max_tokens}")
    check_perceives(
        perceives,
        carries=CARRIED_MEDIA,
    )


def build_request(
    prompt: Prompt,
    *,
    model: str,
    max_tokens: int,
    thinking: ThinkingConfigParam | None,
) -> MessageCreateParamsNonStreaming | None:
    """The prompt as one user message and a system prompt, with the thinking configuration if
    there is one; None if there is nothing to send.

    Text with no non-whitespace character is left out, and the prompt is not sent if a part
    cannot be taken in its form.
    """
    blocks = collect([translate_part(part) for part in prompt.content if not is_blank(part)])
    if not blocks:
        return None
    request: MessageCreateParamsNonStreaming = {
        "model": model,
        "max_tokens": max_tokens,
        "system": prompt.system,
        "messages": [
            {
                "role": "user",
                "content": blocks,
            },
        ],
    }
    if thinking is not None:
        request["thinking"] = thinking
    return request


def is_blank(part: PromptPart) -> bool:
    """Whether a part is text with no non-whitespace character, which the API rejects in a
    message that holds nothing else.
    """
    return (
        isinstance(
            part,
            TextPart,
        )
        and not part.text.strip()
    )


def translate_part(part: PromptPart) -> Block | None:
    """A prompt part as a content block; None if the API cannot take it in its form."""
    match part:
        case TextPart(text=text):
            return {
                "type": "text",
                "text": text,
            }
        case MediaPart(url=str(url)):
            return translate_url(
                part.media_type.lower(),
                url=url,
            )
        case MediaPart(data=bytes(data)):
            return translate_data(
                part.media_type.lower(),
                data=data,
            )


def translate_url(
    media_type: str,
    *,
    url: str,
) -> Block | None:
    """Media at a URL as an image or a PDF document; plain text is not taken by URL."""
    match media_type:
        case "image/*" | "image/jpeg" | "image/png" | "image/gif" | "image/webp":
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": url,
                },
            }
        case "application/pdf":
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": url,
                },
            }
        case "text/plain":
            return None
        case _:
            raise ConfigurationError(f"the Messages API cannot carry {media_type} media")


def translate_data(
    media_type: str,
    *,
    data: bytes,
) -> Block | None:
    """Media bytes as an image, a PDF document or a text document; an image range has no type."""
    encoded = b64encode(data).decode("ascii")
    match media_type:
        case "image/jpeg" | "image/png" | "image/gif" | "image/webp" as image_type:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_type,
                    "data": encoded,
                },
            }
        case "application/pdf":
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": encoded,
                },
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
        "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": text,
        },
    }


def read_reply(reply: Message | Failure) -> GenerationOutcome:
    """The outcome of a reply, or of the failure that stood in for it.

    A reply that is not an assistant's message is unavailable.
    """
    match reply:
        case Message(
            type="message",
            role="assistant",
        ):
            return read_message(reply)
        case APIError() | ValueError():
            return read_failure(reply)
        case _:
            return Unavailable()


def read_failure(failure: Failure) -> Rejected | Unavailable:
    """The outcome of a failure that stood in for a reply.

    A status that says the API cannot use the prompt is rejected with it. Any other client error
    except 408, 409 and 429 raises ConfigurationError naming only the status, and so does a
    plain ValueError, the SDK refusing the request's arguments. Any other status is unavailable
    with it, and any other failure, a ValueError's subclasses included, is unavailable.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    match failure:
        case APIStatusError(
            status_code=status,
            body=body,
        ) if is_rejection(
            status,
            body=body,
        ):
            return Rejected(status=status)
        case APIStatusError(status_code=status) if (
            400 <= status < 500 and status not in _UNAVAILABLE_STATUSES
        ):
            raise ConfigurationError(f"the Messages API answered HTTP {status}")
        case APIStatusError(status_code=status):
            return Unavailable(status=status)
        case ValueError() if type(failure) is ValueError:
            raise ConfigurationError(
                "the Anthropic SDK refused the request's arguments before sending them",
            )
        case _:
            return Unavailable()


def is_rejection(
    status: int,
    *,
    body: object,
) -> bool:
    """Whether a status and its body say the API cannot use the prompt: a 413, or a 400 whose
    message starts "prompt is too long", which is read and never kept.
    """
    match status, body:
        case 413, _:
            return True
        case 400, {
            "error": {
                "type": "invalid_request_error",
                "message": str(message),
            },
        }:
            return message.startswith(_PROMPT_TOO_LONG)
        case _:
            return False


def read_message(message: Message) -> GenerationOutcome:
    """The outcome of an assistant's message: only a turn that ended on its own is complete.

    Any other stop reason, including one the API adds later, is truncated. A message without
    readable content is unavailable.
    """
    match message:
        case Message(
            stop_reason="refusal",
            stop_details=RefusalStopDetails(explanation=str(explanation)),
        ):
            return Refused(explanation=explanation)
        case Message(stop_reason="refusal"):
            return Refused()
        case Message(
            stop_reason=stop_reason,
            content=list(blocks),
        ):
            return read_content(
                stop_reason,
                blocks=blocks,
            )
        case _:
            return Unavailable()


def read_content(
    stop_reason: str | None,
    *,
    blocks: list[ContentBlock],
) -> GenerationOutcome:
    """The text of a reply's blocks as a generation if the turn ended on its own, else cut off."""
    match (
        stop_reason,
        read_text(blocks),
    ):
        case (
            _,
            None,
        ):
            return Unavailable()
        case (
            "end_turn" | "stop_sequence",
            str(text),
        ):
            return Generation(text=text)
        case (
            _,
            str(text),
        ):
            return Truncated(text=text)


def read_text(blocks: list[ContentBlock]) -> str | None:
    """The text blocks of a reply, joined; None if an element is not a block with a type or a
    text block has no text.

    Thinking, tool use and a block type the API adds later are not part of a reply's text. The
    SDK reads a block of a type it does not know as a text block of that type.
    """
    texts: list[str] = []
    for block in blocks:
        match block:
            case TextBlock(
                type="text",
                text=str(text),
            ):
                texts.append(text)
            case TextBlock(type="text"):
                return None
            case BaseModel(type=str()):
                pass
            case _:
                return None
    return "".join(texts)
