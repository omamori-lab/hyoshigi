"""The chat completions format of a prompt's content and of a reply, as LiteLLM documents it.

The request is OpenAI's chat completions request, https://docs.litellm.ai/docs/completion/input,
with the content blocks of https://docs.litellm.ai/docs/completion/vision,
https://docs.litellm.ai/docs/completion/document_understanding and
https://docs.litellm.ai/docs/completion/audio. The reply is the response LiteLLM gives,
https://docs.litellm.ai/docs/completion/output, read through the mapping its model_dump returns.
"""

from base64 import b64encode
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.models import responses
from hyoshigi.models.model import (
    AlternativeToken,
    Generation,
    GenerationOutcome,
    MediaTypes,
    Prompt,
    PromptPart,
    Refused,
    TokenLogProbability,
    Truncated,
    Unavailable,
    collect,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

# The media build_media sends: images, PDF files and WAV audio.
CARRIED_MEDIA = MediaTypes("image/*", "application/pdf", "audio/wav")


class ChatText(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ChatImage(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class UrlFile(TypedDict):
    """A file at a URL, which LiteLLM documents as the file's id."""

    file_id: str
    format: Literal["application/pdf"]


class InlineFile(TypedDict):
    """A file as a data URL with a filename."""

    file_data: str
    filename: str
    format: Literal["application/pdf"]


class ChatFile(TypedDict):
    type: Literal["file"]
    file: UrlFile | InlineFile


class InputAudio(TypedDict):
    data: str
    format: Literal["wav"]


class ChatAudio(TypedDict):
    type: Literal["input_audio"]
    input_audio: InputAudio


type ChatPart = ChatText | ChatImage | ChatFile | ChatAudio


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[ChatPart]


class RequestArguments(TypedDict):
    """The keyword arguments of one call to the caller's function."""

    model: str
    messages: list[SystemMessage | UserMessage]
    logprobs: NotRequired[bool]
    top_logprobs: NotRequired[int]


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Choice:
    """The first choice of a reply: why it stopped, its text, its refusal if it holds one and its
    log-probabilities as the reply gave them.
    """

    finish_reason: str
    text: str
    refusal: str | None
    logprobs: object


def is_utf8(text: str) -> bool:
    """Whether the text has a UTF-8 form, which a lone surrogate has not."""
    try:
        text.encode()
    except UnicodeEncodeError:
        return False
    return True


def build_arguments(
    prompt: Prompt,
    *,
    model: str,
    top_log_probabilities: int,
) -> RequestArguments | None:
    """The prompt as a system message and one user message of its parts in order, with the
    request for log-probabilities when the count is above 0; None if a part, or the system
    prompt, is in a form the API cannot take.

    Every part is built before the result is decided, so media the API can never carry raises
    whatever parts come before it.
    """
    content = collect([build_part(part) for part in prompt.content])
    if content is None or not is_utf8(prompt.system):
        return None
    arguments: RequestArguments = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prompt.system,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    }
    if top_log_probabilities > 0:
        arguments["logprobs"] = True
        arguments["top_logprobs"] = top_log_probabilities
    return arguments


def build_part(part: PromptPart) -> ChatPart | None:
    """A text part as a text block, a media part as the block of its kind; None for text with no
    UTF-8 form.
    """
    match part:
        case TextPart(text=text) if is_utf8(text):
            return {
                "type": "text",
                "text": text,
            }
        case TextPart():
            return None
        case MediaPart():
            return build_media(part)
        case _:
            assert_never(part)


def build_media(part: MediaPart) -> ChatImage | ChatFile | ChatAudio | None:
    """An image as an image URL, a PDF as a file and WAV audio as input audio.

    Raise ConfigurationError for any other media; None for the bytes of a media range, such as
    image/*, which no data URL can name, and for audio at a URL, which the API does not fetch.
    """
    if not CARRIED_MEDIA.covers(part.media_type):
        raise ConfigurationError(f"LiteLLM's chat completions take no {part.media_type} media")
    location = responses.locate(part)
    if location is None:
        return None
    match part.media_type.lower().partition("/")[0]:
        case "image":
            return {
                "type": "image_url",
                "image_url": {"url": location},
            }
        case "application":
            return build_file(
                part,
                location=location,
            )
        case _:
            return build_audio(part)


def build_file(
    part: MediaPart,
    *,
    location: str,
) -> ChatFile:
    """A PDF at a URL by its file id, and PDF bytes as a data URL under a filename, which OpenAI's
    API requires with file data, as the OpenAI models send it.
    """
    if part.url is not None:
        return {
            "type": "file",
            "file": {
                "file_id": location,
                "format": "application/pdf",
            },
        }
    return {
        "type": "file",
        "file": {
            "file_data": location,
            "filename": "attachment.pdf",
            "format": "application/pdf",
        },
    }


def build_audio(part: MediaPart) -> ChatAudio | None:
    """WAV bytes as input audio in base64; None for WAV audio at a URL."""
    match part:
        case MediaPart(data=bytes() as data):
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": b64encode(data).decode(),
                    "format": "wav",
                },
            }
        case _:
            return None


def read_reply(
    reply: object,
    *,
    top_log_probabilities: int,
) -> GenerationOutcome:
    """The outcome of what the function returned: a response, which gives itself as a mapping
    by model_dump, or anything else, such as a stream, which is unavailable.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    match reply:
        case object(model_dump=Callable() as dump):
            return read_dump(
                dump(),
                top_log_probabilities=top_log_probabilities,
            )
        case _:
            return Unavailable()


def read_dump(
    dump: object,
    *,
    top_log_probabilities: int,
) -> GenerationOutcome:
    """The outcome of a response's dump by its first choice; a dump without a choice of the
    documented fields is unavailable.
    """
    choice = read_choice(dump)
    if choice is None:
        return Unavailable()
    return read_outcome(
        choice,
        top_log_probabilities=top_log_probabilities,
    )


def read_outcome(
    choice: Choice,
    *,
    top_log_probabilities: int,
) -> GenerationOutcome:
    """A choice cut off at the output limit is truncated with its text. One stopped by the content
    filter, or with a refusal, is refused with the refusal if it has one. One that stopped on its
    own is a generation. One that stopped for any other reason, such as a tool call, is truncated
    with its text.
    """
    match choice:
        case Choice(
            finish_reason="length",
            text=text,
        ):
            return Truncated(text=text)
        case (
            Choice(
                finish_reason="content_filter",
                refusal=refusal,
            )
            | Choice(refusal=str() as refusal)
        ):
            return Refused(explanation=refusal)
        case Choice(finish_reason="stop"):
            return read_generation(
                choice,
                top_log_probabilities=top_log_probabilities,
            )
        case Choice(text=text):
            return Truncated(text=text)
        case unreachable:
            assert_never(unreachable)


def read_choice(dump: object) -> Choice | None:
    """The first choice of a response's dump; None if it has no choice with a string for its
    finish reason, and a message with a string or null for its content.
    """
    match dump:
        case {
            "choices": [
                {
                    "finish_reason": str() as finish_reason,
                    "message": {"content": str() | None as content} as message,
                } as choice,
                *_,
            ],
        }:
            return Choice(
                finish_reason=finish_reason,
                text=content or "",
                refusal=read_refusal(message),
                logprobs=choice.get("logprobs"),
            )
        case _:
            return None


def read_refusal(message: object) -> str | None:
    """The text of a message's refusal, when it is a string with a character.

    LiteLLM's Message has no field refusal. A refusal in the provider's reply is a key outside
    the fields of Message, which convert_to_model_response_object moves to provider_specific_fields
    (litellm/litellm_core_utils/llm_response_utils/convert_dict_to_response.py at v1.106.5). So a
    refusal is read from the message, as OpenAI's response has it, and from its
    provider_specific_fields.
    """
    match message:
        case {"refusal": str() as refusal} if refusal:
            return refusal
        case {"provider_specific_fields": {"refusal": str() as refusal}} if refusal:
            return refusal
        case _:
            return None


def read_generation(
    choice: Choice,
    *,
    top_log_probabilities: int,
) -> Generation | Unavailable:
    """The choice's text as a generation, with the choice's log-probabilities when the count is
    above 0; unavailable if they are not what LiteLLM documents.
    """
    if top_log_probabilities == 0:
        return Generation(text=choice.text)
    positions = read_positions(choice.logprobs)
    if positions is None:
        return Unavailable()
    return Generation(
        text=choice.text,
        log_probabilities=positions,
    )


def read_positions(logprobs: object) -> tuple[TokenLogProbability, ...] | None:
    """The positions of a choice's logprobs.content, none when there are none; None if they are
    not what LiteLLM documents (litellm/types/utils.py at v1.106.5, ChoiceLogprobs).
    """
    match logprobs:
        case None | {"content": None}:
            return ()
        case {"content": list() as positions}:
            read = collect([read_position(position) for position in positions])
            return None if read is None else tuple(read)
        case _:
            return None


def read_position(position: object) -> TokenLogProbability | None:
    """A token with its log-probability and its alternatives, from a mapping of token, logprob
    and top_logprobs (ChatCompletionTokenLogprob); None if it is not that, or a log-probability
    is not one.
    """
    match position:
        case {
            "token": str() as token,
            "logprob": int() | float() as log_probability,
            "top_logprobs": alternatives,
        }:
            read = read_alternatives(alternatives)
            if read is None:
                return None
            try:
                return TokenLogProbability(
                    token=token,
                    log_probability=log_probability,
                    alternatives=read,
                )
            except ConfigurationError:
                return None
        case _:
            return None


def read_alternatives(alternatives: object) -> tuple[AlternativeToken, ...] | None:
    """The alternatives of a position, none when they are null; None if they are not a list of
    mappings of token and logprob (TopLogprob), or one has a log-probability that is not one.
    """
    match alternatives:
        case None:
            return ()
        case list() as items:
            read = collect([read_alternative(item) for item in items])
            return None if read is None else tuple(read)
        case _:
            return None


def read_alternative(alternative: object) -> AlternativeToken | None:
    """A token with its log-probability; None if it is not a mapping of both, or the
    log-probability is not one.
    """
    match alternative:
        case {
            "token": str() as token,
            "logprob": int() | float() as log_probability,
        }:
            try:
                return AlternativeToken(
                    token=token,
                    log_probability=log_probability,
                )
            except ConfigurationError:
                return None
        case _:
            return None
