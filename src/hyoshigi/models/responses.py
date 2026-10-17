"""The Open Responses format of a prompt's content and of a reply, which the OpenAI and
OpenRouter models share.

The format is the Open Responses specification of 2026-04-24,
https://www.openresponses.org/specification/2026-04-24, which OpenAI's and OpenRouter's
Responses APIs both speak.
"""

import mimetypes
from base64 import b64encode
from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict, assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import (
    Generation,
    GenerationOutcome,
    Prompt,
    PromptPart,
    Refused,
    Truncated,
    Unavailable,
    collect,
)
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import MediaPart, OpaquePart, TextPart

# The built-in table only: the platform's mime.types files differ between machines.
_EXTENSIONS = mimetypes.MimeTypes()


class InputText(TypedDict):
    type: Literal["input_text"]
    text: str


class InputImage(TypedDict):
    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto"]


class InputFile(TypedDict):
    """A file by URL, or as a data URL with a filename."""

    type: Literal["input_file"]
    file_url: NotRequired[str]
    file_data: NotRequired[str]
    filename: NotRequired[str]


type InputPart = InputText | InputImage | InputFile
type Piece = TextPart | Refused | OpaquePart


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    """An assistant's message in a reply: its parts as text, refusals and parts of other
    types, and whether it is whole, completed with no part of another type.
    """

    pieces: list[Piece]
    whole: bool


def build_content(prompt: Prompt) -> list[InputPart] | None:
    """The prompt's content as the parts of one user message.

    Every part is built before the result is decided, so media the API can never carry
    raises whatever parts come before it; None if a part is in a form the API cannot take.
    """
    return collect([build_part(part) for part in prompt.content])


def build_part(part: PromptPart) -> InputPart | None:
    """A text part as input text, a media part as an image or a file."""
    match part:
        case TextPart(text=text):
            return {"type": "input_text", "text": text}
        case MediaPart():
            return build_media(part)
        case _:
            assert_never(part)


def build_media(part: MediaPart) -> InputImage | InputFile | None:
    """Images become input images, application and text media input files.

    Raise ConfigurationError for any other media; None for the bytes of a media range, such
    as image/*, which no data URL can name. The API takes file data only with a filename, as
    every example of https://developers.openai.com/api/docs/guides/pdf-files does.
    """
    match part.media_type.partition("/")[0].lower(), locate(part):
        case "image" | "application" | "text", None:
            return None
        case "image", str(image_url):
            return {"type": "input_image", "image_url": image_url, "detail": "auto"}
        case "application" | "text", str(file_url) if part.url is not None:
            return {"type": "input_file", "file_url": file_url}
        case "application" | "text", str(file_data):
            extension = _EXTENSIONS.guess_extension(part.media_type) or ""
            return {
                "type": "input_file",
                "filename": f"attachment{extension}",
                "file_data": file_data,
            }
        case _:
            raise ConfigurationError(f"the Responses API takes no {part.media_type} media")


def locate(part: MediaPart) -> str | None:
    """The part's URL, or its bytes as a data URL; None for the bytes of a media range."""
    return part.url if part.url is not None else encode(part)


def encode(part: MediaPart) -> str | None:
    """The part's bytes as a base64 data URL; None if it has none or its type is a range."""
    if part.data is None or part.media_type.endswith("/*"):
        return None
    return f"data:{part.media_type};base64,{b64encode(part.data).decode()}"


def names_refusal(code: str) -> bool:
    """Whether an error code is a refusal or names a policy."""
    return code == "refusal" or "policy" in code


def read_reply(body: JSON) -> GenerationOutcome:
    """The outcome of a reply's JSON body by its status; a body whose output is not what the
    format documents is unavailable, whatever its status.
    """
    messages = read_output(body)
    if messages is None:
        return Unavailable()
    match body:
        case {"status": "completed"}:
            return read_completed(messages)
        case {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}:
            return Refused()
        case {"status": "failed"}:
            return Refused() if is_blocked(body) else Unavailable()
        case _:
            return Truncated(text=join_text(messages))


def read_completed(messages: list[Message]) -> GenerationOutcome:
    """A completed reply: its first refusal if it holds one, nothing readable if it holds no
    message, the text so far if a message was not completed or held a part of another type,
    else the text.
    """
    refusals = [
        piece for message in messages for piece in message.pieces if isinstance(piece, Refused)
    ]
    if refusals:
        return refusals[0]
    if not messages:
        return Unavailable()
    if all(message.whole for message in messages):
        return Generation(text=join_text(messages))
    return Truncated(text=join_text(messages))


def join_text(messages: list[Message]) -> str:
    """The text of the messages' output text parts, in order, with no separator."""
    return "".join(
        piece.text
        for message in messages
        for piece in message.pieces
        if isinstance(piece, TextPart)
    )


def is_blocked(body: JSON) -> bool:
    """Whether a failed reply's error code, or its error type, is a refusal or names a policy.

    The error type is OpenRouter's, at the top level of a failed reply, where
    https://openrouter.ai/docs/api_reference/errors-and-debugging puts it.
    """
    match body:
        case {"error": {"code": str(code)}} if names_refusal(code):
            return True
        case {"error_type": str(error_type)} if names_refusal(error_type):
            return True
    return False


def read_output(body: JSON) -> list[Message] | None:
    """The output's assistant messages in order, skipping its items of other types; None if
    the output, an item or a part of a message is not what the format documents.
    """
    match body:
        case {"output": list(items)}:
            read = collect([read_item(item) for item in items])
        case _:
            return None
    return None if read is None else [item for item in read if isinstance(item, Message)]


def read_item(item: JSON) -> Message | OpaquePart | None:
    """An output item: the assistant's message, or an item of another type as opaque; None if
    it has no type, or is a message that is not the assistant's or holds no list of parts.
    """
    match item:
        case {"type": "message", "role": "assistant", "content": list(content)}:
            return read_message(content, completed=item.get("status") == "completed")
        case {"type": str(kind)} if kind != "message":
            return OpaquePart(kind=kind)
        case _:
            return None


def read_message(content: list[JSON], *, completed: bool) -> Message | None:
    """An assistant's message from its parts; None if a part is not what the format documents."""
    pieces = collect([read_part(part) for part in content])
    if pieces is None:
        return None
    unread = any(isinstance(piece, OpaquePart) for piece in pieces)
    return Message(pieces=pieces, whole=completed and not unread)


def read_part(part: JSON) -> Piece | None:
    """A message's part: output text, a refusal, or a part of another type as opaque; None if
    it has no type, or is output text or a refusal without its text.
    """
    match part:
        case {"type": "output_text", "text": str(text)}:
            return TextPart(text=text)
        case {"type": "refusal", "refusal": str(refusal)}:
            return Refused(explanation=refusal)
        case {"type": str(kind)} if kind not in ("output_text", "refusal"):
            return OpaquePart(kind=kind)
        case _:
            return None
