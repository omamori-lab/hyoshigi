"""The Open Responses format: a request's instructions and input items, and a response's output.

The format is the Open Responses specification of 2026-04-24,
https://www.openresponses.org/specification/2026-04-24, and its OpenAPI document,
https://www.openresponses.org/openapi/2026-04-24/openapi.json.
"""

from typing import assert_never
from urllib.parse import urlsplit

from hyoshigi.errors import FormatError
from hyoshigi.trajectory.formats.action_draft import ActionDraft, Thought, close
from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.formats.media import read_data_url
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, MediaPart, OpaquePart, Part, TextPart

_SCAFFOLD_ROLES = ("system", "developer")
# The content parts that carry their text in a text field.
_TEXT_TYPES = ("input_text", "output_text", "text", "summary_text", "reasoning_text")


def read_request(request: JSON, *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a request body into its events, each with the JSONPath it came from."""
    match request:
        case dict():
            events = read_instructions(request) + read_previous_response(request)
        case _:
            raise FormatError(path="$", expected="a request body")
    match request:
        case {"input": str(text)}:
            author = authorship(0, message=text)
            return [*events, ("$.input", Observation(author=author, parts=(TextPart(text=text),)))]
        case {"input": list(items)}:
            return events + read_items(items, authorship=authorship)
        case {"input": items} if items is not None:
            raise FormatError(path="$.input", expected="input as text or a list of items")
        case _:
            return events


def read_response(output: JSON) -> Action:
    """Read a response's output items into the action to rate, with a generated id.

    An item of a type the model did not write, such as a tool the provider ran, becomes an
    opaque part; a message that is not the assistant's, or an item without a type, raises
    FormatError.
    """
    match output:
        case list(items):
            draft = ActionDraft()
        case _:
            raise FormatError(path="$", expected="a list of output items")
    for index, item in enumerate(items):
        for piece in read_output_item(item, path=f"$[{index}]"):
            draft.add(piece)
    return draft.build()


def read_output_item(item: JSON, *, path: str) -> list[Thought | Part | ToolCall]:
    """Read an output item: the model's into an action's pieces, an item of another type as
    opaque; a message must be the assistant's.
    """
    match item:
        case {"type": "reasoning" | "function_call"} | {"type": "message", "role": "assistant"}:
            return read_model_item(item, path=path)
        case {"type": "message"}:
            raise FormatError(path=path, expected="an assistant message")
        case {"type": str(kind)}:
            return [OpaquePart(kind=kind)]
        case _:
            raise FormatError(path=path, expected="an output item with a type")


def read_instructions(request: JSON) -> list[tuple[str, Event]]:
    """Read a request's instructions into the scaffold's observation, if it has any."""
    match request:
        case {"instructions": str(text)}:
            parts = (TextPart(text=text),)
            observation = Observation(author=Author.SCAFFOLD, source="instructions", parts=parts)
            return [("$.instructions", observation)]
        case {"instructions": text} if text is not None:
            raise FormatError(path="$.instructions", expected="instructions as text")
        case _:
            return []


def read_previous_response(request: JSON) -> list[tuple[str, Event]]:
    """Read a reference to a stored response into a placeholder for the turns it holds."""
    match request:
        case {"previous_response_id": str()}:
            parts = (OpaquePart(kind="previous_response"),)
            observation = Observation(
                author=Author.UNKNOWN, source="previous_response", parts=parts
            )
            return [("$.previous_response_id", observation)]
        case {"previous_response_id": response_id} if response_id is not None:
            raise FormatError(path="$.previous_response_id", expected="a response id as text")
        case _:
            return []


def read_items(items: list[JSON], *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read input items into events: the model's consecutive items become one action.

    Each action's id is the JSONPath of its first item.
    """
    events: list[tuple[str, Event]] = []
    draft: ActionDraft | None = None
    for index, item in enumerate(items):
        path = f"$.input[{index}]"
        match read_item(item, index=index, authorship=authorship):
            case Observation() as observation:
                events.extend(close(draft))
                draft = None
                events.append((path, observation))
            case list() as pieces:
                draft = draft or ActionDraft(id=path)
                for piece in pieces:
                    draft.add(piece)
            case unreachable:
                assert_never(unreachable)
    return events + close(draft)


def read_item(
    item: JSON, *, index: int, authorship: Authorship
) -> Observation | list[Thought | Part | ToolCall]:
    """Read an input item: the model's into the pieces of an action, any other into an
    observation; an item of a type the format does not model is an unknown author's.
    """
    path = f"$.input[{index}]"
    kind = read_type(item, path=path)
    match kind, item:
        case ("reasoning" | "function_call", _) | ("message", {"role": "assistant"}):
            return read_model_item(item, path=path)
        case "message", _:
            return read_message(item, index=index, authorship=authorship, path=path)
        case "function_call_output", _:
            return read_function_call_output(item, path=path)
        case "compaction", _:
            parts = (OpaquePart(kind="compaction"),)
            return Observation(author=Author.AGENT, source="compaction", parts=parts)
        case _:
            return Observation(author=Author.UNKNOWN, parts=(OpaquePart(kind=kind),))


def read_message(item: JSON, *, index: int, authorship: Authorship, path: str) -> Observation:
    """Read a user, system or developer message into an observation."""
    match item:
        case {"role": "user", "content": str() | list() as content}:
            parts = read_input_content(content, path=f"{path}.content")
            return Observation(author=authorship(index, message=item), parts=parts)
        case {"role": str(role), "content": str() | list() as content} if role in _SCAFFOLD_ROLES:
            parts = read_input_content(content, path=f"{path}.content")
            return Observation(author=Author.SCAFFOLD, source=role, parts=parts)
        case _:
            raise FormatError(
                path=path, expected="a user, assistant, system or developer message with content"
            )


def read_function_call_output(item: JSON, *, path: str) -> ToolObservation:
    """Read a function call's output into the tool's observation."""
    match item:
        case {"call_id": str(call_id), "output": str() | list() as output}:
            parts = read_input_content(output, path=f"{path}.output")
            return ToolObservation(call_id=call_id, parts=parts)
        case _:
            raise FormatError(
                path=path, expected="a function call output with a call_id and its output"
            )


def read_type(item: JSON, *, path: str) -> str:
    """Return an item's type: a message may omit it, and an item reference may omit it or
    set it to null.
    """
    match item:
        case {"type": str(kind)}:
            return kind
        case {"type": None, "id": str()}:
            return "item_reference"
        case {"role": str()} if "type" not in item:
            return "message"
        case {"id": str()} if "type" not in item:
            return "item_reference"
        case _:
            raise FormatError(path=path, expected="an item with a type")


def read_model_item(item: JSON, *, path: str) -> list[Thought | Part | ToolCall]:
    """Read reasoning, a function call or an assistant message into an action's pieces."""
    match item:
        case {"type": "reasoning"}:
            return [Thought(part) for part in read_reasoning(item, path=path)]
        case {
            "type": "function_call",
            "call_id": str(call_id),
            "name": str(name),
            "arguments": str(arguments),
        }:
            return [ToolCall(id=call_id, name=name, arguments=arguments)]
        case {"type": "function_call"}:
            raise FormatError(
                path=path, expected="a function call with a call_id, a name and its arguments"
            )
        case {"content": str(text)}:
            return [TextPart(text=text)]
        case {"content": list(parts)}:
            return [
                read_model_part(part, path=f"{path}.content[{i}]") for i, part in enumerate(parts)
            ]
        case _:
            raise FormatError(path=path, expected="an assistant message with content")


def read_reasoning(item: JSON, *, path: str) -> list[Part]:
    """Read a reasoning item: its content, then its summary, then its encrypted content as
    opaque.
    """
    match item:
        case {"summary": list(summary)}:
            pass
        case _:
            raise FormatError(path=path, expected="a reasoning item with a summary")
    match item:
        case {"content": list(content)}:
            pass
        case {"content": content} if content is not None:
            raise FormatError(path=f"{path}.content", expected="reasoning content as a list")
        case _:
            content = []
    match item:
        case {"encrypted_content": str()}:
            encrypted = [OpaquePart(kind="reasoning")]
        case {"encrypted_content": encrypted_content} if encrypted_content is not None:
            raise FormatError(
                path=f"{path}.encrypted_content", expected="encrypted content as text"
            )
        case _:
            encrypted = []
    return [
        *(read_model_part(part, path=f"{path}.content[{i}]") for i, part in enumerate(content)),
        *(read_model_part(part, path=f"{path}.summary[{i}]") for i, part in enumerate(summary)),
        *encrypted,
    ]


def read_model_part(part: JSON, *, path: str) -> Part:
    """Read a part the model wrote: text or a refusal; a part of another type becomes opaque."""
    match part:
        case {"type": str(kind), "text": str(text)} if kind in _TEXT_TYPES:
            return TextPart(text=text)
        case {"type": "refusal", "refusal": str(text)}:
            return TextPart(text=text)
        case {"type": str(kind)} if kind in _TEXT_TYPES or kind == "refusal":
            raise FormatError(path=path, expected="a text or refusal part with its text")
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(path=path, expected="a content part with a type")


def read_input_content(content: str | list[JSON], *, path: str) -> tuple[Part, ...]:
    """Read input content: a string, or text, image, file and video parts."""
    if isinstance(content, str):
        return (TextPart(text=content),)
    return tuple(read_input_part(part, path=f"{path}[{i}]") for i, part in enumerate(content))


def read_input_part(part: JSON, *, path: str) -> Part:
    """Read an input part: text, an image, a video or a file; another type becomes opaque."""
    match part:
        case {"type": "input_text", "text": str(text)}:
            return TextPart(text=text)
        case {"type": str(kind)} if kind in ("input_image", "input_video", "input_file"):
            return read_media_part(part, kind=kind, path=path)
        case {"type": "input_text"}:
            raise FormatError(path=path, expected="an input_text part with its text")
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(path=path, expected="a content part with a type")


def read_media_part(part: JSON, *, kind: str, path: str) -> Part:
    """Read an image, a video or a file into media, or as opaque where it names no type.

    File data in a data URL becomes media of the URL's type; a file by URL, by id or as
    bare base64, and an image or a video without a URL, become opaque.
    """
    match part:
        case {"type": "input_image", "image_url": str(url)}:
            path = f"{path}.image_url"
            return read_media_url(url, media_type="image/*", kind="input_image", path=path)
        case {"type": "input_video", "video_url": str(url)}:
            path = f"{path}.video_url"
            return read_media_url(url, media_type="video/*", kind="input_video", path=path)
        case {"type": "input_file", "file_data": str(data)}:
            return read_data_url(data, path=f"{path}.file_data") or OpaquePart(kind="input_file")
        case (
            {"type": "input_image", "image_url": url}
            | {"type": "input_video", "video_url": url}
            | {"type": "input_file", "file_data": url}
        ) if url is not None:
            raise FormatError(path=path, expected="a media part's URL or data as text")
        case _:
            return OpaquePart(kind=kind)


def read_media_url(url: str, *, media_type: str, kind: str, path: str) -> Part:
    """Read a data URL into media, another URL into media of the given range, and a string
    with no scheme, such as bare base64, as opaque.
    """
    if not urlsplit(url).scheme:
        return OpaquePart(kind=kind)
    return read_data_url(url, path=path) or MediaPart(media_type=media_type, url=url)
