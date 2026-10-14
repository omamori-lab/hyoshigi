"""Anthropic's Messages format: a request's system prompt and messages, and a response.

The format is Anthropic's Messages API reference,
https://platform.claude.com/docs/en/api/messages.
"""

import json
from dataclasses import dataclass, field, replace
from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.storage import decode_data

_CONTENT = "a string or a list of content blocks"


@dataclass(frozen=True, slots=True)
class _Thought:
    part: Part


@dataclass(slots=True)
class _ActionDraft:
    """The parts of an assistant message read so far, until they become an action."""

    id: str | None = None
    thought: list[Part] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def add(self, item: _Thought | Part | ToolCall) -> None:
        """Add a thought, a part or a tool call."""
        match item:
            case _Thought(part=part):
                self.thought.append(part)
            case ToolCall():
                self.tool_calls.append(item)
            case TextPart() | MediaPart() | OpaquePart():
                self.parts.append(item)
            case unreachable:
                assert_never(unreachable)

    def build(self) -> Action:
        """Return the action read so far, with its id, or a generated one."""
        action = Action(
            thought=tuple(self.thought), parts=tuple(self.parts), tool_calls=tuple(self.tool_calls)
        )
        return action if self.id is None else replace(action, id=self.id)


def read_request(request: JSON, *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a Messages request body into its events, each with the JSONPath it came from."""
    match request:
        case {"messages": list(messages)}:
            events = read_system(request, path="$.system")
        case _:
            raise FormatError(path="$", expected="a Messages request body with a list of messages")
    for index, message in enumerate(messages):
        events.extend(read_message(message, index=index, authorship=authorship))
    return events


def read_response(message: JSON) -> Action:
    """Read the assistant message of a Messages response into the action to rate.

    A server tool's result in the response becomes an opaque part: the provider ran the
    tool before any monitor saw the call.
    """
    match message:
        case {"role": "assistant", "content": list(blocks), "id": str(message_id)}:
            draft = _ActionDraft(id=message_id)
        case {"role": "assistant", "content": list(blocks)}:
            draft = _ActionDraft()
        case _:
            raise FormatError(
                path="$", expected="an assistant message with a list of content blocks"
            )
    for index, block in enumerate(blocks):
        draft.add(read_assistant_block(block, path=f"$.content[{index}]"))
    return draft.build()


def read_system(request: JSON, *, path: str) -> list[tuple[str, Event]]:
    """Read a request's system prompt, a string or text blocks, into a scaffold's observation."""
    match request:
        case {"system": str(text)}:
            parts: tuple[Part, ...] = (TextPart(text=text),)
        case {"system": list(blocks)}:
            parts = tuple(
                read_text_block(block, path=f"{path}[{i}]") for i, block in enumerate(blocks)
            )
        case {"system": None}:
            return []
        case {"system": _}:
            raise FormatError(path=path, expected="a string or a list of text blocks")
        case _:
            return []
    return [(path, Observation(author=Author.SCAFFOLD, source="system", parts=parts))]


def read_text_block(block: JSON, *, path: str) -> TextPart:
    """Read one text block of a system prompt, which holds text only."""
    match block:
        case {"type": "text", "text": str(text)}:
            return TextPart(text=text)
        case _:
            raise FormatError(path=path, expected="a text block")


def read_message(message: JSON, *, index: int, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a user or assistant message into its events."""
    path = f"$.messages[{index}]"
    match message:
        case {"role": "user", "content": str() | list() as content}:
            author = authorship(index, message=message)
            return read_user_content(content, author=author, path=f"{path}.content")
        case {"role": "assistant", "content": str() | list() as content}:
            return read_assistant_content(content, path=f"{path}.content")
        case {"role": "user" | "assistant"}:
            raise FormatError(path=f"{path}.content", expected=_CONTENT)
        case _:
            raise FormatError(path=path, expected="a user or assistant message with content")


def read_user_content(
    content: str | list[JSON], *, author: Author, path: str
) -> list[tuple[str, Event]]:
    """Read a user message's content: observations by its author, and tool results."""
    if isinstance(content, str):
        return [(path, Observation(author=author, parts=(TextPart(text=content),)))]
    events: list[tuple[str, Event]] = []
    parts: list[Part] = []
    for index, block in enumerate(content):
        block_path = f"{path}[{index}]"
        match block:
            case {"type": "tool_result"}:
                events.extend(observe(parts, author=author, path=path))
                parts = []
                events.append((block_path, read_tool_result(block, path=block_path)))
            case _:
                parts.append(read_content_block(block, path=block_path))
    return events + observe(parts, author=author, path=path)


def observe(parts: list[Part], *, author: Author, path: str) -> list[tuple[str, Event]]:
    """Return the parts read so far as one observation by their author, if there are any."""
    if not parts:
        return []
    return [(path, Observation(author=author, parts=tuple(parts)))]


def read_tool_result(block: JSON, *, path: str) -> ToolObservation:
    """Read a tool_result block into the tool's observation."""
    match block:
        case {"tool_use_id": str(call_id), "is_error": bool(is_error)}:
            pass
        case {"tool_use_id": str(call_id)} if "is_error" not in block:
            is_error = False
        case _:
            raise FormatError(
                path=path, expected="a tool_result with a tool_use_id, and is_error a boolean"
            )
    parts = read_result_content(block, path=f"{path}.content")
    return ToolObservation(call_id=call_id, is_error=is_error, parts=parts)


def read_result_content(block: JSON, *, path: str) -> tuple[Part, ...]:
    """Read a tool result's content: absent, a string, or content blocks."""
    match block:
        case {"content": str(text)}:
            return (TextPart(text=text),)
        case {"content": list(items)}:
            return tuple(
                read_content_block(item, path=f"{path}[{i}]") for i, item in enumerate(items)
            )
        case {"content": None}:
            return ()
        case {"content": _}:
            raise FormatError(path=path, expected=_CONTENT)
        case _:
            return ()


def read_content_block(block: JSON, *, path: str) -> Part:
    """Read a text, image or document block; a block of another type becomes opaque."""
    match block:
        case {"type": "text", "text": str(text)}:
            return TextPart(text=text)
        case {"type": "image", "source": dict() as source}:
            return read_image_source(source, path=f"{path}.source")
        case {"type": "document", "source": dict() as source}:
            return read_document_source(source, path=f"{path}.source")
        case {"type": "text" | "image" | "document" as kind}:
            raise FormatError(path=path, expected=f"a {kind} block in the Messages format")
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(path=path, expected="a content block with a type")


def read_image_source(source: JSON, *, path: str) -> Part:
    """Read an image's source: base64 data, or a URL, whose subtype is unknown."""
    match source:
        case {"type": "base64", "media_type": str(media_type), "data": str(data)}:
            return build_media(media_type, data=decode_data(data, path=f"{path}.data"), path=path)
        case {"type": "url", "url": str(url)}:
            return MediaPart(media_type="image/*", url=url)
        case {"type": "base64" | "url"}:
            raise FormatError(path=path, expected="a base64 or URL image source")
        case {"type": str()}:
            return OpaquePart(kind="image")
        case _:
            raise FormatError(path=path, expected="an image source with a type")


def read_document_source(source: JSON, *, path: str) -> Part:
    """Read a document's source: base64 data, a PDF's URL, or plain text."""
    match source:
        case {"type": "base64", "media_type": str(media_type), "data": str(data)}:
            return build_media(media_type, data=decode_data(data, path=f"{path}.data"), path=path)
        case {"type": "url", "url": str(url)}:
            return MediaPart(media_type="application/pdf", url=url)
        case {"type": "text", "data": str(text)}:
            return TextPart(text=text)
        case {"type": "base64" | "url" | "text"}:
            raise FormatError(path=path, expected="a base64, URL or text document source")
        case {"type": str()}:
            return OpaquePart(kind="document")
        case _:
            raise FormatError(path=path, expected="a document source with a type")


def build_media(media_type: str, *, data: bytes | None, path: str) -> MediaPart:
    """Build a media part from base64 data, or raise FormatError at the path if it cannot be one."""
    try:
        return MediaPart(media_type=media_type, data=data)
    except ConfigurationError:
        raise FormatError(path=path, expected="an IANA media type") from None


def read_assistant_content(content: str | list[JSON], *, path: str) -> list[tuple[str, Event]]:
    """Read an assistant message's content into actions, split by server tools' results.

    Each action's id is the JSONPath of its first block.
    """
    if isinstance(content, str):
        return [(path, Action(id=path, parts=(TextPart(text=content),)))]
    blocks = content
    events: list[tuple[str, Event]] = []
    draft: _ActionDraft | None = None
    for index, block in enumerate(blocks):
        block_path = f"{path}[{index}]"
        match block:
            case {"type": str(kind), "tool_use_id": str(call_id)} if kind.endswith("_tool_result"):
                events.extend(close(draft))
                draft = None
                events.append(
                    (block_path, read_server_result(block, call_id=call_id, path=block_path))
                )
            case _:
                draft = draft or _ActionDraft(id=block_path)
                draft.add(read_assistant_block(block, path=block_path))
    return events + close(draft)


def close(draft: _ActionDraft | None) -> list[tuple[str, Event]]:
    """Return a draft's action with the path it started at, if a draft was started."""
    if draft is None or draft.id is None:
        return []
    return [(draft.id, draft.build())]


def read_server_result(block: JSON, *, call_id: str, path: str) -> ToolObservation:
    """Read a server tool's result: its content, as JSON text, observed from the server.

    A server tool reports a failure as an error object in the content, whose type ends in
    _error, such as web_search_tool_result_error.
    """
    match block:
        case {"content": {"type": str(kind)} as content}:
            is_error = kind.endswith("_error")
        case {"content": content}:
            is_error = False
        case _:
            raise FormatError(path=path, expected="a server tool's result with content")
    text = json.dumps(content, ensure_ascii=False)
    return ToolObservation(
        call_id=call_id, source="server", is_error=is_error, parts=(TextPart(text=text),)
    )


def read_assistant_block(block: JSON, *, path: str) -> _Thought | Part | ToolCall:
    """Read an assistant's block: a thought, a tool call, or a content block."""
    match block:
        case {"type": "thinking", "thinking": str(text)}:
            return _Thought(TextPart(text=text))
        case {"type": "redacted_thinking"}:
            return _Thought(OpaquePart(kind="redacted_thinking"))
        case {
            "type": "tool_use" | "server_tool_use",
            "id": str(call_id),
            "name": str(name),
            "input": dict(arguments),
        }:
            return ToolCall(
                id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False)
            )
        case {"type": "thinking" | "tool_use" | "server_tool_use" as kind}:
            raise FormatError(path=path, expected=f"a {kind} block in the Messages format")
        case _:
            return read_content_block(block, path=path)
