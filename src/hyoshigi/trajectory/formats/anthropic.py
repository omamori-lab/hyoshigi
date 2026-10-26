"""Anthropic's Messages format: a request's system prompt and messages, and a response.

The format is Anthropic's Messages API reference,
https://platform.claude.com/docs/en/api/messages.
"""

import json

from hyoshigi.errors import FormatError
from hyoshigi.trajectory.formats.action_draft import ActionDraft, Thought, close
from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.formats.media import build_media
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.storage import decode_data

_CONTENT = "a string or a list of content blocks"


def read_request(
    request: JSON,
    *,
    authorship: Authorship,
) -> list[tuple[str, Event]]:
    """Read a Messages request body into its events, each with the JSONPath it came from."""
    match request:
        case {"messages": list(messages)}:
            events = read_system(
                request,
                path="$.system",
            )
        case _:
            raise FormatError(
                path="$",
                expected="a Messages request body with a list of messages",
            )
    for index, message in enumerate(messages):
        events.extend(
            read_message(
                message,
                index=index,
                authorship=authorship,
            ),
        )
    return events


def read_response(message: JSON) -> Action:
    """Read the assistant message of a Messages response into the action to rate.

    A server tool's result in the response becomes an opaque part: the provider ran the
    tool before any monitor saw the call.
    """
    match message:
        case {
            "role": "assistant",
            "content": list(blocks),
            "id": str(message_id),
        }:
            draft = ActionDraft(id=message_id)
        case {
            "role": "assistant",
            "content": list(blocks),
        }:
            draft = ActionDraft()
        case _:
            raise FormatError(
                path="$",
                expected="an assistant message with a list of content blocks",
            )
    for index, block in enumerate(blocks):
        for piece in read_assistant_block(
            block,
            path=f"$.content[{index}]",
        ):
            draft.add(piece)
    return draft.build()


def read_system(
    request: JSON,
    *,
    path: str,
) -> list[tuple[str, Event]]:
    """Read a request's system prompt, a string or text blocks, into a scaffold's observation."""
    match request:
        case {"system": str(text)}:
            parts: tuple[Part, ...] = (TextPart(text=text),)
        case {"system": list(blocks)}:
            parts = tuple(
                read_text_block(
                    block,
                    path=f"{path}[{i}]",
                )
                for i, block in enumerate(blocks)
            )
        case {"system": None}:
            return []
        case {"system": _}:
            raise FormatError(
                path=path,
                expected="a string or a list of text blocks",
            )
        case _:
            return []
    return [
        (
            path,
            Observation(
                author=Author.SCAFFOLD,
                source="system",
                parts=parts,
            ),
        ),
    ]


def read_text_block(
    block: JSON,
    *,
    path: str,
) -> TextPart:
    """Read one text block of a system prompt, which holds text only."""
    match block:
        case {
            "type": "text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case _:
            raise FormatError(
                path=path,
                expected="a text block",
            )


def read_message(
    message: JSON,
    *,
    index: int,
    authorship: Authorship,
) -> list[tuple[str, Event]]:
    """Read a user, assistant or system message into its events."""
    path = f"$.messages[{index}]"
    match message:
        case {
            "role": "user",
            "content": str() | list() as content,
        }:
            author = authorship(
                index,
                message=message,
            )
            return read_user_content(
                content,
                author=author,
                path=f"{path}.content",
            )
        case {
            "role": "assistant",
            "content": str() | list() as content,
        }:
            return read_assistant_content(
                content,
                path=f"{path}.content",
            )
        case {
            "role": "system",
            "content": str() | list(),
        }:
            return [
                (
                    f"{path}.content",
                    Observation(
                        author=Author.SCAFFOLD,
                        source="system",
                        parts=read_content(
                            message,
                            path=f"{path}.content",
                        ),
                    ),
                ),
            ]
        case {"role": "user" | "assistant" | "system"}:
            raise FormatError(
                path=f"{path}.content",
                expected=_CONTENT,
            )
        case _:
            raise FormatError(
                path=path,
                expected="a user, assistant or system message with content",
            )


def read_user_content(
    content: str | list[JSON],
    *,
    author: Author,
    path: str,
) -> list[tuple[str, Event]]:
    """Read a user message's content: observations by its author, and tool results."""
    if isinstance(
        content,
        str,
    ):
        return [
            (
                path,
                Observation(
                    author=author,
                    parts=(TextPart(text=content),),
                ),
            ),
        ]
    events: list[tuple[str, Event]] = []
    parts: list[Part] = []
    for index, block in enumerate(content):
        block_path = f"{path}[{index}]"
        match block:
            case {"type": "tool_result"}:
                events.extend(
                    observe(
                        parts,
                        author=author,
                        path=path,
                    ),
                )
                parts = []
                events.append(
                    (
                        block_path,
                        read_tool_result(
                            block,
                            path=block_path,
                        ),
                    ),
                )
            case _:
                parts.extend(
                    read_content_block(
                        block,
                        path=block_path,
                    ),
                )
    return events + observe(
        parts,
        author=author,
        path=path,
    )


def observe(
    parts: list[Part],
    *,
    author: Author,
    path: str,
) -> list[tuple[str, Event]]:
    """Return the parts read so far as one observation by their author, if there are any."""
    if not parts:
        return []
    return [
        (
            path,
            Observation(
                author=author,
                parts=tuple(parts),
            ),
        ),
    ]


def read_tool_result(
    block: JSON,
    *,
    path: str,
) -> ToolObservation:
    """Read a tool_result block into the tool's observation."""
    match block:
        case {
            "tool_use_id": str(call_id),
            "is_error": bool(is_error),
        }:
            pass
        case {"tool_use_id": str(call_id)} if "is_error" not in block:
            is_error = False
        case _:
            raise FormatError(
                path=path,
                expected="a tool_result with a tool_use_id, and is_error a boolean",
            )
    parts = read_content(
        block,
        path=f"{path}.content",
    )
    return ToolObservation(
        call_id=call_id,
        is_error=is_error,
        parts=parts,
    )


def read_content(
    block: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read a tool result's or a system message's content: absent, a string, or content blocks."""
    match block:
        case {"content": str(text)}:
            return (TextPart(text=text),)
        case {"content": list(items)}:
            return tuple(
                part
                for i, item in enumerate(items)
                for part in read_content_block(
                    item,
                    path=f"{path}[{i}]",
                )
            )
        case {"content": None}:
            return ()
        case {"content": _}:
            raise FormatError(
                path=path,
                expected=_CONTENT,
            )
        case _:
            return ()


def read_content_block(
    block: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read a text, image or document block; a block of another type becomes opaque."""
    match block:
        case {
            "type": "text",
            "text": str(text),
        }:
            return (TextPart(text=text),)
        case {
            "type": "image",
            "source": dict() as source,
        }:
            return (
                read_image_source(
                    source,
                    path=f"{path}.source",
                ),
            )
        case {
            "type": "document",
            "source": dict() as source,
        }:
            return read_document(
                block,
                source=source,
                path=path,
            )
        case {"type": "text" | "image" | "document" as kind}:
            raise FormatError(
                path=path,
                expected=f"a {kind} block in the Messages format",
            )
        case {"type": str(kind)}:
            return (OpaquePart(kind=kind),)
        case _:
            raise FormatError(
                path=path,
                expected="a content block with a type",
            )


def read_image_source(
    source: JSON,
    *,
    path: str,
) -> Part:
    """Read an image's source: base64 data, or a URL, whose subtype is unknown."""
    match source:
        case {
            "type": "base64",
            "media_type": str(media_type),
            "data": str(data),
        }:
            return build_media(
                media_type,
                data=decode_data(
                    data,
                    path=f"{path}.data",
                ),
                path=path,
            )
        case {
            "type": "url",
            "url": str(url),
        }:
            return MediaPart(
                media_type="image/*",
                url=url,
            )
        case {"type": "base64" | "url"}:
            raise FormatError(
                path=path,
                expected="a base64 or URL image source",
            )
        case {"type": str()}:
            return OpaquePart(kind="image")
        case _:
            raise FormatError(
                path=path,
                expected="an image source with a type",
            )


def read_document(
    block: JSON,
    *,
    source: JSON,
    path: str,
) -> tuple[Part, ...]:
    """Read a document block: its title, its source, then its context.

    The model reads the title and the context beside the document, so each becomes text.
    """
    match block:
        case {"title": str(text)}:
            title: tuple[Part, ...] = (TextPart(text=text),)
        case {"title": text} if text is not None:
            raise FormatError(
                path=f"{path}.title",
                expected="a document title as text",
            )
        case _:
            title = ()
    match block:
        case {"context": str(text)}:
            context: tuple[Part, ...] = (TextPart(text=text),)
        case {"context": text} if text is not None:
            raise FormatError(
                path=f"{path}.context",
                expected="a document context as text",
            )
        case _:
            context = ()
    return (
        *title,
        read_document_source(
            source,
            path=f"{path}.source",
        ),
        *context,
    )


def read_document_source(
    source: JSON,
    *,
    path: str,
) -> Part:
    """Read a document's source: base64 data, a PDF's URL, or plain text."""
    match source:
        case {
            "type": "base64",
            "media_type": str(media_type),
            "data": str(data),
        }:
            return build_media(
                media_type,
                data=decode_data(
                    data,
                    path=f"{path}.data",
                ),
                path=path,
            )
        case {
            "type": "url",
            "url": str(url),
        }:
            return MediaPart(
                media_type="application/pdf",
                url=url,
            )
        case {
            "type": "text",
            "data": str(text),
        }:
            return TextPart(text=text)
        case {"type": "base64" | "url" | "text"}:
            raise FormatError(
                path=path,
                expected="a base64, URL or text document source",
            )
        case {"type": str()}:
            return OpaquePart(kind="document")
        case _:
            raise FormatError(
                path=path,
                expected="a document source with a type",
            )


def read_assistant_content(
    content: str | list[JSON],
    *,
    path: str,
) -> list[tuple[str, Event]]:
    """Read an assistant message's content into actions, split by server tools' results.

    Each action's id is the JSONPath of its first block.
    """
    if isinstance(
        content,
        str,
    ):
        return [
            (
                path,
                Action(
                    id=path,
                    parts=(TextPart(text=content),),
                ),
            ),
        ]
    blocks = content
    events: list[tuple[str, Event]] = []
    draft: ActionDraft | None = None
    for index, block in enumerate(blocks):
        block_path = f"{path}[{index}]"
        match block:
            case {
                "type": str(kind),
                "tool_use_id": str(call_id),
            } if kind.endswith("_tool_result"):
                events.extend(close(draft))
                draft = None
                events.append(
                    (
                        block_path,
                        read_server_result(
                            block,
                            call_id=call_id,
                            path=block_path,
                        ),
                    ),
                )
            case _:
                draft = draft or ActionDraft(id=block_path)
                for piece in read_assistant_block(
                    block,
                    path=block_path,
                ):
                    draft.add(piece)
    return events + close(draft)


def read_server_result(
    block: JSON,
    *,
    call_id: str,
    path: str,
) -> ToolObservation:
    """Read a server tool's result: its content, a list or a typed object, as JSON text.

    A server tool reports a failure as an error object in the content, whose type ends in
    _error, such as web_search_tool_result_error.
    """
    match block:
        case {"content": {"type": str(kind)} as content}:
            is_error = kind.endswith("_error")
        case {"content": list() as content}:
            is_error = False
        case _:
            raise FormatError(
                path=path,
                expected="a server tool's result whose content is a list or a typed object",
            )
    text = json.dumps(
        content,
        ensure_ascii=False,
    )
    return ToolObservation(
        call_id=call_id,
        source="server",
        is_error=is_error,
        parts=(TextPart(text=text),),
    )


def read_assistant_block(
    block: JSON,
    *,
    path: str,
) -> tuple[Thought | Part | ToolCall, ...]:
    """Read an assistant's block: a thought, a tool call, or a content block's parts."""
    match block:
        case {
            "type": "thinking",
            "thinking": str(text),
        }:
            return (Thought(TextPart(text=text)),)
        case {"type": "redacted_thinking"}:
            return (Thought(OpaquePart(kind="redacted_thinking")),)
        case {
            "type": "tool_use" | "server_tool_use",
            "id": str(call_id),
            "name": str(name),
            "input": dict(arguments),
        }:
            return (
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=json.dumps(
                        arguments,
                        ensure_ascii=False,
                    ),
                ),
            )
        case {"type": "thinking" | "tool_use" | "server_tool_use" as kind}:
            raise FormatError(
                path=path,
                expected=f"a {kind} block in the Messages format",
            )
        case _:
            return read_content_block(
                block,
                path=path,
            )
