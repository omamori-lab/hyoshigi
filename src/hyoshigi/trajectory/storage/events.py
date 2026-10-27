"""How an event, its tool calls and its parts are written to and read from JSON-compatible data."""

from base64 import b64decode, b64encode
from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Description, MediaPart, OpaquePart, Part, TextPart

_AUTHORS = ", ".join(Author)
_MEDIA_PART = "a media part with an IANA media type and exactly one of url and data"


def dump_event(event: Event) -> JSON:
    """Return an action or an observation as JSON-compatible data."""
    match event:
        case Action():
            return {
                "kind": "action",
                "id": event.id,
                "thought": dump_parts(event.thought),
                "parts": dump_parts(event.parts),
                "tool_calls": [dump_tool_call(call) for call in event.tool_calls],
            }
        case ToolObservation():
            return {
                "kind": "tool_observation",
                "call_id": event.call_id,
                "is_error": event.is_error,
                "author": event.author.value,
                "source": event.source,
                "parts": dump_parts(event.parts),
            }
        case Observation():
            return {
                "kind": "observation",
                "author": event.author.value,
                "source": event.source,
                "parts": dump_parts(event.parts),
            }
        case unreachable:
            assert_never(unreachable)


def dump_tool_call(call: ToolCall) -> JSON:
    """Return a tool call as JSON-compatible data."""
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    }


def dump_parts(parts: tuple[Part, ...]) -> list[JSON]:
    """Return parts as JSON-compatible data, media bytes in base64."""
    return [dump_part(part) for part in parts]


def dump_part(part: Part) -> JSON:
    """Return a part as JSON-compatible data, media bytes in base64."""
    match part:
        case TextPart():
            return {
                "kind": "text",
                "text": part.text,
            }
        case MediaPart():
            return {
                "kind": "media",
                "media_type": part.media_type,
                "url": part.url,
                "data": None if part.data is None else b64encode(part.data).decode("ascii"),
                "description": dump_description(part.description),
            }
        case OpaquePart():
            return {
                "kind": "opaque",
                "opaque_kind": part.kind,
            }
        case unreachable:
            assert_never(unreachable)


def dump_description(description: Description | None) -> JSON:
    """Return a media part's description as JSON-compatible data."""
    if description is None:
        return None
    return {
        "text": description.text,
        "author": description.author.value,
    }


def load_event(
    data: JSON,
    *,
    path: str,
) -> Event:
    """Read an action or an observation that dump_event wrote."""
    match data:
        case {
            "kind": "action",
            "id": str(action_id),
            "thought": list(thought),
            "parts": list(parts),
            "tool_calls": list(calls),
        }:
            return Action(
                id=action_id,
                thought=load_parts(
                    thought,
                    path=f"{path}.thought",
                ),
                parts=load_parts(
                    parts,
                    path=f"{path}.parts",
                ),
                tool_calls=tuple(
                    load_tool_call(
                        call,
                        path=f"{path}.tool_calls[{index}]",
                    )
                    for index, call in enumerate(calls)
                ),
            )
        case {
            "kind": "tool_observation",
            "call_id": str(call_id),
            "is_error": bool(is_error),
            "author": str(author),
            "source": str() | None as source,
            "parts": list(parts),
        }:
            return ToolObservation(
                call_id=call_id,
                is_error=is_error,
                author=load_author(
                    author,
                    path=f"{path}.author",
                ),
                source=source,
                parts=load_parts(
                    parts,
                    path=f"{path}.parts",
                ),
            )
        case {
            "kind": "observation",
            "author": str(author),
            "source": str() | None as source,
            "parts": list(parts),
        }:
            return Observation(
                author=load_author(
                    author,
                    path=f"{path}.author",
                ),
                source=source,
                parts=load_parts(
                    parts,
                    path=f"{path}.parts",
                ),
            )
        case _:
            raise FormatError(
                path=path,
                expected="an action, an observation or a tool observation",
            )


def load_tool_call(
    data: JSON,
    *,
    path: str,
) -> ToolCall:
    """Read a tool call that dump_tool_call wrote."""
    match data:
        case {
            "id": str(call_id),
            "name": str(name),
            "arguments": str(arguments),
        }:
            return ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
        case _:
            raise FormatError(
                path=path,
                expected="a tool call with an id, a name and arguments as text",
            )


def load_parts(
    data: list[JSON],
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read the parts that dump_parts wrote."""
    return tuple(
        load_part(
            part,
            path=f"{path}[{index}]",
        )
        for index, part in enumerate(data)
    )


def load_part(
    data: JSON,
    *,
    path: str,
) -> Part:
    """Read a part that dump_part wrote."""
    match data:
        case {
            "kind": "text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case {
            "kind": "opaque",
            "opaque_kind": str(kind),
        }:
            return OpaquePart(kind=kind)
        case {
            "kind": "media",
            "media_type": str(media_type),
            "url": str() | None as url,
            "data": str() | None as encoded,
            "description": None | dict() as description,
        }:
            try:
                return MediaPart(
                    media_type=media_type,
                    url=url,
                    data=decode_data(
                        encoded,
                        path=f"{path}.data",
                    ),
                    description=load_description(
                        description,
                        path=f"{path}.description",
                    ),
                )
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected=_MEDIA_PART,
                ) from None
        case _:
            raise FormatError(
                path=path,
                expected="a text, media or opaque part",
            )


def decode_data(
    encoded: str | None,
    *,
    path: str,
) -> bytes | None:
    """Read media bytes that dump_part wrote in base64."""
    if encoded is None:
        return None
    try:
        return b64decode(
            encoded,
            validate=True,
        )
    except ValueError:
        raise FormatError(
            path=path,
            expected="base64 data",
        ) from None


def load_description(
    data: JSON,
    *,
    path: str,
) -> Description | None:
    """Read a media part's description that dump_description wrote."""
    match data:
        case None:
            return None
        case {
            "text": str(text),
            "author": str(author),
        }:
            return Description(
                text=text,
                author=load_author(
                    author,
                    path=f"{path}.author",
                ),
            )
        case _:
            raise FormatError(
                path=path,
                expected="a description with a text and an author, or null",
            )


def load_author(
    value: str,
    *,
    path: str,
) -> Author:
    """Read an author that dump_event or dump_description wrote."""
    try:
        return Author(value)
    except ValueError:
        raise FormatError(
            path=path,
            expected=f"an author: {_AUTHORS}",
        ) from None
