"""The OpenAI chat completions format, with OpenRouter's reasoning fields and images.

The format is OpenAI's OpenAPI specification, https://github.com/openai/openai-openapi,
and OpenRouter's reasoning tokens guide,
https://openrouter.ai/docs/use-cases/reasoning-tokens.
"""

from dataclasses import replace

from hyoshigi.errors import FormatError
from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.formats.media import read_data_url
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.storage import decode_data

# input_audio's formats as media types; audio/mpeg is RFC 3003's.
_AUDIO_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
}
_SCAFFOLD_ROLES = ("system", "developer")


def read_request(
    request: JSON,
    *,
    authorship: Authorship,
) -> list[tuple[str, Event]]:
    """Read a chat completions request body into its events, each with its JSONPath."""
    match request:
        case {"messages": list(messages)}:
            pass
        case _:
            raise FormatError(
                path="$",
                expected="a request body with a list of messages",
            )
    return [
        read_message(
            message,
            index=index,
            authorship=authorship,
        )
        for index, message in enumerate(messages)
    ]


def read_response(message: JSON) -> Action:
    """Read a response's assistant message into the action to rate, with a generated id."""
    match message:
        case {"role": "assistant"}:
            return read_assistant(
                message,
                path="$",
            )
        case _:
            raise FormatError(
                path="$",
                expected="an assistant message",
            )


def read_message(
    message: JSON,
    *,
    index: int,
    authorship: Authorship,
) -> tuple[str, Event]:
    """Read one message into an event: the assistant's an action, any other an observation."""
    path = f"$.messages[{index}]"
    match message:
        case {"role": "assistant"}:
            return (
                path,
                replace(
                    read_assistant(
                        message,
                        path=path,
                    ),
                    id=path,
                ),
            )
        case {
            "role": "user",
            "content": str() | list() as content,
        }:
            author = authorship(
                index,
                message=message,
            )
            parts = read_user_content(
                content,
                path=f"{path}.content",
            )
            return (
                path,
                Observation(
                    author=author,
                    parts=parts,
                ),
            )
        case _:
            return (
                path,
                read_other_message(
                    message,
                    path=path,
                ),
            )


def read_other_message(
    message: JSON,
    *,
    path: str,
) -> Observation:
    """Read a system, developer, tool or function message into an observation."""
    match message:
        case {
            "role": str(role),
            "content": str() | list() as content,
        } if role in _SCAFFOLD_ROLES:
            parts = read_text_content(
                content,
                path=f"{path}.content",
            )
            return Observation(
                author=Author.SCAFFOLD,
                source=role,
                parts=parts,
            )
        case {
            "role": "tool",
            "tool_call_id": str(call_id),
            "content": str() | list() as content,
        }:
            return ToolObservation(
                call_id=call_id,
                parts=read_text_content(
                    content,
                    path=f"{path}.content",
                ),
            )
        case {
            "role": "function",
            "name": str(name),
            "content": str(text),
        }:
            return Observation(
                author=Author.TOOL,
                source=name,
                parts=(TextPart(text=text),),
            )
        case {
            "role": "function",
            "name": str(name),
            "content": None,
        }:
            return Observation(
                author=Author.TOOL,
                source=name,
                parts=(),
            )
        case _:
            raise FormatError(
                path=path,
                expected="a system, developer, user, assistant, tool or function message",
            )


def read_text_content(
    content: str | list[JSON],
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read content that holds text only: a string, or text parts."""
    if isinstance(
        content,
        str,
    ):
        return (TextPart(text=content),)
    return tuple(
        read_text_part(
            part,
            path=f"{path}[{i}]",
        )
        for i, part in enumerate(content)
    )


def read_text_part(
    part: JSON,
    *,
    path: str,
) -> TextPart:
    """Read one text part of content that may hold text only."""
    match part:
        case {
            "type": "text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case _:
            raise FormatError(
                path=path,
                expected="a text part",
            )


def read_user_content(
    content: str | list[JSON],
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read a user message's content: a string, or text, image, audio and file parts."""
    if isinstance(
        content,
        str,
    ):
        return (TextPart(text=content),)
    return tuple(
        read_user_part(
            part,
            path=f"{path}[{i}]",
        )
        for i, part in enumerate(content)
    )


def read_user_part(
    part: JSON,
    *,
    path: str,
) -> Part:
    """Read a user's content part; a part of an unknown type becomes opaque."""
    match part:
        case {
            "type": "text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case {"type": "image_url" | "input_audio" | "file"}:
            return read_media_part(
                part,
                path=path,
            )
        case {"type": "text"}:
            raise FormatError(
                path=path,
                expected="a text part with text",
            )
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(
                path=path,
                expected="a content part with a type",
            )


def read_media_part(
    part: JSON,
    *,
    path: str,
) -> Part:
    """Read an image, audio or file part; a file by id becomes opaque."""
    match part:
        case {
            "type": "image_url",
            "image_url": {"url": str(url)},
        }:
            return read_image_url(
                url,
                path=f"{path}.image_url.url",
            )
        case {
            "type": "input_audio",
            "input_audio": {
                "data": str(data),
                "format": str(audio_format),
            },
        } if audio_format in _AUDIO_TYPES:
            audio = decode_data(
                data,
                path=f"{path}.input_audio.data",
            )
            return MediaPart(
                media_type=_AUDIO_TYPES[audio_format],
                data=audio,
            )
        case {"type": "file"}:
            return read_file_part(
                part,
                path=path,
            )
        case _:
            raise FormatError(
                path=path,
                expected="an image_url or input_audio part",
            )


def read_image_url(
    url: str,
    *,
    path: str,
) -> MediaPart:
    """Read an image's URL: a data URL into media of its own type, another URL as an image."""
    return read_data_url(
        url,
        path=path,
    ) or MediaPart(
        media_type="image/*",
        url=url,
    )


def read_file_part(
    part: JSON,
    *,
    path: str,
) -> Part:
    """Read a file part: file data in a data URL becomes media of the URL's type, text/plain
    when it names none; a file by id, or file data that is not a data URL, becomes opaque.
    """
    match part:
        case {"file": {"file_data": str(file_data)}}:
            data_path = f"{path}.file.file_data"
            return read_data_url(
                file_data,
                path=data_path,
            ) or OpaquePart(kind="file")
        case {"file": dict()}:
            return OpaquePart(kind="file")
        case _:
            raise FormatError(
                path=path,
                expected="a file part with file data or a file id",
            )


def read_assistant(
    message: JSON,
    *,
    path: str,
) -> Action:
    """Read an assistant message: reasoning into the thought, content into parts, tool calls."""
    calls = read_tool_calls(
        message,
        path=path,
    )
    return Action(
        thought=read_reasoning(
            message,
            path=path,
        ),
        parts=read_assistant_parts(
            message,
            path=path,
        )
        + tuple(
            call
            for call in calls
            if isinstance(
                call,
                OpaquePart,
            )
        ),
        tool_calls=tuple(
            call
            for call in calls
            if isinstance(
                call,
                ToolCall,
            )
        ),
    )


def read_reasoning(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read OpenRouter's reasoning_details, or its reasoning text when it has no details, into
    the thought.
    """
    match message:
        case {"reasoning_details": list(details)} if details:
            return tuple(
                read_reasoning_detail(
                    detail,
                    path=f"{path}.reasoning_details[{i}]",
                )
                for i, detail in enumerate(details)
            )
        case {"reasoning_details": details} if not isinstance(
            details,
            list | None,
        ):
            raise FormatError(
                path=f"{path}.reasoning_details",
                expected="a list of reasoning details",
            )
        case {"reasoning": str(text)}:
            return (TextPart(text=text),)
        case {"reasoning": text} if text is not None:
            raise FormatError(
                path=f"{path}.reasoning",
                expected="reasoning as text",
            )
        case _:
            return ()


def read_reasoning_detail(
    detail: JSON,
    *,
    path: str,
) -> Part:
    """Read one reasoning detail: text or a summary; an encrypted one becomes opaque."""
    match detail:
        case {
            "type": "reasoning.text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case {
            "type": "reasoning.summary",
            "summary": str(summary),
        }:
            return TextPart(text=summary)
        case {"type": "reasoning.text" | "reasoning.summary"}:
            raise FormatError(
                path=path,
                expected="a reasoning detail with its text or summary",
            )
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(
                path=path,
                expected="a reasoning detail with a type",
            )


def read_assistant_parts(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read what an assistant message says: its content, its refusal, its audio and its
    images.
    """
    return (
        read_assistant_content(
            message,
            path=f"{path}.content",
        )
        + read_refusal(
            message,
            path=f"{path}.refusal",
        )
        + read_audio(
            message,
            path=f"{path}.audio",
        )
        + read_images(
            message,
            path=f"{path}.images",
        )
    )


def read_assistant_content(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read an assistant message's content: none, a string, or parts.

    Text and refusal parts become text; parts of other types become opaque.
    """
    match message:
        case {"content": str(text)}:
            return (TextPart(text=text),)
        case {"content": list(parts)}:
            return tuple(
                read_assistant_part(
                    part,
                    path=f"{path}[{i}]",
                )
                for i, part in enumerate(parts)
            )
        case {"content": content} if content is not None:
            raise FormatError(
                path=path,
                expected="content as a string or a list of parts",
            )
        case _:
            return ()


def read_assistant_part(
    part: JSON,
    *,
    path: str,
) -> Part:
    """Read an assistant's content part: text or a refusal; another type becomes opaque."""
    match part:
        case {
            "type": "text",
            "text": str(text),
        }:
            return TextPart(text=text)
        case {
            "type": "refusal",
            "refusal": str(text),
        }:
            return TextPart(text=text)
        case {"type": "text" | "refusal"}:
            raise FormatError(
                path=path,
                expected="a text or refusal part with its text",
            )
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(
                path=path,
                expected="a content part with a type",
            )


def read_refusal(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read an assistant message's refusal, if it has one."""
    match message:
        case {"refusal": str(text)}:
            return (TextPart(text=text),)
        case {"refusal": text} if text is not None:
            raise FormatError(
                path=path,
                expected="a refusal as text",
            )
        case _:
            return ()


def read_audio(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read an assistant message's audio, which the message names by id only, as opaque."""
    match message:
        case {"audio": dict()}:
            return (OpaquePart(kind="audio"),)
        case {"audio": audio} if audio is not None:
            raise FormatError(
                path=path,
                expected="audio as an object",
            )
        case _:
            return ()


def read_images(
    message: JSON,
    *,
    path: str,
) -> tuple[Part, ...]:
    """Read the images an assistant message carries, if it has any: OpenRouter's image
    models return each as an image_url with its URL (the OpenRouter SDK's ChatAssistantMessage).
    """
    match message:
        case {"images": list(images)}:
            return tuple(
                read_image(
                    image,
                    path=f"{path}[{i}]",
                )
                for i, image in enumerate(images)
            )
        case {"images": images} if images is not None:
            raise FormatError(
                path=path,
                expected="a list of images",
            )
        case _:
            return ()


def read_image(
    image: JSON,
    *,
    path: str,
) -> MediaPart:
    """Read one image an assistant message carries."""
    match image:
        case {"image_url": {"url": str(url)}}:
            return read_image_url(
                url,
                path=f"{path}.image_url.url",
            )
        case _:
            raise FormatError(
                path=path,
                expected="an image with an image_url and its url",
            )


def read_tool_calls(
    message: JSON,
    *,
    path: str,
) -> tuple[ToolCall | OpaquePart, ...]:
    """Read an assistant message's tool calls and its deprecated function call."""
    match message:
        case {"tool_calls": list(calls)}:
            read = tuple(
                read_tool_call(
                    call,
                    path=f"{path}.tool_calls[{i}]",
                )
                for i, call in enumerate(calls)
            )
        case {"tool_calls": calls} if calls is not None:
            raise FormatError(
                path=f"{path}.tool_calls",
                expected="a list of tool calls",
            )
        case _:
            read = ()
    return read + read_function_call(
        message,
        path=f"{path}.function_call",
    )


def read_tool_call(
    call: JSON,
    *,
    path: str,
) -> ToolCall | OpaquePart:
    """Read a function or custom tool call; a call of another type becomes opaque."""
    match call:
        case {
            "id": str(call_id),
            "type": "function",
            "function": {
                "name": str(name),
                "arguments": str(arguments),
            },
        }:
            return ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
        case {
            "id": str(call_id),
            "type": "custom",
            "custom": {
                "name": str(name),
                "input": str(text),
            },
        }:
            return ToolCall(
                id=call_id,
                name=name,
                arguments=text,
            )
        case {"type": "function" | "custom"}:
            raise FormatError(
                path=path,
                expected="a tool call with an id, a name and its arguments",
            )
        case {"type": str(kind)}:
            return OpaquePart(kind=kind)
        case _:
            raise FormatError(
                path=path,
                expected="a tool call with a type",
            )


def read_function_call(
    message: JSON,
    *,
    path: str,
) -> tuple[ToolCall, ...]:
    """Read the deprecated function_call, which has no id, as a tool call named by its path."""
    match message:
        case {
            "function_call": {
                "name": str(name),
                "arguments": str(arguments),
            },
        }:
            return (
                ToolCall(
                    id=path,
                    name=name,
                    arguments=arguments,
                ),
            )
        case {"function_call": call} if call is not None:
            raise FormatError(
                path=path,
                expected="a function call with a name and its arguments",
            )
        case _:
            return ()
