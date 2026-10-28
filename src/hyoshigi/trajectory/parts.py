"""The content of actions and observations, and who wrote it."""

import re
from dataclasses import dataclass
from enum import StrEnum

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import check_instance, describe_type

# RFC 6838, section 4.2: a type and a subtype, each a letter or digit followed by up
# to 126 letters, digits or any of ! # $ & - ^ _ . + ; and RFC 9110, section 12.5.1:
# a media range, whose subtype is *, when only the top-level type is known.
_RESTRICTED_NAME = r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}"
MEDIA_TYPE = re.compile(f"{_RESTRICTED_NAME}/(?:{_RESTRICTED_NAME}|\\*)")


class Author(StrEnum):
    """Who wrote a piece of content in a trajectory.

    Only USER, or DELEGATOR inside a subagent's trajectory, authorises what the
    agent does.
    """

    USER = "user"
    DELEGATOR = "delegator"
    AGENT = "agent"
    TOOL = "tool"
    SCAFFOLD = "scaffold"
    MONITOR = "monitor"
    UNKNOWN = "unknown"


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class TextPart:
    """A piece of text, a str; anything else raises ConfigurationError."""

    text: str

    def __post_init__(self) -> None:
        check_instance(
            self.text,
            expected=str,
            name="text",
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Description:
    """A text account of a media part, and who wrote it.

    The text is a str and the author an Author; anything else raises ConfigurationError.
    """

    text: str
    author: Author

    def __post_init__(self) -> None:
        check_instance(
            self.text,
            expected=str,
            name="text",
        )
        check_instance(
            self.author,
            expected=Author,
            name="author",
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class MediaPart:
    """An image, audio, video or file, by URL or as bytes, with its IANA media type.

    The media type may be a range such as image/* when only the top-level type is known.
    The media type is a str, the url a str, the data bytes and the description a Description
    or None, and exactly one of url and data is given; anything else raises
    ConfigurationError.
    """

    media_type: str
    url: str | None = None
    data: bytes | None = None
    description: Description | None = None

    def __post_init__(self) -> None:
        check_instance(
            self.media_type,
            expected=str,
            name="media_type",
        )
        if self.url is not None:
            check_instance(
                self.url,
                expected=str,
                name="url",
            )
        if self.data is not None:
            check_instance(
                self.data,
                expected=bytes,
                name="data",
            )
        if self.description is not None:
            check_instance(
                self.description,
                expected=Description,
                name="description",
            )
        if MEDIA_TYPE.fullmatch(self.media_type) is None:
            raise ConfigurationError("media_type must be an IANA media type, such as image/png")
        if (self.url is None) == (self.data is None):
            raise ConfigurationError("a media part takes exactly one of url and data")


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class OpaquePart:
    """Content recorded but not read, named by its kind in the provider's format.

    A monitor sees a placeholder for it in the history; in the action it rates, the
    score fails closed. The kind is a str; anything else raises ConfigurationError.
    """

    kind: str

    def __post_init__(self) -> None:
        check_instance(
            self.kind,
            expected=str,
            name="kind",
        )


type Part = TextPart | MediaPart | OpaquePart


def read_text(part: MediaPart) -> str | None:
    """Return text media's bytes as UTF-8 text, or None for media of another type, media given
    by URL and bytes that are not UTF-8.
    """
    if not part.media_type.lower().startswith("text/") or part.data is None:
        return None
    try:
        return part.data.decode()
    except UnicodeDecodeError:
        return None


def check_parts(
    parts: tuple[Part, ...],
    *,
    name: str,
) -> None:
    """Raise ConfigurationError, naming the field, unless the parts are a tuple of TextPart,
    MediaPart and OpaquePart.
    """
    check_instance(
        parts,
        expected=tuple,
        name=name,
    )
    for index, part in enumerate(parts):
        if not isinstance(
            part,
            TextPart | MediaPart | OpaquePart,
        ):
            raise ConfigurationError(
                f"{name}[{index}] must be a TextPart, MediaPart or OpaquePart, "
                f"got {describe_type(part)}",
            )
