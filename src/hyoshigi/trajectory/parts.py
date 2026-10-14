"""The content of actions and observations, and who wrote it."""

import re
from dataclasses import dataclass
from enum import StrEnum

from hyoshigi.errors import ConfigurationError

# RFC 6838, section 4.2: a type and a subtype, each a letter or digit followed by up
# to 126 letters, digits or any of ! # $ & - ^ _ . +
_RESTRICTED_NAME = r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}"
_MEDIA_TYPE = re.compile(f"{_RESTRICTED_NAME}/{_RESTRICTED_NAME}")


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


@dataclass(frozen=True, slots=True, kw_only=True)
class TextPart:
    """A piece of text."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Description:
    """A text account of a media part, and who wrote it."""

    text: str
    author: Author


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaPart:
    """An image, audio, video or file, by URL or as bytes, with its IANA media type."""

    media_type: str
    url: str | None = None
    data: bytes | None = None
    description: Description | None = None

    def __post_init__(self) -> None:
        if _MEDIA_TYPE.fullmatch(self.media_type) is None:
            raise ConfigurationError("media_type must be an IANA media type, such as image/png")
        if (self.url is None) == (self.data is None):
            raise ConfigurationError("a media part takes exactly one of url and data")


type Part = TextPart | MediaPart
