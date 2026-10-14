"""The content of actions and observations, and who wrote it."""

from dataclasses import dataclass
from enum import StrEnum


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


type Part = TextPart | MediaPart
