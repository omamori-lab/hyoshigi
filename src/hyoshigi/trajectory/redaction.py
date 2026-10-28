"""What a trajectory's content becomes before it is kept, read by a monitor's model, stored or
reported.
"""

from collections.abc import Callable
from dataclasses import dataclass

from hyoshigi.trajectory.parts import MediaPart, Part


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Redaction:
    """Maps a trajectory's content to what is kept or sent in its place."""

    text: Callable[[str], str]
    media: Callable[[MediaPart], Part]
