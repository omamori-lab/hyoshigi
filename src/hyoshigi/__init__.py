"""Monitors that rate each action an AI agent proposes against its trajectory."""

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.formats.authorship import Authorship, by_role, marked
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Description, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.trajectory import BlackBoxRun, Trajectory

__all__ = [
    "JSON",
    "Action",
    "Author",
    "Authorship",
    "BlackBoxRun",
    "ConfigurationError",
    "Description",
    "Event",
    "FormatError",
    "MediaPart",
    "Observation",
    "OpaquePart",
    "Part",
    "TextPart",
    "ToolCall",
    "ToolObservation",
    "Trajectory",
    "by_role",
    "marked",
]

__version__ = "0.1.0.dev0"
