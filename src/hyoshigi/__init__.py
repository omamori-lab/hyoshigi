"""Monitors that rate each action an AI agent proposes against its trajectory."""

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.models.model import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    PromptPart,
    Question,
    Refused,
    Truncated,
    Unavailable,
)
from hyoshigi.trajectory.formats.authorship import Authorship, by_role, marked
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Description, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.trajectory import BlackBoxRun, Trajectory

__all__ = [
    "JSON",
    "Action",
    "AsyncDecisionModel",
    "AsyncGenerativeModel",
    "Author",
    "Authorship",
    "BlackBoxRun",
    "ConfigurationError",
    "Decision",
    "DecisionModel",
    "DecisionOutcome",
    "Description",
    "Event",
    "FormatError",
    "Generation",
    "GenerationOutcome",
    "GenerativeModel",
    "MediaPart",
    "MediaTypes",
    "Observation",
    "OpaquePart",
    "Part",
    "Prompt",
    "PromptPart",
    "Question",
    "Refused",
    "TextPart",
    "ToolCall",
    "ToolObservation",
    "Trajectory",
    "Truncated",
    "Unavailable",
    "by_role",
    "marked",
]

__version__ = "0.1.0.dev0"
