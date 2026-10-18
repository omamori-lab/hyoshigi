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
from hyoshigi.monitors.composition import (
    AsyncCalibratedMonitor,
    AsyncCascadeMonitor,
    AsyncRepeatedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)
from hyoshigi.monitors.decision import (
    ATTACK_CRITERIA,
    DEFAULT_STATEMENT,
    Aggregation,
    AsyncDecisionMonitor,
    DecisionMonitor,
)
from hyoshigi.monitors.llm import AsyncLLMMonitor, LLMMonitor
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.prompts import DEFAULT_PROMPT, MonitorPrompt
from hyoshigi.monitors.score import (
    Cause,
    Composite,
    Decisions,
    Evidence,
    FailedClosed,
    Rationale,
    StatementDecision,
    SuspicionScore,
    Unperceivable,
    Unreadable,
)
from hyoshigi.monitors.view import Channel, View
from hyoshigi.trajectory.formats.authorship import Authorship, by_role, marked
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Description, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.trajectory import BlackBoxRun, Rating, Trajectory

__all__ = [
    "ATTACK_CRITERIA",
    "DEFAULT_PROMPT",
    "DEFAULT_STATEMENT",
    "JSON",
    "Action",
    "Aggregation",
    "AsyncCalibratedMonitor",
    "AsyncCascadeMonitor",
    "AsyncDecisionModel",
    "AsyncDecisionMonitor",
    "AsyncGenerativeModel",
    "AsyncLLMMonitor",
    "AsyncMonitor",
    "AsyncRepeatedMonitor",
    "Author",
    "Authorship",
    "BlackBoxRun",
    "CalibratedMonitor",
    "CascadeMonitor",
    "Cause",
    "Channel",
    "Composite",
    "ConfigurationError",
    "Decision",
    "DecisionModel",
    "DecisionMonitor",
    "DecisionOutcome",
    "Decisions",
    "Description",
    "Event",
    "Evidence",
    "FailedClosed",
    "FormatError",
    "Generation",
    "GenerationOutcome",
    "GenerativeModel",
    "LLMMonitor",
    "MediaPart",
    "MediaTypes",
    "Monitor",
    "MonitorPrompt",
    "Observation",
    "OpaquePart",
    "Part",
    "Prompt",
    "PromptPart",
    "Question",
    "Rating",
    "Rationale",
    "Refused",
    "RepeatedMonitor",
    "StatementDecision",
    "SuspicionScore",
    "TextPart",
    "ToolCall",
    "ToolObservation",
    "Trajectory",
    "Truncated",
    "Unavailable",
    "Unperceivable",
    "Unreadable",
    "View",
    "by_role",
    "marked",
]

__version__ = "0.1.0.dev0"
