"""What a trajectory records: the agent's actions and the observations that reach it."""

# Action.from_anthropic names Action in its own signature.
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Part


def _create_action_id() -> str:
    return uuid4().hex


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ToolCall:
    """A tool the agent asks to run, with its arguments as the JSON text the model emitted."""

    id: str
    name: str
    arguments: str


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Action:
    """What the agent emits in one step: its thought, what it says and the tools it calls.

    A thought is an action in language space that changes nothing outside the agent
    (Yao et al., "ReAct", ICLR 2023, section 2).
    """

    id: str = field(default_factory=_create_action_id)
    thought: tuple[Part, ...] = ()
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()

    @staticmethod
    def from_anthropic(message: JSON) -> Action:
        """Read the assistant message of an Anthropic Messages response into an action.

        Raises FormatError, naming the JSONPath of the first problem, on a message that
        is not in the Messages format.
        """
        # Imported here because the format module builds actions: a top-level import
        # would be a cycle.
        from hyoshigi.trajectory.formats import anthropic  # lanorme: ignore[IMPORT-001]

        return anthropic.read_response(message)

    @staticmethod
    def from_openai_chat(message: JSON) -> Action:
        """Read a chat completions response's assistant message into an action.

        OpenRouter's reasoning fields are read as the thought. Raises FormatError, naming
        the JSONPath of the first problem, on a message that is not in the format.
        """
        # Imported here because the format module builds actions: a top-level import
        # would be a cycle.
        from hyoshigi.trajectory.formats import openai_chat  # lanorme: ignore[IMPORT-001]

        return openai_chat.read_response(message)

    @staticmethod
    def from_open_responses(output: JSON) -> Action:
        """Read an Open Responses response's output items into one action.

        An item of a type the model did not write, such as a tool the provider ran, becomes
        an opaque part. Raises FormatError, naming the JSONPath of the first problem, on
        output that is not in the format, or that holds a message the assistant did not
        write.
        """
        # Imported here because the format module builds actions: a top-level import
        # would be a cycle.
        from hyoshigi.trajectory.formats import open_responses  # lanorme: ignore[IMPORT-001]

        return open_responses.read_response(output)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Observation:
    """Content that reaches the agent, with who wrote it and where it came from."""

    author: Author
    parts: tuple[Part, ...]
    source: str | None = None


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ToolObservation(Observation):
    """A tool's result, answering one of the agent's tool calls."""

    call_id: str
    is_error: bool = False
    author: Author = field(
        default=Author.TOOL,
        init=False,
    )


# OpenHands calls actions and observations events (Wang et al., "OpenHands", ICLR 2025,
# section 2.1).
type Event = Action | Observation
