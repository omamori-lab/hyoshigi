"""What a trajectory records: the agent's actions and the observations that reach it."""

# Action.from_anthropic names Action in its own signature.
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from hyoshigi.options import check_instance
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Part, TextPart, check_parts


def _create_action_id() -> str:
    return uuid4().hex


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ToolCall:
    """A tool the agent asks to run, with its arguments as the JSON text the model emitted.

    The id, the name and the arguments are str; anything else raises ConfigurationError.
    """

    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        check_instance(
            self.id,
            expected=str,
            name="id",
        )
        check_instance(
            self.name,
            expected=str,
            name="name",
        )
        check_instance(
            self.arguments,
            expected=str,
            name="arguments",
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Action:
    """What the agent emits in one step: its thought, what it says and the tools it calls.

    A thought is an action in language space that changes nothing outside the agent
    (Yao et al., "ReAct", ICLR 2023, section 2). The id is a str, the thought and the parts
    are each a tuple of parts, and the tool calls a tuple of ToolCall; anything else raises
    ConfigurationError.
    """

    id: str = field(default_factory=_create_action_id)
    thought: tuple[Part, ...] = ()
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        check_instance(
            self.id,
            expected=str,
            name="id",
        )
        check_parts(
            self.thought,
            name="thought",
        )
        check_parts(
            self.parts,
            name="parts",
        )
        check_instance(
            self.tool_calls,
            expected=tuple,
            name="tool_calls",
        )
        for index, call in enumerate(self.tool_calls):
            check_instance(
                call,
                expected=ToolCall,
                name=f"tool_calls[{index}]",
            )

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

        OpenRouter's reasoning fields are read as the thought, and its images as media parts.
        Raises FormatError, naming the JSONPath of the first problem, on a message that is
        not in the format.
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
    """Content that reaches the agent, with who wrote it and where it came from.

    The author is an Author, the parts a tuple of parts and the source a str or None;
    anything else raises ConfigurationError.
    """

    author: Author
    parts: tuple[Part, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        check_instance(
            self.author,
            expected=Author,
            name="author",
        )
        check_parts(
            self.parts,
            name="parts",
        )
        if self.source is not None:
            check_instance(
                self.source,
                expected=str,
                name="source",
            )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ToolObservation(Observation):
    """A result answering one of the agent's tool calls, written by the tool unless another
    author is given.

    The author, the parts and the source are those of an Observation, the call id is a str
    and is_error a bool; anything else raises ConfigurationError.
    """

    call_id: str
    is_error: bool = False
    author: Author = Author.TOOL

    def __post_init__(self) -> None:
        # A slotted dataclass is a new class, so super() without arguments fails on Python 3.12.
        Observation.__post_init__(self)
        check_instance(
            self.call_id,
            expected=str,
            name="call_id",
        )
        check_instance(
            self.is_error,
            expected=bool,
            name="is_error",
        )


# OpenHands calls actions and observations events (Wang et al., "OpenHands", ICLR 2025,
# section 2.1).
type Event = Action | Observation


def build_feedback(
    action: Action,
    *,
    reason: str,
) -> list[Observation]:
    """The monitor's reason for blocking an action: an error answering each of its tool calls, in
    order, or one observation when it made none.
    """
    parts = (TextPart(text=reason),)
    return [
        ToolObservation(
            call_id=call.id,
            is_error=True,
            author=Author.MONITOR,
            parts=parts,
        )
        for call in action.tool_calls
    ] or [
        Observation(
            author=Author.MONITOR,
            parts=parts,
        ),
    ]
