"""What a trajectory records: the agent's actions and the observations that reach it."""

from dataclasses import dataclass, field

from hyoshigi.trajectory.parts import Author, Part


def create_action_id() -> str:
    """Return a new action id."""
    return ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """A tool the agent asks to run, with its arguments as the JSON text the model emitted."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """What the agent emits in one step: its thought, what it says and the tools it calls."""

    id: str = field(default_factory=create_action_id)
    thought: tuple[Part, ...] = ()
    parts: tuple[Part, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    """Content that reaches the agent, with who wrote it and where it came from."""

    author: Author
    parts: tuple[Part, ...]
    source: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolObservation(Observation):
    """A tool's result, answering one of the agent's tool calls."""

    call_id: str
    is_error: bool = False
    author: Author = field(default=Author.UNKNOWN, init=False)


type Event = Action | Observation
