"""What a trajectory records: the agent's actions and the observations that reach it."""

from dataclasses import dataclass, field
from uuid import uuid4

from hyoshigi.trajectory.parts import Author, Part


def _create_action_id() -> str:
    return uuid4().hex


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """A tool the agent asks to run, with its arguments as the JSON text the model emitted."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """What the agent emits in one step: its thought, what it says and the tools it calls.

    A thought is an action in language space that changes nothing outside the agent
    (Yao et al., "ReAct", ICLR 2023, section 2).
    """

    id: str = field(default_factory=_create_action_id)
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
    author: Author = field(default=Author.TOOL, init=False)


# OpenHands calls actions and observations events (Wang et al., "OpenHands", ICLR 2025,
# section 2.1).
type Event = Action | Observation
