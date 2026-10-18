"""The pieces of an action read from a provider's payload, until they become the action."""

from dataclasses import dataclass, field, replace
from typing import assert_never

from hyoshigi.trajectory.history import Action, Event, ToolCall
from hyoshigi.trajectory.parts import MediaPart, OpaquePart, Part, TextPart


@dataclass(
    frozen=True,
    slots=True,
)
class Thought:
    part: Part


@dataclass(slots=True)
class ActionDraft:
    """The thought, parts and tool calls read so far, until they become an action."""

    id: str | None = None
    thought: list[Part] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def add(
        self,
        item: Thought | Part | ToolCall,
    ) -> None:
        """Add a thought, a part or a tool call."""
        match item:
            case Thought(part=part):
                self.thought.append(part)
            case ToolCall():
                self.tool_calls.append(item)
            case TextPart() | MediaPart() | OpaquePart():
                self.parts.append(item)
            case unreachable:
                assert_never(unreachable)

    def build(self) -> Action:
        """Return the action read so far, with its id, or a generated one."""
        action = Action(
            thought=tuple(self.thought),
            parts=tuple(self.parts),
            tool_calls=tuple(self.tool_calls),
        )
        return (
            action
            if self.id is None
            else replace(
                action,
                id=self.id,
            )
        )


def close(draft: ActionDraft | None) -> list[tuple[str, Event]]:
    """Return a draft's action with the path it started at, if a draft was started."""
    if draft is None or draft.id is None:
        return []
    return [
        (
            draft.id,
            draft.build(),
        ),
    ]
