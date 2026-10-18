"""What a monitor reads of a trajectory: its channels, and how many recent entries."""

from dataclasses import dataclass
from enum import Flag, auto


class Channel(Flag):
    """The kinds of entry in a trajectory that a monitor can read.

    The default leaves out the agent's thoughts, its parts and raw tool output, following
    Claude Code's auto mode classifier (Hughes, "How we built Claude Code auto mode",
    Anthropic, 2026). Thoughts are one flag away: reading them catches more misbehaviour
    (Baker et al., "Monitoring Reasoning Models for Misbehavior", arXiv:2503.11926) but
    can also mislead a monitor (Arnav et al., "CoT Red-Handed", arXiv:2505.23575). A tool
    error travels with the tool calls, so a view that shows a call shows that it failed.
    """

    TASK = auto()
    CONTEXT = auto()
    THOUGHTS = auto()
    AGENT_PARTS = auto()
    TOOL_CALLS = auto()
    TOOL_OBSERVATIONS = auto()
    SUBAGENT_REPORTS = auto()
    DEFAULT = TASK | TOOL_CALLS | SUBAGENT_REPORTS


@dataclass(frozen=True, slots=True, kw_only=True)
class View:
    """Which entries of a trajectory a monitor reads.

    `most_recent` keeps only that many of the last entries shown, or every entry when it
    is None; the task's entries are kept in addition. `delegation_tools` names the tools
    whose observations are subagent reports.
    """

    channels: Channel = Channel.DEFAULT
    most_recent: int | None = None
    delegation_tools: frozenset[str] = frozenset()
