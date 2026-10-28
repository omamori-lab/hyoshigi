"""What a monitor reads of a trajectory: its channels, and how many recent entries."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Flag, auto

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import check_instance, describe_option_value, read_count
from hyoshigi.trajectory.redaction import Redaction


class Channel(Flag):
    """The kinds of entry in a trajectory that a monitor can read.

    The default is TASK, TOOL_CALLS and SUBAGENT_REPORTS. A tool error travels with the tool
    calls. A view without context notes shows the opaque parts of an unknown author's
    observation with the tool calls.

    <https://omamori-lab.github.io/hyoshigi/explanation/the-default-view/>
    """

    TASK = auto()
    CONTEXT = auto()
    THOUGHTS = auto()
    AGENT_PARTS = auto()
    TOOL_CALLS = auto()
    TOOL_OBSERVATIONS = auto()
    SUBAGENT_REPORTS = auto()
    DEFAULT = TASK | TOOL_CALLS | SUBAGENT_REPORTS


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class View:
    """Which entries of a trajectory a monitor reads.

    `channels` is a Channel. `most_recent` keeps only that many of the last entries shown,
    or every entry when it is None; the task's entries are kept in addition. It is a whole
    number from 1 to sys.maxsize, kept as an int. `delegation_tools` names the tools whose
    observations are subagent reports: any iterable of strings but a string, kept as a
    frozenset. Anything else raises ConfigurationError when the view is built.
    """

    channels: Channel = Channel.DEFAULT
    most_recent: int | None = None
    delegation_tools: frozenset[str] = frozenset()
    redaction: Redaction | None = None

    def __post_init__(self) -> None:
        check_instance(
            self.channels,
            expected=Channel,
            name="channels",
        )
        # The dataclass is frozen, so the checked values are set past its guard.
        if self.most_recent is not None:
            count = read_count(
                self.most_recent,
                name="most_recent",
                minimum=1,
            )
            object.__setattr__(
                self,
                "most_recent",
                count,
            )
        object.__setattr__(
            self,
            "delegation_tools",
            _read_delegation_tools(self.delegation_tools),
        )


def _read_delegation_tools(tools: object) -> frozenset[str]:
    """Read the tool names once, each checked as a string by its position."""
    if isinstance(
        tools,
        str | bytes,
    ) or not isinstance(
        tools,
        Iterable,
    ):
        raise ConfigurationError(
            "delegation_tools must be an iterable of tool names other than a string, "
            f"got {describe_option_value(tools)}",
        )
    names = tuple(tools)
    for position, name in enumerate(names):
        check_instance(
            name,
            expected=str,
            name=f"delegation_tools[{position}]",
        )
    return frozenset(names)
