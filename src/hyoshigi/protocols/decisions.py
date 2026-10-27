"""What a control protocol decides about an action the agent proposes, and what it weighed."""

from dataclasses import dataclass
from enum import StrEnum

from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.trajectory.history import Action


class Policy(StrEnum):
    """Which policy wrote an action: the agent's model, untrusted, or a weaker trusted model.

    The untrusted model U and the trusted model T of Bhatt et al. 2025, "Ctrl-Z", sec. 2.2.
    """

    UNTRUSTED = "untrusted"
    TRUSTED = "trusted"


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Proposal:
    """An action a protocol weighed, its score, and the policy that wrote it.

    A trusted step is not rated, so its score is None.
    """

    action: Action
    score: SuspicionScore | None
    policy: Policy


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Run:
    """The agent's proposed action runs as proposed."""

    action: Action
    proposals: tuple[Proposal, ...]
    position: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Replace:
    """A resample or a trusted step runs in place of the proposed action."""

    action: Action
    proposals: tuple[Proposal, ...]
    position: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Block:
    """Nothing runs, and the agent is told the reason."""

    action: Action
    reason: str
    proposals: tuple[Proposal, ...]
    position: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Audit:
    """The action is held for a person."""

    action: Action
    proposals: tuple[Proposal, ...]
    position: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Halt:
    """The run stops."""

    reason: str
    proposals: tuple[Proposal, ...]
    position: int


type ProtocolDecision = Run | Replace | Block | Audit | Halt
