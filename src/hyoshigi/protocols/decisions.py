"""What a control protocol decides about an action the agent proposes, and what it weighed."""

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.options import check_instance, read_count
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

    The action is an Action and the policy a Policy. An untrusted proposal has a
    SuspicionScore; a trusted step is not rated, so its score is None. Anything else raises
    ConfigurationError.
    """

    action: Action
    score: SuspicionScore | None
    policy: Policy

    def __post_init__(self) -> None:
        check_instance(
            self.action,
            expected=Action,
            name="action",
        )
        check_instance(
            self.policy,
            expected=Policy,
            name="policy",
        )
        match self.policy, self.score:
            case (
                Policy.UNTRUSTED,
                SuspicionScore(),
            ) | (
                Policy.TRUSTED,
                None,
            ):
                pass
            case _:
                raise ConfigurationError(
                    "an untrusted proposal has a SuspicionScore, and a trusted one None",
                )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Run:
    """The agent's proposed action runs as proposed.

    The action is one of the proposals the protocol weighed, at least one, and the position is
    the length of the history when it decided, a whole number of at least 0; anything else
    raises ConfigurationError.
    """

    action: Action
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Flag:
    """The action runs, and a person audits the step afterwards.

    The action is one of the proposals the protocol weighed, at least one, and the position is
    the length of the history when it decided, a whole number of at least 0; anything else
    raises ConfigurationError.

    <https://omamori-lab.github.io/hyoshigi/explanation/control-protocols/>
    """

    action: Action
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Replace:
    """A resample or a trusted step runs in place of the proposed action.

    The action is one of the proposals the protocol weighed, at least one, and the position is
    the length of the history when it decided, a whole number of at least 0; anything else
    raises ConfigurationError.
    """

    action: Action
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Block:
    """Nothing runs, and the agent is told the reason.

    The action is one of the proposals the protocol weighed, at least one, the reason is a
    string, and the position is the length of the history when it decided, a whole number of
    at least 0; anything else raises ConfigurationError.
    """

    action: Action
    reason: str
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Audit:
    """The action is held for a person.

    The action is one of the proposals the protocol weighed, at least one, and the position is
    the length of the history when it decided, a whole number of at least 0; anything else
    raises ConfigurationError.
    """

    action: Action
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Halt:
    """The run stops.

    The proposals are those the protocol weighed, at least one, the reason is a string, and
    the position is the length of the history when it decided, a whole number of at least 0;
    anything else raises ConfigurationError.
    """

    reason: str
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_decision(self)


type ProtocolDecision = Run | Flag | Replace | Block | Audit | Halt


def check_decision(decision: ProtocolDecision) -> None:
    """Raise ConfigurationError unless the decision weighed a tuple of at least one Proposal, at a
    position of at least 0, its action is one of theirs and its reason is a string; set its
    position as an int.
    """
    check_proposals(decision.proposals)
    # The dataclass is frozen, so the int is set past its guard.
    object.__setattr__(
        decision,
        "position",
        read_count(
            decision.position,
            name="position",
            minimum=0,
        ),
    )
    match decision:
        case (
            Run(action=action) | Flag(action=action) | Replace(action=action) | Audit(action=action)
        ):
            check_proposed(
                action,
                proposals=decision.proposals,
            )
        case Block(
            action=action,
            reason=reason,
        ):
            check_proposed(
                action,
                proposals=decision.proposals,
            )
            check_instance(
                reason,
                expected=str,
                name="reason",
            )
        case Halt(reason=reason):
            check_instance(
                reason,
                expected=str,
                name="reason",
            )
        case unreachable:
            assert_never(unreachable)


def check_proposals(proposals: tuple[Proposal, ...]) -> None:
    """Raise ConfigurationError unless the proposals are a tuple of at least one Proposal."""
    check_instance(
        proposals,
        expected=tuple,
        name="proposals",
    )
    for index, proposal in enumerate(proposals):
        check_instance(
            proposal,
            expected=Proposal,
            name=f"proposals[{index}]",
        )
    if not proposals:
        raise ConfigurationError("proposals must hold at least one Proposal")


def check_proposed(
    action: Action,
    *,
    proposals: tuple[Proposal, ...],
) -> None:
    """Raise ConfigurationError unless the action is the action of one of the proposals."""
    if action not in [proposal.action for proposal in proposals]:
        raise ConfigurationError("the action must be the action of one of the proposals")
