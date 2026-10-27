"""What a control protocol asks the program running the agent for, and what it decides when its
monitor got no answer.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import read_count
from hyoshigi.protocols.decisions import Audit, Block, Policy, Proposal, Run, check_proposals


class FailureResponse(StrEnum):
    """What a protocol decides about the agent's proposal when its monitor got no answer.

    AUDIT holds it for a person, BLOCK blocks it with the score's explanation as the reason,
    TRUSTED_STEP asks for an action from the trusted model, and RUN runs it unrated.
    """

    AUDIT = "audit"
    BLOCK = "block"
    TRUSTED_STEP = "trusted_step"
    RUN = "run"


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Resample:
    """A request for up to `count` more proposals from the agent's model, on the same history.

    The count is a whole number of at least 1, the proposals are the agent's proposals rated so
    far, at least one, each untrusted, and the position is the length of the history when the
    protocol asked, a whole number of at least 0; anything else raises ConfigurationError.
    """

    count: int
    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the int is set past its guard.
        object.__setattr__(
            self,
            "count",
            read_count(
                self.count,
                name="count",
                minimum=1,
            ),
        )
        check_request(self)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class TrustedStep:
    """A request for one action from the trusted model, on the same history.

    The proposals are the agent's proposals rated so far, at least one, each untrusted, and the
    position is the length of the history when the protocol asked, a whole number of at least
    0; anything else raises ConfigurationError.
    """

    proposals: tuple[Proposal, ...]
    position: int

    def __post_init__(self) -> None:
        check_request(self)


type ProtocolRequest = Resample | TrustedStep


def check_request(request: ProtocolRequest) -> None:
    """Raise ConfigurationError unless the request holds a tuple of at least one Proposal, each
    untrusted, at a position of at least 0; set its position as an int.
    """
    check_proposals(request.proposals)
    if any(proposal.policy is not Policy.UNTRUSTED for proposal in request.proposals):
        raise ConfigurationError("a request holds the agent's proposals, each untrusted")
    # The dataclass is frozen, so the int is set past its guard.
    object.__setattr__(
        request,
        "position",
        read_count(
            request.position,
            name="position",
            minimum=0,
        ),
    )


def decide_on_unavailable(
    response: Literal[FailureResponse.AUDIT, FailureResponse.BLOCK, FailureResponse.RUN],
    *,
    proposal: Proposal,
    reason: str,
    position: int,
) -> Run | Block | Audit:
    """Decide about the agent's proposal when its monitor got no answer: Audit for AUDIT, Block
    with the reason for BLOCK, and Run for RUN.
    """
    proposals = (proposal,)
    match response:
        case FailureResponse.AUDIT:
            return Audit(
                action=proposal.action,
                proposals=proposals,
                position=position,
            )
        case FailureResponse.BLOCK:
            return Block(
                action=proposal.action,
                reason=reason,
                proposals=proposals,
                position=position,
            )
        case FailureResponse.RUN:
            return Run(
                action=proposal.action,
                proposals=proposals,
                position=position,
            )
        case unreachable:
            assert_never(unreachable)
