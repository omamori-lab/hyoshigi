"""What a control protocol asks the program running the agent for, and what it decides when its
monitor got no answer.
"""

from dataclasses import dataclass
from enum import StrEnum

from hyoshigi.protocols.decisions import Proposal


class FailureResponse(StrEnum):
    """What a protocol decides about the agent's proposal when its monitor got no answer."""

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
    """A request for up to `count` more proposals from the agent's model, on the same history."""

    count: int
    proposals: tuple[Proposal, ...]
    position: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class TrustedStep:
    """A request for one action from the trusted model, on the same history."""

    proposals: tuple[Proposal, ...]
    position: int


type ProtocolRequest = Resample | TrustedStep
