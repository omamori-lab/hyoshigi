"""Block and feed back: a suspicious proposal is blocked and the agent told why, and after too
many blocks a person audits the next proposal.
"""

from dataclasses import dataclass

from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.protocols.decisions import Halt, Policy, Proposal, ProtocolDecision
from hyoshigi.protocols.requests import FailureResponse
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

UNDECIDED = "Not decided yet."


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class BlockAndFeedBack:
    """Blocks the agent's proposal at or above `block_threshold`, with the score's explanation as
    the reason, and runs it below; after too many blocks, a person audits the next proposal.
    """

    monitor: Monitor
    block_threshold: float
    blocks_in_a_row: int = 3
    blocks_in_total: int = 20
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision:
        """Halt, unrated."""
        return Halt(
            reason=UNDECIDED,
            proposals=(
                Proposal(
                    action=proposed,
                    score=None,
                    policy=Policy.TRUSTED,
                ),
            ),
            position=len(trajectory.history),
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncBlockAndFeedBack:
    """Decides as BlockAndFeedBack does, with the monitor awaited."""

    monitor: AsyncMonitor
    block_threshold: float
    blocks_in_a_row: int = 3
    blocks_in_total: int = 20
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    async def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision:
        """Halt, unrated."""
        return Halt(
            reason=UNDECIDED,
            proposals=(
                Proposal(
                    action=proposed,
                    score=None,
                    policy=Policy.TRUSTED,
                ),
            ),
            position=len(trajectory.history),
        )
