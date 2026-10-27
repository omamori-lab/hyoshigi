"""Defer to resample: a suspicious proposal is resampled, and the least suspicious sample decides
the step.
"""

from dataclasses import dataclass

from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.protocols.decisions import Halt, Policy, Proposal, ProtocolDecision
from hyoshigi.protocols.requests import FailureResponse, ProtocolRequest
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

UNDECIDED = "Not decided yet."


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class DeferToResample:
    """Runs the agent's proposal below `defer_threshold`; otherwise asks for up to `max_resamples`
    more proposals from the agent's model, and decides the step from all of them.
    """

    monitor: Monitor
    defer_threshold: float
    audit_threshold: float
    max_resamples: int
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
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

    def resume(
        self,
        request: ProtocolRequest,
        *,
        proposals: tuple[Action, ...],
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Halt, unrated."""
        return Halt(
            reason=UNDECIDED,
            proposals=request.proposals,
            position=request.position,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncDeferToResample:
    """Decides as DeferToResample does, with the monitor awaited and the resamples rated at once."""

    monitor: AsyncMonitor
    defer_threshold: float
    audit_threshold: float
    max_resamples: int
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    async def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
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

    async def resume(
        self,
        request: ProtocolRequest,
        *,
        proposals: tuple[Action, ...],
        trajectory: Trajectory,
    ) -> ProtocolDecision | ProtocolRequest:
        """Halt, unrated."""
        return Halt(
            reason=UNDECIDED,
            proposals=request.proposals,
            position=request.position,
        )
