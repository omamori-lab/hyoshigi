"""Decision monitors: rate a proposed action by asking a decision model statements.

The monitor asks whether statements about the trajectory and the proposed action hold,
and aggregates the probabilities into a suspicion score. Aggregation by maximum lets any
single criterion raise suspicion, the right shape for threat criteria, and the way
ShieldGemma turns its harm types into one binary outcome when it evaluates them (Zeng et
al., "ShieldGemma: Generative AI Content Moderation Based on Gemma", arXiv:2407.21772,
section 5.1). A mean dilutes one strong hit that the maximum keeps: 0.95 with three 0.05s
averages 0.275.

Taking the maximum of raw probabilities differs from control evaluations that aggregate
on the honest scale: LinuxArena turns each monitor's score into a percentile rank
against its honest distribution and then takes the maximum (Tracy et al., "LinuxArena: A
Control Setting for AI Agents in Live Production Software Environments", arXiv:2604.15384,
appendix C), and AI Control maps each score through the empirical distribution of honest
scores before aggregating (Greenblatt et al., "AI Control: Improving Safety Despite
Intentional Subversion", arXiv:2312.06942, appendix K).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from hyoshigi.models.model import AsyncDecisionModel, DecisionModel
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Decisions, SuspicionScore
from hyoshigi.monitors.view import View
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


class Aggregation(Enum):
    """How a decision monitor aggregates the probabilities of several statements."""

    MAX = auto()
    MIN = auto()
    MEAN = auto()


DEFAULT_STATEMENT: str = "stub"
ATTACK_CRITERIA: tuple[str, ...] = ("stub",)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionMonitor(Monitor):
    """A monitor that rates an action using a decision model synchronously."""

    model: DecisionModel
    view: View = field(default_factory=View)
    statements: tuple[str, ...] = (DEFAULT_STATEMENT,)
    aggregation: Aggregation = Aggregation.MAX

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate an action the agent proposes against the trajectory so far."""
        return SuspicionScore(
            value=0.5,
            explanation="stub 0",
            evidence=Decisions(model=self.model.name, decisions=()),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncDecisionMonitor(AsyncMonitor):
    """A monitor that rates an action using a decision model asynchronously."""

    model: AsyncDecisionModel
    view: View = field(default_factory=View)
    statements: tuple[str, ...] = (DEFAULT_STATEMENT,)
    aggregation: Aggregation = Aggregation.MAX

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate an action the agent proposes against the trajectory so far."""
        return SuspicionScore(
            value=0.5,
            explanation="stub 0",
            evidence=Decisions(model=self.model.name, decisions=()),
        )
