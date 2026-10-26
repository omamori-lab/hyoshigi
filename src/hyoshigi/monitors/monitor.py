"""What a monitor is: something that rates an action against a trajectory."""

from abc import ABC, abstractmethod

from hyoshigi.models.model import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    DecisionModel,
    GenerativeModel,
)
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


class Monitor(ABC):
    """A monitor that rates an action by calling its models synchronously.

    A monitor holds configuration only; the state it rates against is the trajectory.
    """

    @abstractmethod
    def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate an action the agent proposes against the trajectory so far.

        The action is not yet in the trajectory. A rating that fails returns a score that
        failed closed; it never raises for a failure the domain expects.
        """

    def rate_latest_step(
        self,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the trajectory's latest action against everything before it.

        A trajectory whose latest entry is not an action raises ConfigurationError.
        """
        step = trajectory.split_latest_step()
        return self.rate(
            step.action,
            trajectory=step.trajectory,
        )


class AsyncMonitor(ABC):
    """A monitor that rates an action by awaiting its models.

    A monitor holds configuration only; the state it rates against is the trajectory.
    """

    @abstractmethod
    async def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate an action the agent proposes against the trajectory so far.

        The action is not yet in the trajectory. A rating that fails returns a score that
        failed closed; it never raises for a failure the domain expects.
        """

    async def rate_latest_step(
        self,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the trajectory's latest action against everything before it.

        A trajectory whose latest entry is not an action raises ConfigurationError.
        """
        step = trajectory.split_latest_step()
        return await self.rate(
            step.action,
            trajectory=step.trajectory,
        )


def name_decision_monitor(
    model: object,
    *,
    otherwise: str,
) -> str:
    """The hint for a generative monitor given a model it does not take: the monitor that takes
    a decision model, and otherwise the hint naming the monitor's twin.
    """
    match model:
        case DecisionModel():
            return "Use DecisionMonitor for a DecisionModel."
        case AsyncDecisionModel():
            return "Use AsyncDecisionMonitor for an AsyncDecisionModel."
        case _:
            return otherwise


def name_generative_monitors(
    model: object,
    *,
    otherwise: str,
) -> str:
    """The hint for a decision monitor given a model it does not take: the monitors that take a
    generative model, and otherwise the hint naming the monitor's twin.
    """
    match model:
        case GenerativeModel():
            return "Use LLMMonitor or GuardMonitor for a GenerativeModel."
        case AsyncGenerativeModel():
            return "Use AsyncLLMMonitor or AsyncGuardMonitor for an AsyncGenerativeModel."
        case _:
            return otherwise
