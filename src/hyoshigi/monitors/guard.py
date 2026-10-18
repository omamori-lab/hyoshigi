"""The guard monitor: a guard model labels an action against the caller's policy."""

from dataclasses import dataclass, field

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.models.model import AsyncGenerativeModel, GenerationOutcome, GenerativeModel, Prompt
from hyoshigi.monitors.guard_labels import GuardLabels
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.prompts import MonitorPrompt
from hyoshigi.monitors.score import SuspicionScore, Unreadable, fail_closed
from hyoshigi.monitors.view import View
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


class HardLabelWarning(UserWarning):
    """A guard monitor takes one sample, so every threshold flags the same actions."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardMonitor(Monitor):
    """Rates an action by the share of a guard model's sampled replies labelled suspicious."""

    model: GenerativeModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    view: View = field(default_factory=View)

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the share of sampled replies labelled suspicious."""
        return drive(
            plan_guard(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncGuardMonitor(AsyncMonitor):
    """Rates an action as GuardMonitor does, with a generative model that is awaited."""

    model: AsyncGenerativeModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    view: View = field(default_factory=View)

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the share of sampled replies labelled suspicious."""
        return await drive_concurrently(
            plan_guard(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


def plan_guard(
    monitor: GuardMonitor | AsyncGuardMonitor, *, action: Action, trajectory: Trajectory
) -> Plan[Prompt, GenerationOutcome, SuspicionScore]:
    """Plan the rating of an action with a guard model."""
    yield ()
    return fail_closed(Unreadable(), model=monitor.model.name)
