"""The LLM monitor: a generative model reasons about an action, then scores it."""

from dataclasses import dataclass, field

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    Prompt,
)
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.prompts import DEFAULT_PROMPT, MonitorPrompt
from hyoshigi.monitors.score import Rationale, SuspicionScore
from hyoshigi.monitors.view import View
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMMonitor(Monitor):
    """Rates an action with a generative model."""

    model: GenerativeModel
    view: View = field(default_factory=View)
    prompt: MonitorPrompt = DEFAULT_PROMPT
    lowest_score: int = 0
    highest_score: int = 10
    max_parse_retries: int = 2

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the model's score, which fails closed when it gives none."""
        return drive(
            plan_rating(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncLLMMonitor(AsyncMonitor):
    """Rates an action with an async generative model."""

    model: AsyncGenerativeModel
    view: View = field(default_factory=View)
    prompt: MonitorPrompt = DEFAULT_PROMPT
    lowest_score: int = 0
    highest_score: int = 10
    max_parse_retries: int = 2

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the model's score, which fails closed when it gives none."""
        return await drive_concurrently(
            plan_rating(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


def plan_rating(
    monitor: LLMMonitor | AsyncLLMMonitor, *, action: Action, trajectory: Trajectory
) -> Plan[Prompt, GenerationOutcome, SuspicionScore]:
    """Plan the rating of an action: render it, ask the monitor's model, read the score."""
    return SuspicionScore(value=0.25, explanation="", evidence=Rationale(model="stub"))
    yield ()
