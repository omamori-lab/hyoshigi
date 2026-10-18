"""Reports of a rating's score to Langfuse and LangSmith, and the monitor that sends them.

The names are in place; what they do is not yet written.
"""

# The SDKs' clients are imported only to type, since each SDK is an optional extra.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langsmith import AsyncClient, Client


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangfuseScores:
    """Sends a score to Langfuse."""

    client: Langfuse
    name: str = "suspicion"

    def send(
        self,
        score: SuspicionScore,
        *,
        trace_id: str,
        observation_id: str,
    ) -> None:
        """Return without sending anything: the request is not built."""


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangSmithFeedback:
    """Sends a score to LangSmith."""

    client: Client
    project_id: str
    name: str = "suspicion"

    def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Return without sending anything: the request is not built."""


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncLangSmithFeedback:
    """Sends a score to LangSmith, awaited."""

    client: AsyncClient
    project_id: str
    name: str = "suspicion"

    async def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Return without sending anything: the request is not built."""


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ReportedMonitor(Monitor):
    """Rates with a monitor and reports its score."""

    monitor: Monitor
    reports: tuple[LangfuseScores | LangSmithFeedback, ...]

    def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the wrapped monitor."""
        return self.monitor.rate(
            action,
            trajectory=trajectory,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncReportedMonitor(AsyncMonitor):
    """Rates with a monitor, awaited, and reports its score."""

    monitor: AsyncMonitor
    reports: tuple[LangfuseScores | AsyncLangSmithFeedback, ...]

    async def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the wrapped monitor."""
        return await self.monitor.rate(
            action,
            trajectory=trajectory,
        )
