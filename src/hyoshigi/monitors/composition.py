"""Monitors that wrap monitors: the mean of repeated ratings, a calibrated rank and a cascade."""

from collections.abc import Iterable
from dataclasses import dataclass

from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Rationale, SuspicionScore
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

_UNRATED = SuspicionScore(value=0.5, explanation="", evidence=Rationale(model=""))


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMonitor(Monitor):
    """Rates an action several times with one monitor."""

    monitor: Monitor
    ensemble_size: int

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncRepeatedMonitor(AsyncMonitor):
    """Rates an action several times with one monitor that is awaited."""

    monitor: AsyncMonitor
    ensemble_size: int

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedMonitor(Monitor):
    """Ranks a monitor's score among honest scores."""

    monitor: Monitor
    honest_scores: Iterable[float]
    random_seed: int | None = None

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncCalibratedMonitor(AsyncMonitor):
    """Ranks the score of a monitor that is awaited among honest scores."""

    monitor: AsyncMonitor
    honest_scores: Iterable[float]
    random_seed: int | None = None

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED


@dataclass(frozen=True, slots=True, kw_only=True)
class CascadeMonitor(Monitor):
    """Rates with a fast monitor, and with a careful one when the fast score is high."""

    fast: Monitor
    careful: Monitor
    escalation_threshold: float

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncCascadeMonitor(AsyncMonitor):
    """Rates with a fast monitor, and with a careful one when the fast score is high."""

    fast: AsyncMonitor
    careful: AsyncMonitor
    escalation_threshold: float

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action."""
        return _UNRATED
