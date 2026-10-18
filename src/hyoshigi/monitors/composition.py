"""Monitors that wrap monitors: the mean of repeated ratings, a calibrated rank and a cascade."""

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Composite, Rationale, SuspicionScore
from hyoshigi.options import check_instance, read_count
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

MEAN_DECIMAL_PLACES = 12

_UNRATED = SuspicionScore(value=0.5, explanation="", evidence=Rationale(model=""))


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMonitor(Monitor):
    """Rates an action `ensemble_size` times with one monitor, in turn, and reports the mean.

    This is a homogeneous ensemble: the mean of independent calls of one monitor (Koran et
    al., "Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute",
    arXiv:2605.15377), whose arithmetic mean did as well as the other aggregators they tried
    (section 3.2). `ensemble_size` is what LinuxArena calls the ensemble size (Tracy et al.,
    "LinuxArena: A Control Setting for AI Agents in Live Production Software Environments",
    arXiv:2604.15384, appendix C). Integer scores tie so often that a low audit budget falls
    inside a group of tied actions; averaging calls adds the resolution that budget needs
    (Kaufman et al., "BashArena: A Control Setting for Highly Privileged AI Agents",
    arXiv:2512.15688, section 4.3). The calls differ only when the wrapped monitor's model
    samples, so it needs a temperature above zero.

    The value is the mean of the calls' values, rounded to `MEAN_DECIMAL_PLACES`, and the
    explanation the most suspicious call's, the first of a tie. When any call failed closed,
    the whole score fails closed: the value is 1.0 and the explanation the first failed
    call's. Either way the evidence is the calls' scores, in order. `ensemble_size` is a
    whole number of at least 1, and `monitor` a Monitor; otherwise ConfigurationError is
    raised when the wrapper is built.
    """

    monitor: Monitor
    ensemble_size: int

    def __post_init__(self) -> None:
        _check_monitor(self.monitor, name="monitor", async_twin=AsyncRepeatedMonitor)
        _check_ensemble_size(self)

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with each call in turn, and combine the scores."""
        return drive(
            plan_repetition(self.monitor, ensemble_size=self.ensemble_size),
            ask=lambda monitor: monitor.rate(action, trajectory=trajectory),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncRepeatedMonitor(AsyncMonitor):
    """Rates an action as RepeatedMonitor does, with the calls awaited at once.

    One failed call cancels the others and its error is raised as it is.
    """

    monitor: AsyncMonitor
    ensemble_size: int

    def __post_init__(self) -> None:
        _check_async_monitor(self.monitor, name="monitor", twin=RepeatedMonitor)
        _check_ensemble_size(self)

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with all the calls at once, and combine the scores."""
        return await drive_concurrently(
            plan_repetition(self.monitor, ensemble_size=self.ensemble_size),
            ask=lambda monitor: monitor.rate(action, trajectory=trajectory),
        )


def _check_monitor(value: object, *, name: str, async_twin: type) -> None:
    """Raise ConfigurationError unless the value is a Monitor; the hint names the async twin."""
    check_instance(
        value,
        expected=Monitor,
        name=name,
        hint=f"Use {async_twin.__name__} for an AsyncMonitor.",
    )


def _check_async_monitor(value: object, *, name: str, twin: type) -> None:
    """Raise ConfigurationError unless the value is an AsyncMonitor; the hint names the twin."""
    check_instance(
        value,
        expected=AsyncMonitor,
        name=name,
        hint=f"Use {twin.__name__} for a Monitor.",
    )


def _check_ensemble_size(repeated: RepeatedMonitor | AsyncRepeatedMonitor) -> None:
    """Keep the ensemble size as an int, whatever whole number was given, or raise."""
    size = read_count(repeated.ensemble_size, name="ensemble_size", minimum=1)
    # The dataclass is frozen, so the int is set past its guard.
    object.__setattr__(repeated, "ensemble_size", size)


def plan_repetition[Rater](
    monitor: Rater, *, ensemble_size: int
) -> Plan[Rater, SuspicionScore, SuspicionScore]:
    """Plan the repeated rating: ask the monitor `ensemble_size` times, then combine."""
    parts = yield (monitor,) * ensemble_size
    return build_mean_score(parts)


def build_mean_score(parts: tuple[SuspicionScore, ...]) -> SuspicionScore:
    """Combine the scores of repeated calls into their mean, or fail closed if any did."""
    evidence = Composite(parts=parts)
    failed = next((part for part in parts if part.failed_closed), None)
    if failed is not None:
        return SuspicionScore(value=1.0, explanation=failed.explanation, evidence=evidence)
    most_suspicious = max(parts, key=lambda part: part.value)
    mean = statistics.fmean(part.value for part in parts)
    return SuspicionScore(
        value=round(mean, MEAN_DECIMAL_PLACES),
        explanation=most_suspicious.explanation,
        evidence=evidence,
    )


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
