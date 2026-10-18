"""Monitors that wrap monitors: the mean of repeated ratings, a calibrated rank and a cascade.

A wrapper is a monitor itself, of the family of the monitors it holds: a wrapper of Monitors is
a Monitor, and one of AsyncMonitors an AsyncMonitor. A monitor of the other family raises
ConfigurationError when the wrapper is built. Wrappers stack around one another. A wrapper
changes a score's value and passes on the explanation of a wrapped score word for word, so
the explanation, the only part of a score an agent may see, never states the new value.

A score that failed closed has the value 1.0 whatever the wrapper: repeated calls fail closed
when any call did, a calibrated monitor returns a failed score unranked, and a cascade
escalates from a failed fast score.
"""

import json
import random
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field, replace
from typing import assert_never

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.errors import ConfigurationError
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Composite, SuspicionScore, average_values
from hyoshigi.options import (
    check_instance,
    describe_option_value,
    read_count,
    read_integer,
    read_threshold,
)
from hyoshigi.trajectory.history import Action, Event, Observation
from hyoshigi.trajectory.storage import dump_event
from hyoshigi.trajectory.trajectory import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedMonitor(Monitor):
    """Rates an action `ensemble_size` times with one monitor, in turn, and reports the mean.

    This is a homogeneous ensemble: the mean of independent calls of one monitor (Koran et
    al., "Ensemble Monitoring for AI Control: Diverse Signals Outweigh More Compute",
    arXiv:2605.15377). Three calls of one monitor gained less there than three different
    monitors (section 5.2), and for ensembles of different monitors the arithmetic mean did
    as well as the other aggregators they tried (section 3.2). `ensemble_size` is what
    LinuxArena calls the ensemble size (Tracy et al., "LinuxArena: A Control Setting for AI
    Agents in Live Production Software Environments", arXiv:2604.15384, appendix C). Integer
    scores tie so often that a low audit budget falls inside a group of tied actions;
    averaging calls adds the resolution that budget needs (Kaufman et al., "BashArena: A
    Control Setting for Highly Privileged AI Agents", arXiv:2512.15688, section 4.3). The
    calls differ only when the wrapped monitor's model samples, so it needs a temperature
    above zero.

    The value is the mean of the calls' values, rounded to `MEAN_DECIMAL_PLACES`, and the
    explanation the most suspicious call's, the first of a tie. When any call failed closed,
    the whole score fails closed: the value is 1.0 and the explanation the first failed
    call's. Either way the evidence is the calls' scores, in order. `ensemble_size` is a
    whole number from 1 to sys.maxsize, and `monitor` a Monitor; otherwise
    ConfigurationError is raised when the wrapper is built.
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
    return SuspicionScore(
        value=average_values(part.value for part in parts),
        explanation=most_suspicious.explanation,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibratedMonitor(Monitor):
    """Reports a monitor's score as its smoothed conformal p-value among honest scores.

    Control evaluations set thresholds as percentiles of honest-run scores, so an audit
    budget of 1% flags the top 1% of honest actions. The p-value ranks the raw score among
    the n honest scores as a share of n + 1, with its ties broken by a uniform random draw
    u: (below + u * (tied + 1)) / (n + 1). That is the smoothed conformal p-value of Vovk et
    al. ("Criteria of efficiency for conformal prediction", arXiv:1603.04416, section 2,
    equation 2) with the raw score as the conformity score, and ties are broken at random as
    Makins et al. ("Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance
    Monitors", arXiv:2607.07368, appendix G.4.1) and LinuxArena (Tracy et al., "LinuxArena: A
    Control Setting for AI Agents in Live Production Software Environments",
    arXiv:2604.15384, section 5.1) break them. When the raw score is exchangeable with the
    honest scores, the p-value is uniform on [0, 1], so a threshold of 0.99 flags 1% of
    such actions on average.

    `honest_scores` are the wrapped monitor's scores on actions of runs the caller trusts,
    from the same monitor with the same settings: any iterable but a string or bytes, read
    once, of at least one score. Each is read as a threshold is, as a float from 0 to 1, so
    a float32 0.1 becomes 0.10000000149011612 and no longer ties with 0.1. They are kept
    sorted in `sorted_honest_scores`. A RepeatedMonitor inside ranks the mean, the score the
    honest scores measured; outside it would average p-values, and a mean of p-values is not
    a p-value.

    Without a `random_seed` the draw is `random.random()`. With one, it is the first number
    of a `random.Random` seeded with the seed, a line break and the key. The key is the JSON
    of the history's events and of the action as `dump_event` writes them, with every
    action's id blanked (ids are generated at random) and the trajectory's ratings left out,
    so the same action against the same history always draws the same share, in every
    process.

    A score that failed closed is returned as it is, unranked. Otherwise the explanation is
    the wrapped score's and the evidence is a Composite of it. `random_seed` is an integer or
    None; anything the monitor cannot work with raises ConfigurationError when it is built.
    """

    monitor: Monitor
    honest_scores: InitVar[Iterable[float]]
    random_seed: int | None = None
    sorted_honest_scores: tuple[float, ...] = field(init=False)

    def __post_init__(self, honest_scores: Iterable[float]) -> None:
        _check_monitor(self.monitor, name="monitor", async_twin=AsyncCalibratedMonitor)
        _check_calibration(self, honest_scores=honest_scores)

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the wrapped monitor, then rank its score among the honest ones."""
        tie_share = draw_tie_share(self.random_seed, action=action, trajectory=trajectory)
        plan = plan_calibration(
            self.monitor, honest_scores=self.sorted_honest_scores, tie_share=tie_share
        )
        return drive(plan, ask=lambda monitor: monitor.rate(action, trajectory=trajectory))


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncCalibratedMonitor(AsyncMonitor):
    """Ranks a score as CalibratedMonitor does, with the wrapped monitor awaited."""

    monitor: AsyncMonitor
    honest_scores: InitVar[Iterable[float]]
    random_seed: int | None = None
    sorted_honest_scores: tuple[float, ...] = field(init=False)

    def __post_init__(self, honest_scores: Iterable[float]) -> None:
        _check_async_monitor(self.monitor, name="monitor", twin=CalibratedMonitor)
        _check_calibration(self, honest_scores=honest_scores)

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the wrapped monitor, then rank its score among the honest ones."""
        tie_share = draw_tie_share(self.random_seed, action=action, trajectory=trajectory)
        plan = plan_calibration(
            self.monitor, honest_scores=self.sorted_honest_scores, tie_share=tie_share
        )
        return await drive_concurrently(
            plan, ask=lambda monitor: monitor.rate(action, trajectory=trajectory)
        )


def _check_calibration(
    calibrated: CalibratedMonitor | AsyncCalibratedMonitor, *, honest_scores: object
) -> None:
    """Keep the honest scores sorted and the seed as an int, or raise ConfigurationError."""
    scores = _read_honest_scores(honest_scores)
    seed = calibrated.random_seed
    if seed is not None:
        seed = read_integer(seed, name="random_seed")
    # The dataclass is frozen, so the values are set past its guard.
    object.__setattr__(calibrated, "sorted_honest_scores", scores)
    object.__setattr__(calibrated, "random_seed", seed)


def _read_honest_scores(honest_scores: object) -> tuple[float, ...]:
    """Read the honest scores once, each as a threshold named by its position, sorted."""
    if isinstance(honest_scores, str | bytes) or not isinstance(honest_scores, Iterable):
        raise ConfigurationError(
            "honest_scores must be an iterable of numbers between 0 and 1, "
            f"got {describe_option_value(honest_scores)}"
        )
    scores = sorted(
        read_threshold(score, name=f"honest_scores[{position}]")
        for position, score in enumerate(honest_scores)
    )
    if not scores:
        raise ConfigurationError("honest_scores must hold at least one score")
    return tuple(scores)


def plan_calibration[Rater](
    monitor: Rater, *, honest_scores: tuple[float, ...], tie_share: float
) -> Plan[Rater, SuspicionScore, SuspicionScore]:
    """Plan the calibrated rating: ask the monitor once, then rank its score unless it failed."""
    (wrapped,) = yield (monitor,)
    if wrapped.failed_closed:
        return wrapped
    return build_calibrated_score(wrapped, honest_scores=honest_scores, tie_share=tie_share)


def build_calibrated_score(
    wrapped: SuspicionScore, *, honest_scores: tuple[float, ...], tie_share: float
) -> SuspicionScore:
    """Rank a score among the sorted honest scores: its smoothed conformal p-value."""
    below = bisect_left(honest_scores, wrapped.value)
    tied = bisect_right(honest_scores, wrapped.value) - below
    return SuspicionScore(
        value=(below + tie_share * (tied + 1)) / (len(honest_scores) + 1),
        explanation=wrapped.explanation,
        evidence=Composite(parts=(wrapped,)),
    )


def draw_tie_share(random_seed: int | None, *, action: Action, trajectory: Trajectory) -> float:
    """Draw the tie share u, uniform on [0, 1): from the seed and the content when seeded.

    A string seed goes through SHA-512 in Python's `random`, so the draw is the same in
    every process.
    """
    if random_seed is None:
        return random.random()
    key = render_tie_break_key(action, trajectory=trajectory)
    return random.Random(f"{random_seed}\n{key}").random()


def render_tie_break_key(action: Action, *, trajectory: Trajectory) -> str:
    """Render the history and the action as JSON, without the ids and ratings.

    Action ids are blanked because they are generated at random: the same action has a
    different id in every run.
    """
    events = (*trajectory.history, action)
    return json.dumps([dump_event(_blank_action_id(event)) for event in events])


def _blank_action_id(event: Event) -> Event:
    match event:
        case Action():
            return replace(event, id="")
        case Observation():
            return event
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True, kw_only=True)
class CascadeMonitor(Monitor):
    """Rates with a fast monitor, and with a careful one only when the fast score is high.

    An action whose fast score is below `escalation_threshold` keeps that score; the rest go
    to the careful monitor, whose score replaces it. A score that failed closed is 1.0, so a
    failed fast score always escalates, and a failed careful score is the result. Both
    scores are returned unchanged. This follows the two stages of Claude Code's auto mode, a
    fast filter tuned for recall in front of a reasoning classifier (Hughes, "How we built
    Claude Code auto mode: a safer way to skip permissions", Anthropic, 2026, section
    "Two-stage classification: efficiency and recall"). `escalation_threshold` is read as a
    threshold is, as a float from 0 to 1; anything else raises ConfigurationError when the
    monitor is built, as does a `fast` or `careful` that is not a Monitor.
    """

    fast: Monitor
    careful: Monitor
    escalation_threshold: float

    def __post_init__(self) -> None:
        _check_monitor(self.fast, name="fast", async_twin=AsyncCascadeMonitor)
        _check_monitor(self.careful, name="careful", async_twin=AsyncCascadeMonitor)
        _check_escalation_threshold(self)

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the fast monitor, and with the careful one if that escalates."""
        plan = plan_cascade(
            self.fast, careful=self.careful, escalation_threshold=self.escalation_threshold
        )
        return drive(plan, ask=lambda monitor: monitor.rate(action, trajectory=trajectory))


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncCascadeMonitor(AsyncMonitor):
    """Rates an action as CascadeMonitor does, with the monitors awaited."""

    fast: AsyncMonitor
    careful: AsyncMonitor
    escalation_threshold: float

    def __post_init__(self) -> None:
        _check_async_monitor(self.fast, name="fast", twin=CascadeMonitor)
        _check_async_monitor(self.careful, name="careful", twin=CascadeMonitor)
        _check_escalation_threshold(self)

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the fast monitor, and with the careful one if that escalates."""
        plan = plan_cascade(
            self.fast, careful=self.careful, escalation_threshold=self.escalation_threshold
        )
        return await drive_concurrently(
            plan, ask=lambda monitor: monitor.rate(action, trajectory=trajectory)
        )


def _check_escalation_threshold(cascade: CascadeMonitor | AsyncCascadeMonitor) -> None:
    """Keep the escalation threshold as a float, whatever real number was given, or raise."""
    threshold = read_threshold(cascade.escalation_threshold, name="escalation_threshold")
    # The dataclass is frozen, so the float is set past its guard.
    object.__setattr__(cascade, "escalation_threshold", threshold)


def plan_cascade[Rater](
    fast: Rater, *, careful: Rater, escalation_threshold: float
) -> Plan[Rater, SuspicionScore, SuspicionScore]:
    """Plan the cascade: ask the fast monitor, then the careful one if the fast score escalates."""
    (fast_score,) = yield (fast,)
    if fast_score.value < escalation_threshold:
        return fast_score
    (careful_score,) = yield (careful,)
    return careful_score
