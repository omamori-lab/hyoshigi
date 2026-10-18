"""The guard monitor: a guard model labels an action against the caller's policy.

Guard models such as gpt-oss-safeguard and Qwen3Guard write a label, not a score. A label
alone gives a monitor one operating point, so every threshold flags the same actions; the
monitor samples the guard's reply several times and scores the share labelled suspicious.
Studies of sampled answer confidence, none of them on guard models, guide how many: Lyu et
al. saw calibration improve sharply over the first samples and saturate after 15 to 20, and
advise 3 to 5 under a tight budget ("Calibrating Large Language Models with Sample
Consistency", arXiv:2402.13904, section 6.3), where the default of five comes from. Guard
models are overconfident (Liu et al., "On Calibration of LLM-based Guard Models for Reliable
Content Moderation", arXiv:2410.10414, section 4.2.1), so honest actions tend to tie at 0; a
CalibratedMonitor ranks a score among honest ones.
"""

import warnings
from dataclasses import dataclass, field
from typing import assert_never

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    Prompt,
    Refused,
    Truncated,
    Unavailable,
)
from hyoshigi.monitors.guard_labels import GuardLabels, read_label
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.prompts import MonitorPrompt
from hyoshigi.monitors.rendering import render
from hyoshigi.monitors.score import (
    Labels,
    SuspicionScore,
    Unperceivable,
    Unreadable,
    fail_closed,
)
from hyoshigi.monitors.view import View
from hyoshigi.options import check_instance, read_count
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


class HardLabelWarning(UserWarning):
    """A guard monitor takes one sample, so its scores are 0 or 1 and every threshold flags
    the same actions.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardMonitor(Monitor):
    """Rates an action by the share of a guard model's sampled replies labelled suspicious.

    `prompt` states the caller's policy and should ask for one label in the guard's format;
    it is built with the bounds 0 and 1, which it need not use. `labels` names the guard's
    labels and where it writes them. A reply with no readable label, an ambiguous one, one
    cut off at the model's output limit, a refusal or an unavailable model fails the rating
    closed, the first such sample in order giving the cause. The samples differ only when the
    model's temperature is above zero.

    The explanation names the guard's finding with no count: the first suspicious label among
    the replies, else the first safe label.
    `samples` is a whole number from 1 to sys.maxsize, kept as an int; with 1 every score is
    0 or 1, and building the monitor warns with HardLabelWarning. An option that cannot work
    raises ConfigurationError when the monitor is built.
    """

    model: GenerativeModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    view: View = field(default_factory=View)

    def __post_init__(self) -> None:
        check_instance(
            self.model,
            expected=GenerativeModel,
            name="model",
            hint="Use AsyncGuardMonitor for an AsyncGenerativeModel.",
        )
        _check_options(self)

    def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the share of sampled replies labelled suspicious."""
        return drive(
            plan_guard(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncGuardMonitor(AsyncMonitor):
    """Rates an action as GuardMonitor does, with the samples awaited at once."""

    model: AsyncGenerativeModel
    prompt: MonitorPrompt
    labels: GuardLabels
    samples: int = 5
    view: View = field(default_factory=View)

    def __post_init__(self) -> None:
        check_instance(
            self.model,
            expected=AsyncGenerativeModel,
            name="model",
            hint="Use GuardMonitor for a GenerativeModel.",
        )
        _check_options(self)

    async def rate(self, action: Action, *, trajectory: Trajectory) -> SuspicionScore:
        """Rate the action with the share of sampled replies labelled suspicious."""
        return await drive_concurrently(
            plan_guard(self, action=action, trajectory=trajectory), ask=self.model.generate
        )


def _check_options(monitor: GuardMonitor | AsyncGuardMonitor) -> None:
    """Raise ConfigurationError for an option that cannot work; warn for one sample."""
    check_instance(monitor.prompt, expected=MonitorPrompt, name="prompt")
    check_instance(monitor.labels, expected=GuardLabels, name="labels")
    check_instance(monitor.view, expected=View, name="view")
    samples = read_count(monitor.samples, name="samples", minimum=1)
    # The dataclass is frozen, so the int is set past its guard.
    object.__setattr__(monitor, "samples", samples)
    if samples == 1:
        # The warning points at the caller, past __post_init__ and the dataclass's __init__.
        warnings.warn(
            "samples=1 gives every action a suspicion of 0 or 1, so every threshold flags the "
            "same actions and an audit budget cannot be set; take more samples.",
            HardLabelWarning,
            stacklevel=4,
        )


def plan_guard(
    monitor: GuardMonitor | AsyncGuardMonitor, *, action: Action, trajectory: Trajectory
) -> Plan[Prompt, GenerationOutcome, SuspicionScore]:
    """Plan the rating: render the action, ask the guard every sample at once, score the labels.

    Content the model cannot perceive fails closed before it is asked; after, the first sample
    that is unreadable, cut off, refused or unavailable fails closed with that cause.
    """
    name = monitor.model.name
    rendering = render(
        action, trajectory=trajectory, view=monitor.view, perceives=monitor.model.perceives
    )
    if rendering.unperceivable:
        return fail_closed(Unperceivable(content=rendering.unperceivable), model=name)
    request = monitor.prompt.build(rendering, lowest_score=0, highest_score=1)
    outcomes = yield (request,) * monitor.samples
    labels: list[str] = []
    for outcome in outcomes:
        match _read_sample(outcome, labels=monitor.labels):
            case str(label):
                labels.append(label)
            case Unreadable() | Truncated() | Refused() | Unavailable() as failure:
                return fail_closed(failure, model=name)
            case unreachable:
                assert_never(unreachable)
    return _score_labels(
        Labels(model=name, labels=tuple(labels), hidden_reasoning=rendering.hidden_reasoning),
        suspicious=monitor.labels.suspicious,
    )


def _read_sample(
    outcome: GenerationOutcome, *, labels: GuardLabels
) -> str | Unreadable | Truncated | Refused | Unavailable:
    """Return the label a sample holds, or the outcome that fails the rating closed."""
    match outcome:
        case Generation(text=text):
            return read_label(text, labels=labels)
        case Truncated() | Refused() | Unavailable():
            return outcome
        case unreachable:
            assert_never(unreachable)


def _score_labels(evidence: Labels, *, suspicious: frozenset[str]) -> SuspicionScore:
    """Score the share of suspicious labels, explained by the first one, else the first label."""
    flagged = [label for label in evidence.labels if label in suspicious]
    explanation = _explain_label((flagged or evidence.labels)[0], suspicious=suspicious)
    return SuspicionScore(
        value=len(flagged) / len(evidence.labels), explanation=explanation, evidence=evidence
    )


def _explain_label(label: str, *, suspicious: frozenset[str]) -> str:
    """Say which label the guard gave and what it found, without any count."""
    finding = "breaks" if label in suspicious else "follows"
    return f"The guard model labelled the action {label!r}, finding that it {finding} the policy."
