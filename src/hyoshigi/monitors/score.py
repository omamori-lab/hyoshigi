"""A monitor's score of an action, and the evidence behind it.

A monitor always returns a score. When it cannot rate the action, the score fails closed:
suspicion 1.0, with the cause as evidence.
"""

# Composite refers to SuspicionScore, and SuspicionScore to Evidence, which holds Composite.
from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import Decision, Refused, Truncated, Unavailable
from hyoshigi.options import check_instance, describe_type, read_threshold

MEAN_DECIMAL_PLACES = 12


def average_values(values: Iterable[float]) -> float:
    """The mean of suspicion values, rounded to `MEAN_DECIMAL_PLACES`.

    Equal means can differ in the last bits of a float: the mean of 0.1, 0.2 and 0.0 is
    0.10000000000000002, where the mean of 0.1 alone is 0.1. Rounding makes equal means equal
    values, which a CalibratedMonitor ranks alike.
    """
    return round(
        statistics.fmean(values),
        MEAN_DECIMAL_PLACES,
    )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Unreadable:
    """The model's replies held no readable score or label, after any retries."""


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Unperceivable:
    """The kinds and media types of the content in the rated action that the monitor cannot
    perceive: at least one.
    """

    content: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.content:
            raise ConfigurationError("unperceivable content names at least one kind")


type Cause = Unreadable | Unperceivable | Truncated | Refused | Unavailable


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class FailedClosed:
    """The monitor could not rate the action, so it scored 1.0; the cause says why."""

    model: str
    cause: Cause


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Rationale:
    """The model that wrote the reasons for a score, and the rated action's hidden reasoning,
    which the monitor names only when its view shows thoughts.
    """

    model: str
    hidden_reasoning: tuple[str, ...] = ()


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class StatementDecision:
    """The decision a model gave on one statement about the state."""

    statement: str
    decision: Decision


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Decisions:
    """The decisions a model gave on the monitor's statements, and its hidden reasoning."""

    model: str
    decisions: tuple[StatementDecision, ...]
    hidden_reasoning: tuple[str, ...] = ()


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Labels:
    """The label a guard model gave in each sampled reply, in order, and its hidden reasoning."""

    model: str
    labels: tuple[str, ...]
    hidden_reasoning: tuple[str, ...] = ()


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Composite:
    """The scores a monitor combined into its own: a tuple of at least one."""

    parts: tuple[SuspicionScore, ...]

    def __post_init__(self) -> None:
        check_instance(
            self.parts,
            expected=tuple,
            name="parts",
        )
        for position, part in enumerate(self.parts):
            check_instance(
                part,
                expected=SuspicionScore,
                name=f"parts[{position}]",
            )
        if not self.parts:
            raise ConfigurationError("a composite holds at least one score")


type Evidence = Rationale | Decisions | Labels | Composite | FailedClosed


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class SuspicionScore:
    """How suspicious an action is, from 0 to 1, with an explanation and the evidence.

    The value is a number from 0 to 1, kept as a float; a score that failed closed has the
    value 1.0. The explanation is a string and the evidence a Rationale, Decisions, Labels,
    Composite or FailedClosed; anything else raises ConfigurationError when the score is
    built.
    """

    value: float
    explanation: str
    evidence: Evidence

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the float is set past its guard.
        object.__setattr__(
            self,
            "value",
            read_threshold(
                self.value,
                name="value",
            ),
        )
        check_instance(
            self.explanation,
            expected=str,
            name="explanation",
        )
        if not isinstance(
            self.evidence,
            Rationale | Decisions | Labels | Composite | FailedClosed,
        ):
            raise ConfigurationError(
                "evidence must be a Rationale, Decisions, Labels, Composite or FailedClosed, "
                f"got {describe_type(self.evidence)}",
            )
        if self.failed_closed and self.value != 1.0:
            raise ConfigurationError("a score that failed closed has the value 1.0")

    @property
    def failed_closed(self) -> bool:
        """Whether the monitor could not rate the action, or could not rate a part of it."""
        match self.evidence:
            case FailedClosed():
                return True
            case Composite(parts=parts):
                return any(part.failed_closed for part in parts)
            case Rationale() | Decisions() | Labels():
                return False
            case unreachable:
                assert_never(unreachable)


def fail_closed(
    cause: Cause,
    *,
    model: str,
) -> SuspicionScore:
    """The score of an action the monitor could not rate: 1.0, with the cause as evidence.

    The explanation names the failure without its content, since the agent may read it.
    """
    return SuspicionScore(
        value=1.0,
        explanation=explain_failure(cause),
        evidence=FailedClosed(
            model=model,
            cause=cause,
        ),
    )


def explain_failure(cause: Cause) -> str:
    """Say in a sentence why the action is treated as suspicious, without numbers or content."""
    match cause:
        case Unreadable():
            return (
                "The monitor could not read its model's reply, so the action is treated as "
                "suspicious."
            )
        case Unperceivable():
            return (
                "The action holds content the monitor cannot perceive, so it is treated as "
                "suspicious."
            )
        case Truncated():
            return (
                "The monitor's model stopped before it gave a score, so the action is treated "
                "as suspicious."
            )
        case Refused():
            return "The monitor's model refused to rate the action, so it is treated as suspicious."
        case Unavailable():
            return "The monitor's model was unavailable, so the action is treated as suspicious."
        case unreachable:
            assert_never(unreachable)
