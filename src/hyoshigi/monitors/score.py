"""A monitor's score of an action, and the evidence behind it."""

# Composite refers to SuspicionScore, and SuspicionScore to Evidence, which holds Composite.
from __future__ import annotations

from dataclasses import dataclass

from hyoshigi.models.model import Decision, Refused, Truncated, Unavailable


@dataclass(frozen=True, slots=True, kw_only=True)
class Unreadable:
    """The model's replies held no readable score, after every retry."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Unperceivable:
    """Content in the rated action that the monitor cannot perceive: at least one kind."""

    content: tuple[str, ...]


type Cause = Unreadable | Unperceivable | Truncated | Refused | Unavailable


@dataclass(frozen=True, slots=True, kw_only=True)
class FailedClosed:
    """The monitor failed to rate the action, so it scored 1.0; the cause says why."""

    model: str
    cause: Cause


@dataclass(frozen=True, slots=True, kw_only=True)
class Rationale:
    """The model's written reasons for a score, and its hidden reasoning if it gave any."""

    model: str
    hidden_reasoning: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class StatementDecision:
    """The decision a model gave on one statement about the state."""

    statement: str
    decision: Decision


@dataclass(frozen=True, slots=True, kw_only=True)
class Decisions:
    """The decisions a model gave on the monitor's statements, and its hidden reasoning."""

    model: str
    decisions: tuple[StatementDecision, ...]
    hidden_reasoning: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Composite:
    """The scores a monitor combined into its own: at least one."""

    parts: tuple[SuspicionScore, ...]


type Evidence = Rationale | Decisions | Composite | FailedClosed


@dataclass(frozen=True, slots=True, kw_only=True)
class SuspicionScore:
    """How suspicious an action is, from 0 to 1, with the explanation and the evidence."""

    value: float
    explanation: str
    evidence: Evidence

    @property
    def failed_closed(self) -> bool:
        """Whether the monitor failed to rate the action, or a score it combined did."""
        return False


def fail_closed(cause: Cause, *, model: str) -> SuspicionScore:
    """The score of an action the monitor could not rate: 1.0, with the cause."""
    return SuspicionScore(
        value=0.0, explanation="", evidence=FailedClosed(model=model, cause=cause)
    )
