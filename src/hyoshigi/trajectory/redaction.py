"""What a trajectory's content becomes before it is kept, read by a monitor's model, stored or
reported.

A redaction reaches text parts, thoughts, tool-call arguments, observations and their sources,
media, their descriptions and the UTF-8 text of text media, a score's explanation and the reply
text its cause holds, and a block's or a halt's reason. Ids, tool names, authors, values and
positions stay as they are, so ratings and records still name their actions.

<https://omamori-lab.github.io/hyoshigi/how-to/redacting-what-leaves-the-process/>
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import Refused, Truncated
from hyoshigi.monitors.score import (
    Composite,
    Decisions,
    Evidence,
    FailedClosed,
    Labels,
    Rationale,
    SuspicionScore,
)
from hyoshigi.options import check_instance, describe_type
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
    Flag,
    Halt,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.trajectory.history import Action, Event, Observation
from hyoshigi.trajectory.parts import MediaPart, OpaquePart, Part, TextPart, read_text


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Redaction:
    """Maps a trajectory's content to what is kept or sent in its place.

    `text` maps a str to a str. `media` maps a MediaPart to a part, such as the media blurred
    or a TextPart in its place, and the text in the part it returns then passes through `text`:
    a TextPart's text, a description, and the bytes of text media that are UTF-8.
    Each is callable, or ConfigurationError is raised when the redaction is built; a function
    that returns another type raises it where the redaction is applied.

    <https://omamori-lab.github.io/hyoshigi/how-to/redacting-what-leaves-the-process/>
    """

    text: Callable[[str], str]
    media: Callable[[MediaPart], Part]

    def __post_init__(self) -> None:
        for function, name in (
            (
                self.text,
                "text",
            ),
            (
                self.media,
                "media",
            ),
        ):
            if not callable(function):
                raise ConfigurationError(f"{name} must be callable, got {describe_type(function)}")


class StoredContent(StrEnum):
    """A stub: what a stored trajectory holds, raw or redacted."""

    RAW = "raw"
    REDACTED = "redacted"


def check_redaction(redaction: object) -> None:
    """Raise ConfigurationError unless the redaction is a Redaction or None."""
    if redaction is not None:
        check_instance(
            redaction,
            expected=Redaction,
            name="redaction",
        )


class Redactor:
    """Applies redactions in turn to a trajectory's entries, scores and records.

    Each distinct action and score is redacted once, and an equal one later gets the same copy,
    so a record's action stays the action in the history whatever a redaction returns on each
    call. With no redaction, everything is returned as it is.
    """

    def __init__(
        self,
        *redactions: Redaction | None,
    ) -> None:
        self._redactions = tuple(redaction for redaction in redactions if redaction is not None)
        self._actions: dict[Action, Action] = {}
        self._scores: dict[SuspicionScore, SuspicionScore] = {}

    def redact_event(
        self,
        event: Event,
    ) -> Event:
        """Return an action or an observation redacted."""
        match event:
            case Action():
                return self.redact_action(event)
            case Observation():
                return self.redact_observation(event)
            case unreachable:
                assert_never(unreachable)

    def redact_action(
        self,
        action: Action,
    ) -> Action:
        """Return the action with its thought, parts and tool calls' arguments redacted."""
        if not self._redactions:
            return action
        if action not in self._actions:
            self._actions[action] = replace(
                action,
                thought=self.redact_parts(action.thought),
                parts=self.redact_parts(action.parts),
                tool_calls=tuple(
                    replace(
                        call,
                        arguments=self.redact_text(call.arguments),
                    )
                    for call in action.tool_calls
                ),
            )
        return self._actions[action]

    def redact_observation(
        self,
        observation: Observation,
    ) -> Observation:
        """Return the observation with its parts and its source redacted."""
        if not self._redactions:
            return observation
        return replace(
            observation,
            parts=self.redact_parts(observation.parts),
            source=None if observation.source is None else self.redact_text(observation.source),
        )

    def redact_score(
        self,
        score: SuspicionScore,
    ) -> SuspicionScore:
        """Return the score with its explanation, and the reply text its evidence holds,
        redacted.
        """
        if not self._redactions:
            return score
        if score not in self._scores:
            self._scores[score] = replace(
                score,
                explanation=self.redact_text(score.explanation),
                evidence=self._redact_evidence(score.evidence),
            )
        return self._scores[score]

    def redact_decision(
        self,
        decision: ProtocolDecision,
    ) -> ProtocolDecision:
        """Return the decision with its proposals, its action and its reason redacted."""
        if not self._redactions:
            return decision
        proposals = tuple(
            replace(
                proposal,
                action=self.redact_action(proposal.action),
                score=None if proposal.score is None else self.redact_score(proposal.score),
            )
            for proposal in decision.proposals
        )
        match decision:
            case Run() | Flag() | Replace() | Audit():
                return replace(
                    decision,
                    action=self.redact_action(decision.action),
                    proposals=proposals,
                )
            case Block():
                return replace(
                    decision,
                    action=self.redact_action(decision.action),
                    reason=self.redact_text(decision.reason),
                    proposals=proposals,
                )
            case Halt():
                return replace(
                    decision,
                    reason=self.redact_text(decision.reason),
                    proposals=proposals,
                )
            case unreachable:
                assert_never(unreachable)

    def redact_parts(
        self,
        parts: tuple[Part, ...],
    ) -> tuple[Part, ...]:
        """Return each part redacted."""
        return tuple(self.redact_part(part) for part in parts)

    def redact_part(
        self,
        part: Part,
    ) -> Part:
        """Return a part through each redaction in turn: media through the media function, then
        the text in the part.
        """
        for redaction in self._redactions:
            part = _redact_part(
                part,
                redaction=redaction,
            )
        return part

    def redact_text(
        self,
        text: str,
    ) -> str:
        """Return the text through each redaction's text function in turn."""
        for redaction in self._redactions:
            text = _redact_text(
                text,
                redaction=redaction,
            )
        return text

    def _redact_evidence(
        self,
        evidence: Evidence,
    ) -> Evidence:
        """Return the evidence with the reply text of its cause, and its parts' scores, redacted."""
        match evidence:
            case FailedClosed(cause=Truncated(text=text) as cause):
                return replace(
                    evidence,
                    cause=replace(
                        cause,
                        text=self.redact_text(text),
                    ),
                )
            case FailedClosed(cause=Refused(explanation=str(explanation)) as cause):
                return replace(
                    evidence,
                    cause=replace(
                        cause,
                        explanation=self.redact_text(explanation),
                    ),
                )
            case Composite(parts=parts):
                return replace(
                    evidence,
                    parts=tuple(self.redact_score(part) for part in parts),
                )
            case Rationale() | Decisions() | Labels() | FailedClosed():
                return evidence
            case unreachable:
                assert_never(unreachable)


def _redact_part(
    part: Part,
    *,
    redaction: Redaction,
) -> Part:
    """Return media through the redaction's media function, then the text in the part through
    its text function; raise ConfigurationError if the media function returns no part.
    """
    if isinstance(
        part,
        MediaPart,
    ):
        part = redaction.media(part)
        if not isinstance(
            part,
            TextPart | MediaPart | OpaquePart,
        ):
            raise ConfigurationError(
                f"media must return a TextPart, MediaPart or OpaquePart, got {describe_type(part)}",
            )
    match part:
        case TextPart(text=text):
            return replace(
                part,
                text=_redact_text(
                    text,
                    redaction=redaction,
                ),
            )
        case MediaPart():
            return _redact_media(
                part,
                redaction=redaction,
            )
        case OpaquePart():
            return part
        case unreachable:
            assert_never(unreachable)


def _redact_media(
    part: MediaPart,
    *,
    redaction: Redaction,
) -> MediaPart:
    """Return media with the UTF-8 text of text media, and its description, through the
    redaction's text function; other bytes and a URL stay as they are.
    """
    text = read_text(part)
    description = part.description
    return replace(
        part,
        data=part.data
        if text is None
        else _redact_text(
            text,
            redaction=redaction,
        ).encode(),
        description=None
        if description is None
        else replace(
            description,
            text=_redact_text(
                description.text,
                redaction=redaction,
            ),
        ),
    )


def _redact_text(
    text: str,
    *,
    redaction: Redaction,
) -> str:
    """Return the text through the redaction's text function; raise ConfigurationError if it
    returns no str.
    """
    redacted = redaction.text(text)
    if not isinstance(
        redacted,
        str,
    ):
        raise ConfigurationError(f"text must return a str, got {describe_type(redacted)}")
    return redacted
