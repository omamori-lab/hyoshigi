"""How a trajectory's events are written to and read from JSON-compatible data."""

from base64 import b64decode, b64encode
from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.models.model import Decision, Refused, Truncated, Unavailable
from hyoshigi.monitors.score import (
    Cause,
    Composite,
    Decisions,
    Evidence,
    FailedClosed,
    Labels,
    Rationale,
    StatementDecision,
    SuspicionScore,
    Unperceivable,
    Unreadable,
)
from hyoshigi.trajectory.history import Action, Event, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, Description, MediaPart, OpaquePart, Part, TextPart

FORMAT = "hyoshigi.trajectory"
VERSION = 1

_AUTHORS = ", ".join(Author)
_MEDIA_PART = "a media part with an IANA media type and exactly one of url and data"
_SCORE = "a score with a value from 0 to 1, an explanation and evidence"
_EVIDENCE = "rationale, decisions, composite or failed-closed evidence"
_DECISION = "a decision with a probability from 0 to 1 and a confidence from 0 to 1, or null"
_CAUSE = "a cause: unreadable, unperceivable, truncated, refused or unavailable"


def dump_event(event: Event) -> JSON:
    """Return an action or an observation as JSON-compatible data."""
    match event:
        case Action():
            return {
                "kind": "action",
                "id": event.id,
                "thought": dump_parts(event.thought),
                "parts": dump_parts(event.parts),
                "tool_calls": [dump_tool_call(call) for call in event.tool_calls],
            }
        case ToolObservation():
            return {
                "kind": "tool_observation",
                "call_id": event.call_id,
                "is_error": event.is_error,
                "source": event.source,
                "parts": dump_parts(event.parts),
            }
        case Observation():
            return {
                "kind": "observation",
                "author": event.author.value,
                "source": event.source,
                "parts": dump_parts(event.parts),
            }
        case unreachable:
            assert_never(unreachable)


def dump_tool_call(call: ToolCall) -> JSON:
    """Return a tool call as JSON-compatible data."""
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def dump_parts(parts: tuple[Part, ...]) -> list[JSON]:
    """Return parts as JSON-compatible data, media bytes in base64."""
    return [dump_part(part) for part in parts]


def dump_part(part: Part) -> JSON:
    """Return a part as JSON-compatible data, media bytes in base64."""
    match part:
        case TextPart():
            return {"kind": "text", "text": part.text}
        case MediaPart():
            return {
                "kind": "media",
                "media_type": part.media_type,
                "url": part.url,
                "data": None if part.data is None else b64encode(part.data).decode("ascii"),
                "description": dump_description(part.description),
            }
        case OpaquePart():
            return {"kind": "opaque", "opaque_kind": part.kind}
        case unreachable:
            assert_never(unreachable)


def dump_description(description: Description | None) -> JSON:
    """Return a media part's description as JSON-compatible data."""
    if description is None:
        return None
    return {"text": description.text, "author": description.author.value}


def load_event(data: JSON, *, path: str) -> Event:
    """Read an action or an observation that dump_event wrote."""
    match data:
        case {
            "kind": "action",
            "id": str(action_id),
            "thought": list(thought),
            "parts": list(parts),
            "tool_calls": list(calls),
        }:
            return Action(
                id=action_id,
                thought=load_parts(thought, path=f"{path}.thought"),
                parts=load_parts(parts, path=f"{path}.parts"),
                tool_calls=tuple(
                    load_tool_call(call, path=f"{path}.tool_calls[{index}]")
                    for index, call in enumerate(calls)
                ),
            )
        case {
            "kind": "tool_observation",
            "call_id": str(call_id),
            "is_error": bool(is_error),
            "source": str() | None as source,
            "parts": list(parts),
        }:
            return ToolObservation(
                call_id=call_id,
                is_error=is_error,
                source=source,
                parts=load_parts(parts, path=f"{path}.parts"),
            )
        case {
            "kind": "observation",
            "author": str(author),
            "source": str() | None as source,
            "parts": list(parts),
        }:
            return Observation(
                author=load_author(author, path=f"{path}.author"),
                source=source,
                parts=load_parts(parts, path=f"{path}.parts"),
            )
        case _:
            raise FormatError(path=path, expected="an action, an observation or a tool observation")


def load_tool_call(data: JSON, *, path: str) -> ToolCall:
    """Read a tool call that dump_tool_call wrote."""
    match data:
        case {"id": str(call_id), "name": str(name), "arguments": str(arguments)}:
            return ToolCall(id=call_id, name=name, arguments=arguments)
        case _:
            raise FormatError(
                path=path, expected="a tool call with an id, a name and arguments as text"
            )


def load_parts(data: list[JSON], *, path: str) -> tuple[Part, ...]:
    """Read the parts that dump_parts wrote."""
    return tuple(load_part(part, path=f"{path}[{index}]") for index, part in enumerate(data))


def load_part(data: JSON, *, path: str) -> Part:
    """Read a part that dump_part wrote."""
    match data:
        case {"kind": "text", "text": str(text)}:
            return TextPart(text=text)
        case {"kind": "opaque", "opaque_kind": str(kind)}:
            return OpaquePart(kind=kind)
        case {
            "kind": "media",
            "media_type": str(media_type),
            "url": str() | None as url,
            "data": str() | None as encoded,
            "description": None | dict() as description,
        }:
            try:
                return MediaPart(
                    media_type=media_type,
                    url=url,
                    data=decode_data(encoded, path=f"{path}.data"),
                    description=load_description(description, path=f"{path}.description"),
                )
            except ConfigurationError:
                raise FormatError(path=path, expected=_MEDIA_PART) from None
        case _:
            raise FormatError(path=path, expected="a text, media or opaque part")


def decode_data(encoded: str | None, *, path: str) -> bytes | None:
    """Read media bytes that dump_part wrote in base64."""
    if encoded is None:
        return None
    try:
        return b64decode(encoded, validate=True)
    except ValueError:
        raise FormatError(path=path, expected="base64 data") from None


def load_description(data: JSON, *, path: str) -> Description | None:
    """Read a media part's description that dump_description wrote."""
    match data:
        case None:
            return None
        case {"text": str(text), "author": str(author)}:
            return Description(text=text, author=load_author(author, path=f"{path}.author"))
        case _:
            raise FormatError(
                path=path, expected="a description with a text and an author, or null"
            )


def load_author(value: str, *, path: str) -> Author:
    """Read an author that dump_event or dump_description wrote."""
    try:
        return Author(value)
    except ValueError:
        raise FormatError(path=path, expected=f"an author: {_AUTHORS}") from None


def dump_score(score: SuspicionScore) -> JSON:
    """Return a score and its evidence as JSON-compatible data."""
    return {
        "value": score.value,
        "explanation": score.explanation,
        "evidence": dump_evidence(score.evidence),
    }


def dump_evidence(evidence: Evidence) -> JSON:
    """Return a score's evidence as JSON-compatible data."""
    match evidence:
        case Rationale():
            return {
                "kind": "rationale",
                "model": evidence.model,
                "hidden_reasoning": [*evidence.hidden_reasoning],
            }
        case Decisions():
            return {
                "kind": "decisions",
                "model": evidence.model,
                "decisions": [dump_statement_decision(each) for each in evidence.decisions],
                "hidden_reasoning": [*evidence.hidden_reasoning],
            }
        case Labels():
            return {"kind": "labels"}
        case Composite():
            return {"kind": "composite", "parts": [dump_score(part) for part in evidence.parts]}
        case FailedClosed():
            return {
                "kind": "failed_closed",
                "model": evidence.model,
                "cause": dump_cause(evidence.cause),
            }
        case unreachable:
            assert_never(unreachable)


def dump_statement_decision(each: StatementDecision) -> JSON:
    """Return a model's decision on one statement as JSON-compatible data."""
    return {
        "statement": each.statement,
        "decision": {
            "probability": each.decision.probability,
            "confidence": each.decision.confidence,
        },
    }


def dump_cause(cause: Cause) -> JSON:
    """Return the cause of a score that failed closed as JSON-compatible data."""
    match cause:
        case Unreadable():
            return {"kind": "unreadable"}
        case Unperceivable():
            return {"kind": "unperceivable", "content": [*cause.content]}
        case Truncated():
            return {"kind": "truncated", "text": cause.text}
        case Refused():
            return {"kind": "refused", "explanation": cause.explanation}
        case Unavailable():
            return {"kind": "unavailable", "status": cause.status}
        case unreachable:
            assert_never(unreachable)


def load_score(data: JSON, *, path: str) -> SuspicionScore:
    """Read a score that dump_score wrote."""
    match data:
        case {
            "value": int() | float() as value,
            "explanation": str(explanation),
            "evidence": dict() as raw,
        }:
            evidence = load_evidence(raw, path=f"{path}.evidence")
            try:
                return SuspicionScore(value=value, explanation=explanation, evidence=evidence)
            except ConfigurationError:
                raise FormatError(path=path, expected=_SCORE) from None
        case _:
            raise FormatError(path=path, expected=_SCORE)


def load_evidence(data: JSON, *, path: str) -> Evidence:
    """Read evidence that dump_evidence wrote."""
    match data:
        case {"kind": "rationale", "model": str(model), "hidden_reasoning": list(reasoning)}:
            return Rationale(
                model=model, hidden_reasoning=load_texts(reasoning, path=f"{path}.hidden_reasoning")
            )
        case {
            "kind": "decisions",
            "model": str(model),
            "decisions": list(decisions),
            "hidden_reasoning": list(reasoning),
        }:
            return Decisions(
                model=model,
                decisions=tuple(
                    load_statement_decision(each, path=f"{path}.decisions[{index}]")
                    for index, each in enumerate(decisions)
                ),
                hidden_reasoning=load_texts(reasoning, path=f"{path}.hidden_reasoning"),
            )
        case {"kind": "labels"}:
            return Labels(model="", labels=())
        case {"kind": "composite", "parts": list(parts)}:
            return load_composite(parts, path=path)
        case {"kind": "failed_closed", "model": str(model), "cause": dict() as cause}:
            return FailedClosed(model=model, cause=load_cause(cause, path=f"{path}.cause"))
        case _:
            raise FormatError(path=path, expected=_EVIDENCE)


def load_composite(parts: list[JSON], *, path: str) -> Composite:
    """Read the parts of a composite that dump_evidence wrote."""
    scores = tuple(
        load_score(part, path=f"{path}.parts[{index}]") for index, part in enumerate(parts)
    )
    try:
        return Composite(parts=scores)
    except ConfigurationError:
        raise FormatError(path=path, expected="a composite of at least one score") from None


def load_statement_decision(data: JSON, *, path: str) -> StatementDecision:
    """Read a model's decision on one statement that dump_statement_decision wrote."""
    match data:
        case {"statement": str(statement), "decision": dict() as decision}:
            return StatementDecision(
                statement=statement, decision=load_decision(decision, path=f"{path}.decision")
            )
        case _:
            raise FormatError(path=path, expected="a statement with a decision")


def load_decision(data: JSON, *, path: str) -> Decision:
    """Read a probability and a confidence, which JSON writes as numbers or null."""
    match data:
        case {
            "probability": int() | float() as probability,
            "confidence": int() | float() | None as confidence,
        } if not isinstance(probability, bool) and not isinstance(confidence, bool):
            try:
                return Decision(probability=probability, confidence=confidence)
            except ConfigurationError:
                raise FormatError(path=path, expected=_DECISION) from None
        case _:
            raise FormatError(path=path, expected=_DECISION)


def load_cause(data: JSON, *, path: str) -> Cause:
    """Read the cause of a score that failed closed that dump_cause wrote."""
    match data:
        case {"kind": "unreadable"}:
            return Unreadable()
        case {"kind": "unperceivable", "content": list(content)}:
            kinds = load_texts(content, path=f"{path}.content")
            try:
                return Unperceivable(content=kinds)
            except ConfigurationError:
                raise FormatError(path=path, expected="content naming at least one kind") from None
        case _:
            return load_model_outcome(data, path=path)


def load_model_outcome(data: JSON, *, path: str) -> Truncated | Refused | Unavailable:
    """Read a cause that is the model's outcome, which dump_cause wrote."""
    match data:
        case {"kind": "truncated", "text": str(text)}:
            return Truncated(text=text)
        case {"kind": "refused", "explanation": str() | None as explanation}:
            return Refused(explanation=explanation)
        case {"kind": "unavailable", "status": int() | None as status} if not isinstance(
            status, bool
        ):
            return Unavailable(status=status)
        case _:
            raise FormatError(path=path, expected=_CAUSE)


def load_texts(data: list[JSON], *, path: str) -> tuple[str, ...]:
    """Read a list of text."""
    return tuple(load_text(text, path=f"{path}[{index}]") for index, text in enumerate(data))


def load_text(data: JSON, *, path: str) -> str:
    """Read a piece of text."""
    match data:
        case str(text):
            return text
        case _:
            raise FormatError(path=path, expected="text")
