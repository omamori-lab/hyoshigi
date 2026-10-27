"""How a score and its evidence are written to and read from JSON-compatible data."""

from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.models.model import Decision, Refused, Rejected, Truncated, Unavailable
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
from hyoshigi.trajectory.json_data import JSON

# How deep composite scores and subagents may each nest in the data load reads. Loading
# recurses once a level, so deeper data raises FormatError well before Python's recursion limit.
NESTING_LIMIT = 32

_SCORE = "a score with a value from 0 to 1, an explanation and evidence"
_EVIDENCE = "rationale, decisions, labels, composite or failed-closed evidence"
_DECISION = "a decision with a probability from 0 to 1 and a confidence from 0 to 1, or null"
_CAUSE = "a cause: unreadable, unperceivable, truncated, refused, rejected or unavailable"


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
            return {
                "kind": "labels",
                "model": evidence.model,
                "labels": [*evidence.labels],
                "unavailable": [dump_cause(sample) for sample in evidence.unavailable],
                "hidden_reasoning": [*evidence.hidden_reasoning],
            }
        case Composite():
            return {
                "kind": "composite",
                "parts": [dump_score(part) for part in evidence.parts],
            }
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
            return {
                "kind": "unperceivable",
                "content": [*cause.content],
            }
        case Truncated():
            return {
                "kind": "truncated",
                "text": cause.text,
            }
        case Refused():
            return {
                "kind": "refused",
                "explanation": cause.explanation,
            }
        case Rejected() | Unavailable():
            return {
                "kind": name_status_kind(cause),
                "status": cause.status,
            }
        case unreachable:
            assert_never(unreachable)


def name_status_kind(cause: Rejected | Unavailable) -> str:
    """The kind a cause stored with its HTTP status is stored as; a subclass takes its base's."""
    match cause:
        case Rejected():
            return "rejected"
        case Unavailable():
            return "unavailable"


def load_score(
    data: JSON,
    *,
    path: str,
    depth: int,
) -> SuspicionScore:
    """Read a score that dump_score wrote, inside `depth` composite scores."""
    match data:
        case {
            "value": int() | float() as value,
            "explanation": str(explanation),
            "evidence": dict() as raw,
        }:
            evidence = load_evidence(
                raw,
                path=f"{path}.evidence",
                depth=depth,
            )
            try:
                return SuspicionScore(
                    value=value,
                    explanation=explanation,
                    evidence=evidence,
                )
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected=_SCORE,
                ) from None
        case _:
            raise FormatError(
                path=path,
                expected=_SCORE,
            )


def load_evidence(
    data: JSON,
    *,
    path: str,
    depth: int,
) -> Evidence:
    """Read evidence that dump_evidence wrote, of a score inside `depth` composite scores."""
    match data:
        case {
            "kind": "rationale",
            "model": str(model),
            "hidden_reasoning": list(reasoning),
        }:
            return Rationale(
                model=model,
                hidden_reasoning=load_texts(
                    reasoning,
                    path=f"{path}.hidden_reasoning",
                ),
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
                    load_statement_decision(
                        each,
                        path=f"{path}.decisions[{index}]",
                    )
                    for index, each in enumerate(decisions)
                ),
                hidden_reasoning=load_texts(
                    reasoning,
                    path=f"{path}.hidden_reasoning",
                ),
            )
        case {
            "kind": "labels",
            "model": str(model),
            "labels": list(labels),
            "hidden_reasoning": list(reasoning),
        }:
            return Labels(
                model=model,
                labels=load_texts(
                    labels,
                    path=f"{path}.labels",
                ),
                unavailable=load_unavailable(
                    data,
                    path=path,
                ),
                hidden_reasoning=load_texts(
                    reasoning,
                    path=f"{path}.hidden_reasoning",
                ),
            )
        case {
            "kind": "composite",
            "parts": list(parts),
        }:
            return load_composite(
                parts,
                path=path,
                depth=depth,
            )
        case {
            "kind": "failed_closed",
            "model": str(model),
            "cause": dict() as cause,
        }:
            return FailedClosed(
                model=model,
                cause=load_cause(
                    cause,
                    path=f"{path}.cause",
                ),
            )
        case _:
            raise FormatError(
                path=path,
                expected=_EVIDENCE,
            )


def load_composite(
    parts: list[JSON],
    *,
    path: str,
    depth: int,
) -> Composite:
    """Read the parts of a composite that dump_evidence wrote, of a score inside `depth`
    composite scores. A composite inside NESTING_LIMIT others raises FormatError.
    """
    if depth == NESTING_LIMIT:
        raise FormatError(
            path=path,
            expected=f"composite scores nested at most {NESTING_LIMIT} deep",
        )
    scores = tuple(
        load_score(
            part,
            path=f"{path}.parts[{index}]",
            depth=depth + 1,
        )
        for index, part in enumerate(parts)
    )
    try:
        return Composite(parts=scores)
    except ConfigurationError:
        raise FormatError(
            path=path,
            expected="a composite of at least one score",
        ) from None


def load_unavailable(
    labels: JSON,
    *,
    path: str,
) -> tuple[Unavailable, ...]:
    """Read the samples that got no answer from labels that dump_evidence wrote, as unavailable
    causes; labels stored without them have none.
    """
    match labels:
        case {"unavailable": list(samples)}:
            return tuple(
                load_unavailable_sample(
                    sample,
                    path=f"{path}.unavailable[{index}]",
                )
                for index, sample in enumerate(samples)
            )
        case {"unavailable": _}:
            raise FormatError(
                path=f"{path}.unavailable",
                expected="a list of unavailable causes",
            )
        case _:
            return ()


def load_unavailable_sample(
    data: JSON,
    *,
    path: str,
) -> Unavailable:
    """Read one sample that got no answer, which dump_cause wrote as an unavailable cause."""
    match load_cause(
        data,
        path=path,
    ):
        case Unavailable() as cause:
            return cause
        case _:
            raise FormatError(
                path=path,
                expected="an unavailable cause",
            )


def load_statement_decision(
    data: JSON,
    *,
    path: str,
) -> StatementDecision:
    """Read a model's decision on one statement that dump_statement_decision wrote."""
    match data:
        case {
            "statement": str(statement),
            "decision": dict() as decision,
        }:
            return StatementDecision(
                statement=statement,
                decision=load_decision(
                    decision,
                    path=f"{path}.decision",
                ),
            )
        case _:
            raise FormatError(
                path=path,
                expected="a statement with a decision",
            )


def load_decision(
    data: JSON,
    *,
    path: str,
) -> Decision:
    """Read a probability and a confidence, which JSON writes as numbers or null."""
    match data:
        case {
            "probability": int() | float() as probability,
            "confidence": int() | float() | None as confidence,
        } if not isinstance(
            probability,
            bool,
        ) and not isinstance(
            confidence,
            bool,
        ):
            try:
                return Decision(
                    probability=probability,
                    confidence=confidence,
                )
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected=_DECISION,
                ) from None
        case _:
            raise FormatError(
                path=path,
                expected=_DECISION,
            )


def load_cause(
    data: JSON,
    *,
    path: str,
) -> Cause:
    """Read the cause of a score that failed closed that dump_cause wrote."""
    match data:
        case {"kind": "unreadable"}:
            return Unreadable()
        case {
            "kind": "unperceivable",
            "content": list(content),
        }:
            kinds = load_texts(
                content,
                path=f"{path}.content",
            )
            try:
                return Unperceivable(content=kinds)
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected="content naming at least one kind",
                ) from None
        case _:
            return load_model_outcome(
                data,
                path=path,
            )


def load_model_outcome(
    data: JSON,
    *,
    path: str,
) -> Truncated | Refused | Rejected | Unavailable:
    """Read a cause that is the model's outcome, which dump_cause wrote."""
    match data:
        case {
            "kind": "truncated",
            "text": str(text),
        }:
            return Truncated(text=text)
        case {
            "kind": "refused",
            "explanation": str() | None as explanation,
        }:
            return Refused(explanation=explanation)
        case {
            "kind": "rejected" | "unavailable" as kind,
            "status": int() | None as status,
        } if not isinstance(
            status,
            bool,
        ):
            return Rejected(status=status) if kind == "rejected" else Unavailable(status=status)
        case _:
            raise FormatError(
                path=path,
                expected=_CAUSE,
            )


def load_texts(
    data: list[JSON],
    *,
    path: str,
) -> tuple[str, ...]:
    """Read a list of text."""
    return tuple(
        load_text(
            text,
            path=f"{path}[{index}]",
        )
        for index, text in enumerate(data)
    )


def load_text(
    data: JSON,
    *,
    path: str,
) -> str:
    """Read a piece of text."""
    match data:
        case str(text):
            return text
        case _:
            raise FormatError(
                path=path,
                expected="text",
            )
