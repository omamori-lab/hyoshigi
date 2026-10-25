"""A span per rating in OpenTelemetry, when the opentelemetry-api package is installed.

Every monitor's rating opens a span named "rate" and the monitor's class, of kind INTERNAL,
as the current span, so the ratings of a wrapped monitor and the calls of an instrumented model
client nest inside it. The span comes from the global tracer provider's tracer "hyoshigi".
Once the rating has its score, the span carries the score in the attributes of OpenTelemetry's
evaluation result, gen_ai.evaluation.name, gen_ai.evaluation.score.value and
gen_ai.evaluation.explanation (the event gen_ai.evaluation.result, OpenTelemetry semantic
conventions for generative AI, docs/gen-ai/gen-ai-events.md, in development), and whether the
score failed closed, with its cause's class when it did. Langfuse's default export filter keeps
a span with a gen_ai.* attribute and drops one without (langfuse 5.0.2,
langfuse/_client/span_filter.py, is_default_export_span). An exception raised during the
rating marks the span as an error by its type alone, since its message can quote the request,
and is raised on. Without the package, a rating opens no span.
"""

# Tracer and Span are imported only to type, since opentelemetry-api is an optional extra.
from __future__ import annotations

import functools
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, assert_never

from hyoshigi.monitors.score import (
    Composite,
    Decisions,
    FailedClosed,
    Labels,
    Rationale,
    SuspicionScore,
    find_failure,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

# The value of gen_ai.evaluation.name: what a monitor's score measures.
EVALUATION_NAME = "suspicion"


@functools.cache
def find_tracer() -> Tracer | None:
    """The tracer "hyoshigi" of the global tracer provider, or None without opentelemetry-api.

    A tracer taken before the application installs its provider uses that provider once it
    is installed. The answer is kept, since Python searches the path again on each import of
    a module it could not find.
    """
    try:
        # Imported here because opentelemetry-api is the optional extra hyoshigi[opentelemetry].
        from opentelemetry import trace
    except ModuleNotFoundError:
        return None
    return trace.get_tracer("hyoshigi")


def trace_rating(monitor: object) -> AbstractContextManager[Span | None]:
    """Open the span of the monitor's rating as the current span, ended when the block exits.

    The block receives the span, or None without opentelemetry-api.
    """
    tracer = find_tracer()
    if tracer is None:
        return nullcontext()
    return _open_span(
        tracer,
        name=f"rate {type(monitor).__name__}",
    )


@contextmanager
def _open_span(
    tracer: Tracer,
    *,
    name: str,
) -> Generator[Span]:
    """Open the span as the current one, and mark it as an error by the exception's type alone."""
    # Imported here because opentelemetry-api is the optional extra hyoshigi[opentelemetry].
    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(
        name,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as error:
            span.set_status(
                Status(
                    StatusCode.ERROR,
                    type(error).__name__,
                ),
            )
            raise


def record_rating(
    score: SuspicionScore,
    *,
    span: Span | None,
) -> SuspicionScore:
    """Set the score on the rating's span, when there is one, and return the score."""
    if span is not None:
        span.set_attributes(describe_rating(score))
    return score


def describe_rating(score: SuspicionScore) -> dict[str, str | float | bool]:
    """The span attributes of a score; the cause's class only when the score failed closed."""
    attributes: dict[str, str | float | bool] = {
        "gen_ai.evaluation.name": EVALUATION_NAME,
        "gen_ai.evaluation.score.value": score.value,
        "gen_ai.evaluation.explanation": score.explanation,
        "hyoshigi.failed_closed": score.failed_closed,
    }
    cause = name_cause(score)
    if cause is not None:
        attributes["hyoshigi.cause"] = cause
    return attributes


def name_cause(score: SuspicionScore) -> str | None:
    """The class of the cause a score failed closed with, or None when it did not.

    A composite's cause is that of the part it fails closed by, searched depth first, as a
    RepeatedMonitor's explanation is that part's.
    """
    match score.evidence:
        case FailedClosed(cause=cause):
            return type(cause).__name__
        case Composite(parts=parts):
            failure = find_failure(parts)
            return None if failure is None else name_cause(failure)
        case Rationale() | Decisions() | Labels():
            return None
        case unreachable:
            assert_never(unreachable)
