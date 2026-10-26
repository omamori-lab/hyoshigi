"""The monitor that sends its rating's score to Langfuse and LangSmith, for the rating's span.

ReportedMonitor rates with the monitor it wraps inside a span of its own and, once the span has
ended, sends the score for that span: to Langfuse the span's trace id as 32 lower-case hex
digits and its span id as 16, to LangSmith the run id that holds the span id's 8 bytes
right-aligned in a 16-byte UUID.

A score is sent only when the span's context is valid and sampled; otherwise a log record says
nothing was sent. A report that raises is logged by the report's class and the exception's
type, the other reports are sent, and the score is returned unchanged.

<https://omamori-lab.github.io/hyoshigi/explanation/observability/>
"""

# OpenTelemetry's classes are imported only to type, since it is an optional extra.
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.reports import AsyncLangSmithFeedback, LangfuseScores, LangSmithFeedback
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.monitors.tracing import find_tracer, record_rating, trace_rating
from hyoshigi.options import check_instance, describe_type, name_class
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanContext

logger = logging.getLogger(__name__)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ReportedMonitor(Monitor):
    """Rates with a monitor inside a span of its own, then sends the score to each report.

    The span is named "rate ReportedMonitor" and holds the wrapped monitor's. Once it ends,
    the score goes to each report in turn: to Langfuse on the span's trace and observation, to
    LangSmith on the span's run. The score returned is the wrapped monitor's, whatever the
    reports do. A `monitor` that is not a Monitor, `reports` that are not a non-empty tuple of
    LangfuseScores and LangSmithFeedback, and a process without opentelemetry-api raise
    ConfigurationError when the monitor is built.
    """

    monitor: Monitor
    reports: tuple[LangfuseScores | LangSmithFeedback, ...]

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=Monitor,
            name="monitor",
            hint="Use AsyncReportedMonitor for an AsyncMonitor.",
        )
        _check_reports(
            self.reports,
            family=(
                LangfuseScores,
                LangSmithFeedback,
            ),
            hint="Use AsyncReportedMonitor for an AsyncLangSmithFeedback.",
        )
        _check_tracing(self)

    def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the wrapped monitor, then send the score to each report."""
        with trace_rating(self) as span:
            score = record_rating(
                self.monitor.rate(
                    action,
                    trajectory=trajectory,
                ),
                span=span,
            )
            recorded = span is not None and span.is_recording()
        context = read_sendable_context(
            span,
            recorded=recorded,
        )
        if context is not None:
            send_reports(
                score,
                context=context,
                reports=self.reports,
            )
        return score


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncReportedMonitor(AsyncMonitor):
    """Rates and reports as ReportedMonitor does, with the monitor and LangSmith awaited.

    `reports` is a non-empty tuple of LangfuseScores and AsyncLangSmithFeedback, sent in turn.
    """

    monitor: AsyncMonitor
    reports: tuple[LangfuseScores | AsyncLangSmithFeedback, ...]

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=AsyncMonitor,
            name="monitor",
            hint="Use ReportedMonitor for a Monitor.",
        )
        _check_reports(
            self.reports,
            family=(
                LangfuseScores,
                AsyncLangSmithFeedback,
            ),
            hint="Use ReportedMonitor for a LangSmithFeedback.",
        )
        _check_tracing(self)

    async def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the wrapped monitor, then send the score to each report."""
        with trace_rating(self) as span:
            score = record_rating(
                await self.monitor.rate(
                    action,
                    trajectory=trajectory,
                ),
                span=span,
            )
            recorded = span is not None and span.is_recording()
        context = read_sendable_context(
            span,
            recorded=recorded,
        )
        if context is not None:
            await await_reports(
                score,
                context=context,
                reports=self.reports,
            )
        return score


def _check_reports(
    reports: object,
    *,
    family: tuple[type, ...],
    hint: str,
) -> None:
    """Raise ConfigurationError unless the reports are a non-empty tuple of the family's
    reports; the hint names the twin monitor.
    """
    if not isinstance(
        reports,
        tuple,
    ):
        raise ConfigurationError(f"reports must be a tuple, got {describe_type(reports)}")
    if not reports:
        raise ConfigurationError("reports must hold at least one report")
    kinds = " or ".join(name_class(kind) for kind in family)
    for position, report in enumerate(reports):
        if not isinstance(
            report,
            family,
        ):
            raise ConfigurationError(
                f"reports[{position}] must be {kinds}, got {describe_type(report)}. {hint}",
            )


def _check_tracing(monitor: ReportedMonitor | AsyncReportedMonitor) -> None:
    """Raise ConfigurationError when opentelemetry-api is not installed, naming the extra."""
    if find_tracer() is None:
        raise ConfigurationError(
            f"{type(monitor).__name__} needs OpenTelemetry: pip install 'hyoshigi[opentelemetry]'",
        )


def read_sendable_context(
    span: Span | None,
    *,
    recorded: bool,
) -> SpanContext | None:
    """The span's context when a backend can hold the span; otherwise log that nothing was sent.

    An unsampled span is a sampler's decision. Any other span that was not recorded means no
    tracer provider is set up, which is a misconfiguration, even when the span carries the
    sampled context of a parent from another process.
    """
    context = _read_valid_context(span)
    if context is not None and not context.trace_flags.sampled:
        logger.debug("No report was sent: the rating's span is not sampled.")
        return None
    if context is None or not recorded:
        logger.warning(
            "No report was sent: no OpenTelemetry tracer provider recorded the rating's span.",
        )
        return None
    return context


def _read_valid_context(span: Span | None) -> SpanContext | None:
    """The span's context, or None when there is no span or its context is not valid."""
    if span is None:
        return None
    context = span.get_span_context()
    return context if context.is_valid else None


def send_reports(
    score: SuspicionScore,
    *,
    context: SpanContext,
    reports: tuple[LangfuseScores | LangSmithFeedback, ...],
) -> None:
    """Send the score to each report in turn for the span; a report that raises is logged."""
    for report in reports:
        # Any error: reporting never loses the score or the reports after it.
        try:
            match report:
                case LangfuseScores():
                    send_to_langfuse(
                        report,
                        score=score,
                        context=context,
                    )
                case LangSmithFeedback():
                    report.send(
                        score,
                        run_id=derive_run_id(context),
                    )
                case unreachable:
                    assert_never(unreachable)
        except Exception as error:
            log_failure(
                report,
                error=error,
            )


async def await_reports(
    score: SuspicionScore,
    *,
    context: SpanContext,
    reports: tuple[LangfuseScores | AsyncLangSmithFeedback, ...],
) -> None:
    """Send the score to each report in turn for the span, awaiting LangSmith; a report that
    raises is logged.
    """
    for report in reports:
        # Any error: reporting never loses the score or the reports after it.
        try:
            match report:
                case LangfuseScores():
                    send_to_langfuse(
                        report,
                        score=score,
                        context=context,
                    )
                case AsyncLangSmithFeedback():
                    await report.send(
                        score,
                        run_id=derive_run_id(context),
                    )
                case unreachable:
                    assert_never(unreachable)
        except Exception as error:
            log_failure(
                report,
                error=error,
            )


def send_to_langfuse(
    report: LangfuseScores,
    *,
    score: SuspicionScore,
    context: SpanContext,
) -> None:
    """Send the score to Langfuse with the span's trace and span ids as lower-case hex."""
    report.send(
        score,
        trace_id=format(
            context.trace_id,
            "032x",
        ),
        observation_id=format(
            context.span_id,
            "016x",
        ),
    )


def derive_run_id(context: SpanContext) -> str:
    """The id of the run LangSmith ingests the span as."""
    # Imported here because langsmith is the optional extra hyoshigi[langsmith].
    from langsmith.integrations.otel import langsmith_run_id_from_otel_span_id

    return str(langsmith_run_id_from_otel_span_id(context.span_id))


def log_failure(
    report: object,
    *,
    error: Exception,
) -> None:
    """Log that a report did not send the score, by the report's class and the error's type
    alone: an error's message can quote the score's explanation or the request.
    """
    logger.warning(
        "%s did not send the score: %s.",
        type(report).__name__,
        type(error).__name__,
    )
