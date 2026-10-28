"""The reports a rating's score can be sent to: Langfuse scores and LangSmith feedback.

Each report sends a score with its explanation as the comment, through the report's redaction
when it has one, and metadata saying whether it failed closed, the class of its cause and the
models behind it. A report built with a ReviewFlag also marks the step for review, with a
boolean score or feedback that carries no comment. A report takes the ids its backend knows
the scored step by; ReportedMonitor gives it the ids of its rating's span.
"""

# The SDKs' clients are imported only to type, since each SDK is an optional extra.
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, Executor, Future
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, assert_never

from hyoshigi.errors import ConfigurationError
from hyoshigi.monitors.score import (
    Composite,
    Decisions,
    FailedClosed,
    Labels,
    Rationale,
    SuspicionScore,
)
from hyoshigi.monitors.tracing import EVALUATION_NAME, name_cause
from hyoshigi.options import check_instance, describe_option_value, read_count, read_threshold
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.redaction import Redaction, Redactor, check_redaction

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langsmith import AsyncClient, Client

# ReportedMonitor's logger, which also takes the failures of the sends on a worker.
logger = logging.getLogger("hyoshigi.monitors.reporting")


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class ReviewFlag:
    """The mark a report puts on a step for a person to review, and the score that earns it.

    `name` is the mark's name in the backend, a non-blank string. A report's `send` flags the
    step when the score is at or above `threshold`, a number from 0 to 1 kept as a float; with
    no threshold, `send` never flags and only the report's `flag` does. A blank name, and a
    threshold that is not a number from 0 to 1, raise ConfigurationError when the flag is
    built.

    <https://omamori-lab.github.io/hyoshigi/how-to/reporting-scores-to-langfuse-and-langsmith/>
    """

    name: str = "flagged"
    threshold: float | None = None

    def __post_init__(self) -> None:
        _check_name(self.name)
        if self.threshold is not None:
            # The dataclass is frozen, so the float is set past its guard.
            object.__setattr__(
                self,
                "threshold",
                read_threshold(
                    self.threshold,
                    name="threshold",
                ),
            )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangfuseScores:
    """Sends a score to Langfuse as a numeric score on one observation of a trace.

    The client queues the score and sends it in the background (Langfuse.create_score), so
    `send` neither waits for the network nor raises for a failed request, which the client
    logs. `name` is the score's name in Langfuse, a non-blank string. The comment is the
    explanation through `redaction`, a Redaction, when one is given. With a `review_flag`, a
    ReviewFlag, `send` first queues a boolean score named by it on the same observation when the
    score reaches its threshold, and `flag` queues it for any step. A `client` that is not a
    langfuse.Langfuse raises ConfigurationError when the report is built, as do a blank name, a
    review flag that is not a ReviewFlag and a redaction that is not a Redaction.
    """

    client: Langfuse
    name: str = EVALUATION_NAME
    review_flag: ReviewFlag | None = None
    redaction: Redaction | None = None

    def __post_init__(self) -> None:
        # Imported here because langfuse is the optional extra hyoshigi[langfuse].
        from langfuse import Langfuse

        check_instance(
            self.client,
            expected=Langfuse,
            name="client",
        )
        _check_name(self.name)
        _check_review_flag(self.review_flag)
        check_redaction(self.redaction)

    def send(
        self,
        score: SuspicionScore,
        *,
        trace_id: str,
        observation_id: str,
    ) -> None:
        """Queue the score for the observation of the trace, the explanation as its comment.

        The trace id is 32 lower-case hex digits and the observation id 16, as Langfuse
        writes OpenTelemetry's ids. When the score reaches the review flag's threshold, the
        flag is queued first.
        """
        if _reaches_threshold(
            self.review_flag,
            score=score,
        ):
            self.flag(
                trace_id=trace_id,
                observation_id=observation_id,
            )
        self.client.create_score(
            name=self.name,
            value=score.value,
            data_type="NUMERIC",
            comment=Redactor(self.redaction).redact_text(score.explanation),
            metadata=describe_score(score),
            trace_id=trace_id,
            observation_id=observation_id,
        )

    def flag(
        self,
        *,
        trace_id: str,
        observation_id: str,
    ) -> None:
        """Queue a boolean score named by the review flag on the observation of the trace, with no
        comment; raises ConfigurationError when the report has no `review_flag`.
        """
        review_flag = _require_review_flag(self.review_flag)
        self.client.create_score(
            name=review_flag.name,
            value=1,
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangSmithFeedback:
    """Sends a score to LangSmith as model feedback on a run, on the caller's executor if given.

    The feedback is posted with no trace id, and leaves the trace's retention as it is; the
    client rounds the score to 4 decimal places and retries as its create_feedback does.
    Without an `executor`, `send` posts and returns once LangSmith answers, and an error from
    the client is raised. With one, `send` submits the post to it and returns at once; a post
    that fails or is cancelled is logged by the report's class and the error's type, and never
    raised, and `executor.shutdown(wait=True)` waits for the posts left. The caller bounds how
    long the interpreter's exit waits for them: a ThreadPoolExecutor's threads are not daemons,
    and a post that LangSmith answers with 404 holds the exit for up to about 511 s at
    `stop_after_attempt=10`. `project_id` is the run's project id, a UUID string.
    `stop_after_attempt` is the number of attempts create_feedback makes, a positive integer,
    10 by default as in LangSmith. The comment is the explanation through `redaction`, a
    Redaction, when one is given. With a `review_flag`, a ReviewFlag, `send` first posts
    boolean feedback named by it on the same run when the score reaches its threshold, and
    `flag` posts it for any step, on the executor if given as `send` does; a post that fails
    is logged as a flag. A `client` that is not a langsmith.Client, a blank `name`, a
    `project_id` that is not a UUID string, a `review_flag` that is not a ReviewFlag, a
    `stop_after_attempt` that is not a positive integer, an `executor` that is not a
    concurrent.futures.Executor or a `redaction` that is not a Redaction raises
    ConfigurationError when the report is built.

    <https://omamori-lab.github.io/hyoshigi/how-to/reporting-scores-to-langfuse-and-langsmith/>
    """

    client: Client
    project_id: str
    name: str = EVALUATION_NAME
    review_flag: ReviewFlag | None = None
    stop_after_attempt: int = 10
    executor: Executor | None = None
    redaction: Redaction | None = None

    def __post_init__(self) -> None:
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith import Client

        check_instance(
            self.client,
            expected=Client,
            name="client",
            hint="Use AsyncLangSmithFeedback for an AsyncClient.",
        )
        _check_name(self.name)
        _check_project_id(self.project_id)
        _check_review_flag(self.review_flag)
        read_count(
            self.stop_after_attempt,
            name="stop_after_attempt",
            minimum=1,
        )
        check_redaction(self.redaction)
        if self.executor is not None:
            check_instance(
                self.executor,
                expected=Executor,
                name="executor",
                hint="Use AsyncLangSmithFeedback for a TaskGroup.",
            )

    def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Post the score as feedback on the run, a UUID string, the explanation as its comment;
        with an executor, submit the post to it. When the score reaches the review flag's
        threshold, the flag is posted first.
        """
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        if _reaches_threshold(
            self.review_flag,
            score=score,
        ):
            self.flag(run_id=run_id)
        self._submit(
            partial(
                self.client.create_feedback,
                run_id,
                key=self.name,
                score=score.value,
                comment=Redactor(self.redaction).redact_text(score.explanation),
                feedback_source_type=FeedbackSourceType.MODEL,
                source_info=describe_score(score),
                session_id=self.project_id,
                extend_trace_retention=False,
                stop_after_attempt=self.stop_after_attempt,
            ),
            noun="score",
        )

    def flag(
        self,
        *,
        run_id: str,
    ) -> None:
        """Post the review flag as boolean feedback on the run, a UUID string, with no comment;
        with an executor, submit the post to it. Raises ConfigurationError when the report has no
        `review_flag`.
        """
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        review_flag = _require_review_flag(self.review_flag)
        self._submit(
            partial(
                self.client.create_feedback,
                run_id,
                key=review_flag.name,
                score=True,
                feedback_source_type=FeedbackSourceType.MODEL,
                session_id=self.project_id,
                extend_trace_retention=False,
                stop_after_attempt=self.stop_after_attempt,
            ),
            noun="flag",
        )

    def _submit(
        self,
        post: Callable[[], object],
        *,
        noun: str,
    ) -> None:
        """Run the post, or submit it to the executor, where a failure is logged as the noun."""
        if self.executor is None:
            post()
        else:
            self.executor.submit(post).add_done_callback(
                partial(
                    _log_failed_send,
                    report=self,
                    noun=noun,
                ),
            )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncLangSmithFeedback:
    """Sends a score to LangSmith as LangSmithFeedback does, through an awaited client, in the
    caller's task group if given.

    Without a `task_group`, `send` awaits the post, and an error from the client is raised.
    With one, `send` creates a task for the post in the group and returns at once; a post that
    fails, or is cancelled with the group, is logged by the report's class and the error's
    type, and never raised, and leaving the group awaits the posts left. Its other fields, and
    their errors, are those of LangSmithFeedback, with an AsyncClient for `client`, and `flag`
    is awaited. A `task_group` that is not an asyncio.TaskGroup raises ConfigurationError when
    the report is built.
    """

    client: AsyncClient
    project_id: str
    name: str = EVALUATION_NAME
    review_flag: ReviewFlag | None = None
    stop_after_attempt: int = 10
    task_group: asyncio.TaskGroup | None = None
    redaction: Redaction | None = None

    def __post_init__(self) -> None:
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith import AsyncClient

        check_instance(
            self.client,
            expected=AsyncClient,
            name="client",
            hint="Use LangSmithFeedback for a Client.",
        )
        _check_name(self.name)
        _check_project_id(self.project_id)
        _check_review_flag(self.review_flag)
        read_count(
            self.stop_after_attempt,
            name="stop_after_attempt",
            minimum=1,
        )
        check_redaction(self.redaction)
        if self.task_group is not None:
            check_instance(
                self.task_group,
                expected=asyncio.TaskGroup,
                name="task_group",
                hint="Use LangSmithFeedback for an Executor.",
            )

    async def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Post the score as feedback on the run, a UUID string, the explanation as its comment;
        with a task group, create a task for the post in it. When the score reaches the review
        flag's threshold, the flag is posted first.
        """
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        if _reaches_threshold(
            self.review_flag,
            score=score,
        ):
            await self.flag(run_id=run_id)
        await self._submit(
            partial(
                self.client.create_feedback,
                run_id,
                key=self.name,
                score=score.value,
                comment=Redactor(self.redaction).redact_text(score.explanation),
                feedback_source_type=FeedbackSourceType.MODEL,
                source_info=describe_score(score),
                session_id=self.project_id,
                extend_trace_retention=False,
                stop_after_attempt=self.stop_after_attempt,
            ),
            noun="score",
        )

    async def flag(
        self,
        *,
        run_id: str,
    ) -> None:
        """Post the review flag as boolean feedback on the run, a UUID string, with no comment;
        with a task group, create a task for the post in it. Raises ConfigurationError when the
        report has no `review_flag`.
        """
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        review_flag = _require_review_flag(self.review_flag)
        await self._submit(
            partial(
                self.client.create_feedback,
                run_id,
                key=review_flag.name,
                score=True,
                feedback_source_type=FeedbackSourceType.MODEL,
                session_id=self.project_id,
                extend_trace_retention=False,
                stop_after_attempt=self.stop_after_attempt,
            ),
            noun="flag",
        )

    async def _submit(
        self,
        post: Callable[[], Awaitable[object]],
        *,
        noun: str,
    ) -> None:
        """Await the post, or create a task for it in the task group, where a failure is logged
        as the noun.
        """
        if self.task_group is None:
            await post()
        else:
            self.task_group.create_task(
                _await_logging_failure(
                    post,
                    report=self,
                    noun=noun,
                ),
            ).add_done_callback(
                partial(
                    _log_failed_send,
                    report=self,
                    noun=noun,
                ),
            )


async def _await_logging_failure(
    post: Callable[[], Awaitable[object]],
    *,
    report: object,
    noun: str,
) -> None:
    """Await the post; an Exception from it is logged in place of being raised."""
    # Any error: a task that raised would cancel the other tasks of the caller's group.
    try:
        await post()
    except Exception as error:
        log_failure(
            report,
            error=error,
            noun=noun,
        )


def _log_failed_send[Result](
    send: Future[Result] | asyncio.Future[Result],
    *,
    report: object,
    noun: str,
) -> None:
    """Log a send that was cancelled or raised, as log_failure does; a CancelledError for one
    that was cancelled.
    """
    error = CancelledError() if send.cancelled() else send.exception()
    if error is not None:
        log_failure(
            report,
            error=error,
            noun=noun,
        )


def log_failure(
    report: object,
    *,
    error: BaseException,
    noun: str = "score",
) -> None:
    """Log that a report did not send the noun, "score" or "flag", by the report's class and
    the error's type.

    <https://omamori-lab.github.io/hyoshigi/explanation/observability/>
    """
    logger.warning(
        "%s did not send the %s: %s.",
        type(report).__name__,
        noun,
        type(error).__name__,
    )


def _check_name(name: object) -> None:
    """Raise ConfigurationError unless the name is a string with a character not blank."""
    if (
        not isinstance(
            name,
            str,
        )
        or not name.strip()
    ):
        raise ConfigurationError(
            f"name must be a non-blank string, got {describe_option_value(name)}",
        )


def _check_review_flag(review_flag: object) -> None:
    """Raise ConfigurationError unless the review flag is a ReviewFlag or None."""
    if review_flag is not None:
        check_instance(
            review_flag,
            expected=ReviewFlag,
            name="review_flag",
        )


def _require_review_flag(review_flag: ReviewFlag | None) -> ReviewFlag:
    """The review flag; raises ConfigurationError when the report has none."""
    if review_flag is None:
        raise ConfigurationError(
            "flag needs a ReviewFlag: build the report with review_flag=ReviewFlag(...)",
        )
    return review_flag


def _reaches_threshold(
    review_flag: ReviewFlag | None,
    *,
    score: SuspicionScore,
) -> bool:
    """Tell whether the score is at or above the review flag's threshold; false with no
    threshold.
    """
    return (
        review_flag is not None
        and review_flag.threshold is not None
        and score.value >= review_flag.threshold
    )


def _check_project_id(project_id: object) -> None:
    """Raise ConfigurationError unless the project id is a string that reads as a UUID."""
    if isinstance(
        project_id,
        str,
    ):
        try:
            uuid.UUID(project_id)
        except ValueError:
            pass
        else:
            return
    raise ConfigurationError(
        "project_id must be a UUID string, such as a LangSmith project's id, "
        f"got {describe_option_value(project_id)}",
    )


def describe_score(score: SuspicionScore) -> dict[str, JSON]:
    """The metadata sent with a score: whether it failed closed, the cause's class, the models."""
    return {
        "failed_closed": score.failed_closed,
        "cause": name_cause(score),
        "model": name_models(score),
    }


def name_models(score: SuspicionScore) -> str | list[JSON]:
    """The model the evidence names; for a composite, its parts' models, nested parts included,
    in order without repeats.
    """
    match score.evidence:
        case Composite(parts=parts):
            models: list[JSON] = []
            for part in parts:
                named = name_models(part)
                models.extend(
                    [named]
                    if isinstance(
                        named,
                        str,
                    )
                    else named,
                )
            return list(dict.fromkeys(models))
        case (
            Rationale(model=model)
            | Decisions(model=model)
            | Labels(model=model)
            | FailedClosed(model=model)
        ):
            return model
        case unreachable:
            assert_never(unreachable)
