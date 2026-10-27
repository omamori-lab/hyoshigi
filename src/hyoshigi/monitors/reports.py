"""The reports a rating's score can be sent to: Langfuse scores and LangSmith feedback.

Each report sends a score with its explanation as the comment and metadata saying whether it
failed closed, the class of its cause and the models behind it. A report takes the ids its
backend knows the scored step by; ReportedMonitor gives it the ids of its rating's span.
"""

# The SDKs' clients are imported only to type, since each SDK is an optional extra.
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass
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
from hyoshigi.options import check_instance, describe_option_value, read_count
from hyoshigi.trajectory.json_data import JSON

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langsmith import AsyncClient, Client


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangfuseScores:
    """Sends a score to Langfuse as a numeric score on one observation of a trace.

    The client queues the score and sends it in the background (Langfuse.create_score), so
    `send` neither waits for the network nor raises for a failed request, which the client
    logs. `name` is the score's name in Langfuse, a non-blank string. A `client` that is not a
    langfuse.Langfuse raises ConfigurationError when the report is built, as does a blank name.
    """

    client: Langfuse
    name: str = EVALUATION_NAME

    def __post_init__(self) -> None:
        # Imported here because langfuse is the optional extra hyoshigi[langfuse].
        from langfuse import Langfuse

        check_instance(
            self.client,
            expected=Langfuse,
            name="client",
        )
        _check_name(self.name)

    def send(
        self,
        score: SuspicionScore,
        *,
        trace_id: str,
        observation_id: str,
    ) -> None:
        """Queue the score for the observation of the trace, the explanation as its comment.

        The trace id is 32 lower-case hex digits and the observation id 16, as Langfuse
        writes OpenTelemetry's ids.
        """
        self.client.create_score(
            name=self.name,
            value=score.value,
            data_type="NUMERIC",
            comment=score.explanation,
            metadata=describe_score(score),
            trace_id=trace_id,
            observation_id=observation_id,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangSmithFeedback:
    """Sends a score to LangSmith as model feedback on a run, and waits for LangSmith's answer.

    The feedback is posted at once, with no trace id, and leaves the trace's retention as it
    is; the client rounds the score to 4 decimal places and retries as its create_feedback
    does. `project_id` is the run's project id, a UUID string. `stop_after_attempt` is the
    number of attempts create_feedback makes, a positive integer, 10 by default as in
    LangSmith. An error from the client is raised. A `client` that is not a langsmith.Client,
    a blank `name`, a `project_id` that is not a UUID string or a `stop_after_attempt` that is
    not a positive integer raises ConfigurationError when the report is built.

    <https://omamori-lab.github.io/hyoshigi/how-to/reporting-scores-to-langfuse-and-langsmith/>
    """

    client: Client
    project_id: str
    name: str = EVALUATION_NAME
    stop_after_attempt: int = 10
    executor: Executor | None = None

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
        read_count(
            self.stop_after_attempt,
            name="stop_after_attempt",
            minimum=1,
        )

    def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Post the score as feedback on the run, a UUID string, the explanation as its comment."""
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        self.client.create_feedback(
            run_id,
            key=self.name,
            score=score.value,
            comment=score.explanation,
            feedback_source_type=FeedbackSourceType.MODEL,
            source_info=describe_score(score),
            session_id=self.project_id,
            extend_trace_retention=False,
            stop_after_attempt=self.stop_after_attempt,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncLangSmithFeedback:
    """Sends a score to LangSmith as LangSmithFeedback does, through an awaited client.

    Its fields other than `client`, and their errors, are those of LangSmithFeedback.
    """

    client: AsyncClient
    project_id: str
    name: str = EVALUATION_NAME
    stop_after_attempt: int = 10
    task_group: asyncio.TaskGroup | None = None

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
        read_count(
            self.stop_after_attempt,
            name="stop_after_attempt",
            minimum=1,
        )

    async def send(
        self,
        score: SuspicionScore,
        *,
        run_id: str,
    ) -> None:
        """Post the score as feedback on the run, a UUID string, the explanation as its comment."""
        # Imported here because langsmith is the optional extra hyoshigi[langsmith].
        from langsmith.schemas import FeedbackSourceType

        await self.client.create_feedback(
            run_id,
            key=self.name,
            score=score.value,
            comment=score.explanation,
            feedback_source_type=FeedbackSourceType.MODEL,
            source_info=describe_score(score),
            session_id=self.project_id,
            extend_trace_retention=False,
            stop_after_attempt=self.stop_after_attempt,
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
