"""Reports of a rating to Langfuse and LangSmith: the request on the wire, and when none is sent."""

import asyncio
import contextlib
import json
import logging
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import signature
from typing import assert_never, cast
from unittest.mock import ANY

import httpx
import pytest
from langfuse import Langfuse
from langsmith import AsyncClient, Client
from langsmith.utils import LangSmithAuthError
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from hyoshigi import (
    JSON,
    Action,
    AsyncCalibratedMonitor,
    AsyncCascadeMonitor,
    AsyncGenerativeModel,
    AsyncLangSmithFeedback,
    AsyncLLMMonitor,
    AsyncMonitor,
    AsyncRepeatedMonitor,
    AsyncReportedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    ConfigurationError,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    LangfuseScores,
    LangSmithFeedback,
    LLMMonitor,
    Monitor,
    Prompt,
    Rationale,
    Redaction,
    Refused,
    RepeatedMonitor,
    ReportedMonitor,
    ReviewFlag,
    SuspicionScore,
    TextPart,
    Trajectory,
    Unavailable,
)

MODEL_NAME = "trusted-1"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))
PROJECT_ID = "6f9a2c4e-1b3d-4e5f-8a7b-9c0d1e2f3a4b"
# The span ReportedMonitor opens is a test's first, so it has the trace id and the first span id
# of the test run's tracer provider (tests/conftest.py): 0x00ABCDEF0123456789ABCDEF01234567
# and 0x00F1E2D3C4B5A697. Langfuse takes them as lower-case hex of 32 and 16 digits; LangSmith
# derives a run id from the span id, its 8 bytes right-aligned in a 16-byte UUID.
TRACE_HEX = "00abcdef0123456789abcdef01234567"
SPAN_HEX = "00f1e2d3c4b5a697"
RUN_ID = "00000000-0000-0000-00f1-e2d3c4b5a697"
CALM_METADATA = {
    "failed_closed": False,
    "cause": None,
    "model": MODEL_NAME,
}
# The body of POST /feedback (langsmith.schemas, FeedbackCreate, without its None fields) for a
# calm rating; the id and the times differ in every run.
CALM_FEEDBACK = {
    "id": ANY,
    "created_at": ANY,
    "modified_at": ANY,
    "run_id": RUN_ID,
    "key": "suspicion",
    "score": 0.3,
    "comment": "calm",
    "feedback_source": {
        "type": "model",
        "metadata": CALM_METADATA,
    },
    "session_id": PROJECT_ID,
    "extend_trace_retention": False,
}
CALM_ATTRIBUTES = {
    "gen_ai.evaluation.name": "suspicion",
    "gen_ai.evaluation.score.value": 0.3,
    "gen_ai.evaluation.explanation": "calm",
    "hyoshigi.failed_closed": False,
}
# A span context from another process whose trace is not sampled.
UNSAMPLED_PARENT = NonRecordingSpan(
    SpanContext(
        trace_id=0x0123456789ABCDEF0123456789ABCDEF,
        span_id=0x0123456789ABCDEF,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.DEFAULT),
    ),
)
UNUSED_URL = "http://127.0.0.1:9"
# How long the stand-in LangSmith server holds an answer that is not released: less than the
# clients' read timeouts, 10 s for AsyncClient and 60 s for Client, so a rating that waits for the
# answer gets it.
ANSWER_HOLD_SECONDS = 5
# Each digit masked; media blurred to new bytes.
MASKED = Redaction(
    text=partial(
        re.sub,
        "[0-9]",
        "#",
    ),
    media=partial(
        replace,
        data=b"blurred",
    ),
)
DIALLING_REPLY = Generation(text="<reasoning>It dials 555-0100.</reasoning><score>3</score>")
# A review flag whose threshold the score of FLAGGED_SCORE reaches exactly.
FLAGGING = ReviewFlag(threshold=0.6)
FLAGGED_SCORE = SuspicionScore(
    value=0.6,
    explanation="calm",
    evidence=Rationale(model=MODEL_NAME),
)


class Script:
    """The outcomes a scripted model returns in order."""

    def __init__(
        self,
        *outcomes: GenerationOutcome,
    ) -> None:
        self.outcomes = list(outcomes)

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.outcomes.pop(0)

    async def generate_asynchronously(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.generate(prompt)


def build_reply(score: int) -> Generation:
    return Generation(text=f"<reasoning>calm</reasoning><score>{score}</score>")


class LangfuseWire:
    """Langfuse's end of the connection: records the scores of each request, accepts them all."""

    def __init__(self) -> None:
        self.scores: list[dict[str, JSON]] = []

    def respond(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.scores.extend(json.loads(request.content))
        # The API accepts a batch with 202 (langfuse.api.scores, CreateScoreBatchResponse).
        return httpx.Response(
            202,
            json={"message": "Accepted"},
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LangfuseStandIn:
    """A Langfuse client on a stand-in wire."""

    client: Langfuse
    wire: LangfuseWire

    def read_scores(self) -> list[dict[str, JSON]]:
        """The scores the wire received, once the client has sent every queued score."""
        self.client.flush()
        return self.wire.scores


@pytest.fixture(scope="module")
def langfuse_stand_in() -> Iterator[LangfuseStandIn]:
    wire = LangfuseWire()
    client = Langfuse(
        # Langfuse keeps one client's resources per public key in a process, so a fresh key
        # gives this module its own wire.
        public_key=f"pk-lf-{uuid.uuid4()}",
        secret_key="sk-lf-test",
        base_url="https://langfuse.test",
        httpx_client=httpx.Client(transport=httpx.MockTransport(wire.respond)),
        # A score is sent as soon as it is queued, not in batches of 15.
        flush_at=1,
        # Langfuse's own provider and exporter, so its span processor neither joins the test
        # run's global provider nor exports over the network.
        tracer_provider=TracerProvider(),
        span_exporter=InMemorySpanExporter(),
    )
    yield LangfuseStandIn(
        client=client,
        wire=wire,
    )
    client.shutdown()


@pytest.fixture
def langfuse(langfuse_stand_in: LangfuseStandIn) -> LangfuseStandIn:
    """The module's Langfuse stand-in, with no score received yet.

    The client sends what an earlier test queued before the wire is cleared, so no score of
    that test reaches this one.
    """
    langfuse_stand_in.client.flush()
    langfuse_stand_in.wire.scores.clear()
    return langfuse_stand_in


class LangSmithServer(ThreadingHTTPServer):
    """LangSmith's end of the connection, on 127.0.0.1: records each request's path and JSON
    body, and answers with an empty JSON object and the next of `first_statuses`, then `status`.

    An answer waits until `release` is set, for at most ANSWER_HOLD_SECONDS, and its status goes
    into `answered` before it is sent.
    """

    def __init__(self) -> None:
        super().__init__(
            ("127.0.0.1", 0),
            FeedbackHandler,
        )
        self.received: list[tuple[str, JSON]] = []
        self.first_statuses: list[int] = []
        self.status = 200
        self.release = threading.Event()
        self.answered: list[int] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class FeedbackHandler(BaseHTTPRequestHandler):
    """Records a POST on the server and answers it."""

    def do_POST(self) -> None:
        server = cast(
            LangSmithServer,
            self.server,
        )
        body = self.rfile.read(int(self.headers["Content-Length"]))
        server.received.append(
            (
                self.path,
                json.loads(body),
            ),
        )
        server.release.wait(ANSWER_HOLD_SECONDS)
        status = server.first_statuses.pop(0) if server.first_statuses else server.status
        server.answered.append(status)
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            "2",
        )
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(
        self,
        format: str,
        *args: object,
    ) -> None:
        """Log nothing: the base class writes each request to standard error."""


@pytest.fixture(scope="session")
def langsmith_server() -> Iterator[LangSmithServer]:
    # LangSmith's AsyncClient builds its own HTTP client and takes no transport, so both
    # clients reach a stand-in server through their public api_url.
    server = LangSmithServer()
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()
    yield server
    server.shutdown()


@pytest.fixture
def langsmith(langsmith_server: LangSmithServer) -> LangSmithServer:
    """The stand-in server, with no request received or answered yet, answering 200 at once."""
    langsmith_server.received.clear()
    langsmith_server.first_statuses.clear()
    langsmith_server.status = 200
    langsmith_server.release.set()
    langsmith_server.answered.clear()
    return langsmith_server


def build_client(url: str = UNUSED_URL) -> Client:
    return Client(
        api_url=url,
        api_key="lsv2-test",
        # The server's details, given so the client does not ask for them.
        info={},
        auto_batch_tracing=False,
    )


def build_async_client(url: str = UNUSED_URL) -> AsyncClient:
    return AsyncClient(
        api_url=url,
        api_key="lsv2-test",
    )


def build_feedback(url: str = UNUSED_URL) -> LangSmithFeedback:
    return LangSmithFeedback(
        client=build_client(url),
        project_id=PROJECT_ID,
    )


def build_async_feedback(url: str = UNUSED_URL) -> AsyncLangSmithFeedback:
    return AsyncLangSmithFeedback(
        client=build_async_client(url),
        project_id=PROJECT_ID,
    )


def build_trajectory() -> Trajectory:
    return Trajectory.from_run(
        "Summarise q3.md",
        answer="Done.",
    ).trajectory


def rate(monitor: Monitor | AsyncMonitor) -> SuspicionScore:
    """Rate the action against a fresh trajectory, awaiting an async monitor on a new loop."""
    trajectory = build_trajectory()
    match monitor:
        case Monitor():
            return monitor.rate(
                ACTION,
                trajectory=trajectory,
            )
        case AsyncMonitor():
            return asyncio.run(
                monitor.rate(
                    ACTION,
                    trajectory=trajectory,
                ),
            )
        case unreachable:
            assert_never(unreachable)


def send_score(
    report: LangfuseScores | LangSmithFeedback | AsyncLangSmithFeedback,
    score: SuspicionScore,
) -> None:
    """Send the score for the step TRACE_HEX, SPAN_HEX and RUN_ID name, awaiting an async report
    on a new loop.
    """
    match report:
        case LangfuseScores():
            report.send(
                score,
                trace_id=TRACE_HEX,
                observation_id=SPAN_HEX,
            )
        case LangSmithFeedback():
            report.send(
                score,
                run_id=RUN_ID,
            )
        case AsyncLangSmithFeedback():
            asyncio.run(
                report.send(
                    score,
                    run_id=RUN_ID,
                ),
            )
        case unreachable:
            assert_never(unreachable)


def flag_step(report: LangfuseScores | LangSmithFeedback | AsyncLangSmithFeedback) -> None:
    """Flag the step TRACE_HEX, SPAN_HEX and RUN_ID name, awaiting an async report on a new loop."""
    match report:
        case LangfuseScores():
            report.flag(
                trace_id=TRACE_HEX,
                observation_id=SPAN_HEX,
            )
        case LangSmithFeedback():
            report.flag(run_id=RUN_ID)
        case AsyncLangSmithFeedback():
            asyncio.run(report.flag(run_id=RUN_ID))
        case unreachable:
            assert_never(unreachable)


def build_score(value: float) -> SuspicionScore:
    return SuspicionScore(
        value=value,
        explanation="calm",
        evidence=Rationale(model=MODEL_NAME),
    )


def read_feedback_field(
    langsmith: LangSmithServer,
    field: str,
) -> list[JSON]:
    """The field of each feedback body LangSmith received, in the order received."""
    return [
        body[field]
        for _, body in langsmith.received
        if isinstance(
            body,
            dict,
        )
    ]


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class WorkerRating:
    """A rating whose LangSmith report sent on a worker: its score, and the statuses LangSmith had
    answered when it returned.
    """

    score: SuspicionScore
    answered: list[int]


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Family[Rater]:
    """The monitors and the LangSmith report of one family, and its ratings with a worker."""

    llm: Callable[..., Rater]
    repeated: Callable[..., Rater]
    calibrated: Callable[..., Rater]
    cascade: Callable[..., Rater]
    reported: Callable[..., Rater]
    feedback: Callable[[str], LangSmithFeedback | AsyncLangSmithFeedback]
    rate_with_a_worker: Callable[[LangSmithServer], WorkerRating]
    rate_and_cancel_the_send: Callable[[LangSmithServer], None]
    send_flagged_with_a_worker: Callable[[LangSmithServer], list[int]]
    flag_with_a_worker: Callable[[LangSmithServer], None]


def build_synchronous_llm(
    script: Script,
    *,
    name: str = MODEL_NAME,
) -> LLMMonitor:
    return LLMMonitor(
        model=GenerativeModel(
            name=name,
            generate=script.generate,
        ),
    )


def build_asynchronous_llm(
    script: Script,
    *,
    name: str = MODEL_NAME,
) -> AsyncLLMMonitor:
    return AsyncLLMMonitor(
        model=AsyncGenerativeModel(
            name=name,
            generate=script.generate_asynchronously,
        ),
    )


def rate_on_an_executor(langsmith: LangSmithServer) -> WorkerRating:
    """Rate calmly with the LangSmith report posting on an executor, release LangSmith's answers
    once the rating returns, then shut the executor down.
    """
    with ThreadPoolExecutor() as executor:
        monitor = ReportedMonitor(
            monitor=build_synchronous_llm(Script(build_reply(3))),
            reports=(
                replace(
                    build_feedback(langsmith.url),
                    executor=executor,
                ),
            ),
        )
        score = rate(monitor)
        answered = list(langsmith.answered)
        langsmith.release.set()
    return WorkerRating(
        score=score,
        answered=answered,
    )


def rate_in_a_task_group(langsmith: LangSmithServer) -> WorkerRating:
    """Rate calmly with the LangSmith report posting in a task group, release LangSmith's answers
    once the rating returns, then leave the group.
    """

    async def rate_in_the_group() -> WorkerRating:
        async with asyncio.TaskGroup() as group:
            score = await AsyncReportedMonitor(
                monitor=build_asynchronous_llm(Script(build_reply(3))),
                reports=(
                    replace(
                        build_async_feedback(langsmith.url),
                        task_group=group,
                    ),
                ),
            ).rate(
                ACTION,
                trajectory=build_trajectory(),
            )
            answered = list(langsmith.answered)
            langsmith.release.set()
        return WorkerRating(
            score=score,
            answered=answered,
        )

    return asyncio.run(rate_in_the_group())


def rate_and_cancel_on_an_executor(langsmith: LangSmithServer) -> None:
    """Rate calmly with the LangSmith report posting on an executor whose one thread is busy, then
    shut the executor down, cancelling the post before it starts.
    """
    busy = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            busy.wait,
            ANSWER_HOLD_SECONDS,
        )
        rate(
            ReportedMonitor(
                monitor=build_synchronous_llm(Script(build_reply(3))),
                reports=(
                    replace(
                        build_feedback(langsmith.url),
                        executor=executor,
                    ),
                ),
            ),
        )
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        busy.set()


def send_flagged_on_an_executor(langsmith: LangSmithServer) -> list[int]:
    """Send a score that reaches the review flag's threshold with the report posting on an
    executor, release LangSmith's answers once send returns, then shut the executor down. Return
    the statuses LangSmith had answered when send returned.
    """
    with ThreadPoolExecutor() as executor:
        report = replace(
            build_feedback(langsmith.url),
            executor=executor,
            review_flag=FLAGGING,
        )
        report.send(
            FLAGGED_SCORE,
            run_id=RUN_ID,
        )
        answered = list(langsmith.answered)
        langsmith.release.set()
    return answered


def send_flagged_in_a_task_group(langsmith: LangSmithServer) -> list[int]:
    """Send a score that reaches the review flag's threshold with the report posting in a task
    group, release LangSmith's answers once send returns, then leave the group. Return the
    statuses LangSmith had answered when send returned.
    """

    async def send_in_the_group() -> list[int]:
        async with asyncio.TaskGroup() as group:
            report = replace(
                build_async_feedback(langsmith.url),
                task_group=group,
                review_flag=FLAGGING,
            )
            await report.send(
                FLAGGED_SCORE,
                run_id=RUN_ID,
            )
            answered = list(langsmith.answered)
            langsmith.release.set()
        return answered

    return asyncio.run(send_in_the_group())


def flag_on_an_executor(langsmith: LangSmithServer) -> None:
    """Flag a step with the report posting on an executor, then shut the executor down."""
    with ThreadPoolExecutor() as executor:
        replace(
            build_feedback(langsmith.url),
            executor=executor,
            review_flag=FLAGGING,
        ).flag(run_id=RUN_ID)


def flag_in_a_task_group(langsmith: LangSmithServer) -> None:
    """Flag a step with the report posting in a task group, then leave the group."""

    async def flag_in_the_group() -> None:
        async with asyncio.TaskGroup() as group:
            await replace(
                build_async_feedback(langsmith.url),
                task_group=group,
                review_flag=FLAGGING,
            ).flag(run_id=RUN_ID)

    asyncio.run(flag_in_the_group())


class CallerError(Exception):
    """An error in the caller's own code inside its task group, which cancels the group's tasks."""


def rate_and_cancel_in_a_task_group(langsmith: LangSmithServer) -> None:
    """Rate calmly with the LangSmith report posting in a task group whose body then raises
    CallerError, which cancels the post.
    """

    async def rate_then_raise() -> None:
        async with asyncio.TaskGroup() as group:
            await AsyncReportedMonitor(
                monitor=build_asynchronous_llm(Script(build_reply(3))),
                reports=(
                    replace(
                        build_async_feedback(langsmith.url),
                        task_group=group,
                    ),
                ),
            ).rate(
                ACTION,
                trajectory=build_trajectory(),
            )
            raise CallerError

    # The group raises an exception group of the CallerError alone, which suppress takes.
    with contextlib.suppress(CallerError):
        asyncio.run(rate_then_raise())


SYNCHRONOUS = Family(
    llm=build_synchronous_llm,
    repeated=RepeatedMonitor,
    calibrated=CalibratedMonitor,
    cascade=CascadeMonitor,
    reported=ReportedMonitor,
    feedback=build_feedback,
    rate_with_a_worker=rate_on_an_executor,
    rate_and_cancel_the_send=rate_and_cancel_on_an_executor,
    send_flagged_with_a_worker=send_flagged_on_an_executor,
    flag_with_a_worker=flag_on_an_executor,
)
ASYNCHRONOUS = Family(
    llm=build_asynchronous_llm,
    repeated=AsyncRepeatedMonitor,
    calibrated=AsyncCalibratedMonitor,
    cascade=AsyncCascadeMonitor,
    reported=AsyncReportedMonitor,
    feedback=build_async_feedback,
    rate_with_a_worker=rate_in_a_task_group,
    rate_and_cancel_the_send=rate_and_cancel_in_a_task_group,
    send_flagged_with_a_worker=send_flagged_in_a_task_group,
    flag_with_a_worker=flag_in_a_task_group,
)


@pytest.fixture(
    params=[
        SYNCHRONOUS,
        ASYNCHRONOUS,
    ],
    ids=["sync", "async"],
)
def family(request: pytest.FixtureRequest) -> Family[Monitor] | Family[AsyncMonitor]:
    return request.param


def build_calm_llm(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.llm(Script(build_reply(3)))


def build_unavailable_llm(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> Monitor | AsyncMonitor:
    return family.llm(Script(Unavailable(status=503)))


def build_repeated_with_a_refusal(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> Monitor | AsyncMonitor:
    return family.repeated(
        monitor=family.llm(
            Script(
                build_reply(3),
                Refused(),
            ),
        ),
        ensemble_size=2,
    )


def build_repeated_cascade(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> Monitor | AsyncMonitor:
    """Three cascades: the first and the last keep the fast score, the second escalates."""
    cascade = family.cascade(
        fast=family.llm(
            Script(
                build_reply(1),
                build_reply(9),
                build_reply(1),
            ),
            name="fast-model",
        ),
        careful=family.llm(
            Script(build_reply(3)),
            name="careful-model",
        ),
        escalation_threshold=0.5,
    )
    return family.repeated(
        monitor=cascade,
        ensemble_size=3,
    )


def build_calibrated_repeated_cascade(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> Monitor | AsyncMonitor:
    return family.calibrated(
        monitor=build_repeated_cascade(family),
        honest_scores=[0.1],
    )


def test_langfuse_receives_the_score_on_the_trace_and_observation_of_the_ratings_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(LangfuseScores(client=langfuse.client),),
    )

    rate(monitor)

    # The body of POST /api/public/scores (langfuse.api.scores, CreateScoresRequest); the id
    # and the timestamp differ in every run.
    assert langfuse.read_scores() == [
        {
            "id": ANY,
            "traceId": TRACE_HEX,
            "observationId": SPAN_HEX,
            "name": "suspicion",
            "value": 0.3,
            "comment": "calm",
            "dataType": "NUMERIC",
            "timestamp": ANY,
            "metadata": CALM_METADATA,
        },
    ]


def test_langsmith_receives_the_score_as_model_feedback_on_the_run_of_the_ratings_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(family.feedback(langsmith.url),),
    )

    rate(monitor)

    assert langsmith.received == [
        (
            "/feedback",
            CALM_FEEDBACK,
        ),
    ]


def test_langfuse_receives_the_explanation_through_the_reports_redaction(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
) -> None:
    monitor = family.reported(
        monitor=family.llm(Script(DIALLING_REPLY)),
        reports=(
            LangfuseScores(
                client=langfuse.client,
                redaction=MASKED,
            ),
        ),
    )

    rate(monitor)

    assert [score["comment"] for score in langfuse.read_scores()] == ["It dials ###-####."]


def test_langsmith_receives_the_explanation_through_the_reports_redaction(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    monitor = family.reported(
        monitor=family.llm(Script(DIALLING_REPLY)),
        reports=(
            replace(
                family.feedback(langsmith.url),
                redaction=MASKED,
            ),
        ),
    )

    rate(monitor)

    assert langsmith.received == [
        (
            "/feedback",
            {
                **CALM_FEEDBACK,
                "comment": "It dials ###-####.",
            },
        ),
    ]


def refuse_to_redact(text: str) -> str:
    """A redaction's text function that fails, as a redaction service that cannot be reached
    does.
    """
    raise RuntimeError(text)


REFUSING = Redaction(
    text=refuse_to_redact,
    media=partial(
        replace,
        data=b"blurred",
    ),
)


@pytest.mark.parametrize(
    ("build_report", "read_sent"),
    [
        (
            lambda family, langfuse, langsmith: LangfuseScores(
                client=langfuse.client,
                redaction=REFUSING,
            ),
            lambda langfuse, langsmith: langfuse.read_scores(),
        ),
        (
            lambda family, langfuse, langsmith: replace(
                family.feedback(langsmith.url),
                redaction=REFUSING,
            ),
            lambda langfuse, langsmith: langsmith.received,
        ),
    ],
    ids=["langfuse", "langsmith"],
)
def test_a_report_whose_redaction_raises_sends_no_score(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    langsmith: LangSmithServer,
    build_report: Callable[..., LangfuseScores | LangSmithFeedback | AsyncLangSmithFeedback],
    read_sent: Callable[
        [
            LangfuseStandIn,
            LangSmithServer,
        ],
        list[object],
    ],
) -> None:
    monitor = family.reported(
        monitor=family.llm(Script(DIALLING_REPLY)),
        reports=(
            build_report(
                family,
                langfuse,
                langsmith,
            ),
        ),
    )

    rate(monitor)

    assert (
        read_sent(
            langfuse,
            langsmith,
        )
        == []
    )


@pytest.mark.parametrize(
    ("build", "metadata"),
    [
        (
            build_unavailable_llm,
            {
                "failed_closed": True,
                "cause": "Unavailable",
                "model": MODEL_NAME,
            },
        ),
        (
            build_repeated_with_a_refusal,
            {
                "failed_closed": True,
                "cause": "Refused",
                "model": [MODEL_NAME],
            },
        ),
        (
            build_repeated_cascade,
            {
                "failed_closed": False,
                "cause": None,
                "model": ["fast-model", "careful-model"],
            },
        ),
        (
            build_calibrated_repeated_cascade,
            {
                "failed_closed": False,
                "cause": None,
                "model": ["fast-model", "careful-model"],
            },
        ),
    ],
    ids=["failed", "failed-part", "models-in-order", "models-of-nested-parts"],
)
def test_the_metadata_says_whether_the_score_failed_closed_why_and_from_which_models(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    build: Callable[[Family[Monitor] | Family[AsyncMonitor]], Monitor | AsyncMonitor],
    metadata: JSON,
) -> None:
    monitor = family.reported(
        monitor=build(family),
        reports=(LangfuseScores(client=langfuse.client),),
    )

    rate(monitor)

    assert [score["metadata"] for score in langfuse.read_scores()] == [metadata]


@pytest.mark.parametrize(
    ("family", "spans_ended"),
    [
        (
            SYNCHRONOUS,
            [
                (
                    "rate LLMMonitor",
                    CALM_ATTRIBUTES,
                ),
                (
                    "rate ReportedMonitor",
                    CALM_ATTRIBUTES,
                ),
            ],
        ),
        (
            ASYNCHRONOUS,
            [
                (
                    "rate AsyncLLMMonitor",
                    CALM_ATTRIBUTES,
                ),
                (
                    "rate AsyncReportedMonitor",
                    CALM_ATTRIBUTES,
                ),
            ],
        ),
    ],
    ids=["sync", "async"],
)
def test_the_reported_monitor_rates_inside_a_span_of_its_own_that_carries_the_score(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    spans_ended: list[tuple[str, dict[str, str | float | bool]]],
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(LangfuseScores(client=langfuse.client),),
    )

    rate(monitor)

    assert [
        (
            span.name,
            span.attributes,
        )
        for span in spans.get_finished_spans()
    ] == spans_ended


def test_a_rating_in_an_unsampled_trace_sends_nothing(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    langsmith: LangSmithServer,
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(
            LangfuseScores(client=langfuse.client),
            family.feedback(langsmith.url),
        ),
    )

    with trace.use_span(UNSAMPLED_PARENT):
        rate(monitor)

    assert (
        langfuse.read_scores(),
        langsmith.received,
    ) == (
        [],
        [],
    )


def test_a_rating_in_an_unsampled_trace_logs_at_debug_that_nothing_was_sent(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.DEBUG,
        logger="hyoshigi",
    )
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(LangfuseScores(client=langfuse.client),),
    )

    with trace.use_span(UNSAMPLED_PARENT):
        rate(monitor)

    assert (
        "hyoshigi.monitors.reporting",
        logging.DEBUG,
        "No report was sent: the rating's span is not sampled.",
    ) in caplog.record_tuples


def test_a_report_that_fails_to_send_leaves_the_score_returned(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    langsmith.status = 401
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(family.feedback(langsmith.url),),
    )

    score = rate(monitor)

    assert score == SuspicionScore(
        value=0.3,
        explanation="calm",
        evidence=Rationale(model=MODEL_NAME),
    )


@pytest.mark.parametrize(
    ("family", "report"),
    [
        (
            SYNCHRONOUS,
            "LangSmithFeedback",
        ),
        (
            ASYNCHRONOUS,
            "AsyncLangSmithFeedback",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_report_that_fails_to_send_is_logged_by_its_class_and_error_type_alone(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    caplog: pytest.LogCaptureFixture,
    report: str,
) -> None:
    langsmith.status = 401
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(family.feedback(langsmith.url),),
    )

    rate(monitor)

    assert [record for record in caplog.record_tuples if record[0].startswith("hyoshigi")] == [
        (
            "hyoshigi.monitors.reporting",
            logging.WARNING,
            f"{report} did not send the score: LangSmithAuthError.",
        ),
    ]


def test_the_other_reports_are_sent_when_one_fails(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
    langsmith: LangSmithServer,
) -> None:
    langsmith.status = 401
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(
            family.feedback(langsmith.url),
            LangfuseScores(client=langfuse.client),
        ),
    )

    rate(monitor)

    assert [score["observationId"] for score in langfuse.read_scores()] == [SPAN_HEX]


@pytest.fixture
def monitor_with_one_attempt(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> Monitor | AsyncMonitor:
    """A reported monitor whose LangSmith report makes one attempt, to a server that answers
    404 to its first request and 200 to the next.
    """
    langsmith.first_statuses = [404]
    return family.reported(
        monitor=build_calm_llm(family),
        reports=(
            replace(
                family.feedback(langsmith.url),
                stop_after_attempt=1,
            ),
        ),
    )


def test_a_report_capped_at_one_attempt_makes_one_request_when_the_run_is_not_found(
    langsmith: LangSmithServer,
    monitor_with_one_attempt: Monitor | AsyncMonitor,
) -> None:
    rate(monitor_with_one_attempt)

    assert len(langsmith.received) == 1


@pytest.mark.parametrize(
    ("family", "report"),
    [
        (
            SYNCHRONOUS,
            "LangSmithFeedback",
        ),
        (
            ASYNCHRONOUS,
            "AsyncLangSmithFeedback",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_score_lost_to_a_run_not_found_after_one_attempt_is_logged_by_its_class_and_error_type(
    monitor_with_one_attempt: Monitor | AsyncMonitor,
    caplog: pytest.LogCaptureFixture,
    report: str,
) -> None:
    rate(monitor_with_one_attempt)

    assert [record for record in caplog.record_tuples if record[0].startswith("hyoshigi")] == [
        (
            "hyoshigi.monitors.reporting",
            logging.WARNING,
            f"{report} did not send the score: LangSmithNotFoundError.",
        ),
    ]


@pytest.mark.parametrize(
    "status",
    [
        200,
        401,
    ],
    ids=["accepted", "rejected"],
)
def test_a_report_with_a_worker_returns_the_score_before_langsmith_answers(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    status: int,
) -> None:
    langsmith.status = status
    langsmith.release.clear()

    rating = family.rate_with_a_worker(langsmith)

    assert rating == WorkerRating(
        score=SuspicionScore(
            value=0.3,
            explanation="calm",
            evidence=Rationale(model=MODEL_NAME),
        ),
        answered=[],
    )


def test_langsmith_receives_the_feedback_a_worker_sends_once_the_worker_finishes(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    family.rate_with_a_worker(langsmith)

    assert langsmith.received == [
        (
            "/feedback",
            CALM_FEEDBACK,
        ),
    ]


@pytest.mark.parametrize(
    ("family", "report"),
    [
        (
            SYNCHRONOUS,
            "LangSmithFeedback",
        ),
        (
            ASYNCHRONOUS,
            "AsyncLangSmithFeedback",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_send_that_fails_on_a_worker_is_logged_by_the_reports_class_and_error_type(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    caplog: pytest.LogCaptureFixture,
    report: str,
) -> None:
    langsmith.status = 401

    family.rate_with_a_worker(langsmith)

    assert [record for record in caplog.record_tuples if record[0].startswith("hyoshigi")] == [
        (
            "hyoshigi.monitors.reporting",
            logging.WARNING,
            f"{report} did not send the score: LangSmithAuthError.",
        ),
    ]


@pytest.mark.parametrize(
    ("family", "report"),
    [
        (
            SYNCHRONOUS,
            "LangSmithFeedback",
        ),
        (
            ASYNCHRONOUS,
            "AsyncLangSmithFeedback",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_send_cancelled_on_its_worker_is_logged_as_cancelled(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    caplog: pytest.LogCaptureFixture,
    report: str,
) -> None:
    family.rate_and_cancel_the_send(langsmith)

    assert [record for record in caplog.record_tuples if record[0].startswith("hyoshigi")] == [
        (
            "hyoshigi.monitors.reporting",
            logging.WARNING,
            f"{report} did not send the score: CancelledError.",
        ),
    ]


# Rates with a reported monitor of each family in a process where no tracer provider is set
# up, with hyoshigi's log records written to standard output.
RATE_WITHOUT_A_TRACER_PROVIDER = """
import logging, sys
from langsmith import Client
from hyoshigi import (
    Action, Generation, GenerativeModel, LangSmithFeedback, LLMMonitor, ReportedMonitor,
    TextPart, Trajectory,
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logging.getLogger("hyoshigi").addHandler(handler)
model = GenerativeModel(
    name="trusted-1",
    generate=lambda prompt: Generation(text="<reasoning>calm</reasoning><score>3</score>"),
)
feedback = LangSmithFeedback(
    client=Client(api_url="http://127.0.0.1:9", api_key="lsv2-test", info={},
                  auto_batch_tracing=False),
    project_id="6f9a2c4e-1b3d-4e5f-8a7b-9c0d1e2f3a4b",
)
monitor = ReportedMonitor(monitor=LLMMonitor(model=model), reports=(feedback,))
trajectory = Trajectory.from_run("Summarise q3.md", answer="Done.").trajectory
monitor.rate(Action(parts=(TextPart(text="Reading q3.md."),)), trajectory=trajectory)
"""
RATE_ASYNCHRONOUSLY_WITHOUT_A_TRACER_PROVIDER = """
import asyncio, logging, sys
from langsmith import AsyncClient
from hyoshigi import (
    Action, AsyncGenerativeModel, AsyncLangSmithFeedback, AsyncLLMMonitor,
    AsyncReportedMonitor, Generation, TextPart, Trajectory,
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
logging.getLogger("hyoshigi").addHandler(handler)
async def generate(prompt):
    return Generation(text="<reasoning>calm</reasoning><score>3</score>")
model = AsyncGenerativeModel(name="trusted-1", generate=generate)
feedback = AsyncLangSmithFeedback(
    client=AsyncClient(api_url="http://127.0.0.1:9", api_key="lsv2-test"),
    project_id="6f9a2c4e-1b3d-4e5f-8a7b-9c0d1e2f3a4b",
)
monitor = AsyncReportedMonitor(monitor=AsyncLLMMonitor(model=model), reports=(feedback,))
trajectory = Trajectory.from_run("Summarise q3.md", answer="Done.").trajectory
action = Action(parts=(TextPart(text="Reading q3.md."),))
asyncio.run(monitor.rate(action, trajectory=trajectory))
"""


@pytest.mark.parametrize(
    "script",
    [
        RATE_WITHOUT_A_TRACER_PROVIDER,
        RATE_ASYNCHRONOUSLY_WITHOUT_A_TRACER_PROVIDER,
    ],
    ids=["sync", "async"],
)
def test_a_rating_with_no_tracer_provider_set_up_warns_that_nothing_was_sent(script: str) -> None:
    command = [
        sys.executable,
        "-c",
        script,
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    assert process.stdout == (
        "WARNING No report was sent: no OpenTelemetry tracer provider recorded the rating's span.\n"
    )


# A server in the rating's process that accepts any request at once, and a sampled span context
# from another process; no tracer provider is set up, so the rating's span carries that context
# without recording.
ACCEPTING_SERVER_AND_REMOTE_PARENT = """
import http.server, threading
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
class Accept(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *arguments):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Accept)
threading.Thread(target=server.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{server.server_port}"
parent = NonRecordingSpan(
    SpanContext(
        trace_id=0x5B8EFFF798038103D269B633813FC60C,
        span_id=0xEEE19B7EC3C1B174,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    ),
)
"""


@pytest.mark.parametrize(
    "script",
    [
        RATE_WITHOUT_A_TRACER_PROVIDER,
        RATE_ASYNCHRONOUSLY_WITHOUT_A_TRACER_PROVIDER,
    ],
    ids=["sync", "async"],
)
def test_a_rating_under_a_remote_parent_with_no_tracer_provider_sends_nothing(script: str) -> None:
    *setup, rating = script.strip().replace('"http://127.0.0.1:9"', "URL").split("\n")
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                ACCEPTING_SERVER_AND_REMOTE_PARENT,
                *setup,
                "with trace.use_span(parent):",
                f"    {rating}",
            ],
        ),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert process.stdout == (
        "WARNING No report was sent: no OpenTelemetry tracer provider recorded the rating's span.\n"
    )


# Builds a reported monitor of each family in a process where importing opentelemetry fails.
BUILD_WITHOUT_OPENTELEMETRY = """
import sys
sys.modules["opentelemetry"] = None
from langsmith import Client
from hyoshigi import (
    ConfigurationError, Generation, GenerativeModel, LangSmithFeedback, LLMMonitor,
    ReportedMonitor,
)
model = GenerativeModel(name="trusted-1", generate=lambda prompt: Generation(text=""))
feedback = LangSmithFeedback(
    client=Client(api_url="http://127.0.0.1:9", api_key="lsv2-test", info={},
                  auto_batch_tracing=False),
    project_id="6f9a2c4e-1b3d-4e5f-8a7b-9c0d1e2f3a4b",
)
try:
    ReportedMonitor(monitor=LLMMonitor(model=model), reports=(feedback,))
except ConfigurationError as error:
    print(error)
"""
BUILD_ASYNCHRONOUSLY_WITHOUT_OPENTELEMETRY = """
import sys
sys.modules["opentelemetry"] = None
from langsmith import AsyncClient
from hyoshigi import (
    AsyncGenerativeModel, AsyncLangSmithFeedback, AsyncLLMMonitor, AsyncReportedMonitor,
    ConfigurationError,
)
async def generate(prompt):
    raise AssertionError("never asked")
model = AsyncGenerativeModel(name="trusted-1", generate=generate)
feedback = AsyncLangSmithFeedback(
    client=AsyncClient(api_url="http://127.0.0.1:9", api_key="lsv2-test"),
    project_id="6f9a2c4e-1b3d-4e5f-8a7b-9c0d1e2f3a4b",
)
try:
    AsyncReportedMonitor(monitor=AsyncLLMMonitor(model=model), reports=(feedback,))
except ConfigurationError as error:
    print(error)
"""


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (
            BUILD_WITHOUT_OPENTELEMETRY,
            "ReportedMonitor needs OpenTelemetry: pip install 'hyoshigi[opentelemetry]'\n",
        ),
        (
            BUILD_ASYNCHRONOUSLY_WITHOUT_OPENTELEMETRY,
            "AsyncReportedMonitor needs OpenTelemetry: pip install 'hyoshigi[opentelemetry]'\n",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_reported_monitor_without_opentelemetry_installed_is_refused_naming_the_extra(
    script: str,
    message: str,
) -> None:
    command = [
        sys.executable,
        "-c",
        script,
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    assert process.stdout == message


@pytest.mark.parametrize(
    ("reported", "monitor", "message"),
    [
        (
            ReportedMonitor,
            build_asynchronous_llm(Script()),
            "monitor must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncReportedMonitor for an AsyncMonitor.",
        ),
        (
            AsyncReportedMonitor,
            build_synchronous_llm(Script()),
            "monitor must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use ReportedMonitor for a Monitor.",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_monitor_of_the_other_family_is_refused_naming_the_twin(
    langfuse: LangfuseStandIn,
    reported: Callable[..., Monitor | AsyncMonitor],
    monitor: Monitor | AsyncMonitor,
    message: str,
) -> None:
    scores = LangfuseScores(client=langfuse.client)

    with pytest.raises(ConfigurationError) as raised:
        reported(
            monitor=monitor,
            reports=(scores,),
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("reported", "monitor", "build_reports", "message"),
    [
        (
            ReportedMonitor,
            build_synchronous_llm(Script()),
            lambda: [build_feedback()],
            "reports must be a tuple, got an instance of list",
        ),
        (
            ReportedMonitor,
            build_synchronous_llm(Script()),
            tuple,
            "reports must hold at least one report",
        ),
        (
            ReportedMonitor,
            build_synchronous_llm(Script()),
            lambda: (
                build_feedback(),
                build_async_feedback(),
            ),
            "reports[1] must be a LangfuseScores or a LangSmithFeedback, got an instance of "
            "AsyncLangSmithFeedback. Use AsyncReportedMonitor for an AsyncLangSmithFeedback.",
        ),
        (
            AsyncReportedMonitor,
            build_asynchronous_llm(Script()),
            lambda: (build_feedback(),),
            "reports[0] must be a LangfuseScores or an AsyncLangSmithFeedback, got an instance "
            "of LangSmithFeedback. Use ReportedMonitor for a LangSmithFeedback.",
        ),
        (
            AsyncReportedMonitor,
            build_asynchronous_llm(Script()),
            lambda: ("suspicion",),
            "reports[0] must be a LangfuseScores or an AsyncLangSmithFeedback, got an instance "
            "of str. Use ReportedMonitor for a LangSmithFeedback.",
        ),
    ],
    ids=["list", "empty", "async-report-in-sync", "sync-report-in-async", "not-a-report"],
)
def test_reports_that_are_not_a_tuple_of_the_familys_reports_are_refused(
    reported: Callable[..., Monitor | AsyncMonitor],
    monitor: Monitor | AsyncMonitor,
    build_reports: Callable[[], object],
    message: str,
) -> None:
    reports = build_reports()

    with pytest.raises(ConfigurationError) as raised:
        reported(
            monitor=monitor,
            reports=reports,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("report", "build_client", "message"),
    [
        (
            partial(
                LangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            build_async_client,
            "client must be a Client, got an instance of langsmith.async_client.AsyncClient. "
            "Use AsyncLangSmithFeedback for an AsyncClient.",
        ),
        (
            partial(
                AsyncLangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            build_client,
            "client must be an AsyncClient, got an instance of langsmith.client.Client. "
            "Use LangSmithFeedback for a Client.",
        ),
        (
            LangfuseScores,
            object,
            "client must be a Langfuse, got an instance of object",
        ),
        (
            partial(
                LangSmithFeedback,
                project_id=PROJECT_ID,
                executor=asyncio.TaskGroup(),
            ),
            build_client,
            "executor must be an Executor, got an instance of asyncio.taskgroups.TaskGroup. "
            "Use AsyncLangSmithFeedback for a TaskGroup.",
        ),
        (
            partial(
                AsyncLangSmithFeedback,
                project_id=PROJECT_ID,
                task_group=ThreadPoolExecutor(),
            ),
            build_async_client,
            "task_group must be a TaskGroup, got an instance of "
            "concurrent.futures.thread.ThreadPoolExecutor. Use LangSmithFeedback for an Executor.",
        ),
    ],
    ids=[
        "async-client-in-sync",
        "sync-client-in-async",
        "not-a-langfuse-client",
        "task-group-in-sync",
        "executor-in-async",
    ],
)
def test_a_client_or_worker_of_the_wrong_kind_is_refused_naming_the_twin(
    report: Callable[..., object],
    build_client: Callable[[], object],
    message: str,
) -> None:
    client = build_client()

    with pytest.raises(ConfigurationError) as raised:
        report(client=client)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("report", "find_client", "message"),
    [
        (
            partial(
                LangfuseScores,
                name="",
            ),
            lambda langfuse: langfuse.client,
            "name must be a non-blank string, got ''",
        ),
        (
            partial(
                LangSmithFeedback,
                project_id=PROJECT_ID,
                name=" ",
            ),
            lambda langfuse: build_client(),
            "name must be a non-blank string, got ' '",
        ),
        (
            partial(
                AsyncLangSmithFeedback,
                project_id=PROJECT_ID,
                name=None,
            ),
            lambda langfuse: build_async_client(),
            "name must be a non-blank string, got None",
        ),
    ],
    ids=["langfuse-empty", "langsmith-blank", "async-langsmith-none"],
)
def test_a_name_that_is_not_a_non_blank_string_is_refused(
    langfuse: LangfuseStandIn,
    report: Callable[..., object],
    find_client: Callable[[LangfuseStandIn], object],
    message: str,
) -> None:
    client = find_client(langfuse)

    with pytest.raises(ConfigurationError) as raised:
        report(client=client)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("report", "build_client", "project_id", "message"),
    [
        (
            LangSmithFeedback,
            build_client,
            "my-project",
            "project_id must be a UUID string, such as a LangSmith project's id, got 'my-project'",
        ),
        (
            AsyncLangSmithFeedback,
            build_async_client,
            uuid.UUID(PROJECT_ID),
            "project_id must be a UUID string, such as a LangSmith project's id, "
            "got an instance of uuid.UUID",
        ),
    ],
    ids=["sync-not-a-uuid", "async-not-a-string"],
)
def test_a_project_id_that_is_not_a_uuid_string_is_refused(
    report: Callable[..., object],
    build_client: Callable[[], object],
    project_id: object,
    message: str,
) -> None:
    client = build_client()

    with pytest.raises(ConfigurationError) as raised:
        report(
            client=client,
            project_id=project_id,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("report", "create_feedback"),
    [
        (
            LangSmithFeedback,
            Client.create_feedback,
        ),
        (
            AsyncLangSmithFeedback,
            AsyncClient.create_feedback,
        ),
    ],
    ids=["sync", "async"],
)
def test_the_default_attempts_are_those_of_the_clients_create_feedback(
    report: Callable[..., object],
    create_feedback: Callable[..., object],
) -> None:
    expected = signature(create_feedback).parameters["stop_after_attempt"].default

    declared = signature(report).parameters["stop_after_attempt"].default

    assert declared == expected


@pytest.mark.parametrize(
    ("report", "build_client"),
    [
        (
            LangSmithFeedback,
            build_client,
        ),
        (
            AsyncLangSmithFeedback,
            build_async_client,
        ),
    ],
    ids=["sync", "async"],
)
@pytest.mark.parametrize(
    ("attempts", "message"),
    [
        (
            0,
            "stop_after_attempt must be at least 1, got 0",
        ),
        (
            -1,
            "stop_after_attempt must be at least 1, got -1",
        ),
        (
            True,
            "stop_after_attempt must be a whole number of at least 1, got True",
        ),
        (
            1.5,
            "stop_after_attempt must be a whole number of at least 1, got 1.5",
        ),
    ],
    ids=["zero", "negative", "bool", "float"],
)
def test_a_stop_after_attempt_that_is_not_a_positive_integer_is_refused(
    report: Callable[..., object],
    build_client: Callable[[], object],
    attempts: object,
    message: str,
) -> None:
    client = build_client()

    with pytest.raises(ConfigurationError) as raised:
        report(
            client=client,
            project_id=PROJECT_ID,
            stop_after_attempt=attempts,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("report", "find_client"),
    [
        (
            LangfuseScores,
            lambda langfuse: langfuse.client,
        ),
        (
            partial(
                LangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            lambda langfuse: build_client(),
        ),
        (
            partial(
                AsyncLangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            lambda langfuse: build_async_client(),
        ),
    ],
    ids=["langfuse", "langsmith", "async-langsmith"],
)
def test_a_redaction_that_is_not_a_redaction_is_refused(
    langfuse: LangfuseStandIn,
    report: Callable[..., object],
    find_client: Callable[[LangfuseStandIn], object],
) -> None:
    client = find_client(langfuse)

    with pytest.raises(ConfigurationError) as raised:
        report(
            client=client,
            redaction=MASKED.text,
        )

    assert (
        str(raised.value) == "redaction must be a Redaction, got an instance of functools.partial"
    )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (
            "",
            "name must be a non-blank string, got ''",
        ),
        (
            " ",
            "name must be a non-blank string, got ' '",
        ),
        (
            None,
            "name must be a non-blank string, got None",
        ),
    ],
    ids=["empty", "blank", "none"],
)
@pytest.mark.xfail(strict=True)
def test_a_review_flag_name_that_is_not_a_non_blank_string_is_refused(
    name: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ReviewFlag(name=name)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("threshold", "message"),
    [
        (
            True,
            "threshold must be a number between 0 and 1, got True",
        ),
        (
            "0.6",
            "threshold must be a number between 0 and 1, got '0.6'",
        ),
        (
            float("nan"),
            "threshold must be between 0 and 1, got nan",
        ),
        (
            1.5,
            "threshold must be between 0 and 1, got 1.5",
        ),
        (
            -0.1,
            "threshold must be between 0 and 1, got -0.1",
        ),
    ],
    ids=["bool", "string", "nan", "above-one", "below-zero"],
)
@pytest.mark.xfail(strict=True)
def test_a_review_flag_threshold_that_is_not_a_number_from_0_to_1_is_refused(
    threshold: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ReviewFlag(threshold=threshold)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message


@pytest.mark.xfail(strict=True)
def test_a_review_flag_threshold_is_kept_as_a_float() -> None:
    review_flag = ReviewFlag(threshold=Fraction(3, 5))  # ty: ignore[invalid-argument-type]

    assert review_flag.threshold == 0.6


@pytest.mark.parametrize(
    ("report", "find_client"),
    [
        (
            LangfuseScores,
            lambda langfuse: langfuse.client,
        ),
        (
            partial(
                LangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            lambda langfuse: build_client(),
        ),
        (
            partial(
                AsyncLangSmithFeedback,
                project_id=PROJECT_ID,
            ),
            lambda langfuse: build_async_client(),
        ),
    ],
    ids=["langfuse", "langsmith", "async-langsmith"],
)
@pytest.mark.xfail(strict=True)
def test_a_review_flag_that_is_not_a_review_flag_is_refused(
    langfuse: LangfuseStandIn,
    report: Callable[..., object],
    find_client: Callable[[LangfuseStandIn], object],
) -> None:
    client = find_client(langfuse)

    with pytest.raises(ConfigurationError) as raised:
        report(
            client=client,
            review_flag="flagged",
        )

    assert str(raised.value) == "review_flag must be a ReviewFlag, got an instance of str"


@pytest.mark.parametrize(
    "build_report",
    [
        lambda langfuse, langsmith: LangfuseScores(client=langfuse.client),
        lambda langfuse, langsmith: build_feedback(langsmith.url),
        lambda langfuse, langsmith: build_async_feedback(langsmith.url),
    ],
    ids=["langfuse", "langsmith", "async-langsmith"],
)
@pytest.mark.xfail(strict=True)
def test_a_report_without_a_review_flag_refuses_to_flag_and_sends_nothing(
    langfuse: LangfuseStandIn,
    langsmith: LangSmithServer,
    build_report: Callable[..., LangfuseScores | LangSmithFeedback | AsyncLangSmithFeedback],
) -> None:
    report = build_report(
        langfuse,
        langsmith,
    )

    with pytest.raises(ConfigurationError) as raised:
        flag_step(report)

    assert str(raised.value) == (
        "flag needs a ReviewFlag: build the report with review_flag=ReviewFlag(...)"
    )
    assert langfuse.read_scores() == []
    assert langsmith.received == []


# A score, the review flag of the report that sends it, and the names of what the report sends,
# the flag before the score.
SEND_CASES = [
    pytest.param(
        0.6,
        ReviewFlag(threshold=0.6),
        [
            "flagged",
            "suspicion",
        ],
        marks=pytest.mark.xfail(strict=True),
        id="at-the-threshold",
    ),
    pytest.param(
        0.9,
        ReviewFlag(threshold=0.6),
        [
            "flagged",
            "suspicion",
        ],
        marks=pytest.mark.xfail(strict=True),
        id="above-the-threshold",
    ),
    pytest.param(
        0.5,
        ReviewFlag(threshold=0.6),
        ["suspicion"],
        id="below-the-threshold",
    ),
    pytest.param(
        0.9,
        ReviewFlag(),
        ["suspicion"],
        id="no-threshold",
    ),
]


@pytest.mark.parametrize(
    ("value", "review_flag", "names"),
    SEND_CASES,
)
def test_langfuse_receives_a_flag_before_the_score_when_the_score_reaches_the_threshold(
    langfuse: LangfuseStandIn,
    value: float,
    review_flag: ReviewFlag,
    names: list[str],
) -> None:
    report = LangfuseScores(
        client=langfuse.client,
        review_flag=review_flag,
    )

    send_score(
        report,
        build_score(value),
    )

    assert [score["name"] for score in langfuse.read_scores()] == names


@pytest.mark.parametrize(
    ("value", "review_flag", "names"),
    SEND_CASES,
)
def test_langsmith_receives_a_flag_before_the_score_when_the_score_reaches_the_threshold(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    value: float,
    review_flag: ReviewFlag,
    names: list[str],
) -> None:
    report = replace(
        family.feedback(langsmith.url),
        review_flag=review_flag,
    )

    send_score(
        report,
        build_score(value),
    )

    assert (
        read_feedback_field(
            langsmith,
            "key",
        )
        == names
    )


@pytest.mark.xfail(strict=True)
def test_langfuse_receives_the_flag_on_the_trace_and_observation_of_the_ratings_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langfuse: LangfuseStandIn,
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(
            LangfuseScores(
                client=langfuse.client,
                review_flag=ReviewFlag(threshold=0.3),
            ),
        ),
    )

    rate(monitor)

    assert {
        score["name"]: (
            score["traceId"],
            score["observationId"],
        )
        for score in langfuse.read_scores()
    } == {
        "flagged": (
            TRACE_HEX,
            SPAN_HEX,
        ),
        "suspicion": (
            TRACE_HEX,
            SPAN_HEX,
        ),
    }


@pytest.mark.xfail(strict=True)
def test_langsmith_receives_the_flag_on_the_run_of_the_ratings_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    monitor = family.reported(
        monitor=build_calm_llm(family),
        reports=(
            replace(
                family.feedback(langsmith.url),
                review_flag=ReviewFlag(threshold=0.3),
            ),
        ),
    )

    rate(monitor)

    assert read_feedback_field(
        langsmith,
        "key",
    ) == [
        "flagged",
        "suspicion",
    ]
    assert read_feedback_field(
        langsmith,
        "run_id",
    ) == [
        RUN_ID,
        RUN_ID,
    ]


@pytest.mark.parametrize(
    ("review_flag", "name"),
    [
        (
            ReviewFlag(),
            "flagged",
        ),
        (
            ReviewFlag(name="needs-review"),
            "needs-review",
        ),
    ],
    ids=["default-name", "named"],
)
@pytest.mark.xfail(strict=True)
def test_langfuse_receives_a_flag_as_a_boolean_score_with_no_comment(
    langfuse: LangfuseStandIn,
    review_flag: ReviewFlag,
    name: str,
) -> None:
    report = LangfuseScores(
        client=langfuse.client,
        review_flag=review_flag,
    )

    flag_step(report)

    # The body of POST /api/public/scores (langfuse.api.scores, CreateScoresRequest), as the
    # score of a rating is, without a comment or metadata. The client turns the value 1 into 1.0
    # (Langfuse.create_score).
    assert langfuse.read_scores() == [
        {
            "id": ANY,
            "traceId": TRACE_HEX,
            "observationId": SPAN_HEX,
            "name": name,
            "value": 1.0,
            "dataType": "BOOLEAN",
            "timestamp": ANY,
        },
    ]


@pytest.mark.parametrize(
    ("review_flag", "name"),
    [
        (
            ReviewFlag(),
            "flagged",
        ),
        (
            ReviewFlag(name="needs-review"),
            "needs-review",
        ),
    ],
    ids=["default-name", "named"],
)
@pytest.mark.xfail(strict=True)
def test_langsmith_receives_a_flag_as_boolean_model_feedback_with_no_comment(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    review_flag: ReviewFlag,
    name: str,
) -> None:
    report = replace(
        family.feedback(langsmith.url),
        review_flag=review_flag,
    )

    flag_step(report)

    assert langsmith.received == [
        (
            "/feedback",
            {
                "id": ANY,
                "created_at": ANY,
                "modified_at": ANY,
                "run_id": RUN_ID,
                "key": name,
                "score": True,
                "feedback_source": {
                    "type": "model",
                    "metadata": {},
                },
                "session_id": PROJECT_ID,
                "extend_trace_retention": False,
            },
        ),
    ]
    # True equals 1, so the comparison above does not tell a boolean from a number.
    assert (
        read_feedback_field(
            langsmith,
            "score",
        )[0]
        is True
    )


def test_a_flagged_score_is_sent_with_a_worker_before_langsmith_answers(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    langsmith.release.clear()

    answered = family.send_flagged_with_a_worker(langsmith)

    assert answered == []


@pytest.mark.xfail(strict=True)
def test_langsmith_receives_the_flag_and_the_score_a_worker_sends_once_the_worker_finishes(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    family.send_flagged_with_a_worker(langsmith)

    # Sorted: the worker may run the two posts in either order.
    assert sorted(
        read_feedback_field(
            langsmith,
            "key",
        ),
        key=str,
    ) == [
        "flagged",
        "suspicion",
    ]


@pytest.mark.parametrize(
    ("family", "report"),
    [
        (
            SYNCHRONOUS,
            "LangSmithFeedback",
        ),
        (
            ASYNCHRONOUS,
            "AsyncLangSmithFeedback",
        ),
    ],
    ids=["sync", "async"],
)
@pytest.mark.xfail(strict=True)
def test_a_flag_that_fails_on_a_worker_is_logged_as_a_flag_by_the_reports_class_and_error_type(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
    caplog: pytest.LogCaptureFixture,
    report: str,
) -> None:
    langsmith.status = 401

    family.flag_with_a_worker(langsmith)

    assert [record for record in caplog.record_tuples if record[0].startswith("hyoshigi")] == [
        (
            "hyoshigi.monitors.reporting",
            logging.WARNING,
            f"{report} did not send the flag: LangSmithAuthError.",
        ),
    ]


@pytest.mark.xfail(strict=True)
def test_a_score_is_not_posted_when_langsmith_rejects_the_flag_sent_before_it(
    family: Family[Monitor] | Family[AsyncMonitor],
    langsmith: LangSmithServer,
) -> None:
    langsmith.status = 401
    report = replace(
        family.feedback(langsmith.url),
        review_flag=FLAGGING,
    )

    with pytest.raises(LangSmithAuthError):
        send_score(
            report,
            FLAGGED_SCORE,
        )

    assert read_feedback_field(
        langsmith,
        "key",
    ) == ["flagged"]
