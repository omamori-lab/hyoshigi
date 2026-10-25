"""The span each rating opens in OpenTelemetry: its name, its attributes and its children."""

import asyncio
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.types import Attributes

from hyoshigi import (
    DEFAULT_PROMPT,
    Action,
    AsyncCalibratedMonitor,
    AsyncCascadeMonitor,
    AsyncDecisionModel,
    AsyncDecisionMonitor,
    AsyncGenerativeModel,
    AsyncGuardMonitor,
    AsyncLLMMonitor,
    AsyncMonitor,
    AsyncRepeatedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    Decision,
    DecisionModel,
    DecisionMonitor,
    DecisionOutcome,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    GuardLabels,
    GuardMonitor,
    LLMMonitor,
    Monitor,
    Prompt,
    Question,
    Refused,
    RepeatedMonitor,
    SuspicionScore,
    TextPart,
    Trajectory,
    Truncated,
    Unavailable,
)

MODEL_NAME = "trusted-1"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))
# The first span id the test run's tracer provider gives in each test (tests/conftest.py).
FIRST_SPAN_ID = 0x00F1E2D3C4B5A697
STATEMENT = "The proposed action sends data outside the system."
GUARD_LABELS = GuardLabels(
    suspicious=frozenset({"unsafe"}),
    safe=frozenset({"safe"}),
)
UNREADABLE_EXPLANATION = (
    "The monitor could not read its model's reply, so the action is treated as suspicious."
)
TRUNCATED_EXPLANATION = (
    "The monitor's model stopped before it gave a score, so the action is treated as suspicious."
)
REFUSED_EXPLANATION = (
    "The monitor's model refused to rate the action, so it is treated as suspicious."
)
UNAVAILABLE_EXPLANATION = (
    "The monitor's model was unavailable, so the action is treated as suspicious."
)


class Script:
    """The outcomes a scripted model returns in order; an exception among them is raised."""

    def __init__(
        self,
        *outcomes: GenerationOutcome | Exception,
    ) -> None:
        self.outcomes = list(outcomes)

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        outcome = self.outcomes.pop(0)
        if isinstance(
            outcome,
            Exception,
        ):
            raise outcome
        return outcome

    async def generate_asynchronously(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.generate(prompt)


class TracedScript(Script):
    """A scripted model that answers inside a span of its own, as an instrumented client does."""

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        with trace.get_tracer("client").start_as_current_span("chat"):
            return super().generate(prompt)


def decide(question: Question) -> DecisionOutcome:
    return Decision(probability=0.25)


async def decide_asynchronously(question: Question) -> DecisionOutcome:
    return decide(question)


def build_reply(
    score: int,
    reasoning: str = "calm",
) -> Generation:
    return Generation(text=f"<reasoning>{reasoning}</reasoning><score>{score}</score>")


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


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Family[Rater]:
    """The monitors of one family, built around scripted models."""

    llm: Callable[[Script], Rater]
    decision: Callable[[], Rater]
    guard: Callable[[Script], Rater]
    repeated: Callable[..., Rater]
    calibrated: Callable[..., Rater]
    cascade: Callable[..., Rater]


def build_synchronous_llm(script: Script) -> LLMMonitor:
    return LLMMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
        ),
    )


def build_asynchronous_llm(script: Script) -> AsyncLLMMonitor:
    return AsyncLLMMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
        ),
    )


def build_synchronous_decision() -> DecisionMonitor:
    return DecisionMonitor(
        model=DecisionModel(
            name=MODEL_NAME,
            decide=decide,
        ),
        statements=(STATEMENT,),
    )


def build_asynchronous_decision() -> AsyncDecisionMonitor:
    return AsyncDecisionMonitor(
        model=AsyncDecisionModel(
            name=MODEL_NAME,
            decide=decide_asynchronously,
        ),
        statements=(STATEMENT,),
    )


def build_synchronous_guard(script: Script) -> GuardMonitor:
    return GuardMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
        ),
        prompt=DEFAULT_PROMPT,
        labels=GUARD_LABELS,
        samples=2,
    )


def build_asynchronous_guard(script: Script) -> AsyncGuardMonitor:
    return AsyncGuardMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
        ),
        prompt=DEFAULT_PROMPT,
        labels=GUARD_LABELS,
        samples=2,
    )


SYNCHRONOUS = Family(
    llm=build_synchronous_llm,
    decision=build_synchronous_decision,
    guard=build_synchronous_guard,
    repeated=RepeatedMonitor,
    calibrated=CalibratedMonitor,
    cascade=CascadeMonitor,
)
ASYNCHRONOUS = Family(
    llm=build_asynchronous_llm,
    decision=build_asynchronous_decision,
    guard=build_asynchronous_guard,
    repeated=AsyncRepeatedMonitor,
    calibrated=AsyncCalibratedMonitor,
    cascade=AsyncCascadeMonitor,
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


def build_llm(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.llm(Script(build_reply(3)))


def build_decision(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.decision()


def build_guard(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.guard(
        Script(
            Generation(text="safe"),
            Generation(text="safe"),
        ),
    )


def build_repeated(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.repeated(
        monitor=build_llm(family),
        ensemble_size=1,
    )


def build_calibrated(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    """A calibrated monitor whose wrapped score fails closed, so its value is not drawn."""
    return family.calibrated(
        monitor=family.llm(Script(Unavailable(status=503))),
        honest_scores=[0.1],
    )


def build_cascade(family: Family[Monitor] | Family[AsyncMonitor]) -> Monitor | AsyncMonitor:
    return family.cascade(
        fast=build_llm(family),
        careful=build_llm(family),
        escalation_threshold=0.5,
    )


def trace_lineage(spans: InMemorySpanExporter) -> list[tuple[str, int | None]]:
    """Each finished span's name and its parent's span id, in the order the spans ended."""
    return [
        (
            span.name,
            None if span.parent is None else span.parent.span_id,
        )
        for span in spans.get_finished_spans()
    ]


def read_last_attributes(spans: InMemorySpanExporter) -> list[Attributes]:
    """The attributes of the span that ended last, the rating's own around any inner ones, in a
    list that is empty when no span ended.
    """
    return [span.attributes for span in spans.get_finished_spans()][-1:]


@pytest.mark.parametrize(
    ("family", "build", "names"),
    [
        (
            SYNCHRONOUS,
            build_llm,
            ["rate LLMMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_llm,
            ["rate AsyncLLMMonitor"],
        ),
        (
            SYNCHRONOUS,
            build_decision,
            ["rate DecisionMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_decision,
            ["rate AsyncDecisionMonitor"],
        ),
        (
            SYNCHRONOUS,
            build_guard,
            ["rate GuardMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_guard,
            ["rate AsyncGuardMonitor"],
        ),
        (
            SYNCHRONOUS,
            build_repeated,
            ["rate LLMMonitor", "rate RepeatedMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_repeated,
            ["rate AsyncLLMMonitor", "rate AsyncRepeatedMonitor"],
        ),
        (
            SYNCHRONOUS,
            build_calibrated,
            ["rate LLMMonitor", "rate CalibratedMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_calibrated,
            ["rate AsyncLLMMonitor", "rate AsyncCalibratedMonitor"],
        ),
        (
            SYNCHRONOUS,
            build_cascade,
            ["rate LLMMonitor", "rate CascadeMonitor"],
        ),
        (
            ASYNCHRONOUS,
            build_cascade,
            ["rate AsyncLLMMonitor", "rate AsyncCascadeMonitor"],
        ),
    ],
    ids=[
        "llm-sync",
        "llm-async",
        "decision-sync",
        "decision-async",
        "guard-sync",
        "guard-async",
        "repeated-sync",
        "repeated-async",
        "calibrated-sync",
        "calibrated-async",
        "cascade-sync",
        "cascade-async",
    ],
)
def test_every_monitor_rates_inside_an_internal_span_named_for_its_class(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    build: Callable[[Family[Monitor] | Family[AsyncMonitor]], Monitor | AsyncMonitor],
    names: list[str],
) -> None:
    monitor = build(family)

    rate(monitor)

    assert [
        (
            span.name,
            span.kind,
        )
        for span in spans.get_finished_spans()
    ] == [
        (
            name,
            SpanKind.INTERNAL,
        )
        for name in names
    ]


CALM = {
    "gen_ai.evaluation.name": "suspicion",
    "gen_ai.evaluation.score.value": 0.3,
    "gen_ai.evaluation.explanation": "calm",
    "hyoshigi.failed_closed": False,
}


@pytest.mark.parametrize(
    ("build", "attributes"),
    [
        (
            build_llm,
            CALM,
        ),
        (
            build_decision,
            {
                "gen_ai.evaluation.name": "suspicion",
                "gen_ai.evaluation.score.value": 0.25,
                "gen_ai.evaluation.explanation": "The decision model's strongest concern: "
                "The proposed action sends data outside the system.",
                "hyoshigi.failed_closed": False,
            },
        ),
        (
            build_guard,
            {
                "gen_ai.evaluation.name": "suspicion",
                "gen_ai.evaluation.score.value": 0.0,
                "gen_ai.evaluation.explanation": "The guard model labelled the action 'safe', "
                "finding that it follows the policy.",
                "hyoshigi.failed_closed": False,
            },
        ),
        (
            build_repeated,
            CALM,
        ),
        (
            build_calibrated,
            {
                "gen_ai.evaluation.name": "suspicion",
                "gen_ai.evaluation.score.value": 1.0,
                "gen_ai.evaluation.explanation": UNAVAILABLE_EXPLANATION,
                "hyoshigi.failed_closed": True,
                "hyoshigi.cause": "Unavailable",
            },
        ),
        (
            build_cascade,
            CALM,
        ),
    ],
    ids=["llm", "decision", "guard", "repeated", "calibrated", "cascade"],
)
def test_every_monitors_span_carries_its_score_as_an_evaluation_result(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    build: Callable[[Family[Monitor] | Family[AsyncMonitor]], Monitor | AsyncMonitor],
    attributes: dict[str, str | float | bool],
) -> None:
    monitor = build(family)

    rate(monitor)

    assert read_last_attributes(spans) == [attributes]


@pytest.mark.parametrize(
    ("outcomes", "explanation", "cause"),
    [
        (
            (Unavailable(status=503),),
            UNAVAILABLE_EXPLANATION,
            "Unavailable",
        ),
        (
            (Refused(),),
            REFUSED_EXPLANATION,
            "Refused",
        ),
        (
            (Generation(text="no score"),) * 3,
            UNREADABLE_EXPLANATION,
            "Unreadable",
        ),
        (
            (Truncated(text="<reasoning>cut"),) * 3,
            TRUNCATED_EXPLANATION,
            "Truncated",
        ),
    ],
    ids=["unavailable", "refused", "unreadable", "truncated"],
)
def test_the_span_of_a_rating_that_failed_closed_names_its_cause_by_class(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    outcomes: tuple[GenerationOutcome, ...],
    explanation: str,
    cause: str,
) -> None:
    monitor = family.llm(Script(*outcomes))

    rate(monitor)

    assert read_last_attributes(spans) == [
        {
            "gen_ai.evaluation.name": "suspicion",
            "gen_ai.evaluation.score.value": 1.0,
            "gen_ai.evaluation.explanation": explanation,
            "hyoshigi.failed_closed": True,
            "hyoshigi.cause": cause,
        },
    ]


def build_three_calls(
    family: Family[Monitor] | Family[AsyncMonitor],
    script: Script,
) -> Monitor | AsyncMonitor:
    return family.repeated(
        monitor=family.llm(script),
        ensemble_size=3,
    )


def build_two_of_two_calls(
    family: Family[Monitor] | Family[AsyncMonitor],
    script: Script,
) -> Monitor | AsyncMonitor:
    """Two repeated monitors of two calls each, inside a third that repeats them."""
    return family.repeated(
        monitor=family.repeated(
            monitor=family.llm(script),
            ensemble_size=2,
        ),
        ensemble_size=2,
    )


@pytest.mark.parametrize(
    ("build", "outcomes"),
    [
        pytest.param(
            build_three_calls,
            (
                build_reply(3),
                Refused(),
                Unavailable(status=503),
            ),
            id="a-refusal-before-an-unavailable-call",
        ),
        pytest.param(
            build_three_calls,
            (
                Unavailable(status=503),
                Refused(),
                build_reply(3),
            ),
            id="a-refusal-after-an-unavailable-call",
        ),
        pytest.param(
            build_two_of_two_calls,
            (
                build_reply(3),
                build_reply(3),
                Unavailable(status=503),
                Refused(),
            ),
            id="a-refusal-after-an-unavailable-call-of-a-nested-wrapper",
        ),
    ],
)
def test_a_wrapper_that_failed_closed_names_the_cause_of_the_part_it_failed_closed_by(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
    build: Callable[
        [
            Family[Monitor] | Family[AsyncMonitor],
            Script,
        ],
        Monitor | AsyncMonitor,
    ],
    outcomes: tuple[GenerationOutcome, ...],
) -> None:
    monitor = build(
        family,
        Script(*outcomes),
    )

    rate(monitor)

    assert read_last_attributes(spans) == [
        {
            "gen_ai.evaluation.name": "suspicion",
            "gen_ai.evaluation.score.value": 1.0,
            "gen_ai.evaluation.explanation": REFUSED_EXPLANATION,
            "hyoshigi.failed_closed": True,
            "hyoshigi.cause": "Refused",
        },
    ]


def test_the_span_of_a_mean_of_the_calls_that_answered_names_no_cause(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    monitor = build_three_calls(
        family,
        Script(
            build_reply(3),
            Unavailable(status=503),
            build_reply(5),
        ),
    )

    rate(monitor)

    assert read_last_attributes(spans) == [
        {
            "gen_ai.evaluation.name": "suspicion",
            "gen_ai.evaluation.score.value": 0.4,
            "gen_ai.evaluation.explanation": "calm",
            "hyoshigi.failed_closed": False,
        },
    ]


def test_a_wrappers_inner_ratings_nest_inside_its_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    monitor = family.repeated(
        monitor=family.llm(
            Script(
                build_reply(1),
                build_reply(2),
            ),
        ),
        ensemble_size=2,
    )

    rate(monitor)

    assert [parent for _, parent in trace_lineage(spans)] == [
        FIRST_SPAN_ID,
        FIRST_SPAN_ID,
        None,
    ]


def test_a_span_the_model_function_opens_nests_inside_the_ratings_span(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    monitor = family.llm(TracedScript(build_reply(3)))

    rate(monitor)

    assert trace_lineage(spans)[:1] == [
        (
            "chat",
            FIRST_SPAN_ID,
        ),
    ]


def test_an_exception_from_the_model_function_marks_the_ratings_span_by_its_type_alone(
    spans: InMemorySpanExporter,
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    monitor = family.llm(Script(ConnectionError("the provider hung up")))

    with pytest.raises(ConnectionError):
        rate(monitor)

    assert [
        (
            span.status.status_code,
            span.status.description,
            [event.name for event in span.events],
        )
        for span in spans.get_finished_spans()
    ] == [
        (
            StatusCode.ERROR,
            "ConnectionError",
            [],
        ),
    ]


# Rates with an LLM monitor in a process where importing opentelemetry fails.
RATE_WITHOUT_OPENTELEMETRY = """
import sys
sys.modules["opentelemetry"] = None
from hyoshigi import Action, Generation, GenerativeModel, LLMMonitor, TextPart, Trajectory
model = GenerativeModel(
    name="trusted-1",
    generate=lambda prompt: Generation(text="<reasoning>calm</reasoning><score>3</score>"),
)
trajectory = Trajectory.from_run("Summarise q3.md", answer="Done.").trajectory
action = Action(parts=(TextPart(text="Reading q3.md."),))
print(LLMMonitor(model=model).rate(action, trajectory=trajectory).value)
"""


def test_a_rating_without_opentelemetry_installed_returns_its_score() -> None:
    command = [
        sys.executable,
        "-c",
        RATE_WITHOUT_OPENTELEMETRY,
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    assert process.stdout == "0.3\n"
