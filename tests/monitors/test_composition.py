"""The wrappers: the mean of repeated ratings, a calibrated rank and a cascade."""

import asyncio
import numbers
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

import pytest

from hyoshigi import (
    Action,
    AsyncGenerativeModel,
    ConfigurationError,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    Prompt,
    Refused,
    TextPart,
    Trajectory,
    Unavailable,
)
from hyoshigi.monitors.composition import AsyncRepeatedMonitor, RepeatedMonitor
from hyoshigi.monitors.llm import AsyncLLMMonitor, LLMMonitor
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Composite, FailedClosed, Rationale, SuspicionScore

MODEL_NAME = "trusted-1"
TIMEOUT_SECONDS = 5
TASK = "Summarise q3.md"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))
UNAVAILABLE_EXPLANATION = (
    "The monitor's model was unavailable, so the action is treated as suspicious."
)
UNAVAILABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation=UNAVAILABLE_EXPLANATION,
    evidence=FailedClosed(model=MODEL_NAME, cause=Unavailable(status=503)),
)
REFUSED_EXPLANATION = (
    "The monitor's model refused to rate the action, so it is treated as suspicious."
)
REFUSED_SCORE = SuspicionScore(
    value=1.0,
    explanation=REFUSED_EXPLANATION,
    evidence=FailedClosed(model=MODEL_NAME, cause=Refused(explanation="I cannot assess this.")),
)


class Script:
    """The outcomes a scripted model returns in order."""

    def __init__(self, *outcomes: GenerationOutcome) -> None:
        self.outcomes = list(outcomes)

    def generate(self, prompt: Prompt) -> GenerationOutcome:
        return self.outcomes.pop(0)

    async def generate_asynchronously(self, prompt: Prompt) -> GenerationOutcome:
        return self.generate(prompt)


def build_reply(score: int, reasoning: str = "r") -> Generation:
    return Generation(text=f"<reasoning>{reasoning}</reasoning><score>{score}</score>")


def build_score(value: float, explanation: str = "r", *, model: str = MODEL_NAME) -> SuspicionScore:
    """The score an LLM monitor gives a reply: a value and the reasoning, from a model."""
    return SuspicionScore(value=value, explanation=explanation, evidence=Rationale(model=model))


def build_trajectory() -> Trajectory:
    return Trajectory.from_run(TASK, answer="Done.").trajectory


def run[Result](awaitable: Awaitable[Result]) -> Result:
    """Run on a fresh event loop; a test that waits longer than the timeout fails."""

    async def finish() -> Result:
        return await asyncio.wait_for(awaitable, timeout=TIMEOUT_SECONDS)

    return asyncio.run(finish())


def build_synchronous_llm(script: Script) -> LLMMonitor:
    model = GenerativeModel(name=MODEL_NAME, generate=script.generate)
    return LLMMonitor(model=model)


def build_asynchronous_llm(script: Script) -> AsyncLLMMonitor:
    model = AsyncGenerativeModel(name=MODEL_NAME, generate=script.generate_asynchronously)
    return AsyncLLMMonitor(model=model)


def rate_synchronously(monitor: Monitor, action: Action, trajectory: Trajectory) -> SuspicionScore:
    return monitor.rate(action, trajectory=trajectory)


def rate_asynchronously(
    monitor: AsyncMonitor, action: Action, trajectory: Trajectory
) -> SuspicionScore:
    return run(monitor.rate(action, trajectory=trajectory))


@dataclass(frozen=True, slots=True, kw_only=True)
class Family[Wrapped]:
    """The wrappers of one family, built around scripted LLM monitors, and how they are rated."""

    build_llm: Callable[[Script], Wrapped]
    repeated: Callable[..., Wrapped]
    rate: Callable[[Wrapped, Action, Trajectory], SuspicionScore]

    def rate_repeated(
        self, script: Script, *, ensemble_size: int, action: Action = ACTION
    ) -> SuspicionScore:
        monitor = self.repeated(monitor=self.build_llm(script), ensemble_size=ensemble_size)
        return self.rate(monitor, action, build_trajectory())


SYNCHRONOUS = Family(
    build_llm=build_synchronous_llm,
    repeated=RepeatedMonitor,
    rate=rate_synchronously,
)
ASYNCHRONOUS = Family(
    build_llm=build_asynchronous_llm,
    repeated=AsyncRepeatedMonitor,
    rate=rate_asynchronously,
)


@pytest.fixture(params=[SYNCHRONOUS, ASYNCHRONOUS], ids=["sync", "async"])
def family(request: pytest.FixtureRequest) -> Family[Monitor] | Family[AsyncMonitor]:
    return request.param


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("scores", "ensemble_size", "value"),
    [
        ([1, 2, 6], 3, 0.3),
        ([10], 1, 1.0),
        ([0, 0], 2, 0.0),
        ([1, 0, 2], 3, 0.1),
        ([0, 0, 3], 3, 0.1),
    ],
    ids=["three-calls", "one-call", "all-lowest", "tenth-from-0.1-0.0-0.2", "tenth-from-0-0-0.3"],
)
def test_the_value_is_the_mean_of_the_calls_rounded_to_12_decimal_places(
    family: Family[Monitor] | Family[AsyncMonitor],
    scores: list[int],
    ensemble_size: int,
    value: float,
) -> None:
    script = Script(*(build_reply(score) for score in scores))

    score = family.rate_repeated(script, ensemble_size=ensemble_size)

    assert score.value == value


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("scores", "explanation"),
    [
        ([(2, "calm"), (6, "odd"), (4, "mild")], "odd"),
        ([(7, "first"), (3, "second"), (3, "third")], "first"),
        ([(3, "first"), (3, "second"), (7, "third")], "third"),
        ([(2, "low"), (5, "tied first"), (5, "tied second")], "tied first"),
        ([(5, "first"), (5, "second"), (5, "third")], "first"),
    ],
    ids=["middle", "first", "last", "first-of-two-tied", "first-of-three-tied"],
)
def test_the_explanation_is_the_most_suspicious_calls_and_the_first_of_a_tie(
    family: Family[Monitor] | Family[AsyncMonitor],
    scores: list[tuple[int, str]],
    explanation: str,
) -> None:
    script = Script(*(build_reply(score, reasoning) for score, reasoning in scores))

    score = family.rate_repeated(script, ensemble_size=3)

    assert score.explanation == explanation


@pytest.mark.xfail(strict=True)
def test_the_evidence_holds_each_calls_score_in_order(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    script = Script(build_reply(1, "a"), build_reply(6, "b"), build_reply(2, "c"))

    score = family.rate_repeated(script, ensemble_size=3)

    assert score.evidence == Composite(
        parts=(build_score(0.1, "a"), build_score(0.6, "b"), build_score(0.2, "c"))
    )


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("outcomes", "explanation", "parts"),
    [
        (
            [build_reply(3), Refused(explanation="I cannot assess this."), Unavailable(status=503)],
            REFUSED_EXPLANATION,
            (build_score(0.3), REFUSED_SCORE, UNAVAILABLE_SCORE),
        ),
        (
            [build_reply(10, "genuine"), Unavailable(status=503), build_reply(2)],
            UNAVAILABLE_EXPLANATION,
            (build_score(1.0, "genuine"), UNAVAILABLE_SCORE, build_score(0.2)),
        ),
    ],
    ids=["first-of-two-failures", "after-a-genuine-top-score"],
)
def test_a_call_that_failed_closed_fails_the_whole_score_closed_with_its_explanation(
    family: Family[Monitor] | Family[AsyncMonitor],
    outcomes: list[GenerationOutcome],
    explanation: str,
    parts: tuple[SuspicionScore, ...],
) -> None:
    script = Script(*outcomes)

    score = family.rate_repeated(script, ensemble_size=3)

    assert score == SuspicionScore(
        value=1.0, explanation=explanation, evidence=Composite(parts=parts)
    )
    assert score.failed_closed


@pytest.mark.xfail(strict=True)
def test_the_async_wrapper_asks_its_calls_at_once() -> None:
    async def rate_at_a_barrier() -> SuspicionScore:
        # Each call waits for the other two, so only calls in flight together finish.
        barrier = asyncio.Barrier(3)

        async def generate(prompt: Prompt) -> GenerationOutcome:
            await barrier.wait()
            return build_reply(4)

        model = AsyncGenerativeModel(name=MODEL_NAME, generate=generate)
        monitor = AsyncRepeatedMonitor(monitor=AsyncLLMMonitor(model=model), ensemble_size=3)
        return await monitor.rate(ACTION, trajectory=build_trajectory())

    score = run(rate_at_a_barrier())

    assert score.value == 0.4


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("ensemble_size", "message"),
    [
        (0, "ensemble_size must be at least 1, got 0"),
        (-2, "ensemble_size must be at least 1, got -2"),
        (2.5, "ensemble_size must be a whole number of at least 1, got 2.5"),
        (3.0, "ensemble_size must be a whole number of at least 1, got 3.0"),
        (True, "ensemble_size must be a whole number of at least 1, got True"),
        ("3", "ensemble_size must be a whole number of at least 1, got '3'"),
        (None, "ensemble_size must be a whole number of at least 1, got None"),
    ],
    ids=["zero", "negative", "float", "whole-float", "bool", "string", "none"],
)
def test_an_ensemble_size_that_is_not_a_count_of_at_least_1_is_refused_naming_it(
    family: Family[Monitor] | Family[AsyncMonitor], ensemble_size: object, message: str
) -> None:
    llm = family.build_llm(Script())

    with pytest.raises(ConfigurationError) as raised:
        family.repeated(monitor=llm, ensemble_size=ensemble_size)

    assert str(raised.value) == message


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


SYNCHRONOUS_LLM = build_synchronous_llm(Script())
ASYNCHRONOUS_LLM = build_asynchronous_llm(Script())


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "build",
    [
        partial(RepeatedMonitor, monitor=SYNCHRONOUS_LLM, ensemble_size=ArrayInteger(2)),
        partial(AsyncRepeatedMonitor, monitor=ASYNCHRONOUS_LLM, ensemble_size=ArrayInteger(2)),
    ],
    ids=["sync", "async"],
)
def test_a_whole_number_that_is_not_an_int_is_kept_as_an_int(
    build: Callable[[], RepeatedMonitor | AsyncRepeatedMonitor],
) -> None:
    repeated = build()

    assert (type(repeated.ensemble_size), repeated.ensemble_size) == (int, 2)


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            partial(RepeatedMonitor, monitor=ASYNCHRONOUS_LLM, ensemble_size=2),
            "monitor must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncRepeatedMonitor for an AsyncMonitor.",
        ),
        (
            partial(AsyncRepeatedMonitor, monitor=SYNCHRONOUS_LLM, ensemble_size=2),
            "monitor must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use RepeatedMonitor for a Monitor.",
        ),
    ],
    ids=["async-monitor-for-the-sync-wrapper", "sync-monitor-for-the-async-wrapper"],
)
def test_a_monitor_of_the_other_family_is_refused_naming_the_wrapper_it_belongs_to(
    build: Callable[[], Monitor | AsyncMonitor], message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build()

    assert str(raised.value) == message
