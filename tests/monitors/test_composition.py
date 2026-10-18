"""The wrappers: the mean of repeated ratings, a calibrated rank and a cascade."""

import asyncio
import numbers
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
from functools import partial
from typing import assert_never

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
from hyoshigi.monitors.composition import (
    AsyncCalibratedMonitor,
    AsyncCascadeMonitor,
    AsyncRepeatedMonitor,
    CalibratedMonitor,
    CascadeMonitor,
    RepeatedMonitor,
)
from hyoshigi.monitors.llm import AsyncLLMMonitor, LLMMonitor
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import Composite, FailedClosed, Rationale, SuspicionScore

MODEL_NAME = "trusted-1"
TIMEOUT_SECONDS = 5
TASK = "Summarise q3.md"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))
# Unsorted, so a monitor that ranks among them in the order given ranks wrongly.
HONEST_SCORES = [0.3, 0.1, 0.2, 0.2]
STEPS = tuple(Action(parts=(TextPart(text=f"step {number}"),)) for number in range(50))
UNAVAILABLE_EXPLANATION = (
    "The monitor's model was unavailable, so the action is treated as suspicious."
)
UNAVAILABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation=UNAVAILABLE_EXPLANATION,
    evidence=FailedClosed(model=MODEL_NAME, cause=Unavailable(status=503)),
)
FAILURES = (Unavailable(status=503),) * 8
REFUSED_EXPLANATION = (
    "The monitor's model refused to rate the action, so it is treated as suspicious."
)
REFUSED_SCORE = SuspicionScore(
    value=1.0,
    explanation=REFUSED_EXPLANATION,
    evidence=FailedClosed(model=MODEL_NAME, cause=Refused(explanation="I cannot assess this.")),
)


class Script:
    """The outcomes a scripted model returns in order, and the prompts it received."""

    def __init__(self, *outcomes: GenerationOutcome) -> None:
        self.outcomes = list(outcomes)
        self.received: list[Prompt] = []

    def generate(self, prompt: Prompt) -> GenerationOutcome:
        self.received.append(prompt)
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


def build_action(text: str = "Reading q3.md.") -> Action:
    return Action(parts=(TextPart(text=text),))


def build_acted_trajectory(
    text: str = "Let me look.", *, rating: SuspicionScore | None = None
) -> Trajectory:
    """A trajectory whose history ends in a new action, which carries the rating if given."""
    trajectory = build_trajectory()
    acted = build_action(text)
    trajectory.act(acted)
    if rating is not None:
        trajectory.attach(rating, to=acted)
    return trajectory


def run[Result](awaitable: Awaitable[Result]) -> Result:
    """Run on a fresh event loop; a test that waits longer than the timeout fails."""

    async def finish() -> Result:
        return await asyncio.wait_for(awaitable, timeout=TIMEOUT_SECONDS)

    return asyncio.run(finish())


def build_synchronous_llm(script: Script, *, highest_score: int = 10) -> LLMMonitor:
    model = GenerativeModel(name=MODEL_NAME, generate=script.generate)
    return LLMMonitor(model=model, highest_score=highest_score)


def build_asynchronous_llm(script: Script, *, highest_score: int = 10) -> AsyncLLMMonitor:
    model = AsyncGenerativeModel(name=MODEL_NAME, generate=script.generate_asynchronously)
    return AsyncLLMMonitor(model=model, highest_score=highest_score)


def rate_synchronously(monitor: Monitor, action: Action, trajectory: Trajectory) -> SuspicionScore:
    return monitor.rate(action, trajectory=trajectory)


def rate_asynchronously(
    monitor: AsyncMonitor, action: Action, trajectory: Trajectory
) -> SuspicionScore:
    return run(monitor.rate(action, trajectory=trajectory))


class Layer(Enum):
    """A wrapper, built with options that do not matter to the property it is used to test."""

    REPEATED = auto()
    CALIBRATED = auto()
    CASCADE = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class Family[Wrapped]:
    """The wrappers of one family, built around scripted LLM monitors, and how they are rated."""

    build_llm: Callable[..., Wrapped]
    repeated: Callable[..., Wrapped]
    calibrated: Callable[..., Wrapped]
    cascade: Callable[..., Wrapped]
    rate: Callable[[Wrapped, Action, Trajectory], SuspicionScore]

    def rate_repeated(
        self, script: Script, *, ensemble_size: int, highest_score: int = 10
    ) -> SuspicionScore:
        llm = self.build_llm(script, highest_score=highest_score)
        monitor = self.repeated(monitor=llm, ensemble_size=ensemble_size)
        return self.rate(monitor, ACTION, build_trajectory())

    def rate_calibrated(
        self,
        script: Script,
        *,
        honest_scores: Iterable[float],
        random_seed: int | None = None,
        action: Action = ACTION,
        trajectory: Trajectory | None = None,
    ) -> SuspicionScore:
        monitor = self.calibrated(
            monitor=self.build_llm(script), honest_scores=honest_scores, random_seed=random_seed
        )
        return self.rate(monitor, action, build_trajectory() if trajectory is None else trajectory)

    def rate_cascade(
        self, fast: Script, careful: Script, *, escalation_threshold: float
    ) -> SuspicionScore:
        monitor = self.cascade(
            fast=self.build_llm(fast),
            careful=self.build_llm(careful),
            escalation_threshold=escalation_threshold,
        )
        return self.rate(monitor, ACTION, build_trajectory())

    def rate_stack(self, script: Script, *, layers: tuple[Layer, ...]) -> SuspicionScore:
        """Rate with the layers wrapped around an LLM monitor, the first layer innermost."""
        monitor = self.build_llm(script)
        for layer in layers:
            monitor = self.wrap(layer, monitor)
        return self.rate(monitor, ACTION, build_trajectory())

    def wrap(self, layer: Layer, monitor: Wrapped) -> Wrapped:
        match layer:
            case Layer.REPEATED:
                return self.repeated(monitor=monitor, ensemble_size=2)
            case Layer.CALIBRATED:
                return self.calibrated(monitor=monitor, honest_scores=HONEST_SCORES)
            case Layer.CASCADE:
                return self.cascade(fast=monitor, careful=monitor, escalation_threshold=0.5)
            case unreachable:
                assert_never(unreachable)


SYNCHRONOUS = Family(
    build_llm=build_synchronous_llm,
    repeated=RepeatedMonitor,
    calibrated=CalibratedMonitor,
    cascade=CascadeMonitor,
    rate=rate_synchronously,
)
ASYNCHRONOUS = Family(
    build_llm=build_asynchronous_llm,
    repeated=AsyncRepeatedMonitor,
    calibrated=AsyncCalibratedMonitor,
    cascade=AsyncCascadeMonitor,
    rate=rate_asynchronously,
)


@pytest.fixture(params=[SYNCHRONOUS, ASYNCHRONOUS], ids=["sync", "async"])
def family(request: pytest.FixtureRequest) -> Family[Monitor] | Family[AsyncMonitor]:
    return request.param


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


@pytest.mark.parametrize(
    ("scores", "value"),
    [([10, 10], 1e-12), ([3, 0], 0.0)],
    ids=["twelfth-place-kept", "thirteenth-place-rounded-away"],
)
def test_the_mean_keeps_the_twelfth_decimal_place_and_rounds_away_the_thirteenth(
    family: Family[Monitor] | Family[AsyncMonitor], scores: list[int], value: float
) -> None:
    script = Script(*(build_reply(score) for score in scores))

    score = family.rate_repeated(script, ensemble_size=2, highest_score=10**13)

    assert score.value == value


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


def test_the_evidence_holds_each_calls_score_in_order(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    script = Script(build_reply(1, "a"), build_reply(6, "b"), build_reply(2, "c"))

    score = family.rate_repeated(script, ensemble_size=3)

    assert score.evidence == Composite(
        parts=(build_score(0.1, "a"), build_score(0.6, "b"), build_score(0.2, "c"))
    )


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
        (
            partial(CalibratedMonitor, monitor=ASYNCHRONOUS_LLM, honest_scores=[0.5]),
            "monitor must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncCalibratedMonitor for an AsyncMonitor.",
        ),
        (
            partial(AsyncCalibratedMonitor, monitor=SYNCHRONOUS_LLM, honest_scores=[0.5]),
            "monitor must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use CalibratedMonitor for a Monitor.",
        ),
        (
            partial(
                CascadeMonitor,
                fast=ASYNCHRONOUS_LLM,
                careful=SYNCHRONOUS_LLM,
                escalation_threshold=0.5,
            ),
            "fast must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncCascadeMonitor for an AsyncMonitor.",
        ),
        (
            partial(
                CascadeMonitor,
                fast=SYNCHRONOUS_LLM,
                careful=ASYNCHRONOUS_LLM,
                escalation_threshold=0.5,
            ),
            "careful must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncCascadeMonitor for an AsyncMonitor.",
        ),
        (
            partial(
                AsyncCascadeMonitor,
                fast=SYNCHRONOUS_LLM,
                careful=ASYNCHRONOUS_LLM,
                escalation_threshold=0.5,
            ),
            "fast must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use CascadeMonitor for a Monitor.",
        ),
        (
            partial(
                AsyncCascadeMonitor,
                fast=ASYNCHRONOUS_LLM,
                careful=SYNCHRONOUS_LLM,
                escalation_threshold=0.5,
            ),
            "careful must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use CascadeMonitor for a Monitor.",
        ),
    ],
    ids=[
        "async-monitor-for-the-sync-repeated",
        "sync-monitor-for-the-async-repeated",
        "async-monitor-for-the-sync-calibrated",
        "sync-monitor-for-the-async-calibrated",
        "async-fast-for-the-sync-cascade",
        "async-careful-for-the-sync-cascade",
        "sync-fast-for-the-async-cascade",
        "sync-careful-for-the-async-cascade",
    ],
)
def test_a_monitor_of_the_other_family_is_refused_naming_the_wrapper_it_belongs_to(
    build: Callable[[], Monitor | AsyncMonitor], message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build()

    assert str(raised.value) == message


@pytest.mark.parametrize("random_seed", [None, 7], ids=["unseeded", "seeded"])
@pytest.mark.parametrize(
    ("reply", "lower", "midpoint", "upper"),
    [
        (0, 0.0, 0.1, 0.2),
        (1, 0.0, 0.2, 0.4),
        (2, 0.2, 0.5, 0.8),
        (3, 0.6, 0.8, 1.0),
        (10, 0.8, 0.9, 1.0),
    ],
    ids=[
        "below-all",
        "tied-with-the-lowest",
        "tied-with-two",
        "tied-with-the-highest",
        "above-all",
    ],
)
def test_a_score_lands_uniformly_in_its_place_among_the_honest_scores(
    family: Family[Monitor] | Family[AsyncMonitor],
    random_seed: int | None,
    reply: int,
    lower: float,
    midpoint: float,
    upper: float,
) -> None:
    values = [
        family.rate_calibrated(
            Script(build_reply(reply)),
            honest_scores=HONEST_SCORES,
            random_seed=random_seed,
            action=step,
        ).value
        for step in STEPS
    ]

    # Of the honest scores 0.1, 0.2, 0.2, 0.3 and a raw score of reply / 10, the rank is
    # (below + u * (tied + 1)) / 5 for u uniform on [0, 1). 50 draws all fall in one half of
    # the place with a chance of 2 * 0.5 ** 50.
    assert all(lower <= value < upper for value in values)
    assert min(values) < midpoint <= max(values)


def test_the_explanation_is_the_wrapped_scores_and_the_evidence_holds_it(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    script = Script(build_reply(2, "It reads the file the task names."))

    score = family.rate_calibrated(script, honest_scores=HONEST_SCORES)

    assert (score.explanation, score.evidence) == (
        "It reads the file the task names.",
        Composite(parts=(build_score(0.2, "It reads the file the task names."),)),
    )


@pytest.mark.parametrize(
    ("failure", "failed_score"),
    [
        (Unavailable(status=503), UNAVAILABLE_SCORE),
        (Refused(explanation="I cannot assess this."), REFUSED_SCORE),
    ],
    ids=["unavailable", "refused"],
)
def test_a_score_that_failed_closed_is_returned_unchanged_and_not_ranked(
    family: Family[Monitor] | Family[AsyncMonitor],
    failure: Unavailable | Refused,
    failed_score: SuspicionScore,
) -> None:
    script = Script(failure)

    score = family.rate_calibrated(script, honest_scores=HONEST_SCORES)

    assert score == failed_score


def test_honest_scores_given_as_a_generator_are_all_ranked_among(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    script = Script(build_reply(9))

    score = family.rate_calibrated(script, honest_scores=(0.1 for _ in range(500)))

    # Above all 500 honest scores, so in the top place: 500 / 501 up to 1.
    assert 500 / 501 <= score.value < 1.0


def rate_seeded(
    family: Family[Monitor] | Family[AsyncMonitor],
    *,
    seed: int = 7,
    action: Action,
    trajectory: Trajectory,
) -> float:
    """The value of a raw score of 0.3, tied with the highest honest score, so in [0.6, 1.0)."""
    return family.rate_calibrated(
        Script(build_reply(3)),
        honest_scores=HONEST_SCORES,
        random_seed=seed,
        action=action,
        trajectory=trajectory,
    ).value


def test_a_seeded_draw_is_the_same_for_the_same_action_and_history_whatever_their_ids(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    values = [
        rate_seeded(family, action=build_action(), trajectory=build_acted_trajectory())
        for _ in range(2)
    ]

    assert 0.6 <= values[0] < 1.0
    assert values[0] == values[1]


def test_a_seeded_draw_ignores_the_ratings_attached_to_the_history(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    trajectories = [build_acted_trajectory(rating=build_score(0.4)), build_acted_trajectory()]
    values = [
        rate_seeded(family, action=build_action(), trajectory=trajectory)
        for trajectory in trajectories
    ]

    assert 0.6 <= values[0] < 1.0
    assert values[0] == values[1]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ((7, "Reading q3.md.", "Let me look."), (8, "Reading q3.md.", "Let me look.")),
        ((7, "Reading q3.md.", "Let me look."), (7, "Reading q4.md.", "Let me look.")),
        ((7, "Reading q3.md.", "Let me look."), (7, "Reading q3.md.", "Let me think.")),
    ],
    ids=["the-seed", "the-action", "the-history"],
)
def test_a_seeded_draw_differs_with_the_seed_the_action_and_the_history(
    family: Family[Monitor] | Family[AsyncMonitor],
    first: tuple[int, str, str],
    second: tuple[int, str, str],
) -> None:
    values = [
        rate_seeded(
            family,
            seed=seed,
            action=build_action(action_text),
            trajectory=build_acted_trajectory(history_text),
        )
        for seed, action_text, history_text in (first, second)
    ]

    assert values[0] != values[1]


def test_an_unseeded_draw_differs_between_calls_for_the_same_action_and_history(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    values = [
        family.rate_calibrated(Script(build_reply(3)), honest_scores=HONEST_SCORES).value
        for _ in range(20)
    ]

    assert len(set(values)) > 1


@pytest.mark.parametrize(
    ("honest_scores", "message"),
    [
        ([], "honest_scores must hold at least one score"),
        ((score for score in ()), "honest_scores must hold at least one score"),
        ("0.5", "honest_scores must be an iterable of numbers between 0 and 1, got '0.5'"),
        (
            b"\x00\x01",
            "honest_scores must be an iterable of numbers between 0 and 1, "
            "got an instance of bytes",
        ),
        (5, "honest_scores must be an iterable of numbers between 0 and 1, got 5"),
        (None, "honest_scores must be an iterable of numbers between 0 and 1, got None"),
        ([0.2, float("nan")], "honest_scores[1] must be between 0 and 1, got nan"),
        ([1.5], "honest_scores[0] must be between 0 and 1, got 1.5"),
        ([0.1, 0.2, -0.1], "honest_scores[2] must be between 0 and 1, got -0.1"),
        ([0.1, True], "honest_scores[1] must be a number between 0 and 1, got True"),
        (["0.5"], "honest_scores[0] must be a number between 0 and 1, got '0.5'"),
    ],
    ids=[
        "empty-list",
        "empty-generator",
        "string",
        "bytes",
        "number",
        "none",
        "nan",
        "above-one",
        "below-zero",
        "bool",
        "string-score",
    ],
)
def test_honest_scores_that_cannot_be_ranked_among_are_refused_naming_the_problem(
    family: Family[Monitor] | Family[AsyncMonitor], honest_scores: object, message: str
) -> None:
    llm = family.build_llm(Script())

    with pytest.raises(ConfigurationError) as raised:
        family.calibrated(monitor=llm, honest_scores=honest_scores)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("random_seed", "message"),
    [
        (1.5, "random_seed must be an integer, got 1.5"),
        (True, "random_seed must be an integer, got True"),
        ("7", "random_seed must be an integer, got '7'"),
    ],
    ids=["float", "bool", "string"],
)
def test_a_random_seed_that_is_not_an_integer_is_refused_naming_it(
    family: Family[Monitor] | Family[AsyncMonitor], random_seed: object, message: str
) -> None:
    llm = family.build_llm(Script())

    with pytest.raises(ConfigurationError) as raised:
        family.calibrated(monitor=llm, honest_scores=HONEST_SCORES, random_seed=random_seed)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    "build",
    [
        partial(
            CalibratedMonitor,
            monitor=SYNCHRONOUS_LLM,
            honest_scores=HONEST_SCORES,
            random_seed=ArrayInteger(7),
        ),
        partial(
            AsyncCalibratedMonitor,
            monitor=ASYNCHRONOUS_LLM,
            honest_scores=HONEST_SCORES,
            random_seed=ArrayInteger(7),
        ),
    ],
    ids=["sync", "async"],
)
def test_a_seed_that_is_a_whole_number_but_not_an_int_is_kept_as_an_int(
    build: Callable[[], CalibratedMonitor | AsyncCalibratedMonitor],
) -> None:
    calibrated = build()

    assert (type(calibrated.random_seed), calibrated.random_seed) == (int, 7)


@pytest.mark.parametrize(
    ("fast_reply", "kept", "careful_asks"),
    [
        (2, build_score(0.2, "fast"), 0),
        (3, build_score(0.9, "careful"), 1),
        (4, build_score(0.9, "careful"), 1),
    ],
    ids=["below-the-threshold", "at-the-threshold", "above-the-threshold"],
)
def test_a_fast_score_below_the_threshold_is_kept_and_one_at_or_above_it_escalates(
    family: Family[Monitor] | Family[AsyncMonitor],
    fast_reply: int,
    kept: SuspicionScore,
    careful_asks: int,
) -> None:
    fast = Script(build_reply(fast_reply, "fast"))
    careful = Script(build_reply(9, "careful"))

    score = family.rate_cascade(fast, careful, escalation_threshold=0.3)

    assert (score, len(careful.received)) == (kept, careful_asks)


@pytest.mark.parametrize("escalation_threshold", [0.5, 1.0], ids=["half", "one"])
def test_a_fast_score_that_failed_closed_always_escalates(
    family: Family[Monitor] | Family[AsyncMonitor], escalation_threshold: float
) -> None:
    fast = Script(Unavailable(status=503))
    careful = Script(build_reply(9, "careful"))

    score = family.rate_cascade(fast, careful, escalation_threshold=escalation_threshold)

    assert score == build_score(0.9, "careful")


def test_a_careful_score_that_failed_closed_is_returned_unchanged(
    family: Family[Monitor] | Family[AsyncMonitor],
) -> None:
    fast = Script(build_reply(5, "fast"))
    careful = Script(Unavailable(status=503))

    score = family.rate_cascade(fast, careful, escalation_threshold=0.3)

    assert score == UNAVAILABLE_SCORE


@pytest.mark.parametrize(
    ("escalation_threshold", "message"),
    [
        (1.2, "escalation_threshold must be between 0 and 1, got 1.2"),
        (-0.1, "escalation_threshold must be between 0 and 1, got -0.1"),
        (float("nan"), "escalation_threshold must be between 0 and 1, got nan"),
        (True, "escalation_threshold must be a number between 0 and 1, got True"),
        ("0.5", "escalation_threshold must be a number between 0 and 1, got '0.5'"),
        (None, "escalation_threshold must be a number between 0 and 1, got None"),
    ],
    ids=["above-one", "below-zero", "nan", "bool", "string", "none"],
)
def test_an_escalation_threshold_that_is_not_from_0_to_1_is_refused_naming_it(
    family: Family[Monitor] | Family[AsyncMonitor], escalation_threshold: object, message: str
) -> None:
    llm = family.build_llm(Script())

    with pytest.raises(ConfigurationError) as raised:
        family.cascade(fast=llm, careful=llm, escalation_threshold=escalation_threshold)

    assert str(raised.value) == message


@pytest.mark.parametrize(
    "build",
    [
        partial(
            CascadeMonitor,
            fast=SYNCHRONOUS_LLM,
            careful=SYNCHRONOUS_LLM,
            escalation_threshold=Fraction(1, 2),
        ),
        partial(
            AsyncCascadeMonitor,
            fast=ASYNCHRONOUS_LLM,
            careful=ASYNCHRONOUS_LLM,
            escalation_threshold=Fraction(1, 2),
        ),
    ],
    ids=["sync", "async"],
)
def test_a_threshold_that_is_a_real_number_but_not_a_float_is_kept_as_a_float(
    build: Callable[[], CascadeMonitor | AsyncCascadeMonitor],
) -> None:
    cascade = build()

    assert (type(cascade.escalation_threshold), cascade.escalation_threshold) == (float, 0.5)


@pytest.mark.parametrize(
    ("layers", "outcomes"),
    [
        ((Layer.REPEATED,), FAILURES),
        ((Layer.CALIBRATED,), FAILURES),
        ((Layer.CASCADE,), FAILURES),
        ((Layer.REPEATED, Layer.CALIBRATED), FAILURES),
        ((Layer.CALIBRATED, Layer.REPEATED), FAILURES),
        ((Layer.CASCADE, Layer.CALIBRATED), FAILURES),
        ((Layer.CALIBRATED, Layer.CASCADE), FAILURES),
        ((Layer.REPEATED, Layer.CASCADE), FAILURES),
        ((Layer.CASCADE, Layer.REPEATED), FAILURES),
        (
            (Layer.REPEATED, Layer.REPEATED),
            (build_reply(1), Unavailable(status=503), build_reply(1), build_reply(1)),
        ),
    ],
    ids=[
        "repeated",
        "calibrated",
        "cascade",
        "calibrated-around-repeated",
        "repeated-around-calibrated",
        "calibrated-around-cascade",
        "cascade-around-calibrated",
        "cascade-around-repeated",
        "repeated-around-cascade",
        "repeated-around-a-repeated-with-one-failed-call",
    ],
)
def test_a_failure_in_any_stack_of_wrappers_fails_the_whole_score_closed_at_1(
    family: Family[Monitor] | Family[AsyncMonitor],
    layers: tuple[Layer, ...],
    outcomes: tuple[GenerationOutcome, ...],
) -> None:
    script = Script(*outcomes)

    score = family.rate_stack(script, layers=layers)

    assert (score.failed_closed, score.value) == (True, 1.0)
