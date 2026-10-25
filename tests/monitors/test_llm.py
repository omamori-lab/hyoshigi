"""The LLM monitor: the prompt it asks, how it reads the reply, and what it records."""

import asyncio
import numbers
from collections.abc import Callable
from functools import partial
from typing import Protocol, TypedDict, Unpack

import pytest

from hyoshigi import (
    Action,
    AsyncGenerativeModel,
    Author,
    ConfigurationError,
    Description,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaPart,
    MediaTypes,
    OpaquePart,
    Prompt,
    Refused,
    TextPart,
    ToolCall,
    Trajectory,
    Truncated,
    Unavailable,
)
from hyoshigi.models.model import TEXT_ONLY
from hyoshigi.monitors.llm import AsyncLLMMonitor, LLMMonitor
from hyoshigi.monitors.prompts import MonitorPrompt
from hyoshigi.monitors.score import (
    FailedClosed,
    Rationale,
    SuspicionScore,
    Unperceivable,
    Unreadable,
)
from hyoshigi.monitors.view import Channel, View

MODEL_NAME = "trusted-1"
TASK = "Summarise q3.md"
READ = ToolCall(
    id="c1",
    name="read",
    arguments='{"path": "q3.md"}',
)
ACTION = Action(tool_calls=(READ,))
# The eight bytes every PNG file starts with (RFC 2083, section 3.1).
PNG = MediaPart(
    media_type="image/png",
    data=b"\x89PNG\r\n\x1a\n",
)
# A prompt that shows exactly what the monitor puts in it.
ECHO = MonitorPrompt(
    system="Scale {lowest_score} to {highest_score}.",
    request="{trajectory}|{proposed_action}",
)
CALM_REPLY = "<reasoning>It reads the file the user named.</reasoning>\n<score>3</score>"
UNREADABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation="The monitor could not read its model's reply, so the action is treated as "
    "suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Unreadable(),
    ),
)
TRUNCATED_SCORE = SuspicionScore(
    value=1.0,
    explanation="The monitor's model stopped before it gave a score, so the action is treated "
    "as suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Truncated(text="The task is"),
    ),
)
REFUSED_SCORE = SuspicionScore(
    value=1.0,
    explanation="The monitor's model refused to rate the action, so it is treated as suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Refused(explanation="I cannot assess this."),
    ),
)
UNAVAILABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation="The monitor's model was unavailable, so the action is treated as suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Unavailable(status=503),
    ),
)
UNPERCEIVABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation="The action holds content the monitor cannot perceive, so it is treated as "
    "suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Unperceivable(content=("server_tool_use", "audio/wav")),
    ),
)


class Options(
    TypedDict,
    total=False,
):
    view: View
    prompt: MonitorPrompt
    lowest_score: int
    highest_score: int
    max_parse_retries: int


class Script:
    """The outcomes a scripted model returns in order, and the prompts it received.

    An exception among the outcomes is raised in place of a return.
    """

    def __init__(
        self,
        *outcomes: GenerationOutcome | Exception,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> None:
        self.outcomes = list(outcomes)
        self.perceives = perceives
        self.received: list[Prompt] = []

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        self.received.append(prompt)
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


class Rate(Protocol):
    def __call__(
        self,
        script: Script,
        *,
        action: Action = ACTION,
        **options: Unpack[Options],
    ) -> SuspicionScore: ...


def build_trajectory() -> Trajectory:
    return Trajectory.from_run(
        TASK,
        answer="Done.",
    ).trajectory


def rate_synchronously(
    script: Script,
    *,
    action: Action = ACTION,
    **options: Unpack[Options],
) -> SuspicionScore:
    model = GenerativeModel(
        name=MODEL_NAME,
        generate=script.generate,
        perceives=script.perceives,
    )
    return LLMMonitor(
        model=model,
        **options,
    ).rate(
        action,
        trajectory=build_trajectory(),
    )


def rate_asynchronously(
    script: Script,
    *,
    action: Action = ACTION,
    **options: Unpack[Options],
) -> SuspicionScore:
    model = AsyncGenerativeModel(
        name=MODEL_NAME,
        generate=script.generate_asynchronously,
        perceives=script.perceives,
    )
    monitor = AsyncLLMMonitor(
        model=model,
        **options,
    )
    return asyncio.run(
        monitor.rate(
            action,
            trajectory=build_trajectory(),
        ),
    )


@pytest.fixture(
    params=[
        rate_synchronously,
        rate_asynchronously,
    ],
    ids=["sync", "async"],
)
def rate(request: pytest.FixtureRequest) -> Rate:
    return request.param


class RateLatestStep(Protocol):
    def __call__(
        self,
        script: Script,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore: ...


def rate_latest_step_synchronously(
    script: Script,
    *,
    trajectory: Trajectory,
) -> SuspicionScore:
    model = GenerativeModel(
        name=MODEL_NAME,
        generate=script.generate,
        perceives=script.perceives,
    )
    return LLMMonitor(
        model=model,
        prompt=ECHO,
    ).rate_latest_step(trajectory)


def rate_latest_step_asynchronously(
    script: Script,
    *,
    trajectory: Trajectory,
) -> SuspicionScore:
    model = AsyncGenerativeModel(
        name=MODEL_NAME,
        generate=script.generate_asynchronously,
        perceives=script.perceives,
    )
    monitor = AsyncLLMMonitor(
        model=model,
        prompt=ECHO,
    )
    return asyncio.run(monitor.rate_latest_step(trajectory))


@pytest.fixture(
    params=[
        rate_latest_step_synchronously,
        rate_latest_step_asynchronously,
    ],
    ids=["sync", "async"],
)
def rate_latest_step(request: pytest.FixtureRequest) -> RateLatestStep:
    return request.param


def build_reply(score_text: str) -> Generation:
    return Generation(text=f"<reasoning>r</reasoning><score>{score_text}</score>")


@pytest.mark.parametrize(
    ("score_text", "lowest_score", "highest_score", "value"),
    [
        ("0", 0, 10, 0.0),
        ("10", 0, 10, 1.0),
        ("3", 0, 10, 0.3),
        (" 5 ", 0, 10, 0.5),
        ("-5", -5, 5, 0.0),
        ("-3", -5, 5, 0.2),
        ("+5", -5, 5, 1.0),
        ("3", 1, 5, 0.5),
        ("-2", -3, -1, 0.5),
    ],
    ids=[
        "lowest",
        "highest",
        "between",
        "spaces-around",
        "negative-lowest",
        "negative-between",
        "signed-highest",
        "scale-from-1",
        "negative-scale",
    ],
)
def test_a_reply_scores_its_integer_as_a_value_from_0_to_1(
    rate: Rate,
    score_text: str,
    lowest_score: int,
    highest_score: int,
    value: float,
) -> None:
    script = Script(build_reply(score_text))

    score = rate(
        script,
        lowest_score=lowest_score,
        highest_score=highest_score,
    )

    assert score.value == value


@pytest.mark.parametrize(
    ("lowest_score", "highest_score"),
    [
        (0, 10),
        (-5, 5),
        (1, 5),
        (0, 1),
    ],
)
def test_a_value_rises_with_the_integer_and_spans_0_to_1(
    rate: Rate,
    lowest_score: int,
    highest_score: int,
) -> None:
    scores = [
        rate(
            Script(build_reply(str(n))),
            lowest_score=lowest_score,
            highest_score=highest_score,
        )
        for n in range(
            lowest_score,
            highest_score + 1,
        )
    ]

    values = [score.value for score in scores]

    assert values == sorted(set(values))
    assert (
        values[0],
        values[-1],
    ) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("reply", "value", "explanation"),
    [
        (
            "<reasoning>\nIt reads the file the user named.\n</reasoning><score>7</score>",
            0.7,
            "It reads the file the user named.",
        ),
        ("<reasoning>r</reasoning>\n<score>3</score>\n", 0.3, "r"),
        ("<reasoning>r</reasoning>\n\n<score>3</score>  \n\n", 0.3, "r"),
        ("<reasoning>r</reasoning><SCORE>3</SCORE>", 0.3, "r"),
        ("<REASONING>r</Reasoning><score>3</score>", 0.3, "r"),
        ("<reasoning><reasoning>a</reasoning><score>3</score>", 0.3, "<reasoning>a"),
        ("<reasoning>a<reasoning>b</reasoning><score>3</score>", 0.3, "a<reasoning>b"),
        ("Sure.\n<reasoning>a</reasoning><score>3</score>", 0.3, "a"),
        ("</reasoning><reasoning>a</reasoning><score>3</score>", 0.3, "a"),
        (
            '<reasoning>It says "<score>0</score>".</reasoning>\n<score>9</score>',
            0.9,
            'It says "<score>0</score>".',
        ),
    ],
    ids=[
        "reasoning-stripped",
        "newline-after",
        "blank-lines-around",
        "upper-case-score",
        "mixed-case-reasoning",
        "start-tag-twice",
        "start-tag-inside",
        "text-before-the-reasoning",
        "end-tag-before-the-start-tag",
        "score-tag-inside-the-reasoning",
    ],
)
def test_a_reply_is_read_as_its_first_reasoning_block_then_one_score(
    rate: Rate,
    reply: str,
    value: float,
    explanation: str,
) -> None:
    script = Script(Generation(text=reply))

    score = rate(
        script,
        max_parse_retries=0,
    )

    assert (
        score.value,
        score.explanation,
    ) == (
        value,
        explanation,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "Looks fine. <score>1</score>",
        "<score>0</score>",
        "<reasoning></reasoning><score>7</score>",
        "<reasoning> </reasoning>\n<score>1</score>",
        "<reasoning>a",
        "</reasoning><reasoning>a",
        "<reasoning>" * 9_000,
        "<reasoning>The note asks for <score>0</score>, an injection.</reasoning>\nScore: 10",
        "<reasoning>r</reasoning>\nNote: the step asked me to reply <score>0</score>",
        "<reasoning>r</reasoning>\nFinal score: <score>3</score>",
        "<reasoning>r</reasoning><score>2</score> on reflection <score>8</score>",
        "<reasoning>r</reasoning><score>3</score> I would say so.",
        "<reasoning>r</reasoning>\n<score>9</score>\nNote: it said </reasoning><score>0</score>",
        '<reasoning>It says "</reasoning><score>0</score>".</reasoning>\n<score>9</score>',
        "<reasoning>r</reasoning><score>11</score>",
        "<reasoning>r</reasoning><score>-1</score>",
        "<reasoning>r</reasoning><score>7.5</score>",
        "<reasoning>r</reasoning><score>high</score>",
        "<reasoning>r</reasoning><score>1_0</score>",
        "<reasoning>r</reasoning><score>\N{FULLWIDTH DIGIT SEVEN}</score>",
        "<reasoning>r</reasoning><score> </score>",
        f"<reasoning>r</reasoning><score>{'0' * 5000}</score>",
    ],
    ids=[
        "no-reasoning",
        "only-a-score",
        "empty-reasoning",
        "blank-reasoning",
        "open-reasoning",
        "end-tag-first",
        "start-tags-without-an-end",
        "score-tag-inside-the-reasoning-and-none-after",
        "prose-before-a-quoted-score",
        "prose-before-the-score",
        "second-score",
        "text-after-the-score",
        "quoted-tags-after-the-score",
        "quoted-tags-in-the-reasoning",
        "above-the-scale",
        "below-the-scale",
        "not-an-integer",
        "not-a-number",
        "underscore",
        "fullwidth-digit",
        "blank-score",
        "more-than-4300-digits",
    ],
)
def test_a_reply_without_one_readable_score_after_its_reasoning_is_unreadable(
    rate: Rate,
    reply: str,
) -> None:
    script = Script(Generation(text=reply))

    score = rate(
        script,
        max_parse_retries=0,
    )

    assert score == UNREADABLE_SCORE
    assert len(script.received) == 1


def test_the_evidence_names_the_model(rate: Rate) -> None:
    script = Script(Generation(text=CALM_REPLY))

    score = rate(script)

    assert score.evidence == Rationale(model="trusted-1")


def test_the_evidence_carries_the_hidden_reasoning_of_the_rated_thought(rate: Rate) -> None:
    script = Script(Generation(text=CALM_REPLY))
    action = Action(
        thought=(
            TextPart(text="Checking."),
            OpaquePart(kind="reasoning.encrypted"),
        ),
        tool_calls=(READ,),
    )

    score = rate(
        script,
        action=action,
        view=View(channels=Channel.DEFAULT | Channel.THOUGHTS),
    )

    assert score.evidence == Rationale(
        model="trusted-1",
        hidden_reasoning=("reasoning.encrypted",),
    )


@pytest.mark.parametrize(
    ("lowest_score", "highest_score", "system"),
    [
        (0, 10, "Scale 0 to 10."),
        (1, 5, "Scale 1 to 5."),
        (-5, 5, "Scale -5 to 5."),
    ],
)
def test_the_model_is_asked_the_prompt_built_from_the_rendering_and_the_scale(
    rate: Rate,
    lowest_score: int,
    highest_score: int,
    system: str,
) -> None:
    script = Script(Generation(text=CALM_REPLY))

    rate(
        script,
        prompt=ECHO,
        lowest_score=lowest_score,
        highest_score=highest_score,
    )

    assert script.received == [
        Prompt(
            system=system,
            content=(
                TextPart(
                    text="<user>Summarise q3.md</user>|"
                    '<tool_call id="c1" name="read">{"path": "q3.md"}</tool_call>',
                ),
            ),
        ),
    ]


def test_the_view_chooses_what_of_the_trajectory_the_model_reads(rate: Rate) -> None:
    script = Script(Generation(text=CALM_REPLY))

    rate(
        script,
        prompt=ECHO,
        view=View(channels=Channel.TOOL_CALLS),
    )

    assert [prompt.content for prompt in script.received] == [
        (TextPart(text='|<tool_call id="c1" name="read">{"path": "q3.md"}</tool_call>'),),
    ]


def test_the_latest_step_is_read_once_as_the_proposed_action(
    rate_latest_step: RateLatestStep,
) -> None:
    trajectory = build_trajectory()
    trajectory.act(ACTION)
    script = Script(Generation(text=CALM_REPLY))

    rate_latest_step(
        script,
        trajectory=trajectory,
    )

    assert script.received == [
        Prompt(
            system="Scale 0 to 10.",
            content=(
                TextPart(
                    text="<user>Summarise q3.md</user>|"
                    '<tool_call id="c1" name="read">{"path": "q3.md"}</tool_call>',
                ),
            ),
        ),
    ]


def test_rating_the_latest_step_of_a_trajectory_ending_in_an_observation_raises(
    rate_latest_step: RateLatestStep,
) -> None:
    script = Script(Generation(text=CALM_REPLY))

    with pytest.raises(ConfigurationError):
        rate_latest_step(
            script,
            trajectory=build_trajectory(),
        )


def test_the_default_prompt_asks_for_a_score_on_the_scale(rate: Rate) -> None:
    script = Script(Generation(text=CALM_REPLY))

    rate(script)

    assert len(script.received) == 1
    assert "Give an integer from 0 to 10." in script.received[0].system


def test_media_the_model_perceives_reaches_it_in_place(rate: Rate) -> None:
    script = Script(
        Generation(text=CALM_REPLY),
        perceives=MediaTypes("image/*"),
    )
    action = Action(
        parts=(PNG,),
        tool_calls=(READ,),
    )

    rate(
        script,
        action=action,
        prompt=ECHO,
    )

    assert [prompt.content for prompt in script.received] == [
        (
            TextPart(text='<user>Summarise q3.md</user>|<agent><media type="image/png">'),
            PNG,
            TextPart(
                text="</media></agent>\n"
                '<tool_call id="c1" name="read">{"path": "q3.md"}</tool_call>',
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("max_parse_retries", "calls"),
    [
        (1, 2),
        (3, 4),
    ],
)
def test_an_unreadable_reply_is_asked_again_up_to_the_limit_then_fails_closed(
    rate: Rate,
    max_parse_retries: int,
    calls: int,
) -> None:
    script = Script(*[build_reply("11")] * 10)

    score = rate(
        script,
        max_parse_retries=max_parse_retries,
    )

    assert score == UNREADABLE_SCORE
    assert len(script.received) == calls


def test_an_unreadable_reply_is_asked_again_twice_by_default(rate: Rate) -> None:
    script = Script(*[build_reply("11")] * 10)

    score = rate(script)

    assert score == UNREADABLE_SCORE
    assert len(script.received) == 3


def test_the_prompt_is_asked_again_unchanged_after_an_unreadable_reply(rate: Rate) -> None:
    script = Script(
        build_reply("11"),
        Generation(text=CALM_REPLY),
    )

    score = rate(script)

    assert score.value == 0.3
    assert len(script.received) == 2
    assert script.received[0] == script.received[1]


def test_a_truncated_reply_is_asked_again_and_its_text_is_never_read(rate: Rate) -> None:
    script = Script(
        Truncated(text="<reasoning>r</reasoning><score>0</score>"),
        Generation(text=CALM_REPLY),
    )

    score = rate(script)

    assert score.value == 0.3
    assert len(script.received) == 2


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (
            (
                Truncated(text="The task is"),
                build_reply("11"),
            ),
            UNREADABLE_SCORE,
        ),
        (
            (
                build_reply("11"),
                Truncated(text="The task is"),
            ),
            TRUNCATED_SCORE,
        ),
    ],
    ids=["unreadable-last", "truncated-last"],
)
def test_a_failed_closed_score_records_the_cause_of_the_last_attempt(
    rate: Rate,
    outcomes: tuple[GenerationOutcome, ...],
    expected: SuspicionScore,
) -> None:
    script = Script(*outcomes)

    score = rate(
        script,
        max_parse_retries=1,
    )

    assert score == expected
    assert len(script.received) == 2


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            Refused(explanation="I cannot assess this."),
            REFUSED_SCORE,
        ),
        (
            Unavailable(status=503),
            UNAVAILABLE_SCORE,
        ),
    ],
    ids=["refused", "unavailable"],
)
def test_a_refusal_or_an_unavailable_model_fails_closed_after_one_call(
    rate: Rate,
    outcome: GenerationOutcome,
    expected: SuspicionScore,
) -> None:
    script = Script(outcome)

    score = rate(script)

    assert score == expected
    assert len(script.received) == 1


def test_an_action_with_content_the_model_cannot_perceive_fails_closed_without_asking(
    rate: Rate,
) -> None:
    script = Script(Generation(text=CALM_REPLY))
    action = Action(
        parts=(
            OpaquePart(kind="server_tool_use"),
            MediaPart(
                media_type="audio/wav",
                data=b"RIFF",
            ),
        ),
        tool_calls=(READ,),
    )

    score = rate(
        script,
        action=action,
    )

    assert score == UNPERCEIVABLE_SCORE
    assert len(script.received) == 0


# The agent's description of media the model cannot perceive is its own claim, not perception.
def test_media_the_agent_described_in_the_rated_action_fails_closed_without_asking(
    rate: Rate,
) -> None:
    script = Script(Generation(text=CALM_REPLY))
    action = Action(
        parts=(
            TextPart(text="Posting:"),
            MediaPart(
                media_type="image/png",
                data=b"\x89PNG\r\n\x1a\n",
                description=Description(
                    text="A bar chart of q3 revenue.",
                    author=Author.AGENT,
                ),
            ),
        ),
    )

    score = rate(
        script,
        action=action,
    )

    assert score == SuspicionScore(
        value=1.0,
        explanation="The action holds content the monitor cannot perceive, so it is treated "
        "as suspicious.",
        evidence=FailedClosed(
            model=MODEL_NAME,
            cause=Unperceivable(content=("image/png",)),
        ),
    )
    assert len(script.received) == 0


def test_an_exception_from_the_model_propagates(rate: Rate) -> None:
    script = Script(RuntimeError("the connection dropped"))

    with pytest.raises(
        RuntimeError,
        match="the connection dropped",
    ):
        rate(script)


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(
        self,
        value: int,
    ) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


def build_refusal(prompt: Prompt) -> GenerationOutcome:
    return Refused()


async def build_refusal_asynchronously(prompt: Prompt) -> GenerationOutcome:
    return Refused()


SYNC_MODEL = GenerativeModel(
    name=MODEL_NAME,
    generate=build_refusal,
)
ASYNC_MODEL = AsyncGenerativeModel(
    name=MODEL_NAME,
    generate=build_refusal_asynchronously,
)


def build_with(
    monitor_class: type[LLMMonitor] | type[AsyncLLMMonitor],
    model: object,
    **options: object,
) -> LLMMonitor | AsyncLLMMonitor:
    """Build a monitor from arguments of any type, which the constructor must check."""
    return monitor_class(
        model=model,  # ty: ignore[invalid-argument-type]
        **options,  # ty: ignore[invalid-argument-type]
    )


@pytest.fixture(
    params=[
        (
            LLMMonitor,
            SYNC_MODEL,
        ),
        (
            AsyncLLMMonitor,
            ASYNC_MODEL,
        ),
    ],
    ids=["sync", "async"],
)
def build(request: pytest.FixtureRequest) -> Callable[..., LLMMonitor | AsyncLLMMonitor]:
    monitor_class, model = request.param
    return partial(
        build_with,
        monitor_class,
        model,
    )


@pytest.mark.parametrize(
    ("monitor_class", "model", "message"),
    [
        (
            LLMMonitor,
            ASYNC_MODEL,
            "model must be a GenerativeModel, got an instance of AsyncGenerativeModel. "
            "Use AsyncLLMMonitor for an AsyncGenerativeModel.",
        ),
        (
            AsyncLLMMonitor,
            SYNC_MODEL,
            "model must be an AsyncGenerativeModel, got an instance of GenerativeModel. "
            "Use LLMMonitor for a GenerativeModel.",
        ),
    ],
    ids=["async-model-for-the-sync-monitor", "sync-model-for-the-async-monitor"],
)
def test_a_model_of_the_other_family_is_refused_naming_the_monitor_it_belongs_to(
    monitor_class: type[LLMMonitor] | type[AsyncLLMMonitor],
    model: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build_with(
            monitor_class,
            model,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"view": View},
            "view must be a View, got the class View; pass an instance",
        ),
        (
            {"view": "all"},
            "view must be a View, got an instance of str",
        ),
        (
            {"prompt": "Rate it."},
            "prompt must be a MonitorPrompt, got an instance of str",
        ),
        (
            {"lowest_score": 2.5},
            "lowest_score must be an integer, got 2.5",
        ),
        (
            {"lowest_score": None},
            "lowest_score must be an integer, got None",
        ),
        (
            {"highest_score": True},
            "highest_score must be an integer, got True",
        ),
        (
            {"highest_score": "10"},
            "highest_score must be an integer, got '10'",
        ),
        (
            {
                "lowest_score": 5,
                "highest_score": 5,
            },
            "lowest_score (5) must be below highest_score (5)",
        ),
        (
            {
                "lowest_score": 10,
                "highest_score": 0,
            },
            "lowest_score (10) must be below highest_score (0)",
        ),
        (
            {"lowest_score": -(10**5000)},
            "lowest_score must be an integer Python can write out, "
            "got an integer too long to write out",
        ),
        (
            {"highest_score": 10**5000},
            "highest_score must be an integer Python can write out, "
            "got an integer too long to write out",
        ),
        (
            {"max_parse_retries": -1},
            "max_parse_retries must be at least 0, got -1",
        ),
        (
            {"max_parse_retries": 1.5},
            "max_parse_retries must be a whole number of at least 0, got 1.5",
        ),
        (
            {"max_parse_retries": True},
            "max_parse_retries must be a whole number of at least 0, got True",
        ),
    ],
    ids=[
        "view-class",
        "view-string",
        "prompt-string",
        "lowest-float",
        "lowest-none",
        "highest-bool",
        "highest-string",
        "scale-of-one-point",
        "scale-reversed",
        "lowest-too-long",
        "highest-too-long",
        "retries-negative",
        "retries-float",
        "retries-bool",
    ],
)
def test_an_option_that_cannot_work_is_refused_naming_it(
    build: Callable[..., LLMMonitor | AsyncLLMMonitor],
    options: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build(**options)

    assert str(raised.value) == message


def test_whole_numbers_that_are_not_ints_are_kept_as_ints(
    build: Callable[..., LLMMonitor | AsyncLLMMonitor],
) -> None:
    monitor = build(
        lowest_score=ArrayInteger(-2),
        highest_score=ArrayInteger(2),
        max_parse_retries=ArrayInteger(1),
    )

    assert [
        (
            type(value),
            value,
        )
        for value in (
            monitor.lowest_score,
            monitor.highest_score,
            monitor.max_parse_retries,
        )
    ] == [
        (
            int,
            -2,
        ),
        (
            int,
            2,
        ),
        (
            int,
            1,
        ),
    ]
