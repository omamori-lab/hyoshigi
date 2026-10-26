"""The LLM monitor: a generative model reasons about an action, then scores it.

The model reasons inside <reasoning> tags and then gives an integer from `lowest_score` to
`highest_score` inside <score> tags. The integer maps linearly onto a suspicion from 0 to 1,
and the reasoning, without the score, becomes the explanation. A reply is readable only in
exactly that shape: the first reasoning block, then one score tag, with nothing but whitespace
between them or after the score.

<https://omamori-lab.github.io/hyoshigi/how-to/rating-with-an-llm-monitor/>
"""

import re
from dataclasses import dataclass, field
from typing import assert_never

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    Prompt,
    Refused,
    Rejected,
    Truncated,
    Unavailable,
    check_outcomes,
)
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor, name_decision_monitor
from hyoshigi.monitors.prompts import DEFAULT_PROMPT, MonitorPrompt
from hyoshigi.monitors.rendering import Rendering, render
from hyoshigi.monitors.score import (
    Rationale,
    SuspicionScore,
    Unperceivable,
    Unreadable,
    fail_closed,
)
from hyoshigi.monitors.tracing import record_rating, trace_rating
from hyoshigi.monitors.view import View
from hyoshigi.options import check_instance, read_count, read_integer
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory

# A plain integer in ASCII digits; int() alone also takes forms such as 1_0 and fullwidth digits.
_INTEGER = re.compile(r"[+-]?[0-9]+")
_REASONING_START = re.compile(
    r"<reasoning>",
    flags=re.IGNORECASE,
)
_REASONING_END = re.compile(
    r"</reasoning>",
    flags=re.IGNORECASE,
)
_FINAL_SCORE = re.compile(
    r"\s*<score>(?P<score>[^<]*)</score>\s*",
    flags=re.IGNORECASE,
)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class LLMMonitor(Monitor):
    """Rates an action with a generative model that reasons, then scores on an integer scale.

    A reply with no readable score, and one cut off at the model's output limit, is asked for
    again, up to `max_parse_retries` more times; with no score by then the rating fails
    closed. A custom `prompt` must ask for the reply's shape, or every reply is unreadable.
    The ends of the scale are integers, the lowest below the highest; an option that cannot
    work raises ConfigurationError when the monitor is built.
    """

    model: GenerativeModel
    view: View = field(default_factory=View)
    prompt: MonitorPrompt = DEFAULT_PROMPT
    lowest_score: int = 0
    highest_score: int = 10
    max_parse_retries: int = 2

    def __post_init__(self) -> None:
        check_instance(
            self.model,
            expected=GenerativeModel,
            name="model",
            hint=name_decision_monitor(
                self.model,
                otherwise="Use AsyncLLMMonitor for an AsyncGenerativeModel.",
            ),
        )
        _check_options(self)

    def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the model's score, which fails closed when it gives none."""
        with trace_rating(self) as span:
            return record_rating(
                drive(
                    plan_rating(
                        self,
                        action=action,
                        trajectory=trajectory,
                    ),
                    ask=self.model.generate,
                ),
                span=span,
            )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncLLMMonitor(AsyncMonitor):
    """Rates an action as LLMMonitor does, with a generative model that is awaited."""

    model: AsyncGenerativeModel
    view: View = field(default_factory=View)
    prompt: MonitorPrompt = DEFAULT_PROMPT
    lowest_score: int = 0
    highest_score: int = 10
    max_parse_retries: int = 2

    def __post_init__(self) -> None:
        check_instance(
            self.model,
            expected=AsyncGenerativeModel,
            name="model",
            hint=name_decision_monitor(
                self.model,
                otherwise="Use LLMMonitor for a GenerativeModel.",
            ),
        )
        _check_options(self)

    async def rate(
        self,
        action: Action,
        *,
        trajectory: Trajectory,
    ) -> SuspicionScore:
        """Rate the action with the model's score, which fails closed when it gives none."""
        with trace_rating(self) as span:
            return record_rating(
                await drive_concurrently(
                    plan_rating(
                        self,
                        action=action,
                        trajectory=trajectory,
                    ),
                    ask=self.model.generate,
                ),
                span=span,
            )


def _check_options(monitor: LLMMonitor | AsyncLLMMonitor) -> None:
    """Raise ConfigurationError for an option that cannot work, naming it.

    The scale's ends and the retry count are kept as ints, whatever whole numbers were given.
    """
    check_instance(
        monitor.view,
        expected=View,
        name="view",
    )
    check_instance(
        monitor.prompt,
        expected=MonitorPrompt,
        name="prompt",
    )
    lowest = read_integer(
        monitor.lowest_score,
        name="lowest_score",
    )
    highest = read_integer(
        monitor.highest_score,
        name="highest_score",
    )
    if lowest >= highest:
        raise ConfigurationError(f"lowest_score ({lowest}) must be below highest_score ({highest})")
    retries = read_count(
        monitor.max_parse_retries,
        name="max_parse_retries",
        minimum=0,
    )
    # The dataclass is frozen, so the ints are set past its guard.
    object.__setattr__(
        monitor,
        "lowest_score",
        lowest,
    )
    object.__setattr__(
        monitor,
        "highest_score",
        highest,
    )
    object.__setattr__(
        monitor,
        "max_parse_retries",
        retries,
    )


# The cause of an attempt that is asked again.
type _Retry = Unreadable | Truncated


def plan_rating(
    monitor: LLMMonitor | AsyncLLMMonitor,
    *,
    action: Action,
    trajectory: Trajectory,
) -> Plan[Prompt, GenerationOutcome, SuspicionScore]:
    """Plan the rating of an action: render it, ask the monitor's model, read the score.

    The model is asked again, with the same prompt, after a reply that holds no readable
    score and after one cut off at its output limit, until `max_parse_retries` further asks
    are spent. Without a score by then, the rating fails closed with the last cause. A
    refusal, an unavailable model, and content the model cannot perceive fail closed at once.
    """
    name = monitor.model.name
    rendering = render(
        action,
        trajectory=trajectory,
        view=monitor.view,
        perceives=monitor.model.perceives,
    )
    if rendering.unperceivable:
        return fail_closed(
            Unperceivable(content=rendering.unperceivable),
            model=name,
        )
    request = monitor.prompt.build(
        rendering,
        lowest_score=monitor.lowest_score,
        highest_score=monitor.highest_score,
    )
    cause: _Retry = Unreadable()
    for _ in range(monitor.max_parse_retries + 1):
        outcomes = yield (request,)
        check_outcomes(
            outcomes,
            union=GenerationOutcome,
            function="generate",
        )
        (outcome,) = outcomes
        settled = _settle(
            outcome,
            monitor=monitor,
            rendering=rendering,
        )
        if isinstance(
            settled,
            SuspicionScore,
        ):
            return settled
        cause = settled
    return fail_closed(
        cause,
        model=name,
    )


def _settle(
    outcome: GenerationOutcome,
    *,
    monitor: LLMMonitor | AsyncLLMMonitor,
    rendering: Rendering,
) -> SuspicionScore | _Retry:
    """The score an outcome ends the rating with, or the cause that earns another ask."""
    match outcome:
        case Generation(text=text):
            return _score_reply(
                text,
                monitor=monitor,
                rendering=rendering,
            )
        case Truncated():
            return outcome
        case Refused() | Rejected() | Unavailable():
            return fail_closed(
                outcome,
                model=monitor.model.name,
            )
        case unreachable:
            assert_never(unreachable)


def _score_reply(
    text: str,
    *,
    monitor: LLMMonitor | AsyncLLMMonitor,
    rendering: Rendering,
) -> SuspicionScore | Unreadable:
    """Score a reply on the monitor's scale, mapped onto 0 to 1, or return Unreadable."""
    match _read_reply(
        text,
        lowest_score=monitor.lowest_score,
        highest_score=monitor.highest_score,
    ):
        case _Reply(
            reasoning=reasoning,
            score=score,
        ):
            span = monitor.highest_score - monitor.lowest_score
            return SuspicionScore(
                value=(score - monitor.lowest_score) / span,
                explanation=reasoning,
                evidence=Rationale(
                    model=monitor.model.name,
                    hidden_reasoning=rendering.hidden_reasoning,
                ),
            )
        case Unreadable() as unreadable:
            return unreadable
        case unreachable:
            assert_never(unreachable)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class _Reply:
    """A readable reply: its reasoning, stripped, and its score."""

    reasoning: str
    score: int


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class _ReasoningBlock:
    """The text of a reply's first reasoning block, and the offset where the block ends."""

    text: str
    end: int


def _read_reply(
    text: str,
    *,
    lowest_score: int,
    highest_score: int,
) -> _Reply | Unreadable:
    """Read a reply's reasoning and score, or return Unreadable.

    The reasoning must hold some text, and the score tag, alone after it, an integer in the
    scale. A score tag quoted inside the reasoning is part of it and never read. Text before
    the first reasoning block is ignored, so a reply that gives its own verdict there and then
    quotes a reasoning block and a score is read as the quoted one.
    """
    match _find_reasoning(text):
        case _ReasoningBlock(
            text=reasoning,
            end=end,
        ) if reasoning.strip():
            return _read_score(
                text,
                start=end,
                reasoning=reasoning.strip(),
                lowest_score=lowest_score,
                highest_score=highest_score,
            )
        case _ReasoningBlock() | Unreadable():
            return Unreadable()
        case unreachable:
            assert_never(unreachable)


def _read_score(
    text: str,
    *,
    start: int,
    reasoning: str,
    lowest_score: int,
    highest_score: int,
) -> _Reply | Unreadable:
    """Read the score tag that follows the reasoning, alone to the text's end, as an integer
    in the scale; matched from there, so a score tag elsewhere never counts.
    """
    final = _FINAL_SCORE.fullmatch(
        text,
        start,
    )
    if final is None:
        return Unreadable()
    match _parse_score(
        final["score"].strip(),
        lowest_score=lowest_score,
        highest_score=highest_score,
    ):
        case int(score):
            return _Reply(
                reasoning=reasoning,
                score=score,
            )
        case Unreadable() as unreadable:
            return unreadable
        case unreachable:
            assert_never(unreachable)


def _find_reasoning(text: str) -> _ReasoningBlock | Unreadable:
    """Return the first reasoning block, from the first start tag to the first end tag after
    it, or Unreadable when the reply has none.

    When the first start tag has no end tag after it, no later one has. Two linear scans find
    that; a lazy search for a start tag, any text and an end tag starts again at each start
    tag, so a reply of many start tags and no end tag takes time quadratic in its length.
    """
    start = _REASONING_START.search(text)
    if start is None:
        return Unreadable()
    end = _REASONING_END.search(
        text,
        start.end(),
    )
    if end is None:
        return Unreadable()
    return _ReasoningBlock(
        text=text[start.end() : end.start()],
        end=end.end(),
    )


def _parse_score(
    score_text: str,
    *,
    lowest_score: int,
    highest_score: int,
) -> int | Unreadable:
    """Read the score as an integer in the scale, or return Unreadable.

    Python refuses to convert an integer of more than 4,300 digits, by default; such a score
    is no integer in the scale either.
    """
    if not _INTEGER.fullmatch(score_text):
        return Unreadable()
    try:
        score = int(score_text)
    except ValueError:
        return Unreadable()
    return score if lowest_score <= score <= highest_score else Unreadable()
