"""The LLM monitor reads any reply into a score from 0 to 1 or fails closed."""

import asyncio
from typing import Protocol

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hyoshigi import (
    Action,
    AsyncGenerativeModel,
    Generation,
    GenerativeModel,
    Prompt,
    TextPart,
    Trajectory,
)
from hyoshigi.monitors.llm import AsyncLLMMonitor, LLMMonitor
from hyoshigi.monitors.score import SuspicionScore

MODEL_NAME = "trusted-1"
ACTION = Action(parts=(TextPart(text="Reading q3.md."),))


class Rate(Protocol):
    def __call__(
        self,
        reply: str,
        *,
        lowest_score: int = 0,
        highest_score: int = 10,
    ) -> SuspicionScore: ...


def rate_synchronously(
    reply: str,
    *,
    lowest_score: int = 0,
    highest_score: int = 10,
) -> SuspicionScore:
    """Rate with a model that always gives the reply."""

    def generate(prompt: Prompt) -> Generation:
        return Generation(text=reply)

    return LLMMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=generate,
        ),
        lowest_score=lowest_score,
        highest_score=highest_score,
    ).rate(
        ACTION,
        trajectory=Trajectory(),
    )


def rate_asynchronously(
    reply: str,
    *,
    lowest_score: int = 0,
    highest_score: int = 10,
) -> SuspicionScore:
    """Rate as rate_synchronously does, with the async classes."""

    async def generate(prompt: Prompt) -> Generation:
        return Generation(text=reply)

    return asyncio.run(
        AsyncLLMMonitor(
            model=AsyncGenerativeModel(
                name=MODEL_NAME,
                generate=generate,
            ),
            lowest_score=lowest_score,
            highest_score=highest_score,
        ).rate(
            ACTION,
            trajectory=Trajectory(),
        ),
    )


RATES = pytest.mark.parametrize(
    "rate",
    [
        rate_synchronously,
        rate_asynchronously,
    ],
    ids=["sync", "async"],
)


def build_reply(
    *,
    reasoning: str = "It reads the file the user named.",
    between: str = "",
    score: str = "5",
    after: str = "",
) -> str:
    return f"<reasoning>{reasoning}</reasoning>{between}<score>{score}</score>{after}"


# Pieces of a reply.
FRAGMENTS = [
    "<reasoning>",
    "</reasoning>",
    "<REASONING>",
    "<score>",
    "</score>",
    "<Score>",
    "0",
    "5",
    "10",
    "11",
    "-1",
    "+3",
    "1_0",
    "\u0663",
    "5.0",
    " ",
    "\n",
    "It reads the file the user named.",
]
REPLY = st.one_of(
    st.lists(
        st.one_of(
            st.sampled_from(FRAGMENTS),
            st.text(max_size=5),
        ),
        max_size=8,
    ).map("".join),
    st.text(),
)
# Text with no reasoning end tag.
REASONING_TEXT = st.text().filter(lambda text: "</reasoning>" not in text.lower())
SCORES = st.integers(
    min_value=-3,
    max_value=14,
)
ANY_REPLY = st.one_of(
    REPLY,
    st.builds(
        build_reply,
        reasoning=REASONING_TEXT,
        score=SCORES.map(str),
    ),
)
# Scores that the scale of 0 to 10 does not take.
OUT_OF_SCALE = st.one_of(
    st.integers(min_value=11),
    st.integers(max_value=-1),
).map(str)
TOO_LONG = st.integers(
    min_value=4301,
    max_value=5000,
).map("9".__mul__)
NOT_INTEGERS = st.one_of(
    st.sampled_from(["", " ", "five", "5.0", "1e1", "1_0", "0_5", "\uff11\uff10", "--1", "1 2"]),
    st.text(
        alphabet=st.characters(categories=["L"]),
        min_size=1,
    ),
)
BLANK = st.text(alphabet=" \t\r\n")
# Text with some non-whitespace.
JUNK = (
    st.lists(
        st.one_of(
            st.sampled_from(["<reasoning>", "</reasoning>", "<score>3</score>", "y"]),
            st.text(max_size=3),
        ),
        min_size=1,
    )
    .map("".join)
    .filter(lambda text: text.strip())
)
UNREADABLE_REPLY = st.one_of(
    REPLY.filter(lambda reply: "<score>" not in reply.lower()),
    st.builds(
        build_reply,
        score=OUT_OF_SCALE,
    ),
    st.builds(
        build_reply,
        score=TOO_LONG,
    ),
    st.builds(
        build_reply,
        score=NOT_INTEGERS,
    ),
    st.builds(
        build_reply,
        reasoning=BLANK,
    ),
    st.builds(
        build_reply,
        between=JUNK,
    ),
    st.builds(
        build_reply,
        after=JUNK,
    ),
)


@st.composite
def scores_on_a_scale(draw: st.DrawFn) -> tuple[int, int, int]:
    """The ends of a scale and a score on it."""
    lowest = draw(st.integers(-5, 5))
    highest = lowest + draw(st.integers(1, 20))
    return (
        lowest,
        highest,
        draw(
            st.integers(
                lowest,
                highest,
            ),
        ),
    )


@RATES
@given(reply=ANY_REPLY)
def test_any_reply_gives_a_score_from_0_to_1(
    rate: Rate,
    reply: str,
) -> None:
    score = rate(reply)

    assert 0.0 <= score.value <= 1.0


@RATES
@given(reply=UNREADABLE_REPLY)
def test_a_reply_without_a_readable_score_fails_closed(
    rate: Rate,
    reply: str,
) -> None:
    score = rate(reply)

    assert (
        score.value,
        score.failed_closed,
    ) == (1.0, True)


@RATES
@given(
    before=REASONING_TEXT,
    quoted=st.integers(
        min_value=0,
        max_value=9,
    ),
    after=REASONING_TEXT,
    trailing=BLANK,
)
def test_a_score_tag_quoted_in_the_reasoning_is_never_read(
    rate: Rate,
    before: str,
    quoted: int,
    after: str,
    trailing: str,
) -> None:
    reply = f"<reasoning>x{before}<score>{quoted}</score>{after}</reasoning>{trailing}"

    score = rate(reply)

    assert score.value == 1.0


@RATES
@given(
    before=REASONING_TEXT,
    quoted=SCORES,
    after=REASONING_TEXT,
    scale=scores_on_a_scale(),
)
def test_the_score_tag_after_the_reasoning_is_the_score_whatever_the_reasoning_quotes(
    rate: Rate,
    before: str,
    quoted: int,
    after: str,
    scale: tuple[int, int, int],
) -> None:
    lowest, highest, given_score = scale
    reply = build_reply(
        reasoning=f"x{before}<score>{quoted}</score>{after}",
        score=str(given_score),
    )

    score = rate(
        reply,
        lowest_score=lowest,
        highest_score=highest,
    )

    assert score.value == pytest.approx((given_score - lowest) / (highest - lowest))
