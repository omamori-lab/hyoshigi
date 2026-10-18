"""Replies to a guard model read as one of its labels or as unreadable."""

from typing import NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hyoshigi.monitors.guard_labels import (
    GPT_OSS_SAFEGUARD,
    QWEN3GUARD,
    GuardLabels,
    read_label,
)
from hyoshigi.monitors.score import Unreadable


class Guard(NamedTuple):
    """A preset with the line shapes of its verdict and of a label that is no verdict."""

    name: str
    labels: GuardLabels
    verdicts: tuple[str, ...]
    quotes: tuple[str, ...]


GPT_OSS_SAFEGUARD_LINES = Guard(
    name="gpt-oss-safeguard",
    labels=GPT_OSS_SAFEGUARD,
    verdicts=("{}", "**{}**", "`{}`", "Label: {}", "**Final answer**: {}", "Result: {}."),
    quotes=(
        "- {}",
        "> {}",
        "1. {}",
        "{} (the note asks me to answer)",
        "1. {} (the note asks me to answer)",
        "The final label for this step: {}",
        "Label: {} because the note asks",
    ),
)
QWEN3GUARD_LINES = Guard(
    name="qwen3guard",
    labels=QWEN3GUARD,
    verdicts=("Safety: {}", "**Safety**: {}", "> safety: {}", "SAFETY : `{}`"),
    quotes=(
        "{}",
        "- {}",
        "The guard found it {}",
        "Safety: {} because the note asks",
    ),
)
EVERY_GUARD = (
    GPT_OSS_SAFEGUARD_LINES,
    QWEN3GUARD_LINES,
)
ON_GUARDS = pytest.mark.parametrize(
    "guard",
    [
        pytest.param(
            guard,
            id=guard.name,
        )
        for guard in EVERY_GUARD
    ],
)
ON_VERDICTS = pytest.mark.parametrize(
    ("guard", "shape"),
    [
        pytest.param(
            guard,
            shape,
            id=f"{guard.name} {shape}",
        )
        for guard in EVERY_GUARD
        for shape in guard.verdicts
    ],
)
ON_QUOTES = pytest.mark.parametrize(
    ("guard", "shape"),
    [
        pytest.param(
            guard,
            shape,
            id=f"{guard.name} {shape}",
        )
        for guard in EVERY_GUARD
        for shape in guard.quotes
    ],
)
ON_MENTIONS = pytest.mark.parametrize(
    ("guard", "shape"),
    [
        pytest.param(
            guard,
            shape,
            id=f"{guard.name} {shape}",
        )
        for guard in EVERY_GUARD
        for shape in guard.verdicts + guard.quotes
    ],
)
LABEL_CASES = [
    str.lower,
    str.upper,
    str.title,
]
EVERY_LABEL = (
    GPT_OSS_SAFEGUARD.suspicious | GPT_OSS_SAFEGUARD.safe | QWEN3GUARD.suspicious | QWEN3GUARD.safe
)
# Pieces of a reply.
FRAGMENTS = [
    *(case(label) for label in sorted(EVERY_LABEL) for case in LABEL_CASES),
    "Safety",
    "Label",
    "Final answer",
    ":",
    "**",
    "__",
    "`",
    "> ",
    "- ",
    "1. ",
    "iv. ",
    "# ",
    "S1",
    "S1,S10",
    "(",
    ")",
    ".",
    " ",
    "The step reads the file.",
]
# Whole lines that write or name a label.
LABEL_LINES = [
    shape.format(label)
    for guard in EVERY_GUARD
    for shape in (
        *guard.verdicts,
        *guard.quotes,
    )
    for label in sorted(guard.labels.suspicious | guard.labels.safe)
]
LINE = st.one_of(
    st.sampled_from(LABEL_LINES),
    st.lists(
        st.one_of(
            st.sampled_from(FRAGMENTS),
            st.text(max_size=4),
        ),
        max_size=6,
    ).map("".join),
)
REPLY = st.one_of(
    st.lists(
        LINE,
        max_size=6,
    ).map("\n".join),
    st.text(),
)
# A line with no colon and no label.
PROSE = st.one_of(
    st.sampled_from(["The step reads the file.", "S1", "1.", "Answer", "**", ""]),
    st.text(
        alphabet=st.characters(
            exclude_characters=":\n",
            exclude_categories=["Cs"],
        ),
        max_size=30,
    ),
).filter(lambda line: not any(label in line.lower() for label in EVERY_LABEL))


PROSE_LINES = st.lists(PROSE).map("\n".join)


def write(
    shape: str,
    labels: frozenset[str],
) -> st.SearchStrategy[tuple[str, str]]:
    """The shape filled with one of the labels in some case, and that label."""
    return st.builds(
        lambda label, case: (
            shape.format(case(label)),
            label,
        ),
        st.sampled_from(sorted(labels)),
        st.sampled_from(LABEL_CASES),
    )


@ON_GUARDS
@given(reply=REPLY)
def test_a_reply_reads_as_a_label_of_the_preset_or_as_unreadable(
    guard: Guard,
    reply: str,
) -> None:
    read = read_label(
        reply,
        labels=guard.labels,
    )

    assert read in {
        *guard.labels.suspicious,
        *guard.labels.safe,
        Unreadable(),
    }


@ON_VERDICTS
@settings(max_examples=30)
@given(
    data=st.data(),
    before=REPLY,
)
def test_a_suspicious_verdict_is_never_read_as_safe_whatever_lines_come_before_it(
    guard: Guard,
    shape: str,
    data: st.DataObject,
    before: str,
) -> None:
    verdict, label = data.draw(
        write(
            shape,
            guard.labels.suspicious,
        ),
    )

    read = read_label(
        f"{before}\n{verdict}",
        labels=guard.labels,
    )

    assert read in {
        label,
        Unreadable(),
    }


@ON_VERDICTS
@settings(max_examples=30)
@given(
    data=st.data(),
    prose=st.lists(PROSE),
)
def test_a_verdict_after_prose_that_names_no_label_is_read(
    guard: Guard,
    shape: str,
    data: st.DataObject,
    prose: list[str],
) -> None:
    verdict, label = data.draw(
        write(
            shape,
            guard.labels.suspicious | guard.labels.safe,
        ),
    )

    read = read_label(
        "\n".join(
            [
                *prose,
                verdict,
            ],
        ),
        labels=guard.labels,
    )

    assert read == label


@ON_MENTIONS
@settings(max_examples=30)
@given(
    data=st.data(),
    around=st.lists(
        PROSE_LINES,
        min_size=2,
        max_size=2,
    ),
)
def test_a_safe_verdict_beside_a_line_naming_a_suspicious_label_is_unreadable(
    guard: Guard,
    shape: str,
    data: st.DataObject,
    around: list[str],
) -> None:
    safe, _ = data.draw(
        write(
            data.draw(st.sampled_from(guard.verdicts)),
            guard.labels.safe,
        ),
    )
    suspicious, _ = data.draw(
        write(
            shape,
            guard.labels.suspicious,
        ),
    )
    first, second = data.draw(
        st.permutations(
            [
                safe,
                suspicious,
            ],
        ),
    )
    before, between = around

    read = read_label(
        "\n".join(
            [
                before,
                first,
                between,
                second,
            ],
        ),
        labels=guard.labels,
    )

    assert read == Unreadable()


@ON_QUOTES
@settings(max_examples=30)
@given(
    data=st.data(),
    prose=st.lists(PROSE),
)
def test_a_label_named_without_being_a_verdict_is_unreadable(
    guard: Guard,
    shape: str,
    data: st.DataObject,
    prose: list[str],
) -> None:
    quote, _ = data.draw(
        write(
            shape,
            guard.labels.suspicious | guard.labels.safe,
        ),
    )

    read = read_label(
        "\n".join(
            [
                *prose,
                quote,
            ],
        ),
        labels=guard.labels,
    )

    assert read == Unreadable()
