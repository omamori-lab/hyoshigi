"""The view: its channels, how many recent entries a monitor reads, and its delegation tools."""

import numbers
from collections.abc import Iterator

import pytest

from hyoshigi import ConfigurationError
from hyoshigi.monitors.view import Channel, View


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


def yield_tool_names() -> Iterator[str]:
    yield "task"


@pytest.mark.parametrize(
    ("most_recent", "message"),
    [
        (0, "most_recent must be at least 1, got 0"),
        (-1, "most_recent must be at least 1, got -1"),
        (True, "most_recent must be a whole number of at least 1, got True"),
        (2.5, "most_recent must be a whole number of at least 1, got 2.5"),
        ("5", "most_recent must be a whole number of at least 1, got '5'"),
    ],
    ids=["zero", "negative", "bool", "float", "string"],
)
def test_a_view_whose_most_recent_is_not_a_count_of_at_least_one_raises_when_built(
    most_recent: object, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        View(most_recent=most_recent)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message


def test_a_view_of_one_recent_entry_keeps_its_count() -> None:
    view = View(most_recent=1)

    assert view.most_recent == 1


def test_a_view_keeps_the_count_of_recent_entries_as_an_int() -> None:
    view = View(most_recent=ArrayInteger(3))  # ty: ignore[invalid-argument-type]

    assert (view.most_recent, type(view.most_recent)) == (3, int)


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        (123, "channels must be a Channel, got 123"),
        ("TASK", "channels must be a Channel, got 'TASK'"),
        (None, "channels must be a Channel, got None"),
        (Channel, "channels must be a Channel, got the class Channel; pass an instance"),
    ],
    ids=["integer", "string", "none", "the-class"],
)
def test_a_view_whose_channels_are_not_a_channel_raises_when_built(
    channels: object, message: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        View(channels=channels)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("tools", "described"),
    [("Task", "'Task'"), (b"Task", "an instance of bytes"), (123, "123"), (None, "None")],
    ids=["string", "bytes", "integer", "none"],
)
def test_a_view_whose_delegation_tools_are_not_an_iterable_of_names_raises_when_built(
    tools: object, described: str
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        View(delegation_tools=tools)  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == (
        f"delegation_tools must be an iterable of tool names other than a string, got {described}"
    )


def test_a_view_whose_delegation_tools_hold_a_non_string_raises_naming_its_position() -> None:
    with pytest.raises(ConfigurationError) as raised:
        View(delegation_tools=("task", 5))  # ty: ignore[invalid-argument-type]

    assert str(raised.value) == "delegation_tools[1] must be a str, got 5"


@pytest.mark.parametrize(
    "tools",
    [
        frozenset({"task"}),
        ["task"],
        ("task",),
        {"task"},
        yield_tool_names(),
    ],
    ids=["frozenset", "list", "tuple", "set", "generator"],
)
def test_a_views_delegation_tools_are_kept_as_a_frozenset(tools: object) -> None:
    view = View(delegation_tools=tools)  # ty: ignore[invalid-argument-type]

    assert (view.delegation_tools, type(view.delegation_tools)) == (
        frozenset({"task"}),
        frozenset,
    )
