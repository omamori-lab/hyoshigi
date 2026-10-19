"""Payloads with one node replaced."""

from base64 import b64decode
from collections.abc import Callable, Iterator
from enum import Enum, auto
from functools import partial

from hypothesis import strategies as st

from hyoshigi import JSON, FormatError

MARKER = "SECRET"


class Deleted(Enum):
    """The edit that removes a node."""

    DELETED = auto()


type Path = tuple[str | int, ...]
type Edit = JSON | Deleted


def branches(value: JSON) -> Iterator[tuple[str | int, JSON]]:
    """The keys or indexes of a node with its children."""
    match value:
        case dict():
            yield from value.items()
        case list():
            yield from enumerate(value)


def walk_nodes(value: JSON) -> Iterator[tuple[Path, JSON]]:
    """Every node with its path, starting with the root."""
    yield (), value
    for key, child in branches(value):
        for path, node in walk_nodes(child):
            yield (
                (
                    key,
                    *path,
                ),
                node,
            )


def replace_node(
    value: JSON,
    path: Path,
    edit: Edit,
) -> Edit:
    """The value with the node at the path replaced by the edit."""
    if not path:
        return edit
    head, *rest = path
    replace_child = partial(
        replace_node,
        path=tuple(rest),
        edit=edit,
    )
    match value:
        case dict():
            edited = {
                key: replace_child(child) if key == head else child for key, child in value.items()
            }
            return {key: child for key, child in edited.items() if child is not Deleted.DELETED}
        case list():
            items = [
                replace_child(child) if index == head else child
                for index, child in enumerate(value)
            ]
            return [child for child in items if child is not Deleted.DELETED]
        case _:
            return value


MARKED_STRINGS = st.text().map(lambda text: MARKER + text)
# Bytes whose base64 text starts with the marker.
MARKED_BYTES = st.binary().map(lambda tail: b64decode(MARKER * 2) + tail)
# Strings that break readers of URLs, each followed by the marker.
HOSTILE_STRINGS = st.sampled_from(
    [
        "//[",
        "http://[::1",
        "http://a]b",
        "http://\uff03",
        "data:",
        "data:;base64,",
        "data:image/png;base64,@",
        "data:text/plain,%",
    ],
).map(lambda text: text + MARKER)
JUNK = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(),
        MARKED_STRINGS,
    ),
    lambda children: st.one_of(
        st.lists(
            children,
            max_size=3,
        ),
        st.dictionaries(
            MARKED_STRINGS,
            children,
            max_size=3,
        ),
    ),
    max_leaves=6,
)


@st.composite
def draw_edit(
    draw: st.DrawFn,
    payload: JSON,
) -> Edit:
    """What can replace a node: junk, a hostile string, nothing or another node."""
    return draw(
        st.one_of(
            JUNK,
            HOSTILE_STRINGS,
            st.just(Deleted.DELETED),
            st.sampled_from([node for _, node in walk_nodes(payload)]),
        ),
    )


def replace_each_node(
    payload: JSON,
    edit: Edit,
) -> Iterator[JSON]:
    """The payload with each of its nodes in turn replaced by the edit."""
    for path, _ in walk_nodes(payload):
        edited = replace_node(
            payload,
            path,
            edit,
        )
        yield None if edited is Deleted.DELETED else edited


def capture_error(
    read: Callable[[JSON], object],
    payload: JSON,
) -> Exception | None:
    """The exception that reading the payload raises, if any."""
    try:
        read(payload)
    except Exception as error:
        return error
    return None


def reads_cleanly(
    read: Callable[[JSON], object],
    payload: JSON,
) -> bool:
    """Whether reading succeeds or raises a FormatError that quotes no value after its path."""
    error = capture_error(
        read,
        payload,
    )
    if error is None:
        return True
    if not isinstance(
        error,
        FormatError,
    ):
        return False
    return MARKER not in str(error).removeprefix(f"{error.path}: ")
