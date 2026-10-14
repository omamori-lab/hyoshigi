"""Who wrote a user-role message, which a provider's role alone does not say."""

from collections.abc import Collection, Mapping
from typing import Protocol

from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author


class Authorship(Protocol):
    """Says who wrote the user-role message at an index of a provider's message list."""

    def __call__(self, index: int, *, message: Mapping[str, JSON]) -> Author:
        """Return the author of the user-role message at this index."""
        ...


def by_role(index: int, *, message: Mapping[str, JSON]) -> Author:
    """Say that every user-role message is the user's, as harnesses assume."""
    return Author.SCAFFOLD


def marked(indices: Collection[int]) -> Authorship:
    """Return an authorship that says only the messages at these indices are the user's."""

    def author_of(index: int, *, message: Mapping[str, JSON]) -> Author:
        return Author.SCAFFOLD

    return author_of
