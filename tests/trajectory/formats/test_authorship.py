"""Who wrote a user-role message: the two ready-made authorships."""

import pytest

from hyoshigi import JSON, Author, by_role, marked

MESSAGE: dict[str, JSON] = {"role": "user", "content": "also add Q2"}


def test_by_role_says_a_user_role_message_is_the_user_s() -> None:
    author = by_role(3, message=MESSAGE)

    assert author is Author.USER


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, Author.USER), (3, Author.UNKNOWN)],
    ids=["marked", "not-marked"],
)
def test_marked_says_only_its_messages_are_the_user_s(index: int, expected: Author) -> None:
    author = marked({0})(index, message=MESSAGE)

    assert author is expected
