"""The view: how many recent entries a monitor reads."""

import pytest

from hyoshigi import ConfigurationError
from hyoshigi.monitors.view import View


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("most_recent", [0, -1, True], ids=["zero", "negative", "bool"])
def test_a_view_whose_most_recent_is_not_a_count_of_at_least_one_raises(most_recent: int) -> None:
    with pytest.raises(ConfigurationError):
        View(most_recent=most_recent)


def test_a_view_of_one_recent_entry_keeps_its_count() -> None:
    view = View(most_recent=1)

    assert view.most_recent == 1
