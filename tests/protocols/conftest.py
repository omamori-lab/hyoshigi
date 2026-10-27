"""The family of protocols a test runs under: the sync and the async classes."""

import pytest
from families import ASYNCHRONOUS, SYNCHRONOUS, AnyFamily


@pytest.fixture(
    params=[
        SYNCHRONOUS,
        ASYNCHRONOUS,
    ],
    ids=["sync", "async"],
)
def family(request: pytest.FixtureRequest) -> AnyFamily:
    return request.param
