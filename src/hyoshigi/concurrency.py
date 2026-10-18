"""Plans, which hold a rating's logic once for the sync and the async monitors."""

from collections.abc import Awaitable, Callable, Generator, Iterable
from typing import cast

type Plan[Request, Reply, Result] = Generator[tuple[Request, ...], tuple[Reply, ...], Result]


def drive[Request, Reply, Result](
    plan: Plan[Request, Reply, Result], *, ask: Callable[[Request], Reply]
) -> Result:
    """Run a plan, asking each request in turn."""
    return cast(Result, None)


async def drive_concurrently[Request, Reply, Result](
    plan: Plan[Request, Reply, Result], *, ask: Callable[[Request], Awaitable[Reply]]
) -> Result:
    """Run a plan, asking each round's requests at once."""
    return cast(Result, None)


async def run_concurrently[Result](awaitables: Iterable[Awaitable[Result]]) -> list[Result]:
    """Run the awaitables at once and return their results in order."""
    return []
