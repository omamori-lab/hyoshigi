"""Plans, which hold a rating's logic once for the sync and the async monitors.

A plan is a generator that yields a tuple of requests and receives the tuple of their
replies, in order, until it returns its result. It does no I/O: a driver asks the requests,
in turn or at once.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Generator, Iterable

logger = logging.getLogger(__name__)

type Plan[Request, Reply, Result] = Generator[tuple[Request, ...], tuple[Reply, ...], Result]


def drive[Request, Reply, Result](
    plan: Plan[Request, Reply, Result],
    *,
    ask: Callable[[Request], Reply],
) -> Result:
    """Run a plan, asking each yielded request in turn, and return its result."""
    try:
        requests = next(plan)
        while True:
            requests = plan.send(tuple(ask(request) for request in requests))
    except StopIteration as finished:
        return finished.value


async def drive_concurrently[Request, Reply, Result](
    plan: Plan[Request, Reply, Result],
    *,
    ask: Callable[[Request], Awaitable[Reply]],
) -> Result:
    """Run a plan, asking each yielded tuple's requests at once, and return its result."""
    try:
        requests = next(plan)
        while True:
            replies = await run_concurrently(ask(request) for request in requests)
            requests = plan.send(tuple(replies))
    except StopIteration as finished:
        return finished.value


async def run_concurrently[Result](awaitables: Iterable[Awaitable[Result]]) -> list[Result]:
    """Run the awaitables at once and return their results in order.

    When one fails, the others are cancelled and the first failure is raised as it is, not
    wrapped in an ExceptionGroup, so a caller sees the error a single call raises. Any
    further failures are logged by their type alone, since an error's message can quote
    the request.
    """
    first_failure: BaseException | None = None
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(await_result(awaitable)) for awaitable in awaitables]
    except BaseExceptionGroup as failures:
        first_failure, *other_failures = failures.exceptions
        for other_failure in other_failures:
            logger.warning(
                "A concurrent call also failed, with %s.",
                type(other_failure).__name__,
            )
    # Raised outside the except block, so the error is not chained onto the exception group.
    if first_failure is not None:
        raise first_failure
    return [task.result() for task in tasks]


async def await_result[Result](awaitable: Awaitable[Result]) -> Result:
    """Await an awaitable inside a coroutine, which a task group needs to run it."""
    return await awaitable
