"""Plans are driven a request at a time or a round at once, and one failure stops the rest."""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

import pytest

from hyoshigi.concurrency import Plan, drive, drive_concurrently, run_concurrently

TIMEOUT_SECONDS = 5
LOGGER = "hyoshigi.concurrency"
SECRET = "s3cret-0f3c"

type Driver = Callable[
    [
        Plan[str, str, object],
        Callable[[str], str],
    ],
    object,
]


def run[Result](awaitable: Awaitable[Result]) -> Result:
    """Run on a fresh event loop; a test that waits longer than the timeout fails."""

    async def finish() -> Result:
        return await asyncio.wait_for(
            awaitable,
            timeout=TIMEOUT_SECONDS,
        )

    return asyncio.run(finish())


def drive_in_turn(
    plan: Plan[str, str, object],
    ask: Callable[[str], str],
) -> object:
    return drive(
        plan,
        ask=ask,
    )


def drive_at_once(
    plan: Plan[str, str, object],
    ask: Callable[[str], str],
) -> object:
    async def ask_awaited(request: str) -> str:
        return ask(request)

    return run(
        drive_concurrently(
            plan,
            ask=ask_awaited,
        ),
    )


DRIVERS = pytest.mark.parametrize(
    "driver",
    [
        drive_in_turn,
        drive_at_once,
    ],
    ids=["drive", "drive-concurrently"],
)


def add_a_bang(request: str) -> str:
    return request + "!"


def plan_two_rounds() -> Plan[str, str, object]:
    first = yield ("a", "b")
    second = yield first
    return second


def plan_no_rounds() -> Plan[str, str, object]:
    return "done"
    yield ()


def plan_an_empty_round() -> Plan[str, str, object]:
    replies = yield ()
    return replies


def plan_one_round_of_two() -> Plan[str, str, object]:
    replies = yield ("left", "right")
    return replies


async def fail(text: str) -> str:
    raise RuntimeError(text)


async def wait_at(
    barrier: asyncio.Barrier,
    name: str,
) -> str:
    await barrier.wait()
    return name


async def wait_to_be_cancelled(cancelled: list[str]) -> str:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        cancelled.append("cancelled")
        raise
    return "never"


async def finish_after(
    gate: asyncio.Event,
    name: str,
) -> str:
    await gate.wait()
    return name


async def open_gate(
    gate: asyncio.Event,
    name: str,
) -> str:
    gate.set()
    return name


async def list_cancelled_after_a_failure(cancelled: list[str]) -> list[str]:
    with contextlib.suppress(RuntimeError):
        await run_concurrently(
            [
                wait_to_be_cancelled(cancelled),
                fail("first call failed"),
            ],
        )
    return list(cancelled)


@DRIVERS
@pytest.mark.parametrize(
    ("plan", "result"),
    [
        (
            plan_two_rounds,
            ("a!!", "b!!"),
        ),
        (
            plan_no_rounds,
            "done",
        ),
        (
            plan_an_empty_round,
            (),
        ),
    ],
    ids=["two-rounds-of-two", "no-rounds", "an-empty-round"],
)
def test_a_driven_plan_returns_the_result_it_ends_with(
    driver: Driver,
    plan: Callable[[], Plan[str, str, object]],
    result: object,
) -> None:
    driven = driver(
        plan(),
        add_a_bang,
    )

    assert driven == result


@DRIVERS
def test_a_driven_plan_has_its_requests_asked_in_order(driver: Driver) -> None:
    asked: list[str] = []

    def record(request: str) -> str:
        asked.append(request)
        return add_a_bang(request)

    driver(
        plan_two_rounds(),
        record,
    )

    assert asked == ["a", "b", "a!", "b!"]


@DRIVERS
def test_an_error_from_the_asker_reaches_the_caller_of_a_driven_plan(driver: Driver) -> None:
    def ask_and_fail(request: str) -> str:
        raise RuntimeError("the model failed")

    with pytest.raises(
        RuntimeError,
        match="the model failed",
    ):
        driver(
            plan_two_rounds(),
            ask_and_fail,
        )


def test_awaitables_that_each_wait_for_the_other_finish_only_when_run_at_once() -> None:
    barrier = asyncio.Barrier(2)

    results = run(
        run_concurrently(
            [
                wait_at(
                    barrier,
                    "left",
                ),
                wait_at(
                    barrier,
                    "right",
                ),
            ],
        ),
    )

    assert results == ["left", "right"]


def test_results_come_back_in_the_order_of_the_awaitables_not_of_their_finishing() -> None:
    gate = asyncio.Event()

    results = run(
        run_concurrently(
            [
                finish_after(
                    gate,
                    "first",
                ),
                open_gate(
                    gate,
                    "second",
                ),
            ],
        ),
    )

    assert results == ["first", "second"]


def test_one_failure_cancels_the_awaitables_still_running() -> None:
    cancelled: list[str] = []

    seen = run(list_cancelled_after_a_failure(cancelled))

    assert seen == ["cancelled"]


def test_the_first_failure_is_raised_as_it_is_not_in_a_group() -> None:
    with pytest.raises(
        RuntimeError,
        match="first call failed",
    ):
        run(
            run_concurrently(
                [
                    fail("first call failed"),
                    fail("second call failed"),
                ],
            ),
        )


def test_a_second_failure_is_logged_by_its_type_and_never_by_its_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.DEBUG,
        logger=LOGGER,
    )

    with pytest.raises(
        RuntimeError,
        match="first call failed",
    ):
        run(
            run_concurrently(
                [
                    fail("first call failed"),
                    fail(f"second call sent {SECRET}"),
                ],
            ),
        )

    assert [record.getMessage() for record in caplog.records] == [
        "A concurrent call also failed, with RuntimeError.",
    ]
    assert SECRET not in caplog.text


def test_the_requests_of_a_round_are_asked_at_once_by_drive_concurrently() -> None:
    barrier = asyncio.Barrier(2)

    async def ask_when_both_are_asked(request: str) -> str:
        return await wait_at(
            barrier,
            request,
        )

    driven = run(
        drive_concurrently(
            plan_one_round_of_two(),
            ask=ask_when_both_are_asked,
        ),
    )

    assert driven == ("left", "right")
