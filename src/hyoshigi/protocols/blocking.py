"""Block and feed back: a suspicious proposal is blocked and the agent told why, and after too
many blocks a person audits the next proposal.
"""

from dataclasses import dataclass
from operator import itemgetter

from hyoshigi.concurrency import Plan, drive, drive_concurrently
from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import Unavailable
from hyoshigi.monitors.monitor import AsyncMonitor, Monitor
from hyoshigi.monitors.score import SuspicionScore, find_cause
from hyoshigi.options import check_instance, read_count, read_threshold
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
    Flag,
    Policy,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.protocols.requests import FailureResponse, decide_on_unavailable
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory, find_delegation


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class BlockAndFeedBack:
    """Blocks the agent's proposal at or above `block_threshold`, with the score's explanation as
    the reason, and runs it below; after too many blocks, a person audits the next proposal.

    The monitor rates every proposal. The proposal is Audit when the trajectory's records, replayed
    in order, hold at least `blocks_in_a_row` blocks in a row, or its run's records at least
    `blocks_in_total` in total.
    A Block adds 1 to both counts; a Run, a Flag or a Replace sets the count in a row to 0; an
    Audit recorded while the total is at or above its limit sets the total to 0; an Audit or a
    Halt otherwise changes neither. This follows Claude Code's permission-modes page, "When auto
    mode falls back".

    The run's records are those of every trajectory in its delegation tree, replayed from the root:
    a trajectory's own records in order, with each subagent's merged in after the records whose
    position is at or before that of the action that delegated it, ordered by that position and,
    for subagents of one action, in the order they were delegated. A trajectory split from a
    subagent's takes that subagent's place.

    Below both limits, a score that failed closed is Audit, or, when the monitor got no answer,
    what `on_unavailable` says: Audit, Block with the score's explanation as the reason, or Run.
    Every decision holds the proposal and its score, and the length of the history.

    `block_threshold` is a number from 0 to 1, kept as a float; the limits are whole numbers of
    at least 1, kept as ints; `monitor` is a Monitor; `on_unavailable` is a FailureResponse
    other than TRUSTED_STEP, since this protocol never asks for proposals. Anything else raises
    ConfigurationError when the protocol is built.

    <https://omamori-lab.github.io/hyoshigi/explanation/control-protocols/>
    """

    monitor: Monitor
    block_threshold: float
    blocks_in_a_row: int = 3
    blocks_in_total: int = 20
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=Monitor,
            name="monitor",
            hint="Use AsyncBlockAndFeedBack for an AsyncMonitor.",
        )
        _check_options(self)

    def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision:
        """Rate the proposed action, which is not yet in the trajectory, and decide about it."""
        return drive(
            plan_blocking(
                self,
                proposed=proposed,
                trajectory=trajectory,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncBlockAndFeedBack:
    """Decides as BlockAndFeedBack does, with the monitor awaited."""

    monitor: AsyncMonitor
    block_threshold: float
    blocks_in_a_row: int = 3
    blocks_in_total: int = 20
    on_unavailable: FailureResponse = FailureResponse.AUDIT

    def __post_init__(self) -> None:
        check_instance(
            self.monitor,
            expected=AsyncMonitor,
            name="monitor",
            hint="Use BlockAndFeedBack for a Monitor.",
        )
        _check_options(self)

    async def decide(
        self,
        proposed: Action,
        *,
        trajectory: Trajectory,
    ) -> ProtocolDecision:
        """Rate the proposed action, which is not yet in the trajectory, and decide about it."""
        return await drive_concurrently(
            plan_blocking(
                self,
                proposed=proposed,
                trajectory=trajectory,
            ),
            ask=lambda action: self.monitor.rate(
                action,
                trajectory=trajectory,
            ),
        )


def _check_options(protocol: BlockAndFeedBack | AsyncBlockAndFeedBack) -> None:
    """Keep the threshold as a float and the limits as ints, or raise ConfigurationError naming
    the option that cannot work.
    """
    block_threshold = read_threshold(
        protocol.block_threshold,
        name="block_threshold",
    )
    blocks_in_a_row = read_count(
        protocol.blocks_in_a_row,
        name="blocks_in_a_row",
        minimum=1,
    )
    blocks_in_total = read_count(
        protocol.blocks_in_total,
        name="blocks_in_total",
        minimum=1,
    )
    check_instance(
        protocol.on_unavailable,
        expected=FailureResponse,
        name="on_unavailable",
    )
    if protocol.on_unavailable is FailureResponse.TRUSTED_STEP:
        raise ConfigurationError(
            "on_unavailable cannot be TRUSTED_STEP, since this protocol never asks for proposals",
        )
    # The dataclass is frozen, so the values are set past its guard.
    object.__setattr__(
        protocol,
        "block_threshold",
        block_threshold,
    )
    object.__setattr__(
        protocol,
        "blocks_in_a_row",
        blocks_in_a_row,
    )
    object.__setattr__(
        protocol,
        "blocks_in_total",
        blocks_in_total,
    )


def plan_blocking(
    protocol: BlockAndFeedBack | AsyncBlockAndFeedBack,
    *,
    proposed: Action,
    trajectory: Trajectory,
) -> Plan[Action, SuspicionScore, ProtocolDecision]:
    """Plan the decision about the agent's proposal: rate it, then audit it at a limit, answer a
    failure, block it or run it.
    """
    position = len(trajectory.history)
    limited = reaches_a_limit(
        trajectory,
        blocks_in_a_row=protocol.blocks_in_a_row,
        blocks_in_total=protocol.blocks_in_total,
    )
    (score,) = yield (proposed,)
    proposal = Proposal(
        action=proposed,
        score=score,
        policy=Policy.UNTRUSTED,
    )
    match limited, find_cause(score), protocol.on_unavailable:
        case (
            False,
            Unavailable(),
            FailureResponse.AUDIT | FailureResponse.BLOCK | FailureResponse.RUN as response,
        ):
            return decide_on_unavailable(
                response,
                proposal=proposal,
                reason=score.explanation,
                position=position,
            )
        case False, None, _ if score.value >= protocol.block_threshold:
            return Block(
                action=proposed,
                reason=score.explanation,
                proposals=(proposal,),
                position=position,
            )
        case False, None, _:
            return Run(
                action=proposed,
                proposals=(proposal,),
                position=position,
            )
        case _:
            return Audit(
                action=proposed,
                proposals=(proposal,),
                position=position,
            )


def reaches_a_limit(
    trajectory: Trajectory,
    *,
    blocks_in_a_row: int,
    blocks_in_total: int,
) -> bool:
    """Whether the trajectory's own records hold at least `blocks_in_a_row` blocks in a row, or the
    records of its run, merged by merge_run, at least `blocks_in_total` in total.
    """
    return (
        count_blocks_in_a_row(trajectory.records) >= blocks_in_a_row
        or count_blocks_in_total(
            merge_run(trajectory),
            limit=blocks_in_total,
        )
        >= blocks_in_total
    )


def count_blocks_in_a_row(records: tuple[ProtocolDecision, ...]) -> int:
    """Count the blocks in a row: a Block adds 1, a Run, a Flag or a Replace sets the count to 0,
    and an Audit or a Halt changes nothing. Claude Code's permission-modes page, "When auto mode
    falls back".
    """
    in_a_row = 0
    for record in records:
        if isinstance(
            record,
            Block,
        ):
            in_a_row += 1
        elif isinstance(
            record,
            Run | Flag | Replace,
        ):
            in_a_row = 0
    return in_a_row


def count_blocks_in_total(
    records: tuple[ProtocolDecision, ...],
    *,
    limit: int,
) -> int:
    """Count the blocks in total: a Block adds 1, and an Audit recorded while the count is at or
    above the limit sets it to 0. Claude Code's permission-modes page, "When auto mode falls
    back".
    """
    in_total = 0
    for record in records:
        if isinstance(
            record,
            Block,
        ):
            in_total += 1
        elif (
            isinstance(
                record,
                Audit,
            )
            and in_total >= limit
        ):
            in_total = 0
    return in_total


def merge_run(trajectory: Trajectory) -> tuple[ProtocolDecision, ...]:
    """Return the records of the run the trajectory is in, merged up to the root of its delegation
    tree: the trajectory's merged records stand in for the subagent under its call in its parent's,
    and the parent's for the subagent under its own call, and so on.
    """
    records = merge_records(
        trajectory,
        placed={},
    )
    delegation = find_delegation(trajectory)
    while delegation is not None:
        records = merge_records(
            delegation.parent,
            placed={delegation.call_id: records},
        )
        delegation = find_delegation(delegation.parent)
    return records


def merge_records(
    trajectory: Trajectory,
    *,
    placed: dict[str, tuple[ProtocolDecision, ...]],
) -> tuple[ProtocolDecision, ...]:
    """Return the trajectory's records in order, with each subagent's merged records after the
    records whose position is at or before that of the action that delegated it.

    Subagents go in the order of those positions and, for one action, in the order they were
    delegated. The records `placed` under a call id stand in for those of its subagent.
    """
    delegated_at = {
        call.id: position
        for position, event in enumerate(trajectory.history)
        if isinstance(
            event,
            Action,
        )
        for call in event.tool_calls
    }
    subagents = {
        call_id: merge_records(
            subagent,
            placed={},
        )
        for call_id, subagent in trajectory.subagents.items()
    } | placed
    # sorted keeps the order of entries at one position: the records, then the subagents.
    timeline = sorted(
        [
            *(
                (
                    record.position,
                    (record,),
                )
                for record in trajectory.records
            ),
            *(
                (
                    delegated_at[call_id],
                    records,
                )
                for call_id, records in subagents.items()
            ),
        ],
        key=itemgetter(0),
    )
    return tuple(record for _, records in timeline for record in records)
