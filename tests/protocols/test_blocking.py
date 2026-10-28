"""Block and feed back: what it decides about a proposal, given the blocks already recorded."""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pytest
from families import (
    FAILURE_IDS,
    FAILURES_WITH_AN_ANSWER,
    UNAVAILABLE_EXPLANATION,
    UNAVAILABLE_SCORE,
    AnyFamily,
    Script,
    build_action,
    build_asynchronous_llm,
    build_proposal,
    build_reply,
    build_score,
    build_synchronous_llm,
    build_trajectory,
)

from hyoshigi import (
    JSON,
    Action,
    AsyncBlockAndFeedBack,
    Audit,
    Block,
    BlockAndFeedBack,
    ConfigurationError,
    FailureResponse,
    Flag,
    GenerationOutcome,
    Halt,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
    SuspicionScore,
    ToolCall,
    Trajectory,
    Unavailable,
)

PROPOSED = build_action("proposed")
REASON = "It writes outside the task."
CALM_PROPOSAL = build_proposal(
    PROPOSED,
    build_score(0.2),
)
UNAVAILABLE = Unavailable(status=503)


def build_block(
    *,
    action: Action,
    proposals: tuple[Proposal, ...],
    position: int,
) -> Block:
    return Block(
        action=action,
        reason=REASON,
        proposals=proposals,
        position=position,
    )


def build_halt(
    *,
    action: Action,
    proposals: tuple[Proposal, ...],
    position: int,
) -> Halt:
    return Halt(
        reason=REASON,
        proposals=proposals,
        position=position,
    )


class Step(NamedTuple):
    """A decision of a kind on the action of a name."""

    kind: Callable[..., ProtocolDecision]
    name: str


BLOCK = partial(
    Step,
    build_block,
)
RUN = partial(
    Step,
    Run,
)
REPLACE = partial(
    Step,
    Replace,
)
FLAG = partial(
    Step,
    Flag,
)
AUDIT = partial(
    Step,
    Audit,
)
HALT = partial(
    Step,
    build_halt,
)


def record_steps(
    trajectory: Trajectory,
    steps: tuple[Step, ...],
) -> None:
    """Record each step's decision on the action of its name, as a program would: at the length
    of the history, with the action's proposal scored 0.8.
    """
    for kind, name in steps:
        action = build_action(name)
        trajectory.record(
            kind(
                action=action,
                proposals=(
                    build_proposal(
                        action,
                        build_score(
                            0.8,
                            REASON,
                        ),
                    ),
                ),
                position=len(trajectory.history),
            ),
        )


def build_audit(position: int) -> Audit:
    """The audit of the calm proposal at the position."""
    return Audit(
        action=PROPOSED,
        proposals=(CALM_PROPOSAL,),
        position=position,
    )


def build_run(position: int) -> Run:
    """The run of the calm proposal at the position."""
    return Run(
        action=PROPOSED,
        proposals=(CALM_PROPOSAL,),
        position=position,
    )


@pytest.mark.parametrize(
    ("reply", "outcome"),
    [
        pytest.param(
            build_reply(
                5,
                "It deletes the notes.",
            ),
            Block(
                action=PROPOSED,
                reason="It deletes the notes.",
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(
                            0.5,
                            "It deletes the notes.",
                        ),
                    ),
                ),
                position=1,
            ),
            id="at-the-threshold-blocks",
        ),
        pytest.param(
            build_reply(
                4,
                "It reads the notes.",
            ),
            Run(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        build_score(
                            0.4,
                            "It reads the notes.",
                        ),
                    ),
                ),
                position=1,
            ),
            id="below-the-threshold-runs",
        ),
    ],
)
def test_a_proposal_at_or_above_the_block_threshold_is_blocked_with_the_scores_explanation(
    family: AnyFamily,
    reply: GenerationOutcome,
    outcome: ProtocolDecision,
) -> None:
    script = Script(reply)

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("steps", "limits", "outcome"),
    [
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                BLOCK("a3"),
            ),
            {},
            build_audit(7),
            id="three-in-a-row-audit",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
            ),
            {},
            build_run(5),
            id="two-in-a-row-do-not",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                RUN("a3"),
                BLOCK("a4"),
                BLOCK("a5"),
            ),
            {},
            build_run(10),
            id="a-run-ends-the-run-of-blocks",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                REPLACE("a3"),
                BLOCK("a4"),
                BLOCK("a5"),
            ),
            {},
            build_run(10),
            id="a-replace-ends-the-run-of-blocks",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                FLAG("a3"),
                BLOCK("a4"),
                BLOCK("a5"),
            ),
            {},
            build_run(10),
            id="a-flag-ends-the-run-of-blocks",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                HALT("a3"),
                BLOCK("a3"),
            ),
            {},
            build_audit(7),
            id="a-halt-changes-nothing",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                AUDIT("a3"),
                BLOCK("a3"),
            ),
            {},
            build_audit(7),
            id="an-audit-changes-nothing",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                BLOCK("a3"),
                AUDIT("a4"),
                RUN("a4"),
            ),
            {},
            build_run(8),
            id="a-persons-run-after-the-audit-ends-the-run",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                BLOCK("a3"),
                AUDIT("a4"),
                BLOCK("a4"),
            ),
            {},
            build_audit(9),
            id="a-persons-block-after-the-audit-keeps-the-run",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                HALT("a3"),
            ),
            {},
            build_run(5),
            id="a-halt-adds-nothing-in-a-row",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                AUDIT("a3"),
            ),
            {},
            build_run(5),
            id="an-audit-adds-nothing-in-a-row",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                RUN("a2"),
                BLOCK("a3"),
            ),
            {"blocks_in_total": 2},
            build_audit(6),
            id="two-in-total-with-a-run-between-audit",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                RUN("a2"),
            ),
            {"blocks_in_total": 2},
            build_run(4),
            id="one-in-total-does-not",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                RUN("a2"),
                BLOCK("a3"),
                AUDIT("a4"),
                RUN("a4"),
            ),
            {"blocks_in_total": 2},
            build_run(7),
            id="the-limits-audit-resets-the-total",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                RUN("a2"),
                BLOCK("a3"),
                AUDIT("a4"),
                BLOCK("a4"),
            ),
            {"blocks_in_total": 2},
            build_run(8),
            id="a-persons-block-after-the-limits-audit-counts-from-0",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                RUN("a2"),
                BLOCK("a3"),
                HALT("a4"),
            ),
            {"blocks_in_total": 2},
            build_audit(6),
            id="a-halt-at-the-total-limit-leaves-the-total",
        ),
        pytest.param(
            (
                BLOCK("a1"),
                BLOCK("a2"),
                AUDIT("a3"),
                RUN("a3"),
                BLOCK("a4"),
            ),
            {
                "blocks_in_a_row": 2,
                "blocks_in_total": 3,
            },
            build_audit(8),
            id="an-audit-below-the-total-limit-leaves-the-total",
        ),
    ],
)
def test_a_proposal_is_audited_only_once_the_blocks_reach_a_limit(
    family: AnyFamily,
    steps: tuple[Step, ...],
    limits: dict[str, int],
    outcome: ProtocolDecision,
) -> None:
    trajectory = build_trajectory()
    record_steps(
        trajectory,
        steps,
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=trajectory,
        **limits,
    )

    assert decided == outcome


INSTRUCTION = "Find the q3 revenue"
DELEGATING = Action(
    id="delegating",
    tool_calls=(
        ToolCall(
            id="call_1",
            name="delegate",
            arguments='{"task": "Find the q3 revenue"}',
        ),
    ),
)
NINETEEN_BLOCKS = tuple(BLOCK(f"b{number}") for number in range(19))
SUSPICIOUS_PROPOSAL = build_proposal(
    PROPOSED,
    build_score(0.9),
)
# The audit of the calm proposal of action s2, the latest action of a subagent, decided again on
# the trajectory before it.
REDECIDED = Audit(
    action=build_action("s2"),
    proposals=(
        build_proposal(
            build_action("s2"),
            build_score(0.2),
        ),
    ),
    position=3,
)


def start_subagent(trajectory: Trajectory) -> Trajectory:
    """Act DELEGATING, and return the trajectory of the subagent its call delegates."""
    trajectory.act(DELEGATING)
    return trajectory.delegate(
        INSTRUCTION,
        call_id="call_1",
    )


def reload(trajectory: Trajectory) -> Trajectory:
    """Load the trajectory's dump."""
    return Trajectory.load(trajectory.dump())


def split_off_a_later_action(trajectory: Trajectory) -> Trajectory:
    """Act one more action, and return the trajectory before it that split_latest_step returns."""
    trajectory.act(build_action("later"))
    return trajectory.split_latest_step().trajectory


@pytest.fixture
def split_at_the_limit() -> Trajectory:
    """The trajectory before the latest action of a delegator that delegated a subagent, was
    blocked twice, and acted once more.
    """
    trajectory = build_trajectory()
    start_subagent(trajectory)
    record_steps(
        trajectory,
        (
            BLOCK("a2"),
            BLOCK("a3"),
        ),
    )
    return split_off_a_later_action(trajectory)


@pytest.mark.parametrize(
    ("limits", "outcome"),
    [
        pytest.param(
            {},
            Audit(
                action=PROPOSED,
                proposals=(SUSPICIOUS_PROPOSAL,),
                position=3,
            ),
            id="in-total",
        ),
        pytest.param(
            {"blocks_in_total": 21},
            Block(
                action=PROPOSED,
                reason="r",
                proposals=(SUSPICIOUS_PROPOSAL,),
                position=3,
            ),
            id="not-in-a-row",
        ),
    ],
)
def test_a_subagent_counts_its_delegators_blocks_in_its_total_only(
    family: AnyFamily,
    limits: dict[str, int],
    outcome: ProtocolDecision,
) -> None:
    trajectory = build_trajectory()
    record_steps(
        trajectory,
        NINETEEN_BLOCKS,
    )
    subagent = start_subagent(trajectory)
    record_steps(
        subagent,
        (BLOCK("s1"),),
    )
    script = Script(build_reply(9))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=subagent,
        **limits,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("limits", "outcome"),
    [
        pytest.param(
            {"blocks_in_total": 3},
            build_audit(2),
            id="in-total",
        ),
        pytest.param(
            {},
            build_run(2),
            id="not-in-a-row",
        ),
    ],
)
def test_a_delegator_counts_its_subagents_blocks_in_its_total_only(
    family: AnyFamily,
    limits: dict[str, int],
    outcome: ProtocolDecision,
) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    record_steps(
        subagent,
        (
            BLOCK("s1"),
            BLOCK("s2"),
            BLOCK("s3"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=trajectory,
        **limits,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    ("deciding", "position"),
    [
        pytest.param(
            lambda root: root,
            4,
            id="the-root",
        ),
        pytest.param(
            lambda root: root.subagents["call_1"],
            4,
            id="the-middle-subagent",
        ),
        pytest.param(
            lambda root: root.subagents["call_1"].subagents["call_1"],
            3,
            id="the-deepest-subagent",
        ),
    ],
)
def test_the_total_counts_the_blocks_of_subagents_nested_two_levels_deep(
    family: AnyFamily,
    deciding: Callable[[Trajectory], Trajectory],
    position: int,
) -> None:
    root = build_trajectory()
    record_steps(
        root,
        (BLOCK("r1"),),
    )
    subagent = start_subagent(root)
    record_steps(
        subagent,
        (BLOCK("s1"),),
    )
    nested = start_subagent(subagent)
    record_steps(
        nested,
        (BLOCK("n1"),),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=deciding(root),
        blocks_in_total=3,
    )

    assert decided == build_audit(position)


DELEGATING_TWICE = Action(
    id="delegating",
    tool_calls=(
        ToolCall(
            id="call_1",
            name="delegate",
            arguments='{"task": "Find the q3 revenue"}',
        ),
        ToolCall(
            id="call_2",
            name="delegate",
            arguments='{"task": "Find the q4 revenue"}',
        ),
    ),
)


def test_subagents_of_one_action_are_counted_in_the_order_their_records_were_recorded(
    family: AnyFamily,
) -> None:
    trajectory = build_trajectory()
    trajectory.act(DELEGATING_TWICE)
    first = trajectory.delegate(
        INSTRUCTION,
        call_id="call_1",
    )
    second = trajectory.delegate(
        "Find the q4 revenue",
        call_id="call_2",
    )
    record_steps(
        second,
        (BLOCK("t1"),),
    )
    record_steps(
        first,
        (
            BLOCK("f1"),
            AUDIT("f2"),
            RUN("f2"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=second,
        blocks_in_total=2,
    )

    assert decided == build_run(3)


@pytest.mark.parametrize(
    ("delegators", "subagents", "outcome"),
    [
        pytest.param(
            (
                AUDIT("a2"),
                RUN("a2"),
            ),
            (
                BLOCK("s1"),
                BLOCK("s2"),
            ),
            build_audit(5),
            id="an-audit-below-the-limit-leaves-the-total",
        ),
        pytest.param(
            (
                BLOCK("a2"),
                BLOCK("a3"),
            ),
            (
                AUDIT("s1"),
                RUN("s1"),
            ),
            build_run(2),
            id="an-audit-at-the-limit-resets-the-total",
        ),
    ],
)
def test_the_reset_applies_over_the_runs_records_in_the_order_they_were_recorded(
    family: AnyFamily,
    delegators: tuple[Step, ...],
    subagents: tuple[Step, ...],
    outcome: ProtocolDecision,
) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    record_steps(
        trajectory,
        delegators,
    )
    record_steps(
        subagent,
        subagents,
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=subagent,
        blocks_in_total=2,
    )

    assert decided == outcome


@pytest.mark.parametrize(
    "keeping",
    [
        pytest.param(
            lambda trajectory: trajectory,
            id="as-built",
        ),
        pytest.param(
            reload,
            id="loaded-from-a-dump",
        ),
        pytest.param(
            split_off_a_later_action,
            id="copied-by-a-split",
        ),
    ],
)
def test_a_subagents_split_copy_leaves_out_the_records_at_the_position_it_drops(
    family: AnyFamily,
    keeping: Callable[[Trajectory], Trajectory],
) -> None:
    trajectory = build_trajectory()
    record_steps(
        trajectory,
        (BLOCK("a1"),),
    )
    nested = start_subagent(start_subagent(trajectory))
    record_steps(
        nested,
        (
            BLOCK("s1"),
            AUDIT("s2"),
            RUN("s2"),
        ),
    )
    kept = keeping(trajectory).subagents["call_1"].subagents["call_1"]
    step = kept.split_latest_step()
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=step.action,
        trajectory=step.trajectory,
        blocks_in_total=2,
    )

    assert decided == REDECIDED


def test_a_subagent_delegated_by_a_split_copy_counts_its_own_blocks(family: AnyFamily) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    subagent.act(DELEGATING)
    subagent.act(build_action("later"))
    nested = subagent.split_latest_step().trajectory.delegate(
        INSTRUCTION,
        call_id="call_1",
    )
    record_steps(
        nested,
        (
            BLOCK("n1"),
            BLOCK("n2"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=nested,
        blocks_in_total=2,
    )

    assert decided == build_audit(5)


@pytest.mark.parametrize(
    "keeping",
    [
        pytest.param(
            lambda trajectory: trajectory,
            id="as-built",
        ),
        pytest.param(
            reload,
            id="loaded-from-a-dump",
        ),
    ],
)
def test_a_subagent_kept_by_a_split_copy_counts_the_copys_blocks(
    family: AnyFamily,
    keeping: Callable[[Trajectory], Trajectory],
) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    start_subagent(subagent)
    earlier = split_off_a_later_action(subagent)
    record_steps(
        earlier,
        (
            BLOCK("e1"),
            BLOCK("e2"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=keeping(earlier).subagents["call_1"],
        blocks_in_total=2,
    )

    assert decided == build_audit(1)


@pytest.mark.parametrize(
    "keeping",
    [
        pytest.param(
            lambda trajectory: trajectory,
            id="as-built",
        ),
        pytest.param(
            reload,
            id="loaded-from-a-dump",
        ),
    ],
)
def test_a_subagent_kept_by_a_split_copy_counts_the_copys_reset(
    family: AnyFamily,
    keeping: Callable[[Trajectory], Trajectory],
    split_at_the_limit: Trajectory,
) -> None:
    record_steps(
        split_at_the_limit,
        (
            AUDIT("a4"),
            RUN("a4"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=keeping(split_at_the_limit).subagents["call_1"],
        blocks_in_total=2,
    )

    assert decided == build_run(1)


def test_a_subagent_two_levels_below_a_split_copy_counts_the_copys_blocks(
    family: AnyFamily,
) -> None:
    trajectory = build_trajectory()
    start_subagent(start_subagent(trajectory))
    earlier = split_off_a_later_action(trajectory)
    record_steps(
        earlier,
        (
            BLOCK("e1"),
            BLOCK("e2"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=earlier.subagents["call_1"].subagents["call_1"],
        blocks_in_total=2,
    )

    assert decided == build_audit(1)


def test_new_records_of_a_split_copys_subagent_follow_the_runs(
    family: AnyFamily,
    split_at_the_limit: Trajectory,
) -> None:
    copied = split_at_the_limit.subagents["call_1"]
    record_steps(
        copied,
        (
            AUDIT("s1"),
            RUN("s1"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=copied,
        blocks_in_total=2,
    )

    assert decided == build_run(2)


def test_a_block_on_a_split_copys_subagent_stays_out_of_the_original_runs_total(
    family: AnyFamily,
) -> None:
    trajectory = build_trajectory()
    original = start_subagent(trajectory)
    record_steps(
        trajectory,
        (BLOCK("a2"),),
    )
    record_steps(
        split_off_a_later_action(trajectory).subagents["call_1"],
        (BLOCK("s1"),),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=original,
        blocks_in_total=2,
    )

    assert decided == build_run(1)


def test_a_split_copys_new_records_follow_the_runs(family: AnyFamily) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    subagent.act(build_action("later"))
    record_steps(
        trajectory,
        (
            BLOCK("a2"),
            BLOCK("a3"),
        ),
    )
    earlier = subagent.split_latest_step().trajectory
    record_steps(
        earlier,
        (
            AUDIT("s1"),
            RUN("s1"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=earlier,
        blocks_in_total=2,
    )

    assert decided == build_run(2)


def test_a_run_loaded_from_a_dump_replays_its_records_in_the_order_they_were_recorded(
    family: AnyFamily,
) -> None:
    trajectory = build_trajectory()
    subagent = start_subagent(trajectory)
    record_steps(
        subagent,
        (
            AUDIT("s1"),
            RUN("s1"),
        ),
    )
    record_steps(
        trajectory,
        (
            BLOCK("a2"),
            AUDIT("a3"),
            RUN("a3"),
        ),
    )
    record_steps(
        subagent,
        (BLOCK("s2"),),
    )
    record_steps(
        trajectory,
        (BLOCK("a4"),),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=reload(trajectory),
        blocks_in_total=2,
    )

    assert decided == build_audit(7)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            (
                BLOCK("a2"),
                BLOCK("a3"),
            ),
            (),
            id="blocks-stored",
        ),
        pytest.param(
            (),
            (
                BLOCK("a2"),
                BLOCK("a3"),
            ),
            id="nothing-stored",
        ),
    ],
)
def test_a_loaded_runs_new_records_follow_its_stored_ones(
    family: AnyFamily,
    before: tuple[Step, ...],
    after: tuple[Step, ...],
) -> None:
    trajectory = build_trajectory()
    start_subagent(trajectory)
    record_steps(
        trajectory,
        before,
    )
    loaded = reload(trajectory)
    record_steps(
        loaded,
        after,
    )
    record_steps(
        loaded.subagents["call_1"],
        (
            AUDIT("s1"),
            RUN("s1"),
        ),
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=reload(loaded).subagents["call_1"],
        blocks_in_total=2,
    )

    assert decided == build_run(2)


def work_apart(
    stored: JSON,
    *,
    call_id: str,
    steps: tuple[Step, ...],
) -> Trajectory:
    """Load the stored run, as a process of its own does, record the steps on the subagent the
    call delegates there, and return that subagent loaded from its dump.
    """
    subagent = Trajectory.load(stored).delegate(
        INSTRUCTION,
        call_id=call_id,
    )
    record_steps(
        subagent,
        steps,
    )
    return Trajectory.load(subagent.dump())


@pytest.mark.parametrize(
    ("first", "second", "outcome"),
    [
        pytest.param(
            "call_1",
            "call_2",
            build_audit(2),
            id="call-1-then-call-2",
        ),
        pytest.param(
            "call_2",
            "call_1",
            build_run(2),
            id="call-2-then-call-1",
        ),
    ],
)
def test_parallel_subagents_replay_in_the_order_they_were_joined(
    family: AnyFamily,
    first: str,
    second: str,
    outcome: ProtocolDecision,
) -> None:
    trajectory = build_trajectory()
    trajectory.act(DELEGATING_TWICE)
    stored = trajectory.dump()
    apart = {
        "call_1": work_apart(
            stored,
            call_id="call_1",
            steps=(
                BLOCK("f1"),
                AUDIT("f2"),
                RUN("f2"),
            ),
        ),
        "call_2": work_apart(
            stored,
            call_id="call_2",
            steps=(
                AUDIT("s1"),
                BLOCK("s1"),
            ),
        ),
    }
    trajectory.join(
        apart[first],
        call_id=first,
    )
    trajectory.join(
        apart[second],
        call_id=second,
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=trajectory,
        blocks_in_total=2,
    )

    assert decided == outcome


@pytest.fixture
def blocked_then_delegating() -> Trajectory:
    """A trajectory whose action a1 was blocked, and which then acted DELEGATING."""
    trajectory = build_trajectory()
    record_steps(
        trajectory,
        (BLOCK("a1"),),
    )
    trajectory.act(DELEGATING)
    return trajectory


@pytest.mark.parametrize(
    ("deciding", "position"),
    [
        pytest.param(
            lambda delegator, joined: delegator,
            4,
            id="the-delegator",
        ),
        pytest.param(
            lambda delegator, joined: joined,
            3,
            id="the-joined-subagent",
        ),
    ],
)
def test_a_joined_subagents_blocks_count_in_its_runs_total(
    family: AnyFamily,
    deciding: Callable[
        [
            Trajectory,
            Trajectory,
        ],
        Trajectory,
    ],
    position: int,
    blocked_then_delegating: Trajectory,
) -> None:
    joined = blocked_then_delegating.join(
        work_apart(
            blocked_then_delegating.dump(),
            call_id="call_1",
            steps=(BLOCK("s1"),),
        ),
        call_id="call_1",
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=deciding(
            blocked_then_delegating,
            joined,
        ),
        blocks_in_total=2,
    )

    assert decided == build_audit(position)


def test_a_joined_runs_records_replay_in_the_order_they_were_recorded_across_its_subagents(
    family: AnyFamily,
) -> None:
    trajectory = build_trajectory()
    trajectory.act(DELEGATING)
    subagent = Trajectory.load(trajectory.dump()).delegate(
        INSTRUCTION,
        call_id="call_1",
    )
    nested = start_subagent(subagent)
    record_steps(
        subagent,
        (BLOCK("s1"),),
    )
    record_steps(
        nested,
        (AUDIT("n1"),),
    )
    record_steps(
        subagent,
        (BLOCK("s2"),),
    )
    trajectory.join(
        Trajectory.load(subagent.dump()),
        call_id="call_1",
    )
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=trajectory,
        blocks_in_total=2,
    )

    assert decided == build_audit(2)


def test_a_joined_subagents_split_copy_leaves_out_the_records_at_the_position_it_drops(
    family: AnyFamily,
    blocked_then_delegating: Trajectory,
) -> None:
    joined = blocked_then_delegating.join(
        work_apart(
            blocked_then_delegating.dump(),
            call_id="call_1",
            steps=(
                BLOCK("s1"),
                AUDIT("s2"),
                RUN("s2"),
            ),
        ),
        call_id="call_1",
    )
    step = joined.split_latest_step()
    script = Script(build_reply(2))

    decided = family.decide_blocking(
        script,
        proposed=step.action,
        trajectory=step.trajectory,
        blocks_in_total=2,
    )

    assert decided == REDECIDED


@pytest.mark.parametrize(
    ("on_unavailable", "outcome"),
    [
        pytest.param(
            FailureResponse.AUDIT,
            Audit(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="audit",
        ),
        pytest.param(
            FailureResponse.BLOCK,
            Block(
                action=PROPOSED,
                reason=UNAVAILABLE_EXPLANATION,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="block",
        ),
        pytest.param(
            FailureResponse.RUN,
            Run(
                action=PROPOSED,
                proposals=(
                    build_proposal(
                        PROPOSED,
                        UNAVAILABLE_SCORE,
                    ),
                ),
                position=1,
            ),
            id="run",
        ),
    ],
)
def test_on_unavailable_decides_about_a_proposal_whose_monitor_got_no_answer(
    family: AnyFamily,
    on_unavailable: FailureResponse,
    outcome: ProtocolDecision,
) -> None:
    script = Script(UNAVAILABLE)

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_unavailable=on_unavailable,
    )

    assert decided == outcome


def test_a_proposal_whose_monitor_got_no_answer_is_audited_by_default(family: AnyFamily) -> None:
    script = Script(UNAVAILABLE)

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
    )

    assert decided == Audit(
        action=PROPOSED,
        proposals=(
            build_proposal(
                PROPOSED,
                UNAVAILABLE_SCORE,
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("failure", "score"),
    FAILURES_WITH_AN_ANSWER,
    ids=FAILURE_IDS,
)
def test_a_proposal_that_failed_closed_with_an_answer_is_audited_whatever_on_unavailable_says(
    family: AnyFamily,
    failure: GenerationOutcome,
    score: SuspicionScore,
) -> None:
    script = Script(failure)

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=build_trajectory(),
        on_unavailable=FailureResponse.RUN,
    )

    assert decided == Audit(
        action=PROPOSED,
        proposals=(
            build_proposal(
                PROPOSED,
                score,
            ),
        ),
        position=1,
    )


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        pytest.param(
            UNAVAILABLE,
            UNAVAILABLE_SCORE,
            id="unavailable-with-run-on-unavailable",
        ),
        pytest.param(
            build_reply(9),
            build_score(0.9),
            id="above-the-block-threshold",
        ),
    ],
)
def test_a_limit_audits_the_proposal_whatever_its_score(
    family: AnyFamily,
    reply: GenerationOutcome,
    score: SuspicionScore,
) -> None:
    trajectory = build_trajectory()
    record_steps(
        trajectory,
        (
            BLOCK("a1"),
            BLOCK("a2"),
            BLOCK("a3"),
        ),
    )
    script = Script(reply)

    decided = family.decide_blocking(
        script,
        proposed=PROPOSED,
        trajectory=trajectory,
        on_unavailable=FailureResponse.RUN,
    )

    assert decided == Audit(
        action=PROPOSED,
        proposals=(
            build_proposal(
                PROPOSED,
                score,
            ),
        ),
        position=7,
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        pytest.param(
            {"block_threshold": -0.1},
            "block_threshold must be between 0 and 1, got -0.1",
            id="block-threshold-below-0",
        ),
        pytest.param(
            {"blocks_in_a_row": 0},
            "blocks_in_a_row must be at least 1, got 0",
            id="blocks-in-a-row-0",
        ),
        pytest.param(
            {"blocks_in_a_row": True},
            "blocks_in_a_row must be a whole number of at least 1, got True",
            id="blocks-in-a-row-a-bool",
        ),
        pytest.param(
            {"blocks_in_total": 0},
            "blocks_in_total must be at least 1, got 0",
            id="blocks-in-total-0",
        ),
        pytest.param(
            {"on_unavailable": "block"},
            "on_unavailable must be a FailureResponse, got an instance of str",
            id="on-unavailable-a-string",
        ),
        pytest.param(
            {"on_unavailable": FailureResponse.TRUSTED_STEP},
            "on_unavailable cannot be TRUSTED_STEP, since this protocol never asks for proposals",
            id="on-unavailable-a-trusted-step",
        ),
    ],
)
def test_an_option_that_cannot_work_is_refused_naming_it(
    family: AnyFamily,
    options: dict[str, object],
    message: str,
) -> None:
    script = Script()

    with pytest.raises(ConfigurationError) as raised:
        family.build_blocking(
            script,
            **options,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            partial(
                BlockAndFeedBack,
                monitor=build_asynchronous_llm(Script()),
                block_threshold=0.5,
            ),
            "monitor must be a Monitor, got an instance of AsyncLLMMonitor. "
            "Use AsyncBlockAndFeedBack for an AsyncMonitor.",
            id="async-monitor-for-the-sync-protocol",
        ),
        pytest.param(
            partial(
                AsyncBlockAndFeedBack,
                monitor=build_synchronous_llm(Script()),
                block_threshold=0.5,
            ),
            "monitor must be an AsyncMonitor, got an instance of LLMMonitor. "
            "Use BlockAndFeedBack for a Monitor.",
            id="sync-monitor-for-the-async-protocol",
        ),
    ],
)
def test_a_monitor_of_the_other_family_is_refused_naming_the_protocol_it_belongs_to(
    build: Callable[[], BlockAndFeedBack | AsyncBlockAndFeedBack],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build()

    assert str(raised.value) == message
