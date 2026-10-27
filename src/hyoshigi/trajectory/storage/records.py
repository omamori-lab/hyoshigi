"""How a control protocol's decision and the proposals it weighed are written to and read from
JSON-compatible data.
"""

from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
    Flag,
    Halt,
    Policy,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.trajectory.history import Action, Event
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.storage.events import dump_event, load_event
from hyoshigi.trajectory.storage.scores import dump_score, load_score

_RECORD = "a run, flag, replace, block, audit or halt record, with its proposals and position"
_PROPOSAL = "a proposal with an action, a score, null for a trusted one, and a policy"
# The decisions on an action that hold nothing else.
_ACTION_KINDS = {
    "run": Run,
    "flag": Flag,
    "replace": Replace,
    "audit": Audit,
}


def dump_record(decision: ProtocolDecision) -> dict[str, JSON]:
    """Return a decision and the proposals it weighed as JSON-compatible data.

    The decision's action is written as the index of the first proposal of that action.
    """
    match decision:
        case Run() | Flag() | Replace() | Audit():
            fields: dict[str, JSON] = {
                "proposal": index_proposal(
                    decision.action,
                    proposals=decision.proposals,
                ),
            }
        case Block():
            fields = {
                "proposal": index_proposal(
                    decision.action,
                    proposals=decision.proposals,
                ),
                "reason": decision.reason,
            }
        case Halt():
            fields = {"reason": decision.reason}
        case unreachable:
            assert_never(unreachable)
    return {
        "kind": name_kind(decision),
        **fields,
        "proposals": [dump_proposal(proposal) for proposal in decision.proposals],
        "position": decision.position,
    }


def name_kind(decision: ProtocolDecision) -> str:
    """The kind a decision is stored as; a subclass takes its base's."""
    match decision:
        case Run() | Flag() | Replace() | Audit():
            return next(
                kind
                for kind, decision_type in _ACTION_KINDS.items()
                if isinstance(
                    decision,
                    decision_type,
                )
            )
        case Block():
            return "block"
        case Halt():
            return "halt"
        case unreachable:
            assert_never(unreachable)


def index_proposal(
    action: Action,
    *,
    proposals: tuple[Proposal, ...],
) -> int:
    """The index of the first of the proposals whose action is this one."""
    return [proposal.action for proposal in proposals].index(action)


def dump_proposal(proposal: Proposal) -> JSON:
    """Return a proposal as JSON-compatible data: its action as an event, its score or null."""
    return {
        "action": dump_event(proposal.action),
        "score": None if proposal.score is None else dump_score(proposal.score),
        "policy": proposal.policy.value,
    }


def load_records(
    records: list[JSON],
    *,
    path: str,
    history: list[Event],
) -> list[ProtocolDecision]:
    """Read the decisions that dump_record wrote, in the order they were recorded.

    Each decision's position is at least the end of what the decisions before it added to the
    history; otherwise FormatError is raised at its path.
    """
    decisions: list[ProtocolDecision] = []
    earliest = 0
    for index, data in enumerate(records):
        record_path = f"{path}[{index}]"
        decision = load_record(
            data,
            path=record_path,
            history=history,
        )
        if decision.position < earliest:
            raise FormatError(
                path=record_path,
                expected="a record at or after what the records before it added to the history",
            )
        earliest = decision.position + count_added(decision)
        decisions.append(decision)
    return decisions


def count_added(decision: ProtocolDecision) -> int:
    """The number of entries record adds to the history for a decision: a run's, a flag's or a
    replace's action; a block's action and its feedback, one entry for each tool call or one
    when there are none; nothing for an audit or a halt.
    """
    match decision:
        case Run() | Flag() | Replace():
            return 1
        case Block(action=action):
            return 1 + max(
                len(action.tool_calls),
                1,
            )
        case Audit() | Halt():
            return 0
        case unreachable:
            assert_never(unreachable)


def load_record(
    data: JSON,
    *,
    path: str,
    history: list[Event],
) -> ProtocolDecision:
    """Read a decision that dump_record wrote, on the history it was recorded on.

    A run, a flag, a replace or a block names the action at its position in the history.
    """
    match data:
        case {
            "kind": str(),
            "proposals": list(proposals),
            "position": int(position),
        }:
            if not 0 <= position <= len(history):
                raise FormatError(
                    path=f"{path}.position",
                    expected="a position from 0 to the length of the history",
                )
            loaded = tuple(
                load_proposal(
                    proposal,
                    path=f"{path}.proposals[{index}]",
                )
                for index, proposal in enumerate(proposals)
            )
            try:
                decision = build_decision(
                    data,
                    proposals=loaded,
                    position=position,
                    path=path,
                )
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected=_RECORD,
                ) from None
            check_history(
                decision,
                history=history,
                path=path,
            )
            return decision
        case _:
            raise FormatError(
                path=path,
                expected=_RECORD,
            )


def check_history(
    decision: ProtocolDecision,
    *,
    history: list[Event],
    path: str,
) -> None:
    """Raise FormatError unless a run, a flag, a replace or a block names the action at its
    position in the history.
    """
    # The slice is empty when the position is the length of the history.
    if isinstance(
        decision,
        Run | Flag | Replace | Block,
    ) and history[decision.position : decision.position + 1] != [decision.action]:
        raise FormatError(
            path=path,
            expected="a run, flag, replace or block of the action at its position in the history",
        )


def build_decision(
    data: JSON,
    *,
    proposals: tuple[Proposal, ...],
    position: int,
    path: str,
) -> ProtocolDecision:
    """Build the decision of the kind a record names, on its loaded proposals."""
    match data:
        case {"kind": str(kind)} if kind in _ACTION_KINDS:
            return _ACTION_KINDS[kind](
                action=read_action(
                    data,
                    proposals=proposals,
                    path=path,
                ),
                proposals=proposals,
                position=position,
            )
        case {
            "kind": "block",
            "reason": str(reason),
        }:
            return Block(
                action=read_action(
                    data,
                    proposals=proposals,
                    path=path,
                ),
                reason=reason,
                proposals=proposals,
                position=position,
            )
        case {
            "kind": "halt",
            "reason": str(reason),
        }:
            return Halt(
                reason=reason,
                proposals=proposals,
                position=position,
            )
        case _:
            raise FormatError(
                path=path,
                expected=_RECORD,
            )


def read_action(
    data: JSON,
    *,
    proposals: tuple[Proposal, ...],
    path: str,
) -> Action:
    """Read a record's action, which dump_record wrote as the index of one of its proposals."""
    match data:
        case {"proposal": int(index)} if not isinstance(
            index,
            bool,
        ) and 0 <= index < len(proposals):
            return proposals[index].action
        case _:
            raise FormatError(
                path=f"{path}.proposal",
                expected="the index of one of the record's proposals",
            )


def load_proposal(
    data: JSON,
    *,
    path: str,
) -> Proposal:
    """Read a proposal that dump_proposal wrote."""
    match data:
        case {
            "action": dict() as action,
            "score": dict() | None as score,
            "policy": str(policy),
        } if policy in Policy:
            try:
                return Proposal(
                    action=load_action(
                        action,
                        path=f"{path}.action",
                    ),
                    score=None
                    if score is None
                    else load_score(
                        score,
                        path=f"{path}.score",
                        depth=0,
                    ),
                    policy=Policy(policy),
                )
            except ConfigurationError:
                raise FormatError(
                    path=path,
                    expected=_PROPOSAL,
                ) from None
        case _:
            raise FormatError(
                path=path,
                expected=_PROPOSAL,
            )


def load_action(
    data: JSON,
    *,
    path: str,
) -> Action:
    """Read a proposal's action, which dump_proposal wrote as an event."""
    match load_event(
        data,
        path=path,
    ):
        case Action() as action:
            return action
        case _:
            raise FormatError(
                path=path,
                expected="an action",
            )


def read_sequence_numbers(
    records: list[JSON],
    *,
    path: str,
    seen: set[int],
) -> list[int]:
    """Read each stored record's sequence number, and add it to the numbers `seen` in the run.

    A number that is not a whole number of at least 0, greater than the one before it, that no
    other record in the run has raises FormatError at its path.
    """
    numbers: list[int] = []
    for index, data in enumerate(records):
        match data:
            case {"sequence": int(number)} if (
                not isinstance(
                    number,
                    bool,
                )
                and number > (numbers[-1] if numbers else -1)
                and number not in seen
            ):
                numbers.append(number)
                seen.add(number)
            case _:
                raise FormatError(
                    path=f"{path}[{index}].sequence",
                    expected="a whole number of at least 0, greater than the one before it, that "
                    "no other record in the run has",
                )
    return numbers
