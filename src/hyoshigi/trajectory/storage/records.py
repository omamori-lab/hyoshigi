"""How a control protocol's decision and the proposals it weighed are written to and read from
JSON-compatible data.
"""

from typing import assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
    Halt,
    Policy,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.storage.events import dump_event, load_event
from hyoshigi.trajectory.storage.scores import dump_score, load_score

_RECORD = "a run, replace, block, audit or halt record, with its proposals and position"
_PROPOSAL = "a proposal with an action, a score, null for a trusted one, and a policy"
# The decisions on an action that hold nothing else.
_ACTION_KINDS = {
    "run": Run,
    "replace": Replace,
    "audit": Audit,
}


def dump_record(decision: ProtocolDecision) -> JSON:
    """Return a decision and the proposals it weighed as JSON-compatible data.

    The decision's action is written as the index of the first proposal of that action.
    """
    match decision:
        case Run() | Replace() | Audit():
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
        case Run():
            return "run"
        case Replace():
            return "replace"
        case Block():
            return "block"
        case Audit():
            return "audit"
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


def load_record(
    data: JSON,
    *,
    path: str,
    length: int,
) -> ProtocolDecision:
    """Read a decision that dump_record wrote, on a history `length` entries long."""
    match data:
        case {
            "kind": str(),
            "proposals": list(proposals),
            "position": int(position),
        }:
            if not 0 <= position <= length:
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
                return build_decision(
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
        case _:
            raise FormatError(
                path=path,
                expected=_RECORD,
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
