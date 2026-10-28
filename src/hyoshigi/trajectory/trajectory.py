"""A run's trajectory, which only grows, and the black-box run built from it."""

# Trajectory and BlackBoxRun refer to each other.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from operator import itemgetter
from types import MappingProxyType
from typing import NamedTuple, Self, assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.options import check_instance, describe_type
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
    Flag,
    Halt,
    Proposal,
    ProtocolDecision,
    Replace,
    Run,
)
from hyoshigi.trajectory.formats import anthropic, open_responses, openai_chat
from hyoshigi.trajectory.formats.authorship import Authorship, by_role
from hyoshigi.trajectory.history import Action, Event, Observation, ToolObservation
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, TextPart
from hyoshigi.trajectory.redaction import Redaction
from hyoshigi.trajectory.storage.events import dump_event, load_event
from hyoshigi.trajectory.storage.paths import select_member
from hyoshigi.trajectory.storage.records import dump_record, load_records, read_sequence_numbers
from hyoshigi.trajectory.storage.scores import NESTING_LIMIT, dump_score, load_score

FORMAT = "hyoshigi.trajectory"
VERSION = 3


class BlackBoxRun(NamedTuple):
    """A black-box agent's run: the trajectory holding its task, and its answer to rate."""

    trajectory: Trajectory
    answer: Action


class Step(NamedTuple):
    """An action to rate, and the trajectory before it."""

    trajectory: Trajectory
    action: Action


class Delegation(NamedTuple):
    """A subagent's place in its run: the trajectory that delegated it, and the delegating call's
    id.
    """

    parent: Trajectory
    call_id: str


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Rating:
    """A monitor's score of an action in the trajectory."""

    action_id: str
    score: SuspicionScore


class Trajectory:
    """One run's actions and observations, in order; nothing is removed or rewritten.

    A trajectory is a history of actions and observations (Sutton and Barto,
    "Reinforcement Learning: An Introduction", second edition, sections 3.1 and
    17.3). Each observation also carries who wrote it.
    """

    def __init__(
        self,
        *,
        redaction: Redaction | None = None,
    ) -> None:
        self._history: list[Event] = []
        self._action_ids: set[str] = set()
        self._call_ids: set[str] = set()
        self._ratings: list[Rating] = []
        self._subagents: dict[str, Trajectory] = {}
        self._records: list[ProtocolDecision] = []
        self._sequence_numbers: list[int] = []
        # Every trajectory of a run shares it, so each record takes a number no other record has.
        self._sequence = count()
        self._delegation: Delegation | None = None

    @classmethod
    def from_run(
        cls,
        task: str,
        *,
        answer: str,
        redaction: Redaction | None = None,
    ) -> BlackBoxRun:
        """Build the trajectory of a black-box run and its answer as the action to rate."""
        trajectory = cls()
        trajectory.observe(
            Observation(
                author=Author.USER,
                parts=(TextPart(text=task),),
            ),
        )
        return BlackBoxRun(
            trajectory=trajectory,
            answer=Action(parts=(TextPart(text=answer),)),
        )

    @classmethod
    def from_anthropic(
        cls,
        request: JSON,
        *,
        authorship: Authorship = by_role,
        redaction: Redaction | None = None,
    ) -> Self:
        """Build the trajectory of an Anthropic Messages request body.

        Raises FormatError, naming the JSONPath of the first problem, on a body that is
        not in the Messages format.
        """
        trajectory = cls()
        for path, event in anthropic.read_request(
            request,
            authorship=authorship,
        ):
            trajectory._add_event(
                event,
                path=path,
            )
        return trajectory

    @classmethod
    def from_openai_chat(
        cls,
        request: JSON,
        *,
        authorship: Authorship = by_role,
        redaction: Redaction | None = None,
    ) -> Self:
        """Build the trajectory of an OpenAI chat completions request body.

        OpenRouter's reasoning fields are read as the assistant's thought, and its images
        as media parts. Raises FormatError, naming the JSONPath of the first problem, on a
        body that is not in the chat completions format.
        """
        trajectory = cls()
        for path, event in openai_chat.read_request(
            request,
            authorship=authorship,
        ):
            trajectory._add_event(
                event,
                path=path,
            )
        return trajectory

    @classmethod
    def from_open_responses(
        cls,
        request: JSON,
        *,
        authorship: Authorship = by_role,
        redaction: Redaction | None = None,
    ) -> Self:
        """Build the trajectory of an Open Responses request body.

        The model's consecutive items become one action. Raises FormatError, naming the
        JSONPath of the first problem, on a body that is not in the Open Responses format.
        """
        trajectory = cls()
        for path, event in open_responses.read_request(
            request,
            authorship=authorship,
        ):
            trajectory._add_event(
                event,
                path=path,
            )
        return trajectory

    @property
    def history(self) -> tuple[Event, ...]:
        """The actions and observations, in the order they happened."""
        return tuple(self._history)

    @property
    def subagents(self) -> Mapping[str, Trajectory]:
        """Each subagent's trajectory, by the id of the tool call that delegated it."""
        return MappingProxyType(self._subagents)

    @property
    def ratings(self) -> tuple[Rating, ...]:
        """The ratings of the actions in the history, in the order they were attached."""
        return tuple(self._ratings)

    @property
    def records(self) -> tuple[ProtocolDecision, ...]:
        """The control protocol's decisions on this trajectory's steps, in the order they were
        recorded.
        """
        return tuple(self._records)

    def split_latest_step(self) -> Step:
        """Return the latest action, and a new trajectory of everything before it.

        The new trajectory keeps the ratings and subagents of the earlier actions, the records
        whose position is before the latest action's, and this trajectory's place in its run. It
        holds copies of the subagents' trajectories, at every depth, so a record on a copy leaves
        the original unchanged, and this trajectory is unchanged. A trajectory whose latest entry
        is not an action raises ConfigurationError.
        """
        match self._history:
            case [
                *earlier_history,
                Action() as action,
            ]:
                call_ids = {call.id for call in action.tool_calls}
                earlier = Trajectory()
                earlier._history = earlier_history
                earlier._action_ids = self._action_ids - {action.id}
                earlier._call_ids = self._call_ids - call_ids
                earlier._ratings = [
                    rating for rating in self._ratings if rating.action_id != action.id
                ]
                earlier._subagents = {
                    call_id: child
                    for call_id, child in self._subagents.items()
                    if call_id not in call_ids
                }
                earlier._records = [
                    record for record in self._records if record.position < len(earlier_history)
                ]
                # The records kept come first, since their positions never decrease.
                earlier._sequence_numbers = self._sequence_numbers[: len(earlier._records)]
                earlier._sequence = self._sequence
                earlier._delegation = self._delegation
                earlier._copy_subagents()
                return Step(
                    trajectory=earlier,
                    action=action,
                )
            case _:
                raise ConfigurationError("the latest entry of the trajectory is not an action")

    def _copy_subagents(self) -> None:
        """Replace each subagent below this trajectory, at every depth, with a copy delegated by
        its new parent.
        """
        parents: list[Trajectory] = [self]
        while parents:
            parent = parents.pop()
            parent._subagents = {
                call_id: subagent._copy(
                    Delegation(
                        parent=parent,
                        call_id=call_id,
                    ),
                )
                for call_id, subagent in parent._subagents.items()
            }
            parents.extend(parent._subagents.values())

    def _copy(
        self,
        delegation: Delegation,
    ) -> Trajectory:
        """Return a trajectory with this one's entries, ratings, records and subagents in lists of
        its own, at the place in the run that `delegation` names.
        """
        copied = Trajectory()
        copied._history = list(self._history)
        copied._action_ids = set(self._action_ids)
        copied._call_ids = set(self._call_ids)
        copied._ratings = list(self._ratings)
        copied._subagents = dict(self._subagents)
        copied._records = list(self._records)
        copied._sequence_numbers = list(self._sequence_numbers)
        copied._sequence = self._sequence
        copied._delegation = delegation
        return copied

    def attach(
        self,
        score: SuspicionScore,
        *,
        to: Action,
    ) -> None:
        """Record a monitor's score of an action in the history.

        The score must be a SuspicionScore and the action's id in the history; otherwise
        ConfigurationError is raised and nothing is recorded. An action can be rated more than
        once, and each rating is kept.
        """
        check_instance(
            score,
            expected=SuspicionScore,
            name="score",
        )
        check_instance(
            to,
            expected=Action,
            name="to",
        )
        if to.id not in self._action_ids:
            raise ConfigurationError("attach takes an action in this trajectory")
        self._ratings.append(
            Rating(
                action_id=to.id,
                score=score,
            ),
        )

    def record(
        self,
        decision: ProtocolDecision,
    ) -> None:
        """Record a control protocol's decision, and add to the history what it ran or blocked.

        A Run, a Flag or a Replace adds its action, as act does, and attaches the score of the
        first proposal of that action, unless that proposal is a trusted step. A Block adds its
        action, then the reason, written by the monitor: an error answering each of the action's
        tool calls, in order, or one observation when it made none. An Audit or a Halt adds the
        record only. A decision that is not one of the six, a position that is not the length
        of the history, and an action act refuses raise ConfigurationError, and nothing is
        recorded.

        <https://omamori-lab.github.io/hyoshigi/explanation/control-protocols/>
        """
        if not isinstance(
            decision,
            Run | Flag | Replace | Block | Audit | Halt,
        ):
            raise ConfigurationError(
                "decision must be a Run, Flag, Replace, Block, Audit or Halt, "
                f"got {describe_type(decision)}",
            )
        if decision.position != len(self._history):
            raise ConfigurationError("the decision's position is not the length of the history")
        match decision:
            case Run(action=action) | Flag(action=action) | Replace(action=action):
                self._run(
                    action,
                    proposals=decision.proposals,
                )
            case Block(
                action=action,
                reason=reason,
            ):
                self.act(action)
                self._history.extend(
                    build_feedback(
                        action,
                        reason=reason,
                    ),
                )
            case Audit() | Halt():
                pass
            case unreachable:
                assert_never(unreachable)
        self._records.append(decision)
        self._sequence_numbers.append(next(self._sequence))

    def _run(
        self,
        action: Action,
        *,
        proposals: tuple[Proposal, ...],
    ) -> None:
        """Act the action, and attach the score of its first proposal, if that has one."""
        self.act(action)
        score = next(proposal.score for proposal in proposals if proposal.action == action)
        if score is not None:
            self.attach(
                score,
                to=action,
            )

    def act(
        self,
        action: Action,
    ) -> None:
        """Record an action the agent took.

        The action must be an Action, and its id and each of its tool calls' ids new to the
        trajectory; otherwise ConfigurationError is raised and nothing is recorded.
        """
        check_instance(
            action,
            expected=Action,
            name="action",
        )
        call_ids = [call.id for call in action.tool_calls]
        if action.id in self._action_ids:
            raise ConfigurationError("an action with this id is already in the trajectory")
        if len(set(call_ids)) < len(call_ids) or not self._call_ids.isdisjoint(call_ids):
            raise ConfigurationError(
                "each tool call needs an id no other call in the trajectory has",
            )
        self._action_ids.add(action.id)
        self._call_ids.update(call_ids)
        self._history.append(action)

    def observe(
        self,
        observation: Observation,
    ) -> None:
        """Record an observation that reached the agent; anything but an Observation raises
        ConfigurationError.
        """
        check_instance(
            observation,
            expected=Observation,
            name="observation",
        )
        self._history.append(observation)

    def delegate(
        self,
        instruction: str,
        *,
        call_id: str,
    ) -> Trajectory:
        """Start a subagent's trajectory, held under the tool call that delegated it.

        The call must be one an action in this trajectory made, and the subagent at most 32
        levels below the root of the delegation tree; otherwise ConfigurationError is raised.
        The subagent's task is the instruction, authored by the delegator.
        """
        if call_id not in self._call_ids:
            raise ConfigurationError("delegate takes the id of a tool call in this trajectory")
        if call_id in self._subagents:
            raise ConfigurationError("a subagent is already delegated by this call")
        if len(trace_path(self)) > NESTING_LIMIT:
            raise ConfigurationError(f"subagents are nested at most {NESTING_LIMIT} deep")
        child = Trajectory()
        child.observe(
            Observation(
                author=Author.DELEGATOR,
                parts=(TextPart(text=instruction),),
            ),
        )
        child._sequence = self._sequence
        child._delegation = Delegation(
            parent=self,
            call_id=call_id,
        )
        self._subagents[call_id] = child
        return child

    def dump(
        self,
        *,
        redaction: Redaction | None = None,
    ) -> dict[str, JSON]:
        """Return the trajectory, its ratings, its records with their sequence numbers, and its
        subagents' as JSON-compatible data.
        """
        subagents: dict[str, JSON] = {
            call_id: child.dump() for call_id, child in self._subagents.items()
        }
        return {
            "format": FORMAT,
            "version": VERSION,
            "history": [dump_event(event) for event in self._history],
            "ratings": [
                {
                    "action_id": rating.action_id,
                    "score": dump_score(rating.score),
                }
                for rating in self._ratings
            ],
            "records": [
                {
                    **dump_record(decision),
                    "sequence": number,
                }
                for decision, number in zip(
                    self._records,
                    self._sequence_numbers,
                    strict=True,
                )
            ],
            "subagents": subagents,
        }

    @classmethod
    def load(
        cls,
        data: JSON,
        *,
        redaction: Redaction | None = None,
    ) -> Self:
        """Rebuild a trajectory from what dump returned.

        Raises FormatError, naming the JSONPath of the first problem, on data that
        lacks what dump writes or holds it in another shape, on composite scores or
        subagents nested more than 32 levels deep, and on a record's sequence number that
        is not a whole number of at least 0, greater than the one before it, that no other
        record in the run has. Keys dump does not write are ignored. Records recorded after
        loading take numbers past the largest loaded.
        """
        seen: set[int] = set()
        trajectory = cls._load(
            data,
            path="$",
            depth=0,
            seen=seen,
        )
        sequence = count(
            max(
                seen,
                default=-1,
            )
            + 1,
        )
        for member in list_run(trajectory):
            member._sequence = sequence
        return trajectory

    @classmethod
    def _load(
        cls,
        data: JSON,
        *,
        path: str,
        depth: int,
        seen: set[int],
    ) -> Self:
        match data:
            case {
                "format": "hyoshigi.trajectory",
                "version": int(version),
                "history": list(history),
                "ratings": list(ratings),
                "records": list(records),
                "subagents": dict(subagents),
            } if version == VERSION and not isinstance(
                version,
                bool,
            ):
                trajectory = cls()
                trajectory._load_history(
                    history,
                    path=f"{path}.history",
                )
                trajectory._load_ratings(
                    ratings,
                    path=f"{path}.ratings",
                )
                trajectory._records = load_records(
                    records,
                    path=f"{path}.records",
                    history=trajectory._history,
                )
                trajectory._sequence_numbers = read_sequence_numbers(
                    records,
                    path=f"{path}.records",
                    seen=seen,
                )
                trajectory._load_subagents(
                    subagents,
                    path=f"{path}.subagents",
                    depth=depth,
                    seen=seen,
                )
                return trajectory
            case _:
                raise FormatError(
                    path=path,
                    expected=f"a {FORMAT}, format version {VERSION}",
                )

    def _load_history(
        self,
        history: list[JSON],
        *,
        path: str,
    ) -> None:
        for index, data in enumerate(history):
            event_path = f"{path}[{index}]"
            self._add_event(
                load_event(
                    data,
                    path=event_path,
                ),
                path=event_path,
            )

    def _load_ratings(
        self,
        ratings: list[JSON],
        *,
        path: str,
    ) -> None:
        for index, data in enumerate(ratings):
            rating_path = f"{path}[{index}]"
            match data:
                case {
                    "action_id": str(action_id),
                    "score": dict() as score,
                }:
                    if action_id not in self._action_ids:
                        raise FormatError(
                            path=f"{rating_path}.action_id",
                            expected="the id of an action in the history",
                        )
                    loaded = load_score(
                        score,
                        path=f"{rating_path}.score",
                        depth=0,
                    )
                    self._ratings.append(
                        Rating(
                            action_id=action_id,
                            score=loaded,
                        ),
                    )
                case _:
                    raise FormatError(
                        path=rating_path,
                        expected="a rating with an action id and a score",
                    )

    def _add_event(
        self,
        event: Event,
        *,
        path: str,
    ) -> None:
        """Act or observe an event read from a payload, or raise FormatError at its path."""
        match event:
            case Action():
                try:
                    self.act(event)
                except ConfigurationError:
                    raise FormatError(
                        path=path,
                        expected="an action with ids new to the trajectory",
                    ) from None
            case Observation():
                self.observe(event)
            case unreachable:
                assert_never(unreachable)

    def _load_subagents(
        self,
        subagents: dict[str, JSON],
        *,
        path: str,
        depth: int,
        seen: set[int],
    ) -> None:
        """Load the subagents of a trajectory `depth` levels deep, adding their records'
        sequence numbers to `seen`. A subagent deeper than NESTING_LIMIT raises FormatError.
        """
        for call_id, data in subagents.items():
            match call_id:
                case str():
                    subagent_path = select_member(
                        call_id,
                        path=path,
                    )
                case _:
                    raise FormatError(
                        path=path,
                        expected="subagents keyed by tool call id",
                    )
            if call_id not in self._call_ids:
                raise FormatError(
                    path=subagent_path,
                    expected="a subagent under a tool call in the history",
                )
            if depth == NESTING_LIMIT:
                raise FormatError(
                    path=subagent_path,
                    expected=f"subagents nested at most {NESTING_LIMIT} deep",
                )
            subagent = Trajectory._load(
                data,
                path=subagent_path,
                depth=depth + 1,
                seen=seen,
            )
            subagent._delegation = Delegation(
                parent=self,
                call_id=call_id,
            )
            self._subagents[call_id] = subagent


def trace_path(trajectory: Trajectory) -> list[Trajectory]:
    """The trajectory and those that delegated it, up to the root of its delegation tree."""
    path = [trajectory]
    while (delegation := path[-1]._delegation) is not None:
        path.append(delegation.parent)
    return path


def list_run(trajectory: Trajectory) -> list[Trajectory]:
    """The trajectories of the run the trajectory is in, from the root of its delegation tree.

    Each trajectory on the path from the root to this one stands in for the subagent held under
    the call that delegated it.
    """
    path = trace_path(trajectory)
    # A trajectory compares and hashes by identity, so a delegation names one place in the tree.
    placed = {member._delegation: member for member in path[:-1]}
    run: list[Trajectory] = []
    unvisited = [path[-1]]
    while unvisited:
        visited = unvisited.pop()
        run.append(visited)
        unvisited.extend(
            placed.get(
                Delegation(
                    parent=visited,
                    call_id=call_id,
                ),
                subagent,
            )
            for call_id, subagent in visited._subagents.items()
        )
    return run


def list_run_records(trajectory: Trajectory) -> tuple[ProtocolDecision, ...]:
    """The records of the trajectories list_run returns, in the order they were recorded."""
    numbered = sorted(
        (
            pair
            for member in list_run(trajectory)
            for pair in zip(
                member._sequence_numbers,
                member._records,
                strict=True,
            )
        ),
        key=itemgetter(0),
    )
    return tuple(record for _, record in numbered)


def build_feedback(
    action: Action,
    *,
    reason: str,
) -> list[Observation]:
    """The monitor's reason for blocking an action: an error answering each of its tool calls, in
    order, or one observation when it made none.
    """
    parts = (TextPart(text=reason),)
    return [
        ToolObservation(
            call_id=call.id,
            is_error=True,
            author=Author.MONITOR,
            parts=parts,
        )
        for call in action.tool_calls
    ] or [
        Observation(
            author=Author.MONITOR,
            parts=parts,
        ),
    ]
