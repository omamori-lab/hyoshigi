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
from hyoshigi.trajectory.history import Action, Event, Observation, build_feedback
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import Author, TextPart
from hyoshigi.trajectory.redaction import Redaction, Redactor, StoredContent, check_redaction
from hyoshigi.trajectory.storage.events import dump_event
from hyoshigi.trajectory.storage.records import dump_record
from hyoshigi.trajectory.storage.scores import NESTING_LIMIT, dump_score

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
    17.3). Each observation also carries who wrote it. With a `redaction`, a Redaction, each
    entry, rating and record is redacted before it is kept, and a split copy, a subagent and a
    monitor's reading of an action rated against it redact the same; anything else raises
    ConfigurationError.

    <https://omamori-lab.github.io/hyoshigi/how-to/redacting-what-leaves-the-process/>
    """

    def __init__(
        self,
        *,
        redaction: Redaction | None = None,
    ) -> None:
        check_redaction(redaction)
        self._redaction = redaction
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
        # The latest action split off this trajectory, which it kept redacted.
        self._split_off: Action | None = None

    @classmethod
    def from_run(
        cls,
        task: str,
        *,
        answer: str,
        redaction: Redaction | None = None,
    ) -> BlackBoxRun:
        """Build the trajectory of a black-box run and its answer as the action to rate."""
        trajectory = cls(redaction=redaction)
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
        trajectory = cls(redaction=redaction)
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
        trajectory = cls(redaction=redaction)
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
        trajectory = cls(redaction=redaction)
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
    def redaction(self) -> Redaction | None:
        """The redaction each entry passes through before it is kept, or None."""
        return self._redaction

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
                earlier = Trajectory(redaction=self._redaction)
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
                earlier._split_off = action
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
        copied = Trajectory(redaction=self._redaction)
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
        once, and each rating is kept, its score redacted.
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
                score=Redactor(self._redaction).redact_score(score),
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
        record only. The decision is redacted once, and its action is the action it adds. A
        decision that is not one of the six, a position that is not the length of the history,
        and an action act refuses raise ConfigurationError, and nothing is recorded.

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
        decision = Redactor(self._redaction).redact_decision(decision)
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
                self._keep(action)
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
        """Keep the action, and rate it with the score of its first proposal, if that has one."""
        self._keep(action)
        score = next(proposal.score for proposal in proposals if proposal.action == action)
        if score is not None:
            self._ratings.append(
                Rating(
                    action_id=action.id,
                    score=score,
                ),
            )

    def act(
        self,
        action: Action,
    ) -> None:
        """Record an action the agent took, redacted.

        The action must be an Action, and its id and each of its tool calls' ids new to the
        trajectory; otherwise ConfigurationError is raised and nothing is recorded.
        """
        check_instance(
            action,
            expected=Action,
            name="action",
        )
        self._keep(Redactor(self._redaction).redact_action(action))

    def _keep(
        self,
        action: Action,
    ) -> None:
        """Append an action whose id and tool calls' ids are new, or raise ConfigurationError."""
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
        """Record an observation that reached the agent, redacted; anything but an Observation
        raises ConfigurationError.
        """
        check_instance(
            observation,
            expected=Observation,
            name="observation",
        )
        self._history.append(Redactor(self._redaction).redact_observation(observation))

    def delegate(
        self,
        instruction: str,
        *,
        call_id: str,
    ) -> Trajectory:
        """Start a subagent's trajectory, held under the tool call that delegated it.

        The call id must be a str naming a call an action in this trajectory made, and the
        subagent at most 32 levels below the root of the delegation tree; otherwise
        ConfigurationError is raised.
        The subagent's task is the instruction, authored by the delegator.
        """
        check_instance(
            call_id,
            expected=str,
            name="call_id",
        )
        if call_id not in self._call_ids:
            raise ConfigurationError("delegate takes the id of a tool call in this trajectory")
        if call_id in self._subagents:
            raise ConfigurationError("a subagent is already delegated by this call")
        if len(trace_path(self)) > NESTING_LIMIT:
            raise ConfigurationError(f"subagents are nested at most {NESTING_LIMIT} deep")
        child = Trajectory(redaction=self._redaction)
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

    def join(
        self,
        subagent: Trajectory,
        *,
        call_id: str,
    ) -> Trajectory:
        """Hold a copy of a subagent's trajectory, kept apart from this run, under the tool call
        that delegated it, and return the copy.

        The subagent must be a Trajectory at the root of a run apart from this one, such as one
        loaded from a subagent's dump, the call id a str naming a call an action in this
        trajectory made with no subagent yet, and each trajectory of the subagent's tree at most
        32 levels below the root once joined; otherwise ConfigurationError is raised and nothing
        changes. The copy is the load of the subagent's dump through this trajectory's
        redaction, as load(data, redaction=...) is, so it marks no action as split off. Its
        trajectories take this run's sequence, and their records the run's next numbers, in the
        order of their own. The subagent is unchanged.

        <https://omamori-lab.github.io/hyoshigi/explanation/control-protocols/>
        """
        check_instance(
            subagent,
            expected=Trajectory,
            name="subagent",
        )
        check_instance(
            call_id,
            expected=str,
            name="call_id",
        )
        if call_id not in self._call_ids:
            raise ConfigurationError("join takes the id of a tool call in this trajectory")
        if call_id in self._subagents:
            raise ConfigurationError("a subagent is already delegated by this call")
        # Every trajectory of a run, and every copy split from one, shares the run's sequence.
        if subagent._delegation is not None or subagent._sequence is self._sequence:
            raise ConfigurationError("join takes the root of a run apart from this one")
        joined = Trajectory.load(
            subagent.dump(),
            redaction=self._redaction,
        )
        members = list_run(joined)
        joined._delegation = Delegation(
            parent=self,
            call_id=call_id,
        )
        # The path from a trajectory to the root holds one trajectory more than its depth.
        if max(len(trace_path(member)) for member in members) > NESTING_LIMIT + 1:
            raise ConfigurationError(f"subagents are nested at most {NESTING_LIMIT} deep")
        self._take_into_run(members)
        self._subagents[call_id] = joined
        return joined

    def _take_into_run(
        self,
        members: list[Trajectory],
    ) -> None:
        """Give each trajectory this run's sequence, and its records the sequence's next numbers,
        in the order of their own numbers.
        """
        numbers = {
            number: next(self._sequence)
            for number in sorted(
                number for member in members for number in member._sequence_numbers
            )
        }
        for member in members:
            member._sequence_numbers = [numbers[number] for number in member._sequence_numbers]
            member._sequence = self._sequence

    def dump(
        self,
        *,
        redaction: Redaction | None = None,
    ) -> dict[str, JSON]:
        """Return the trajectory, its ratings, its records with their sequence numbers, and its
        subagents' as JSON-compatible data, through the redaction when one is given.

        The trajectory is unchanged. A redaction that is not a Redaction raises
        ConfigurationError.
        """
        check_redaction(redaction)
        return self._dump(Redactor(redaction))

    def _dump(
        self,
        redactor: Redactor,
    ) -> dict[str, JSON]:
        subagents: dict[str, JSON] = {
            call_id: child._dump(redactor) for call_id, child in self._subagents.items()
        }
        return {
            "format": FORMAT,
            "version": VERSION,
            "history": [dump_event(redactor.redact_event(event)) for event in self._history],
            "ratings": [
                {
                    "action_id": rating.action_id,
                    "score": dump_score(redactor.redact_score(rating.score)),
                }
                for rating in self._ratings
            ],
            "records": [
                {
                    **dump_record(redactor.redact_decision(decision)),
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
        content: StoredContent = StoredContent.RAW,
    ) -> Self:
        """Rebuild a trajectory from what dump returned.

        Raises FormatError, naming the JSONPath of the first problem, on data that
        lacks what dump writes or holds it in another shape, on composite scores or
        subagents nested more than 32 levels deep, and on a record's sequence number that
        is not a whole number of at least 0, greater than the one before it, that no other
        record in the run has. Keys dump does not write are ignored. Records recorded after
        loading take numbers past the largest loaded. With a redaction, the trajectory and its
        subagents keep it for later entries, as Trajectory(redaction=...) does, and content
        StoredContent.RAW, the default, is the load of what dump writes through it; content
        StoredContent.REDACTED is loaded as it was stored. A redaction that is not a Redaction,
        content that is not a StoredContent, and REDACTED without a redaction raise
        ConfigurationError before the data is read.
        """
        # Imported here because the reader builds trajectories: a top-level import would be a
        # cycle.
        from hyoshigi.trajectory.storage import trajectories  # lanorme: ignore[IMPORT-001]

        check_redaction(redaction)
        check_instance(
            content,
            expected=StoredContent,
            name="content",
        )
        if content is StoredContent.REDACTED and redaction is None:
            raise ConfigurationError(
                "content=StoredContent.REDACTED takes the redaction for the entries added later",
            )
        if content is StoredContent.RAW and redaction is not None:
            data = cls.load(data).dump(redaction=redaction)
        seen: set[int] = set()
        trajectory = trajectories.load_trajectory(
            data,
            kind=cls,
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
            member._redaction = redaction
        return trajectory

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


def trace_path(trajectory: Trajectory) -> list[Trajectory]:
    """The trajectory and those that delegated it, up to the root of its delegation tree."""
    path = [trajectory]
    while (delegation := path[-1]._delegation) is not None:
        path.append(delegation.parent)
    return path


def is_split_off(
    action: Action,
    *,
    trajectory: Trajectory,
) -> bool:
    """Whether the action is the one split_latest_step split off the trajectory, which the
    trajectory's redaction already reached.
    """
    return action is trajectory._split_off


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
