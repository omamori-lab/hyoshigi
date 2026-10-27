"""A run's trajectory, which only grows, and the black-box run built from it."""

# Trajectory and BlackBoxRun refer to each other.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple, Self, assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.monitors.score import SuspicionScore
from hyoshigi.options import check_instance, describe_type
from hyoshigi.protocols.decisions import (
    Audit,
    Block,
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
from hyoshigi.trajectory.storage.events import dump_event, load_event
from hyoshigi.trajectory.storage.records import dump_record, load_record
from hyoshigi.trajectory.storage.scores import NESTING_LIMIT, dump_score, load_score

FORMAT = "hyoshigi.trajectory"
VERSION = 2
_NAME_ESCAPES = {
    **{code: f"\\u{code:04x}" for code in range(0x20)},
    **{code: f"\\u{code:04x}" for code in range(0xD800, 0xE000)},
    ord("\b"): "\\b",
    ord("\t"): "\\t",
    ord("\n"): "\\n",
    ord("\f"): "\\f",
    ord("\r"): "\\r",
    ord("'"): "\\'",
    ord("\\"): "\\\\",
}


class BlackBoxRun(NamedTuple):
    """A black-box agent's run: the trajectory holding its task, and its answer to rate."""

    trajectory: Trajectory
    answer: Action


class Step(NamedTuple):
    """An action to rate, and the trajectory before it."""

    trajectory: Trajectory
    action: Action


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

    def __init__(self) -> None:
        self._history: list[Event] = []
        self._action_ids: set[str] = set()
        self._call_ids: set[str] = set()
        self._ratings: list[Rating] = []
        self._subagents: dict[str, Trajectory] = {}
        self._records: list[ProtocolDecision] = []

    @classmethod
    def from_run(
        cls,
        task: str,
        *,
        answer: str,
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

        The new trajectory keeps the ratings and subagents of the earlier actions, and the
        records whose position is before the latest action's; it shares the subagents'
        trajectories, and this trajectory is unchanged. A trajectory whose latest entry is not
        an action raises ConfigurationError.
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
                return Step(
                    trajectory=earlier,
                    action=action,
                )
            case _:
                raise ConfigurationError("the latest entry of the trajectory is not an action")

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

        A Run or a Replace adds its action, as act does, and attaches the score of the first
        proposal of that action, unless that proposal is a trusted step. A Block adds its
        action, then the reason, written by the monitor: an error answering each of the action's
        tool calls, in order, or one observation when it made none. An Audit or a Halt adds the
        record only. A decision that is not one of the five, a position that is not the length
        of the history, and an action act refuses raise ConfigurationError, and nothing is
        recorded.
        """
        if not isinstance(
            decision,
            Run | Replace | Block | Audit | Halt,
        ):
            raise ConfigurationError(
                "decision must be a Run, Replace, Block, Audit or Halt, "
                f"got {describe_type(decision)}",
            )
        if decision.position != len(self._history):
            raise ConfigurationError("the decision's position is not the length of the history")
        match decision:
            case Run(action=action) | Replace(action=action):
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

        The action's id and each of its tool calls' ids must be new to the
        trajectory; otherwise nothing is recorded.
        """
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
        """Record an observation that reached the agent."""
        self._history.append(observation)

    def delegate(
        self,
        instruction: str,
        *,
        call_id: str,
    ) -> Trajectory:
        """Start a subagent's trajectory, held under the tool call that delegated it.

        The call must be one an action in this trajectory made. The subagent's task
        is the instruction, authored by the delegator.
        """
        if call_id not in self._call_ids:
            raise ConfigurationError("delegate takes the id of a tool call in this trajectory")
        if call_id in self._subagents:
            raise ConfigurationError("a subagent is already delegated by this call")
        child = Trajectory()
        child.observe(
            Observation(
                author=Author.DELEGATOR,
                parts=(TextPart(text=instruction),),
            ),
        )
        self._subagents[call_id] = child
        return child

    def dump(self) -> dict[str, JSON]:
        """Return the trajectory, its ratings, its records and its subagents' as JSON-compatible
        data.
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
            "records": [dump_record(decision) for decision in self._records],
            "subagents": subagents,
        }

    @classmethod
    def load(
        cls,
        data: JSON,
    ) -> Self:
        """Rebuild a trajectory from what dump returned.

        Raises FormatError, naming the JSONPath of the first problem, on data that
        lacks what dump writes or holds it in another shape, and on composite scores or
        subagents nested more than 32 levels deep. Keys dump does not write are ignored.
        """
        return cls._load(
            data,
            path="$",
            depth=0,
        )

    @classmethod
    def _load(
        cls,
        data: JSON,
        *,
        path: str,
        depth: int,
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
                trajectory._records = [
                    load_record(
                        record,
                        path=f"{path}.records[{index}]",
                        history=trajectory._history,
                    )
                    for index, record in enumerate(records)
                ]
                trajectory._load_subagents(
                    subagents,
                    path=f"{path}.subagents",
                    depth=depth,
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
    ) -> None:
        """Load the subagents of a trajectory `depth` levels deep. A subagent deeper than
        NESTING_LIMIT raises FormatError.
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
            self._subagents[call_id] = Trajectory._load(
                data,
                path=subagent_path,
                depth=depth + 1,
            )


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


def select_member(
    name: str,
    *,
    path: str,
) -> str:
    r"""Return the path to the member `name` of the object at `path`.

    The member is named in single quotes and escaped as in a Normalized Path (RFC 9535,
    sec. 2.7). An unpaired surrogate, which no JSONPath can name, is written as an
    escape, such as `\ud800`.
    """
    return f"{path}['{name.translate(_NAME_ESCAPES)}']"
