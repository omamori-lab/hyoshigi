"""A run's trajectory, which only grows, and the black-box run built from it."""

# Trajectory and BlackBoxRun refer to each other.
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple, Self, assert_never

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.history import Action, Event, Observation
from hyoshigi.trajectory.parts import Author, TextPart
from hyoshigi.trajectory.storage import FORMAT, JSON, VERSION, dump_event, load_event


class BlackBoxRun(NamedTuple):
    """A black-box agent's run: the trajectory holding its task, and its answer to rate."""

    trajectory: Trajectory
    answer: Action


class Trajectory:
    """One run's actions and observations, in order; nothing is removed or rewritten.

    A trajectory is a history of actions and observations (Sutton and Barto,
    "Reinforcement Learning: An Introduction", second edition, sections 3.1 and
    17.3). Each observation also carries who wrote it.
    """

    def __init__(self) -> None:
        self._history: list[Event] = []
        self._action_ids: set[str] = set()
        self._subagents: dict[str, Trajectory] = {}

    @classmethod
    def from_run(cls, task: str, *, answer: str) -> BlackBoxRun:
        """Build the trajectory of a black-box run and its answer as the action to rate."""
        trajectory = cls()
        trajectory.observe(Observation(author=Author.USER, parts=(TextPart(text=task),)))
        return BlackBoxRun(trajectory=trajectory, answer=Action(parts=(TextPart(text=answer),)))

    @property
    def history(self) -> tuple[Event, ...]:
        """The actions and observations, in the order they happened."""
        return tuple(self._history)

    @property
    def subagents(self) -> Mapping[str, Trajectory]:
        """Each subagent's trajectory, by the id of the tool call that delegated it."""
        return MappingProxyType(self._subagents)

    def act(self, action: Action) -> None:
        """Record an action the decision allowed."""
        if action.id in self._action_ids:
            raise ConfigurationError("an action with this id is already in the trajectory")
        self._action_ids.add(action.id)
        self._history.append(action)

    def observe(self, observation: Observation) -> None:
        """Record an observation that reached the agent."""
        self._history.append(observation)

    def delegate(self, instruction: str, *, call_id: str) -> Trajectory:
        """Start a subagent's trajectory, held under the tool call that delegated it.

        The subagent's task is the instruction, authored by the delegator.
        """
        if call_id in self._subagents:
            raise ConfigurationError("a subagent is already delegated by this call")
        child = Trajectory()
        child.observe(Observation(author=Author.DELEGATOR, parts=(TextPart(text=instruction),)))
        self._subagents[call_id] = child
        return child

    def dump(self) -> dict[str, JSON]:
        """Return the trajectory and its subagents' as JSON-compatible data."""
        subagents: dict[str, JSON] = {
            call_id: child.dump() for call_id, child in self._subagents.items()
        }
        return {
            "format": FORMAT,
            "version": VERSION,
            "history": [dump_event(event) for event in self._history],
            "subagents": subagents,
        }

    @classmethod
    def load(cls, data: JSON) -> Self:
        """Rebuild a trajectory from what dump returned.

        Raises FormatError, naming the JSONPath of the first problem, on data that
        dump could not have returned.
        """
        return cls._load(data, path="$")

    @classmethod
    def _load(cls, data: JSON, *, path: str) -> Self:
        match data:
            case {
                "format": "hyoshigi.trajectory",
                "version": int(version),
                "history": list(history),
                "subagents": dict(subagents),
            } if version == VERSION and not isinstance(version, bool):
                trajectory = cls()
                trajectory._load_history(history, path=f"{path}.history")
                trajectory._load_subagents(subagents, path=f"{path}.subagents")
                return trajectory
            case _:
                raise FormatError(path=path, expected=f"a {FORMAT}, format version {VERSION}")

    def _load_history(self, history: list[JSON], *, path: str) -> None:
        for index, data in enumerate(history):
            match load_event(data, path=f"{path}[{index}]"):
                case Action() as action if action.id in self._action_ids:
                    raise FormatError(
                        path=f"{path}[{index}].id", expected="an id no earlier action has"
                    )
                case Action() as action:
                    self.act(action)
                case Observation() as observation:
                    self.observe(observation)
                case unreachable:
                    assert_never(unreachable)

    def _load_subagents(self, subagents: dict[str, JSON], *, path: str) -> None:
        for index, (call_id, data) in enumerate(subagents.items()):
            self._subagents[call_id] = Trajectory._load(data, path=f"{path}[{index}]")
