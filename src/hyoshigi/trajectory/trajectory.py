"""A run's trajectory, which only grows, and the black-box run built from it."""

# Trajectory and BlackBoxRun refer to each other.
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, Self

from hyoshigi.trajectory.history import Action, Event, Observation
from hyoshigi.trajectory.storage import JSON


class BlackBoxRun(NamedTuple):
    """A black-box agent's run: the trajectory holding its task, and its answer to rate."""

    trajectory: Trajectory
    answer: Action


class Trajectory:
    """One run's actions and observations, in order; nothing is removed or rewritten."""

    def __init__(self) -> None:
        self._history: list[Event] = []
        self._subagents: dict[str, Trajectory] = {}

    @classmethod
    def from_run(cls, task: str, *, answer: str) -> BlackBoxRun:
        """Build the trajectory of a black-box run and its answer as the action to rate."""
        return BlackBoxRun(trajectory=cls(), answer=Action())

    @property
    def history(self) -> tuple[Event, ...]:
        """The actions and observations, in the order they happened."""
        return ()

    @property
    def subagents(self) -> Mapping[str, Trajectory]:
        """Each subagent's trajectory, by the id of the tool call that delegated it."""
        return {}

    def act(self, action: Action) -> None:
        """Record an action the decision allowed."""

    def observe(self, observation: Observation) -> None:
        """Record an observation that reached the agent."""

    def delegate(self, instruction: str, *, call_id: str) -> Trajectory:
        """Start a subagent's trajectory, held under the tool call that delegated it."""
        return Trajectory()

    def dump(self) -> dict[str, JSON]:
        """Return the trajectory as JSON-compatible data."""
        return {}

    @classmethod
    def load(cls, data: object) -> Self:
        """Rebuild a trajectory from what dump returned."""
        return cls()
