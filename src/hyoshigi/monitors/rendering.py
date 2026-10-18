"""A trajectory and the action a monitor rates, rendered as the parts its model reads."""

from dataclasses import dataclass

from hyoshigi.models.model import MediaTypes, PromptPart
from hyoshigi.monitors.view import View
from hyoshigi.trajectory.history import Action
from hyoshigi.trajectory.trajectory import Trajectory


@dataclass(frozen=True, slots=True, kw_only=True)
class Rendering:
    """The history as a view shows it, the rated action, and what of the action is unseen.

    `hidden_reasoning` names the kinds of the rated thought's opaque parts and the media
    types of its media not shown, and is empty when the view does not show the thought.
    `unperceivable` names the same of the action's parts and tool calls.
    """

    trajectory: tuple[PromptPart, ...]
    proposed_action: tuple[PromptPart, ...]
    hidden_reasoning: tuple[str, ...]
    unperceivable: tuple[str, ...]


def render(
    action: Action, *, trajectory: Trajectory, view: View, perceives: MediaTypes
) -> Rendering:
    """Render the trajectory through the view, and the action rated against it."""
    return Rendering(trajectory=(), proposed_action=(), hidden_reasoning=(), unperceivable=())
