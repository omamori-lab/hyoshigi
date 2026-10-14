"""Anthropic's Messages format: a request's system prompt and messages, and a response."""

from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.history import Action, Event
from hyoshigi.trajectory.json_data import JSON


def read_request(request: JSON, *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a Messages request body into its events, each with the JSONPath it came from."""
    return []


def read_response(message: JSON) -> Action:
    """Read the assistant message of a Messages response into the action to rate."""
    return Action()
