"""The Open Responses format: a request's instructions and input items, and a response's output.

The format is the Open Responses specification of 2026-04-24,
https://www.openresponses.org/specification/2026-04-24, and its OpenAPI document,
https://www.openresponses.org/openapi/2026-04-24/openapi.json.
"""

from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.history import Action, Event
from hyoshigi.trajectory.json_data import JSON


def read_request(request: JSON, *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a request body into its events, each with the JSONPath it came from."""
    return []


def read_response(output: JSON) -> Action:
    """Read a response's output items into the action to rate, with a generated id."""
    return Action()
