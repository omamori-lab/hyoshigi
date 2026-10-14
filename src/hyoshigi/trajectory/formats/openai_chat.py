"""The OpenAI chat completions format, with OpenRouter's reasoning fields.

The format is OpenAI's OpenAPI specification, https://github.com/openai/openai-openapi,
and OpenRouter's reasoning tokens guide,
https://openrouter.ai/docs/use-cases/reasoning-tokens.
"""

from hyoshigi.trajectory.formats.authorship import Authorship
from hyoshigi.trajectory.history import Action, Event
from hyoshigi.trajectory.json_data import JSON


def read_request(request: JSON, *, authorship: Authorship) -> list[tuple[str, Event]]:
    """Read a chat completions request body into its events, each with its JSONPath."""
    return []


def read_response(message: JSON) -> Action:
    """Read a response's assistant message into the action to rate."""
    return Action()
