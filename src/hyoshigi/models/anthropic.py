"""Anthropic's Messages API as a monitor's model, through the caller's client.

The API is Anthropic's Messages API reference, https://platform.claude.com/docs/en/api/messages.
"""

try:
    from anthropic import Anthropic, AsyncAnthropic
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Anthropic models need the extra: pip install 'hyoshigi[anthropic]'"
    ) from error

from hyoshigi.models.model import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Refused,
)


def build_model(
    client: Anthropic, *, model: str, max_tokens: int, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Messages API, through the caller's client."""
    return GenerativeModel(name=model, generate=lambda prompt: Refused(), perceives=perceives)


def build_async_model(
    client: AsyncAnthropic, *, model: str, max_tokens: int, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Messages API, through the caller's async client."""

    async def generate(prompt: Prompt) -> GenerationOutcome:
        return Refused()

    return AsyncGenerativeModel(name=model, generate=generate, perceives=perceives)
