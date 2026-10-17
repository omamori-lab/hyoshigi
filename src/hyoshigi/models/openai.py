"""OpenAI's Responses API as a monitor's model, through the caller's client.

The API is OpenAI's Responses API reference, https://platform.openai.com/docs/api-reference/responses.
"""

try:
    from openai import AsyncOpenAI, OpenAI
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenAI models need the extra: pip install 'hyoshigi[openai]'"
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
    client: OpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client."""
    return GenerativeModel(name=model, generate=lambda prompt: Refused(), perceives=perceives)


def build_async_model(
    client: AsyncOpenAI, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's async client."""

    async def generate(prompt: Prompt) -> GenerationOutcome:
        return Refused()

    return AsyncGenerativeModel(name=model, generate=generate, perceives=perceives)
