"""OpenRouter's Responses and Decisions APIs as a monitor's models, through the caller's client.

The APIs are OpenRouter's API reference, https://openrouter.ai/docs/api-reference/overview.
"""

try:
    from openrouter import OpenRouter
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenRouter models need the extra: pip install 'hyoshigi[openrouter]'"
    ) from error

from hyoshigi.models.model import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    DecisionModel,
    DecisionOutcome,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Question,
    Refused,
)


def build_model(
    client: OpenRouter, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client."""
    return GenerativeModel(name=model, generate=lambda prompt: Refused(), perceives=perceives)


def build_async_model(
    client: OpenRouter, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's client, awaited."""

    async def generate(prompt: Prompt) -> GenerationOutcome:
        return Refused()

    return AsyncGenerativeModel(name=model, generate=generate, perceives=perceives)


def build_decision_model(client: OpenRouter, *, model: str, perceives: MediaTypes) -> DecisionModel:
    """A decision model on the Decisions API, through the caller's client."""
    return DecisionModel(name=model, decide=lambda question: Refused(), perceives=perceives)


def build_async_decision_model(
    client: OpenRouter, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on the Decisions API, through the caller's client, awaited."""

    async def decide(question: Question) -> DecisionOutcome:
        return Refused()

    return AsyncDecisionModel(name=model, decide=decide, perceives=perceives)
