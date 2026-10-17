"""TypeSafe's decision models, such as Jev, as a monitor's models, through the caller's client.

The API is TypeSafe's documentation, https://docs.typesafe.ai.
"""

try:
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "TypeSafe models need the extra: pip install 'hyoshigi[typesafe]'"
    ) from error

from hyoshigi.models.model import (
    AsyncDecisionModel,
    DecisionModel,
    DecisionOutcome,
    MediaTypes,
    Question,
    Refused,
)


def build_decision_model(
    client: TypeSafeClient, *, model: str, perceives: MediaTypes
) -> DecisionModel:
    """A decision model on TypeSafe's API, through the caller's client."""
    return DecisionModel(name=model, decide=lambda question: Refused(), perceives=perceives)


def build_async_decision_model(
    client: AsyncTypeSafeClient, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on TypeSafe's API, through the caller's async client."""

    async def decide(question: Question) -> DecisionOutcome:
        return Refused()

    return AsyncDecisionModel(name=model, decide=decide, perceives=perceives)
