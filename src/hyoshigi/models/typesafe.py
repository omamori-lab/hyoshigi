"""TypeSafe's decision models, such as Jev, as a monitor's models, through the caller's client.

The API is TypeSafe's documentation, https://docs.typesafe.ai.
"""

from typing import assert_never

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Image,
        Noul,
        NoulAnswer,
        SystemOneResponse,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
        TypeSafeClient,
        TypeSafeInternalServerError,
    )
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "TypeSafe models need the extra: pip install 'hyoshigi[typesafe]'"
    ) from error

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import (
    AsyncDecisionModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    MediaTypes,
    PromptPart,
    Question,
    Unavailable,
    check_name,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

type StateItem = str | Image

QUESTION_KEY = "statement"
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/*"})
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 415, 429})


def format_model_name(model: str) -> str:
    """Validate the TypeSafe model identifier and prefix it with the provider name."""
    check_name(model)
    return f"typesafe:{model}"


def translate_media(part: MediaPart) -> Image | None:
    """Translate a media part to a TypeSafe image, or None when its bytes are unusable."""
    if part.media_type.lower() not in IMAGE_MEDIA_TYPES:
        raise ConfigurationError(f"TypeSafe does not support media type {part.media_type}")
    if part.data is None or not part.data.startswith(IMAGE_SIGNATURES):
        return None
    return Image(part.data)


def translate_part(part: PromptPart) -> StateItem | None:
    """Translate a question part to a TypeSafe state item, or None when its bytes are unusable."""
    match part:
        case TextPart(text=text):
            return text
        case MediaPart() as media:
            return translate_media(media)
        case _ as unreachable:
            assert_never(unreachable)


def translate_state(parts: tuple[PromptPart, ...]) -> list[StateItem] | None:
    """Translate state parts to a TypeSafe state list, or None when an image is unusable."""
    state: list[StateItem] = []
    for part in parts:
        item = translate_part(part)
        if item is None:
            return None
        state.append(item)
    return state


def build_questions(question: Question) -> dict[str, Noul]:
    """Build the single Noul question dictionary for TypeSafe's System One API."""
    return {QUESTION_KEY: Noul(instructions=question.statement)}


def read_decision(response: SystemOneResponse) -> DecisionOutcome:
    """Read a Decision from TypeSafe's response, or Unavailable when the answer is invalid."""
    match response.answers.get(QUESTION_KEY):
        case NoulAnswer(noul=probability):
            try:
                return Decision(probability=probability)
            except ConfigurationError:
                return Unavailable()
        case _:
            return Unavailable()


def translate_error(
    error: TypeSafeAPIConnectionError | TypeSafeAPIError,
) -> Unavailable:
    """Translate a TypeSafe SDK error to Unavailable, or raise ConfigurationError on 4xx."""
    if isinstance(error, TypeSafeAPIConnectionError | TypeSafeAPIResponseValidationError):
        return Unavailable()
    if isinstance(error, TypeSafeInternalServerError) or error.status in UNAVAILABLE_STATUSES:
        return Unavailable(status=error.status)
    raise ConfigurationError(f"TypeSafe request failed with HTTP {error.status}") from None


def build_decision_model(
    client: TypeSafeClient, *, model: str, perceives: MediaTypes
) -> DecisionModel:
    """A decision model on TypeSafe's API, through the caller's client."""
    name = format_model_name(model)

    def decide(question: Question) -> DecisionOutcome:
        state = translate_state(question.state)
        if state is None:
            return Unavailable()
        try:
            response = client.system_one(
                state=state,
                questions=build_questions(question),
                model=model,
            )
        except (TypeSafeAPIConnectionError, TypeSafeAPIError) as error:
            return translate_error(error)
        return read_decision(response)

    return DecisionModel(name=name, decide=decide, perceives=perceives)


def build_async_decision_model(
    client: AsyncTypeSafeClient, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on TypeSafe's API, through the caller's async client."""
    name = format_model_name(model)

    async def decide(question: Question) -> DecisionOutcome:
        state = translate_state(question.state)
        if state is None:
            return Unavailable()
        try:
            response = await client.system_one(
                state=state,
                questions=build_questions(question),
                model=model,
            )
        except (TypeSafeAPIConnectionError, TypeSafeAPIError) as error:
            return translate_error(error)
        return read_decision(response)

    return AsyncDecisionModel(name=name, decide=decide, perceives=perceives)
