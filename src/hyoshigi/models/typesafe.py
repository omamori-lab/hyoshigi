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
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
        TypeSafeClient,
        TypeSafeError,
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
    collect,
)
from hyoshigi.trajectory.parts import MediaPart, TextPart

type StateItem = str | Image

QUESTION_KEY = "statement"
IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/*"})
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
# https://docs.typesafe.ai/images.md, "Turning on images".
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
# https://docs.typesafe.ai/api.md, "Errors": 415 is an unsupported Content-Encoding or
# multiple compression layers (client configuration), so only transient or oversized 4xx
# statuses map to Unavailable.
UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 429})


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


def build_headers(state: list[StateItem]) -> dict[str, str] | None:
    """Return the opt-in header required by TypeSafe when the state contains an image."""
    if any(isinstance(item, Image) for item in state):
        return IMAGE_HEADERS
    return None


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


def is_unavailable_error(error: TypeSafeAPIError) -> bool:
    """Whether an HTTP failure from TypeSafe is a transient or content-length outcome."""
    match error.body:
        case {"detail": {"error_type": "max_tokens_exceeded"}}:
            return True
        case _:
            return (
                isinstance(error, TypeSafeInternalServerError)
                or error.status < 400
                or error.status in UNAVAILABLE_STATUSES
            )


def translate_error(error: TypeSafeError) -> Unavailable:
    """Translate a TypeSafe SDK error to Unavailable, or raise ConfigurationError on 4xx."""
    if not isinstance(error, TypeSafeAPIError) or isinstance(
        error, TypeSafeAPIResponseValidationError
    ):
        return Unavailable()
    if is_unavailable_error(error):
        return Unavailable(status=error.status)
    raise ConfigurationError(f"TypeSafe request failed with HTTP {error.status}")


def build_decision_model(
    client: TypeSafeClient, *, model: str, perceives: MediaTypes
) -> DecisionModel:
    """A decision model on TypeSafe's API, through the caller's client.

    Declaring image media types in ``perceives`` opts in to sending
    ``TypeSafe-Danger-Zone-Images: true`` on requests that contain an image.
    """
    name = format_model_name(model)

    def decide(question: Question) -> DecisionOutcome:
        state = collect([translate_part(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            response = client.system_one(
                state=state,
                questions=build_questions(question),
                model=model,
                extra_headers=build_headers(state),
            )
        except TypeSafeError as caught:
            error = caught
        else:
            return read_decision(response)
        return translate_error(error)

    return DecisionModel(name=name, decide=decide, perceives=perceives)


def build_async_decision_model(
    client: AsyncTypeSafeClient, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on TypeSafe's API, through the caller's async client.

    Declaring image media types in ``perceives`` opts in to sending
    ``TypeSafe-Danger-Zone-Images: true`` on requests that contain an image.
    """
    name = format_model_name(model)

    async def decide(question: Question) -> DecisionOutcome:
        state = collect([translate_part(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            response = await client.system_one(
                state=state,
                questions=build_questions(question),
                model=model,
                extra_headers=build_headers(state),
            )
        except TypeSafeError as caught:
            error = caught
        else:
            return read_decision(response)
        return translate_error(error)

    return AsyncDecisionModel(name=name, decide=decide, perceives=perceives)
