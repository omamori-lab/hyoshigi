"""OpenRouter's Responses and Decisions APIs as a monitor's models, through the caller's client.

The generative model is the Responses API, https://openrouter.ai/docs/api_reference/responses/overview.
The decision model is the Decisions API, https://openrouter.ai/docs/guides/community/jev.
"""

import base64
import json
import mimetypes
from dataclasses import dataclass
from typing import Literal, TypedDict, assert_never

try:
    import httpx
    from openrouter import OpenRouter, components, errors
    from openrouter.types import UNSET
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "OpenRouter models need the extra: pip install 'hyoshigi[openrouter]'"
    ) from error

from hyoshigi.errors import ConfigurationError
from hyoshigi.models.model import (
    AsyncDecisionModel,
    AsyncGenerativeModel,
    Decision,
    DecisionModel,
    DecisionOutcome,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    PromptPart,
    Question,
    Refused,
    Truncated,
    Unavailable,
)
from hyoshigi.trajectory.json_data import JSON
from hyoshigi.trajectory.parts import MediaPart, TextPart

# The name of the one question a decision model is asked, and so of its answer.
QUESTION_NAME = "statement"

# https://openrouter.ai/docs/guides/community/multimodal-decisions: the Decisions API reads an
# image only as a base64 data URL of one of these types.
DECISION_IMAGES = MediaTypes("image/png", "image/jpeg", "image/webp")

# A fresh table of the standard library's own media types, so a filename's extension does not
# depend on the files of the platform.
MIME_TYPES = mimetypes.MimeTypes()

# https://openrouter.ai/docs/api_reference/errors-and-debugging: a timeout, a conflict, a
# request body too large and a rate limit mean no answer now.
UNAVAILABLE_STATUSES = frozenset({408, 409, 413, 429})

# https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes: the error
# types for input the provider rejects, which an agent's own trajectory can cause.
REJECTED_INPUT = frozenset(
    {
        "context_length_exceeded",
        "string_too_long",
        "payload_too_large",
        "invalid_image",
        "image_too_large",
        "image_too_small",
        "unsupported_image_format",
        "image_download_failed",
    }
)

FAILURES = (httpx.TransportError, errors.OpenRouterError)
type Failure = httpx.TransportError | errors.OpenRouterError


class ImageUrl(TypedDict):
    url: str


class ImageObject(TypedDict):
    """An image in a Decisions request's state, as in OpenRouter's multimodal Decisions guide."""

    type: Literal["image_url"]
    image_url: ImageUrl


def locate(part: MediaPart) -> str:
    """The part's URL, or its bytes as a base64 data URL; bytes need a concrete media type."""
    match part:
        case MediaPart(url=str(url)):
            return url
        case MediaPart(media_type=media_type, data=bytes() as data) if "*" not in media_type:
            return f"data:{media_type};base64,{base64.b64encode(data).decode()}"
        case _:
            raise ConfigurationError(f"bytes of {part.media_type} need a concrete media type")


def build_file(part: MediaPart) -> components.InputFileTypedDict:
    """A file by URL, or as a data URL with the filename the providers behind OpenRouter require."""
    if part.url is not None:
        return {"type": "input_file", "file_url": part.url}
    extension = MIME_TYPES.guess_extension(part.media_type) or ""
    return {"type": "input_file", "file_data": locate(part), "filename": f"attachment{extension}"}


def build_media_content(part: MediaPart) -> components.InputMessageItemContentUnionTypedDict:
    """An image as an input image, a document or text as an input file."""
    match part.media_type.partition("/")[0].lower():
        case "image":
            return {"type": "input_image", "detail": "auto", "image_url": locate(part)}
        case "application" | "text":
            return build_file(part)
        case _:
            raise ConfigurationError(
                f"OpenRouter's Responses API carries images and files, not {part.media_type}"
            )


def build_input(prompt: Prompt) -> components.InputsUnionTypedDict:
    """The prompt's content as the one user message of a Responses API request."""
    content: list[components.InputMessageItemContentUnionTypedDict] = []
    for part in prompt.content:
        match part:
            case TextPart(text=text):
                content.append({"type": "input_text", "text": text})
            case MediaPart():
                content.append(build_media_content(part))
            case _:
                assert_never(part)
    return [{"type": "message", "role": "user", "content": content}]


def build_state_item(part: PromptPart) -> str | ImageObject:
    """A question's state part as a Decisions API state item: a string or an image object."""
    match part:
        case TextPart(text=text):
            return text
        case MediaPart(media_type=media_type) if DECISION_IMAGES.covers(media_type):
            url = locate(part)
            if not url.startswith("data:"):
                raise ConfigurationError(f"OpenRouter's Decisions API does not fetch {media_type}")
            return {"type": "image_url", "image_url": {"url": url}}
        case MediaPart(media_type=media_type):
            raise ConfigurationError(
                f"OpenRouter's Decisions API reads png, jpeg and webp images, not {media_type}"
            )
        case _:
            assert_never(part)


def read_codes(payload: JSON) -> tuple[str, ...]:
    """The machine-readable codes of an error: its error code and its error type.

    The error's message is not read; it can echo the prompt.
    """
    match payload:
        case {"error": {"code": str(code)}, "error_type": str(error_type)}:
            return (code, error_type)
        case {"error": {"code": str(code)}}:
            return (code,)
        case {"error_type": str(error_type)}:
            return (error_type,)
        case _:
            return ()


def read_body_codes(body: str) -> tuple[str, ...]:
    """The machine-readable codes of an HTTP error body; a body that is not JSON has none."""
    try:
        payload = json.loads(body)
    except ValueError:
        return ()
    return read_codes(payload)


def names_refusal(code: str) -> bool:
    """Whether an error code says a policy or the model's refusal declined the request."""
    return "policy" in code or code == "refusal"


def is_temporary_failure(status: int | None) -> bool:
    """Whether a failure's HTTP status, if it has one, says to ask again later."""
    return status is None or status >= 500 or status in UNAVAILABLE_STATUSES


def read_failure(status: int | None, *, codes: tuple[str, ...]) -> Refused | Unavailable:
    """The outcome of a failure with this HTTP status, if any, and these error codes.

    A refusal is refused. No answer now, or input the provider rejects, is unavailable. Any
    other rejection raises, naming the status and nothing the provider said.
    """
    if any(map(names_refusal, codes)):
        return Refused()
    if is_temporary_failure(status) or not REJECTED_INPUT.isdisjoint(codes):
        return Unavailable(status=status)
    raise ConfigurationError(f"OpenRouter answered HTTP {status}") from None


def read_error(error: Failure) -> Refused | Unavailable:
    """The outcome of an exception the SDK raised: a transport failure or an HTTP error.

    A reply the SDK could not read has an HTTP status below 400 and no status here.
    """
    match error:
        case httpx.TransportError():
            return Unavailable()
        case errors.OpenRouterError(status_code=status, body=body):
            return read_failure(status if status >= 400 else None, codes=read_body_codes(body))
        case _:
            assert_never(error)


@dataclass(frozen=True, slots=True, kw_only=True)
class Written:
    """What a reply's messages hold: the text, the model's refusal if it gave one, and whether
    every part was one the SDK could classify.
    """

    text: str
    refusal: str | None
    classified: bool


def list_parts(reply: components.OpenResponsesResult) -> list[components.OutputMessageItemContent]:
    """The content parts of the reply's messages, in order; its other items are skipped."""
    return [
        part
        for item in reply.output
        if isinstance(item, components.OutputMessageItem)
        for part in item.content
    ]


def read_written(reply: components.OpenResponsesResult) -> Written:
    """What the reply's messages hold."""
    parts = list_parts(reply)
    texts = [part.text for part in parts if isinstance(part, components.ResponseOutputText)]
    refusals = [
        part.refusal for part in parts if isinstance(part, components.OpenAIResponsesRefusalContent)
    ]
    unknown = [
        part for part in parts if isinstance(part, components.UnknownOutputMessageItemContent)
    ]
    return Written(
        text="".join(texts), refusal=refusals[0] if refusals else None, classified=not unknown
    )


def read_completed(written: Written) -> GenerationOutcome:
    """A completed reply is the refusal or the text, or Truncated if a part was unclassified."""
    if written.refusal is not None:
        return Refused(explanation=written.refusal)
    if written.classified:
        return Generation(text=written.text)
    return Truncated(text=written.text)


def read_reply(reply: components.OpenResponsesResult) -> GenerationOutcome:
    """The outcome of a Responses API reply, by its status.

    A failed reply is read by its error codes; any status other than completed, incomplete
    or failed is Truncated, as is an incomplete reply that was not cut by a content filter.
    """
    written = read_written(reply)
    match reply.status, reply.incomplete_details:
        case "completed", _:
            return read_completed(written)
        case "incomplete", components.IncompleteDetails(reason="content_filter"):
            return Refused()
        case "failed", _:
            return read_failure(None, codes=read_codes(reply.model_dump()))
        case _:
            return Truncated(text=written.text)


def read_decision(reply: components.DecisionsResponse) -> DecisionOutcome:
    """The probability a Decisions API reply gives the question; any other answer is unavailable."""
    match reply.answers.get(QUESTION_NAME):
        case components.DecisionsNoulAnswer(noul=probability) if 0 <= probability <= 1:
            return Decision(probability=probability)
        case _:
            return Unavailable()


def build_model(
    client: OpenRouter, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> GenerativeModel:
    """A model on the Responses API, through the caller's client.

    The API is stateless and the SDK sends store false, so no stored response holds the
    monitored agent's content.
    """
    limit = UNSET if max_output_tokens is None else max_output_tokens

    def generate(prompt: Prompt) -> GenerationOutcome:
        try:
            reply = client.responses.send(
                model=model,
                instructions=prompt.system,
                input=build_input(prompt),
                max_output_tokens=limit,
            )
        except FAILURES as error:
            return read_error(error)
        return read_reply(reply)

    return GenerativeModel(name=f"openrouter:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: OpenRouter, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's client, awaited."""
    limit = UNSET if max_output_tokens is None else max_output_tokens

    async def generate(prompt: Prompt) -> GenerationOutcome:
        try:
            reply = await client.responses.send_async(
                model=model,
                instructions=prompt.system,
                input=build_input(prompt),
                max_output_tokens=limit,
            )
        except FAILURES as error:
            return read_error(error)
        return read_reply(reply)

    return AsyncGenerativeModel(name=f"openrouter:{model}", generate=generate, perceives=perceives)


def build_decision_model(client: OpenRouter, *, model: str, perceives: MediaTypes) -> DecisionModel:
    """A decision model on the Decisions API, through the caller's client.

    The question is one noul question, whose instructions are the statement; the probability
    of its answer is the decision's.
    """

    def decide(question: Question) -> DecisionOutcome:
        state = [build_state_item(part) for part in question.state]
        try:
            reply = client.alpha.decisions.create(
                model=model,
                state=state,
                questions={QUESTION_NAME: {"type": "noul", "instructions": question.statement}},
            )
        except FAILURES as error:
            return read_error(error)
        return read_decision(reply)

    return DecisionModel(name=f"openrouter:{model}", decide=decide, perceives=perceives)


def build_async_decision_model(
    client: OpenRouter, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on the Decisions API, through the caller's client, awaited."""

    async def decide(question: Question) -> DecisionOutcome:
        state = [build_state_item(part) for part in question.state]
        try:
            reply = await client.alpha.decisions.create_async(
                model=model,
                state=state,
                questions={QUESTION_NAME: {"type": "noul", "instructions": question.statement}},
            )
        except FAILURES as error:
            return read_error(error)
        return read_decision(reply)

    return AsyncDecisionModel(name=f"openrouter:{model}", decide=decide, perceives=perceives)
