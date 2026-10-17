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
    check_name,
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

# The codes of input the provider rejects, which an agent's own trajectory can cause: the error
# types of https://openrouter.ai/docs/api_reference/errors-and-debugging#typed-error-codes, and
# the codes of an unusable image in the schema OpenResponsesErrorField of
# https://openrouter.ai/docs/api/api-reference/responses/create-a-response
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
        "image_not_found",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_parse_error",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    }
)

# Every failure the SDK raises: the HTTP library's request errors (connection, timeout, decoding),
# the SDK's errors for a status or an unreadable reply, and its error for no response.
FAILURES = (httpx.RequestError, errors.OpenRouterError, errors.NoResponseError)
type Failure = httpx.RequestError | errors.OpenRouterError | errors.NoResponseError


class ImageUrl(TypedDict):
    url: str


class ImageObject(TypedDict):
    """An image in a Decisions request's state, as in OpenRouter's multimodal Decisions guide."""

    type: Literal["image_url"]
    image_url: ImageUrl


@dataclass(frozen=True, slots=True, kw_only=True)
class Rejection:
    """A request the provider rejected as misuse, with the HTTP status."""

    status: int


def collect[Item](items: list[Item | None]) -> list[Item] | None:
    """The items, or None if any is None; every item is built before this decides."""
    kept = [item for item in items if item is not None]
    return kept if len(kept) == len(items) else None


def encode(part: MediaPart) -> str | None:
    """The part's bytes as a base64 data URL; None if it has none or its type is a range."""
    if part.data is None or "*" in part.media_type:
        return None
    return f"data:{part.media_type};base64,{base64.b64encode(part.data).decode()}"


def locate(part: MediaPart) -> str | None:
    """The part's URL, or its bytes as a data URL; None for the bytes of a media range."""
    return part.url if part.url is not None else encode(part)


def build_file(part: MediaPart) -> components.InputFileTypedDict | None:
    """A file by URL, or as a data URL with the filename the providers behind OpenRouter require."""
    if part.url is not None:
        return {"type": "input_file", "file_url": part.url}
    data_url = encode(part)
    if data_url is None:
        return None
    extension = MIME_TYPES.guess_extension(part.media_type) or ""
    return {"type": "input_file", "file_data": data_url, "filename": f"attachment{extension}"}


def build_media_content(
    part: MediaPart,
) -> components.InputMessageItemContentUnionTypedDict | None:
    """An image as an input image, a document or text as an input file; None if the part is
    in a form the API cannot take.
    """
    match part.media_type.partition("/")[0].lower():
        case "image":
            image_url = locate(part)
            return (
                None
                if image_url is None
                else {"type": "input_image", "detail": "auto", "image_url": image_url}
            )
        case "application" | "text":
            return build_file(part)
        case _:
            raise ConfigurationError(
                f"OpenRouter's Responses API carries images and files, not {part.media_type}"
            )


def build_content(part: PromptPart) -> components.InputMessageItemContentUnionTypedDict | None:
    """A prompt part as a Responses API input part, or None if the API cannot take its form."""
    match part:
        case TextPart(text=text):
            return {"type": "input_text", "text": text}
        case MediaPart():
            return build_media_content(part)
        case _:
            assert_never(part)


def build_input(prompt: Prompt) -> components.InputsUnionTypedDict | None:
    """The prompt's content as the one user message of a Responses API request; None if a part
    is in a form the API cannot take, after every part has been checked for a type it cannot.
    """
    content = collect([build_content(part) for part in prompt.content])
    return None if content is None else [{"type": "message", "role": "user", "content": content}]


def build_image(part: MediaPart) -> ImageObject | None:
    """An image as an image object; None unless it is bytes of a concrete type, since the
    Decisions API reads data URLs and does not fetch a URL, even one that is a data URL.
    """
    data_url = encode(part)
    return None if data_url is None else {"type": "image_url", "image_url": {"url": data_url}}


def build_state_item(part: PromptPart) -> str | ImageObject | None:
    """A question's state part as a Decisions API state item: a string or an image object;
    None if the part is an image the API can read but not in this form.
    """
    match part:
        case TextPart(text=text):
            return text
        case MediaPart(media_type=media_type) if (
            DECISION_IMAGES.covers(media_type) or media_type.lower() == "image/*"
        ):
            return build_image(part)
        case MediaPart(media_type=media_type):
            raise ConfigurationError(
                f"OpenRouter's Decisions API reads png, jpeg and webp images, not {media_type}"
            )
        case _:
            assert_never(part)


def read_codes(payload: JSON) -> tuple[str, ...]:
    """The machine-readable codes of an error: its error code and its error type.

    The error type is where https://openrouter.ai/docs/api_reference/errors-and-debugging puts
    it: in the error's metadata, in the error itself (the Anthropic skin) and at the top level
    (a failed Responses reply). The error's message is not read; it can echo the prompt.
    """
    codes: list[str] = []
    match payload:
        case {"error": {"code": str(code)}}:
            codes.append(code)
    match payload:
        case {"error": {"error_type": str(error_type)}}:
            codes.append(error_type)
    match payload:
        case {"error": {"metadata": {"error_type": str(error_type)}}}:
            codes.append(error_type)
    match payload:
        case {"error_type": str(error_type)}:
            codes.append(error_type)
    return tuple(codes)


def read_body_codes(body: str) -> tuple[str, ...]:
    """The machine-readable codes of an HTTP error body; a body that is not JSON has none."""
    try:
        payload = json.loads(body)
    except ValueError:
        return ()
    return read_codes(payload)


def names_refusal(codes: tuple[str, ...]) -> bool:
    """Whether an error code says a policy or the model's refusal declined the request."""
    return any("policy" in code or code == "refusal" for code in codes)


def is_temporary_failure(status: int) -> bool:
    """Whether a status says no answer now: below 400 (no error), above 499, or to ask later."""
    return status < 400 or status >= 500 or status in UNAVAILABLE_STATUSES


def read_failure(status: int, *, codes: tuple[str, ...]) -> Refused | Unavailable | Rejection:
    """The outcome of an HTTP failure with this status and these error codes.

    A refusal is refused. No answer now, or input the provider rejects, is unavailable. Any
    other rejection is misuse, to be raised by the caller outside the handling of the failure.
    """
    if names_refusal(codes):
        return Refused()
    if is_temporary_failure(status) or not REJECTED_INPUT.isdisjoint(codes):
        return Unavailable(status=status)
    return Rejection(status=status)


def read_error(error: Failure) -> Refused | Unavailable | Rejection:
    """The outcome of an exception the SDK raised.

    A success status with a reply the SDK could not read has no status to report.
    """
    match error:
        case httpx.RequestError() | errors.NoResponseError():
            return Unavailable()
        case errors.OpenRouterError(status_code=status) if 200 <= status < 300:
            return Unavailable()
        case errors.OpenRouterError(status_code=status, body=body):
            return read_failure(status, codes=read_body_codes(body))
        case _:
            assert_never(error)


def settle[Outcome](outcome: Outcome | Rejection) -> Outcome:
    """The outcome, or ConfigurationError naming the status of a rejection.

    It is called after the handling of the failure ends, so that no exception of the
    provider, whose message can echo the prompt, is in the raised error's chain.
    """
    if isinstance(outcome, Rejection):
        raise ConfigurationError(f"OpenRouter answered HTTP {outcome.status}")
    return outcome


@dataclass(frozen=True, slots=True, kw_only=True)
class Written:
    """What a reply's messages hold: the text, the model's refusal if it gave one, whether there
    was a message, and whether every part was one the SDK could classify.
    """

    text: str
    refusal: str | None
    has_message: bool
    classified: bool


def list_parts(result: components.OpenResponsesResult) -> list[components.OutputMessageItemContent]:
    """The content parts of the result's messages, in order; its other items are skipped."""
    return [
        part
        for item in result.output
        if isinstance(item, components.OutputMessageItem)
        for part in item.content
    ]


def is_cracked(unknown: object) -> bool:
    """Whether something the SDK built as unknown names a type this module reads: a message, a
    text or a refusal. It is a documented type whose fields do not match, not an unlisted type.
    """
    match unknown:
        case components.UnknownOutputItems(raw={"type": "message"}):
            return True
        case components.UnknownOutputMessageItemContent(raw={"type": "output_text" | "refusal"}):
            return True
    return False


def read_written(result: components.OpenResponsesResult) -> Written | None:
    """What the result's messages hold; None if an item or a part is cracked."""
    parts = list_parts(result)
    if any(map(is_cracked, [*result.output, *parts])):
        return None
    texts = [part.text for part in parts if isinstance(part, components.ResponseOutputText)]
    refusals = [
        part.refusal for part in parts if isinstance(part, components.OpenAIResponsesRefusalContent)
    ]
    unknown = [
        part for part in parts if isinstance(part, components.UnknownOutputMessageItemContent)
    ]
    return Written(
        text="".join(texts),
        refusal=refusals[0] if refusals else None,
        has_message=any(isinstance(item, components.OutputMessageItem) for item in result.output),
        classified=not unknown,
    )


def read_completed(written: Written) -> GenerationOutcome:
    """A completed reply is the refusal or the text; Truncated if a part was unclassified, and
    unavailable if it holds no message.
    """
    if not written.has_message:
        return Unavailable()
    if written.refusal is not None:
        return Refused(explanation=written.refusal)
    if written.classified:
        return Generation(text=written.text)
    return Truncated(text=written.text)


def read_result(result: components.OpenResponsesResult) -> GenerationOutcome:
    """The outcome of a Responses API result, by its status.

    A failed reply is read by its error codes; any status other than completed, incomplete
    or failed is Truncated, as is an incomplete reply that was not cut by a content filter.
    """
    written = read_written(result)
    if written is None:
        return Unavailable()
    match result.status, result.incomplete_details:
        case "completed", _:
            return read_completed(written)
        case "incomplete", components.IncompleteDetails(reason="content_filter"):
            return Refused()
        case "failed", _:
            return Refused() if names_refusal(read_codes(result.model_dump())) else Unavailable()
        case _:
            return Truncated(text=written.text)


def read_reply(reply: object) -> GenerationOutcome:
    """The outcome of what the SDK returned: a result, or an event stream, which is not one."""
    match reply:
        case components.OpenResponsesResult():
            return read_result(reply)
        case _:
            return Unavailable()


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
    check_name(model)
    limit = UNSET if max_output_tokens is None else max_output_tokens

    def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_input(prompt)
        if request is None:
            return Unavailable()
        try:
            reply = client.responses.send(
                model=model, instructions=prompt.system, input=request, max_output_tokens=limit
            )
        except FAILURES as error:
            outcome = read_error(error)
        else:
            outcome = read_reply(reply)
        return settle(outcome)

    return GenerativeModel(name=f"openrouter:{model}", generate=generate, perceives=perceives)


def build_async_model(
    client: OpenRouter, *, model: str, max_output_tokens: int | None, perceives: MediaTypes
) -> AsyncGenerativeModel:
    """A model on the Responses API, through the caller's client, awaited."""
    check_name(model)
    limit = UNSET if max_output_tokens is None else max_output_tokens

    async def generate(prompt: Prompt) -> GenerationOutcome:
        request = build_input(prompt)
        if request is None:
            return Unavailable()
        try:
            reply = await client.responses.send_async(
                model=model, instructions=prompt.system, input=request, max_output_tokens=limit
            )
        except FAILURES as error:
            outcome = read_error(error)
        else:
            outcome = read_reply(reply)
        return settle(outcome)

    return AsyncGenerativeModel(name=f"openrouter:{model}", generate=generate, perceives=perceives)


def build_decision_model(client: OpenRouter, *, model: str, perceives: MediaTypes) -> DecisionModel:
    """A decision model on the Decisions API, through the caller's client.

    The question is one noul question, whose instructions are the statement; the probability
    of its answer is the decision's.
    """
    check_name(model)

    def decide(question: Question) -> DecisionOutcome:
        state = collect([build_state_item(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            reply = client.alpha.decisions.create(
                model=model,
                state=state,
                questions={QUESTION_NAME: {"type": "noul", "instructions": question.statement}},
            )
        except FAILURES as error:
            outcome = read_error(error)
        else:
            outcome = read_decision(reply)
        return settle(outcome)

    return DecisionModel(name=f"openrouter:{model}", decide=decide, perceives=perceives)


def build_async_decision_model(
    client: OpenRouter, *, model: str, perceives: MediaTypes
) -> AsyncDecisionModel:
    """A decision model on the Decisions API, through the caller's client, awaited."""
    check_name(model)

    async def decide(question: Question) -> DecisionOutcome:
        state = collect([build_state_item(part) for part in question.state])
        if state is None:
            return Unavailable()
        try:
            reply = await client.alpha.decisions.create_async(
                model=model,
                state=state,
                questions={QUESTION_NAME: {"type": "noul", "instructions": question.statement}},
            )
        except FAILURES as error:
            outcome = read_error(error)
        else:
            outcome = read_decision(reply)
        return settle(outcome)

    return AsyncDecisionModel(name=f"openrouter:{model}", decide=decide, perceives=perceives)
