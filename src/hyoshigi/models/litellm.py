"""LiteLLM's chat completions as a model, through the caller's litellm.completion or
litellm.acompletion.

The module imports neither LiteLLM nor the OpenAI SDK's models. LiteLLM's exceptions subclass the
SDK's, which the module reads by class, and the request and the reply are those of
hyoshigi.models.chat_completions.
"""

import inspect
import sys
from collections.abc import Awaitable, Callable

from openai import APIConnectionError, APIError, APIResponseValidationError

from hyoshigi.errors import ConfigurationError
from hyoshigi.models import chat_completions
from hyoshigi.models.model import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Refused,
    Rejected,
    Unavailable,
    check_name,
    check_perceives,
)
from hyoshigi.models.openai_errors import read_status
from hyoshigi.options import describe_type, read_count

# LiteLLM defines its errors in the module litellm.exceptions, which is loaded once LiteLLM is
# imported. Most of them subclass the OpenAI SDK's errors. GuardrailRaisedException,
# BlockedPiiEntityError and BudgetExceededError subclass Exception. The classes are found in that
# module, and never by their names alone.
_EXCEPTIONS_MODULE = "litellm.exceptions"
# Errors whose content a policy declined: a content policy violation, a guardrail's block and a
# blocked entity of personal information.
_REFUSALS = (
    "ContentPolicyViolationError",
    "GuardrailRaisedException",
    "BlockedPiiEntityError",
)
# Errors for input the provider cannot use, which LiteLLM raises with the status 400 and no code.
_UNUSABLE_INPUTS = (
    "ContextWindowExceededError",
    "ImageFetchError",
)
# The error of a spent budget, which read_error makes unavailable as it does any error without a
# status.
_BUDGETS = ("BudgetExceededError",)


def find_classes(names: tuple[str, ...]) -> tuple[type[Exception], ...]:
    """The classes of these names in LiteLLM's exceptions module; none while LiteLLM is not
    imported, since an instance of its classes can exist only after it is.
    """
    module = sys.modules.get(_EXCEPTIONS_MODULE)
    found = [
        getattr(
            module,
            name,
            None,
        )
        for name in names
    ]
    return tuple(kind for kind in found if kind is not None)


def find_failures() -> tuple[type[Exception], ...]:
    """What a call to the function raises when it gets no usable reply: the SDK's errors, which
    LiteLLM's mostly subclass, the error of encoding text that has no UTF-8 form and the
    LiteLLM classes that are not the SDK's.
    """
    return (
        APIError,
        UnicodeError,
        *find_classes(_REFUSALS + _UNUSABLE_INPUTS + _BUDGETS),
    )


def read_failure(error: Exception) -> Refused | Rejected | Unavailable:
    """The outcome of a call that failed.

    An error of a LiteLLM class for a content policy, a guardrail or a blocked entity is a
    refusal; one for a context window or an image it could not fetch is rejected with 400. Any
    other error is read by read_error, which makes one for a spent budget unavailable.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    if isinstance(
        error,
        find_classes(_REFUSALS),
    ):
        return Refused()
    if isinstance(
        error,
        find_classes(_UNUSABLE_INPUTS),
    ):
        return Rejected(status=400)
    return read_error(error)


def read_error(error: Exception) -> Refused | Rejected | Unavailable:
    """The outcome of an error of the SDK's classes, or of encoding text.

    A connection error, a timeout and an invalid reply are unavailable with no status, whatever
    status LiteLLM gives them. An error with a status is read as an OpenAI status is, which raises
    ConfigurationError naming LiteLLM and only the status for a client error that blames the
    request. Any other error is unavailable.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    match error:
        case APIConnectionError() | APIResponseValidationError():
            return Unavailable()
        case APIError(
            status_code=int() as status,
            code=code,
        ):
            return read_status(
                status,
                code=code,
                provider="LiteLLM",
            )
        case _:
            return Unavailable()


def check_completion(completion: object) -> None:
    """Raise ConfigurationError unless the function is callable and is no coroutine function,
    pointing a coroutine function to the constructor that takes it.
    """
    if not callable(completion):
        raise ConfigurationError(f"completion must be callable, got {describe_type(completion)}")
    if inspect.iscoroutinefunction(completion):
        raise ConfigurationError(
            "completion must not be a coroutine function. "
            "Use AsyncGenerativeModel.from_litellm for litellm.acompletion.",
        )


def check_acompletion(acompletion: object) -> None:
    """Raise ConfigurationError unless the function is a coroutine function, pointing any other
    callable to the constructor that takes it.
    """
    if not callable(acompletion):
        raise ConfigurationError(f"acompletion must be callable, got {describe_type(acompletion)}")
    if not inspect.iscoroutinefunction(acompletion):
        raise ConfigurationError(
            "acompletion must be a coroutine function. "
            "Use GenerativeModel.from_litellm for litellm.completion.",
        )


def build_model(
    completion: Callable[..., object],
    *,
    model: str,
    perceives: MediaTypes,
    top_log_probabilities: int,
) -> GenerativeModel:
    """A model on LiteLLM's chat completions, through the caller's litellm.completion.

    One call to the function makes one request. LiteLLM keeps the key, the retries and the
    timeout, which the caller binds to the function with functools.partial. Raise
    ConfigurationError if the function is not callable or is a coroutine function, the model id
    is empty, perceives names media the API cannot carry, or top_log_probabilities is not a whole
    number from 0.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    check_completion(completion)
    check_name(model)
    check_perceives(
        perceives,
        carries=chat_completions.CARRIED_MEDIA,
    )
    # The count as an int, which the model keeps and the request sends.
    count = read_count(
        top_log_probabilities,
        name="top_log_probabilities",
        minimum=0,
    )

    def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = chat_completions.build_arguments(
            prompt,
            model=model,
            top_log_probabilities=count,
        )
        if arguments is None:
            return Unavailable()
        try:
            reply = completion(**arguments)
        except find_failures() as caught:
            error = caught
        else:
            return chat_completions.read_reply(
                reply,
                top_log_probabilities=count,
            )
        return read_failure(error)

    return GenerativeModel(
        name=f"litellm:{model}",
        generate=generate,
        perceives=perceives,
        top_log_probabilities=count,
    )


def build_async_model(
    acompletion: Callable[..., Awaitable[object]],
    *,
    model: str,
    perceives: MediaTypes,
    top_log_probabilities: int,
) -> AsyncGenerativeModel:
    """A model on LiteLLM's chat completions, through the caller's litellm.acompletion.

    One call to the function makes one request. Raise ConfigurationError if the function is not
    a coroutine function, the model id is empty, perceives names media the API cannot carry, or
    top_log_probabilities is not a whole number from 0.

    <https://omamori-lab.github.io/hyoshigi/reference/providers/>
    """
    check_acompletion(acompletion)
    check_name(model)
    check_perceives(
        perceives,
        carries=chat_completions.CARRIED_MEDIA,
    )
    # The count as an int, which the model keeps and the request sends.
    count = read_count(
        top_log_probabilities,
        name="top_log_probabilities",
        minimum=0,
    )

    async def generate(prompt: Prompt) -> GenerationOutcome:
        arguments = chat_completions.build_arguments(
            prompt,
            model=model,
            top_log_probabilities=count,
        )
        if arguments is None:
            return Unavailable()
        try:
            reply = await acompletion(**arguments)
        except find_failures() as caught:
            error = caught
        else:
            return chat_completions.read_reply(
                reply,
                top_log_probabilities=count,
            )
        return read_failure(error)

    return AsyncGenerativeModel(
        name=f"litellm:{model}",
        generate=generate,
        perceives=perceives,
        top_log_probabilities=count,
    )
