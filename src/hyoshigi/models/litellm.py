"""LiteLLM's chat completions as a model, through the caller's litellm.completion."""

from collections.abc import Awaitable, Callable

from hyoshigi.models.model import (
    AsyncGenerativeModel,
    GenerationOutcome,
    GenerativeModel,
    MediaTypes,
    Prompt,
    Truncated,
)


def build_model(
    completion: Callable[..., object],
    *,
    model: str,
    perceives: MediaTypes,
    top_log_probabilities: int,
) -> GenerativeModel:
    """A stub that declares the name build_model."""

    def generate(prompt: Prompt) -> GenerationOutcome:
        return Truncated(text="stub")

    # A count no test asks for, so that no test passes under the stub.
    return GenerativeModel(
        name="stub",
        generate=generate,
        top_log_probabilities=7,
    )


def build_async_model(
    acompletion: Callable[..., Awaitable[object]],
    *,
    model: str,
    perceives: MediaTypes,
    top_log_probabilities: int,
) -> AsyncGenerativeModel:
    """A stub that declares the name build_async_model."""

    async def generate(prompt: Prompt) -> GenerationOutcome:
        return Truncated(text="stub")

    # A count no test asks for, so that no test passes under the stub.
    return AsyncGenerativeModel(
        name="stub",
        generate=generate,
        top_log_probabilities=7,
    )
