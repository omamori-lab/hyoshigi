"""The models a monitor asks, each around the caller's function.

A generative model writes a reply for an LLM monitor to parse; a decision model gives
the probability that a statement about the state holds. A model carries a name, since a
score records which model rated it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import read_threshold
from hyoshigi.trajectory.parts import MEDIA_TYPE, MediaPart, TextPart

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import ThinkingConfigParam
    from openai import AsyncOpenAI, OpenAI
    from openai.types import ReasoningEffort
    from openrouter import OpenRouter, components
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

type PromptPart = TextPart | MediaPart


@dataclass(frozen=True, slots=True, init=False)
class MediaTypes:
    """The media a model perceives, as IANA media types or ranges such as image/*."""

    patterns: frozenset[str]

    def __init__(self, *patterns: str) -> None:
        if any(MEDIA_TYPE.fullmatch(pattern) is None for pattern in patterns):
            raise ConfigurationError("each pattern must be an IANA media type, such as image/*")
        object.__setattr__(self, "patterns", frozenset(map(str.lower, patterns)))

    def covers(self, media_type: str) -> bool:
        """Whether these patterns cover media of this type; a range only through its range."""
        media_type = media_type.lower()
        top_level = media_type.partition("/")[0]
        return media_type in self.patterns or f"{top_level}/*" in self.patterns


# A model that perceives no media: media reaches it only as a description or a placeholder.
TEXT_ONLY = MediaTypes()


def check_parts(parts: tuple[PromptPart, ...]) -> None:
    """Raise ConfigurationError unless every part is text or media."""
    if not all(isinstance(part, TextPart | MediaPart) for part in parts):
        raise ConfigurationError("a model is given text and media only")


def collect[Item](items: list[Item | None]) -> list[Item] | None:
    """The items, or None if any is None; every item is built before this decides."""
    kept = [item for item in items if item is not None]
    return kept if len(kept) == len(items) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class Prompt:
    """What a generative model is asked: the monitor's instructions, and the content."""

    system: str
    content: tuple[PromptPart, ...]

    def __post_init__(self) -> None:
        check_parts(self.content)


@dataclass(frozen=True, slots=True, kw_only=True)
class Question:
    """What a decision model is asked: whether the statement holds of the state."""

    statement: str
    state: tuple[PromptPart, ...]

    def __post_init__(self) -> None:
        check_parts(self.state)


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """A complete reply."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Truncated:
    """A reply cut off at the model's output limit, with the text written so far."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Refused:
    """The model or its provider declined, with the provider's explanation if it gave one."""

    explanation: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Unavailable:
    """No answer after the caller's retries, with the HTTP status if there was one."""

    status: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """The probability that the statement holds, and the model's confidence if it gave one.

    Each is read as a threshold is, as a float from 0 to 1, so that a stored trajectory
    writes it as a JSON number; anything else raises ConfigurationError.
    """

    probability: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the floats are set past its guard.
        object.__setattr__(
            self, "probability", read_threshold(self.probability, name="probability")
        )
        if self.confidence is not None:
            confidence = read_threshold(self.confidence, name="confidence")
            object.__setattr__(self, "confidence", confidence)


type GenerationOutcome = Generation | Truncated | Refused | Unavailable
type DecisionOutcome = Decision | Refused | Unavailable


def check_name(name: str) -> None:
    """Raise ConfigurationError if a model's name is empty."""
    if not name:
        raise ConfigurationError("a model needs a name, such as the provider's model id")


def check_perceives(perceives: MediaTypes, *, carries: MediaTypes) -> None:
    """Raise ConfigurationError naming the first pattern, in sorted order, that the media an
    API carries do not cover; a range such as image/* is covered only by a carried range.
    """
    for pattern in sorted(perceives.patterns):
        if not carries.covers(pattern):
            raise ConfigurationError(f"the API cannot carry {pattern} media")


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerativeModel:
    """A model that writes a reply to a prompt, through the caller's function.

    The function may be called more than once per rating, and returns an outcome rather
    than raising for a failure it expects.
    """

    name: str
    generate: Callable[[Prompt], GenerationOutcome]
    perceives: MediaTypes = TEXT_ONLY

    def __post_init__(self) -> None:
        check_name(self.name)

    @classmethod
    def from_openai(
        cls,
        client: OpenAI,
        *,
        model: str,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> GenerativeModel:
        """A model on OpenAI's Responses API, through the caller's client.

        A reasoning effort, when given, is sent as the request's reasoning effort; without
        one, OpenAI's default for the model applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_model(
            client,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            perceives=perceives,
        )

    @classmethod
    def from_anthropic(
        cls,
        client: Anthropic,
        *,
        model: str,
        max_tokens: int,
        thinking: ThinkingConfigParam | None = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> GenerativeModel:
        """A model on Anthropic's Messages API, through the caller's client.

        A thinking configuration, when given, is sent as the request's thinking; without
        one, the model's default applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[anthropic].
        from hyoshigi.models import anthropic as provider

        return provider.build_model(
            client, model=model, max_tokens=max_tokens, thinking=thinking, perceives=perceives
        )

    @classmethod
    def from_openrouter(
        cls,
        client: OpenRouter,
        *,
        model: str,
        max_output_tokens: int | None = None,
        reasoning_effort: components.ReasoningEffort | None = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> GenerativeModel:
        """A model on OpenRouter's Responses API, through the caller's client.

        A reasoning effort, when given, is sent as the request's reasoning effort; without
        one, the model's default applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_model(
            client,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            perceives=perceives,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncGenerativeModel:
    """A model that writes a reply to a prompt, through the caller's async function.

    The function may be called more than once per rating, concurrently, and returns an
    outcome rather than raising for a failure it expects.
    """

    name: str
    generate: Callable[[Prompt], Awaitable[GenerationOutcome]]
    perceives: MediaTypes = TEXT_ONLY

    def __post_init__(self) -> None:
        check_name(self.name)

    @classmethod
    def from_openai(
        cls,
        client: AsyncOpenAI,
        *,
        model: str,
        max_output_tokens: int | None = None,
        reasoning_effort: ReasoningEffort = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncGenerativeModel:
        """A model on OpenAI's Responses API, through the caller's async client.

        A reasoning effort, when given, is sent as the request's reasoning effort; without
        one, OpenAI's default for the model applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_async_model(
            client,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            perceives=perceives,
        )

    @classmethod
    def from_anthropic(
        cls,
        client: AsyncAnthropic,
        *,
        model: str,
        max_tokens: int,
        thinking: ThinkingConfigParam | None = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncGenerativeModel:
        """A model on Anthropic's Messages API, through the caller's async client.

        A thinking configuration, when given, is sent as the request's thinking; without
        one, the model's default applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[anthropic].
        from hyoshigi.models import anthropic as provider

        return provider.build_async_model(
            client, model=model, max_tokens=max_tokens, thinking=thinking, perceives=perceives
        )

    @classmethod
    def from_openrouter(
        cls,
        client: OpenRouter,
        *,
        model: str,
        max_output_tokens: int | None = None,
        reasoning_effort: components.ReasoningEffort | None = None,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncGenerativeModel:
        """A model on OpenRouter's Responses API, through the caller's client, awaited.

        A reasoning effort, when given, is sent as the request's reasoning effort; without
        one, the model's default applies.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_async_model(
            client,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            perceives=perceives,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionModel:
    """A model that gives the probability a statement holds, through the caller's function.

    The function may be called more than once per rating, and returns an outcome rather
    than raising for a failure it expects.
    """

    name: str
    decide: Callable[[Question], DecisionOutcome]
    perceives: MediaTypes = TEXT_ONLY

    def __post_init__(self) -> None:
        check_name(self.name)

    @classmethod
    def from_openai(
        cls, client: OpenAI, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> DecisionModel:
        """A decision model on OpenAI's Decisions API, such as gpt-6-luna, through the caller's
        client.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_decision_model(client, model=model, perceives=perceives)

    @classmethod
    def from_openrouter(
        cls, client: OpenRouter, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> DecisionModel:
        """A decision model on OpenRouter's Decisions API, such as typesafe/jev-1.14."""
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_decision_model(client, model=model, perceives=perceives)

    @classmethod
    def from_typesafe(
        cls, client: TypeSafeClient, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> DecisionModel:
        """A decision model on TypeSafe's API, such as jev-1.14, through the caller's client."""
        # Imported here because the SDK is the optional extra hyoshigi[typesafe].
        from hyoshigi.models import typesafe as provider

        return provider.build_decision_model(client, model=model, perceives=perceives)


@dataclass(frozen=True, slots=True, kw_only=True)
class AsyncDecisionModel:
    """A model that gives the probability a statement holds, through the caller's async
    function.

    The function may be called more than once per rating, concurrently, and returns an
    outcome rather than raising for a failure it expects.
    """

    name: str
    decide: Callable[[Question], Awaitable[DecisionOutcome]]
    perceives: MediaTypes = TEXT_ONLY

    def __post_init__(self) -> None:
        check_name(self.name)

    @classmethod
    def from_openai(
        cls, client: AsyncOpenAI, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> AsyncDecisionModel:
        """A decision model on OpenAI's Decisions API, through the caller's async client."""
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_async_decision_model(client, model=model, perceives=perceives)

    @classmethod
    def from_openrouter(
        cls, client: OpenRouter, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> AsyncDecisionModel:
        """A decision model on OpenRouter's Decisions API, through the caller's client,
        awaited.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_async_decision_model(client, model=model, perceives=perceives)

    @classmethod
    def from_typesafe(
        cls, client: AsyncTypeSafeClient, *, model: str, perceives: MediaTypes = TEXT_ONLY
    ) -> AsyncDecisionModel:
        """A decision model on TypeSafe's API, through the caller's async client."""
        # Imported here because the SDK is the optional extra hyoshigi[typesafe].
        from hyoshigi.models import typesafe as provider

        return provider.build_async_decision_model(client, model=model, perceives=perceives)
