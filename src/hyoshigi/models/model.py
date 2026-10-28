"""The models a monitor asks, each around the caller's function.

A generative model writes a reply for an LLM monitor to parse; a decision model gives
the probability that a statement about the state holds. A model carries a name, since a
score records which model rated it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAliasType, get_args

from hyoshigi.errors import ConfigurationError
from hyoshigi.options import (
    check_instance,
    describe_type,
    name_class,
    read_count,
    read_log_probability,
    read_threshold,
)
from hyoshigi.trajectory.parts import MEDIA_TYPE, MediaPart, TextPart

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import ThinkingConfigParam
    from openai import AsyncOpenAI, OpenAI
    from openai.types import ReasoningEffort
    from openrouter import OpenRouter, components
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

type PromptPart = TextPart | MediaPart


@dataclass(
    frozen=True,
    slots=True,
    init=False,
)
class MediaTypes:
    """The media a model perceives, as IANA media types or ranges such as image/*."""

    patterns: frozenset[str]

    def __init__(
        self,
        *patterns: str,
    ) -> None:
        if any(MEDIA_TYPE.fullmatch(pattern) is None for pattern in patterns):
            raise ConfigurationError("each pattern must be an IANA media type, such as image/*")
        object.__setattr__(
            self,
            "patterns",
            frozenset(
                map(
                    str.lower,
                    patterns,
                ),
            ),
        )

    def covers(
        self,
        media_type: str,
    ) -> bool:
        """Whether these patterns cover media of this type; a range only through its range."""
        media_type = media_type.lower()
        top_level = media_type.partition("/")[0]
        return media_type in self.patterns or f"{top_level}/*" in self.patterns


# A model that perceives no media: media reaches it only as a description or a placeholder.
TEXT_ONLY = MediaTypes()


def check_parts(parts: tuple[PromptPart, ...]) -> None:
    """Raise ConfigurationError unless every part is text or media."""
    if not all(
        isinstance(
            part,
            TextPart | MediaPart,
        )
        for part in parts
    ):
        raise ConfigurationError("a model is given text and media only")


def holds_media(parts: tuple[PromptPart, ...]) -> bool:
    """Whether any of the parts is media."""
    return any(
        isinstance(
            part,
            MediaPart,
        )
        for part in parts
    )


def collect[Item](items: list[Item | None]) -> list[Item] | None:
    """The items, or None if any is None; every item is built before this decides."""
    kept = [item for item in items if item is not None]
    return kept if len(kept) == len(items) else None


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Prompt:
    """What a generative model is asked: the monitor's instructions, and the content."""

    system: str
    content: tuple[PromptPart, ...]

    def __post_init__(self) -> None:
        check_parts(self.content)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Question:
    """What a decision model is asked: whether the statement holds of the state."""

    statement: str
    state: tuple[PromptPart, ...]

    def __post_init__(self) -> None:
        check_parts(self.state)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AlternativeToken:
    """A token a model could have written at a position, with its log-probability.

    The log-probability is a number of at most 0, kept as a float; minus infinity is a
    probability of 0. Anything else raises ConfigurationError.
    """

    token: str
    log_probability: float

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the float is set past its guard.
        object.__setattr__(
            self,
            "log_probability",
            read_token_log_probability(
                self.token,
                log_probability=self.log_probability,
            ),
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class TokenLogProbability:
    """A generated token with its log-probability and the likeliest alternatives to it.

    The log-probability is read as an AlternativeToken's is. The alternatives are a tuple of
    AlternativeToken, which may leave out the generated token itself; anything else raises
    ConfigurationError.
    """

    token: str
    log_probability: float
    alternatives: tuple[AlternativeToken, ...] = ()

    def __post_init__(self) -> None:
        # The dataclass is frozen, so the float is set past its guard.
        object.__setattr__(
            self,
            "log_probability",
            read_token_log_probability(
                self.token,
                log_probability=self.log_probability,
            ),
        )
        check_items(
            self.alternatives,
            expected=AlternativeToken,
            name="alternatives",
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Generation:
    """A complete reply, with the log-probability of each generated token if the model gave them.

    `log_probabilities` is a tuple of TokenLogProbability, one per generated token in order,
    and empty when the model gave none; anything else raises ConfigurationError.

    <https://omamori-lab.github.io/hyoshigi/how-to/choosing-a-monitor/>
    """

    text: str
    log_probabilities: tuple[TokenLogProbability, ...] = ()

    def __post_init__(self) -> None:
        check_items(
            self.log_probabilities,
            expected=TokenLogProbability,
            name="log_probabilities",
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Truncated:
    """A reply cut off at the model's output limit, with the text written so far."""

    text: str


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Refused:
    """The model or its provider declined, with the provider's explanation if it gave one."""

    explanation: str | None = None


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Rejected:
    """The provider rejected the prompt as one it cannot use, with the HTTP status if there was
    one.
    """

    status: int | None = None


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Unavailable:
    """No answer after the caller's retries, with the HTTP status if there was one."""

    status: int | None = None


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
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
            self,
            "probability",
            read_threshold(
                self.probability,
                name="probability",
            ),
        )
        if self.confidence is not None:
            confidence = read_threshold(
                self.confidence,
                name="confidence",
            )
            object.__setattr__(
                self,
                "confidence",
                confidence,
            )


type GenerationOutcome = Generation | Truncated | Refused | Rejected | Unavailable
type DecisionOutcome = Decision | Refused | Rejected | Unavailable


def check_outcomes(
    outcomes: tuple[object, ...],
    *,
    union: TypeAliasType,
    function: str,
) -> None:
    """Raise ConfigurationError if a model's function returned a value outside its outcome
    union, naming the function, the union's classes and the first such value's type, never
    the value, which can hold a model's reply.
    """
    kinds: tuple[type, ...] = get_args(union.__value__)
    for outcome in outcomes:
        if not isinstance(
            outcome,
            kinds,
        ):
            *others, last = (kind.__name__ for kind in kinds)
            raise ConfigurationError(
                f"{function} must return a {', '.join(others)} or {last}, "
                f"got {describe_type(outcome)}",
            )


def check_name(name: str) -> None:
    """Raise ConfigurationError if a model's name is empty."""
    if not name:
        raise ConfigurationError("a model needs a name, such as the provider's model id")


def read_token_log_probability(
    token: object,
    *,
    log_probability: object,
) -> float:
    """Raise ConfigurationError unless the token is a str; return the log-probability as a float."""
    check_instance(
        token,
        expected=str,
        name="token",
    )
    return read_log_probability(
        log_probability,
        name="log_probability",
    )


def check_items(
    items: tuple[object, ...],
    *,
    expected: type,
    name: str,
) -> None:
    """Raise ConfigurationError unless items is a tuple of instances of the expected class,
    naming the first wrong item by its position.
    """
    check_instance(
        items,
        expected=tuple,
        name=name,
    )
    for position, item in enumerate(items):
        check_instance(
            item,
            expected=expected,
            name=f"{name}[{position}]",
        )


def keep_top_log_probabilities(model: GenerativeModel | AsyncGenerativeModel) -> None:
    """Keep the model's top_log_probabilities as an int, or raise ConfigurationError."""
    # The dataclass is frozen, so the int is set past its guard.
    object.__setattr__(
        model,
        "top_log_probabilities",
        read_count(
            model.top_log_probabilities,
            name="top_log_probabilities",
            minimum=0,
        ),
    )


def check_model(
    name: str,
    *,
    function: Callable[..., object],
    function_name: str,
    perceives: MediaTypes,
) -> None:
    """Raise ConfigurationError unless a model's name is a non-empty str, its function is
    callable and perceives is a MediaTypes; the message names the field and the type given.
    """
    check_instance(
        name,
        expected=str,
        name="name",
    )
    check_name(name)
    if not callable(function):
        raise ConfigurationError(f"{function_name} must be callable, got {describe_type(function)}")
    check_instance(
        perceives,
        expected=MediaTypes,
        name="perceives",
    )


def check_client(
    client: object,
    *,
    expected: type,
    twin: type,
    twin_constructor: str,
) -> None:
    """Raise ConfigurationError unless the client is an instance of the expected class; a client
    of the twin class, the SDK's other family, is pointed to the constructor that takes it.
    """
    check_instance(
        client,
        expected=expected,
        name="client",
        hint=(
            f"Use {twin_constructor} for {name_class(twin)}."
            if isinstance(
                client,
                twin,
            )
            else ""
        ),
    )


def check_perceives(
    perceives: MediaTypes,
    *,
    carries: MediaTypes,
) -> None:
    """Raise ConfigurationError unless perceives is a MediaTypes, or naming the first pattern, in
    sorted order, that the media an API carries do not cover; a range such as image/* is covered
    only by a carried range.
    """
    check_instance(
        perceives,
        expected=MediaTypes,
        name="perceives",
    )
    for pattern in sorted(perceives.patterns):
        if not carries.covers(pattern):
            raise ConfigurationError(f"the API cannot carry {pattern} media")


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class GenerativeModel:
    """A model that writes a reply to a prompt, through the caller's function.

    The function may be called more than once per rating, and returns an outcome rather
    than raising for a failure it expects.

    `top_log_probabilities` is the number of likeliest alternative tokens the function returns
    with each generated token's log-probability, a whole number from 0 to sys.maxsize kept as
    an int; 0 says the function returns none. Above 0, the function asks its provider for them
    on every call and returns them on every Generation, and a GuardMonitor scores from them.
    Anything else raises ConfigurationError.

    <https://omamori-lab.github.io/hyoshigi/how-to/choosing-a-monitor/>
    """

    name: str
    generate: Callable[[Prompt], GenerationOutcome]
    perceives: MediaTypes = TEXT_ONLY
    top_log_probabilities: int = 0

    def __post_init__(self) -> None:
        check_model(
            self.name,
            function=self.generate,
            function_name="generate",
            perceives=self.perceives,
        )
        keep_top_log_probabilities(self)

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
            client,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            perceives=perceives,
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

    @classmethod
    def from_litellm(
        cls,
        completion: Callable[..., object],
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
        top_log_probabilities: int = 0,
    ) -> GenerativeModel:
        """A model on LiteLLM's chat completions, through the caller's litellm.completion.

        Options of the call, such as max_completion_tokens or drop_params, are bound to the
        function with functools.partial. Above 0, top_log_probabilities asks for that many
        alternatives to each generated token, which the model returns and declares.

        <https://omamori-lab.github.io/hyoshigi/reference/providers/>
        """
        # Imported here because the module imports the OpenAI SDK's errors, which only the
        # adapters read.
        from hyoshigi.models import litellm as provider

        return provider.build_model(
            completion,
            model=model,
            perceives=perceives,
            top_log_probabilities=top_log_probabilities,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class AsyncGenerativeModel:
    """A model that writes a reply to a prompt, through the caller's async function.

    The function may be called more than once per rating, concurrently, and returns an
    outcome rather than raising for a failure it expects. `top_log_probabilities` is read as
    GenerativeModel reads it.

    <https://omamori-lab.github.io/hyoshigi/how-to/choosing-a-monitor/>
    """

    name: str
    generate: Callable[[Prompt], Awaitable[GenerationOutcome]]
    perceives: MediaTypes = TEXT_ONLY
    top_log_probabilities: int = 0

    def __post_init__(self) -> None:
        check_model(
            self.name,
            function=self.generate,
            function_name="generate",
            perceives=self.perceives,
        )
        keep_top_log_probabilities(self)

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
            client,
            model=model,
            max_tokens=max_tokens,
            thinking=thinking,
            perceives=perceives,
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

    @classmethod
    def from_litellm(
        cls,
        acompletion: Callable[..., Awaitable[object]],
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
        top_log_probabilities: int = 0,
    ) -> AsyncGenerativeModel:
        """A model on LiteLLM's chat completions, through the caller's litellm.acompletion.

        Options of the call, such as max_completion_tokens or drop_params, are bound to the
        function with functools.partial. Above 0, top_log_probabilities asks for that many
        alternatives to each generated token, which the model returns and declares.

        <https://omamori-lab.github.io/hyoshigi/reference/providers/>
        """
        # Imported here because the module imports the OpenAI SDK's errors, which only the
        # adapters read.
        from hyoshigi.models import litellm as provider

        return provider.build_async_model(
            acompletion,
            model=model,
            perceives=perceives,
            top_log_probabilities=top_log_probabilities,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class DecisionModel:
    """A model that gives the probability a statement holds, through the caller's function.

    The function may be called more than once per rating, and returns an outcome rather
    than raising for a failure it expects.
    """

    name: str
    decide: Callable[[Question], DecisionOutcome]
    perceives: MediaTypes = TEXT_ONLY

    def __post_init__(self) -> None:
        check_model(
            self.name,
            function=self.decide,
            function_name="decide",
            perceives=self.perceives,
        )

    @classmethod
    def from_openai(
        cls,
        client: OpenAI,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> DecisionModel:
        """A decision model on OpenAI's Decisions API, such as gpt-6-luna, through the caller's
        client.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_decision_model(
            client,
            model=model,
            perceives=perceives,
        )

    @classmethod
    def from_openrouter(
        cls,
        client: OpenRouter,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> DecisionModel:
        """A decision model on OpenRouter's Decisions API, such as typesafe/jev-1.14."""
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_decision_model(
            client,
            model=model,
            perceives=perceives,
        )

    @classmethod
    def from_typesafe(
        cls,
        client: TypeSafeClient,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> DecisionModel:
        """A decision model on TypeSafe's API, such as jev-1.14, through the caller's client."""
        # Imported here because the SDK is the optional extra hyoshigi[typesafe].
        from hyoshigi.models import typesafe as provider

        return provider.build_decision_model(
            client,
            model=model,
            perceives=perceives,
        )


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
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
        check_model(
            self.name,
            function=self.decide,
            function_name="decide",
            perceives=self.perceives,
        )

    @classmethod
    def from_openai(
        cls,
        client: AsyncOpenAI,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncDecisionModel:
        """A decision model on OpenAI's Decisions API, through the caller's async client."""
        # Imported here because the SDK is the optional extra hyoshigi[openai].
        from hyoshigi.models import openai as provider

        return provider.build_async_decision_model(
            client,
            model=model,
            perceives=perceives,
        )

    @classmethod
    def from_openrouter(
        cls,
        client: OpenRouter,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncDecisionModel:
        """A decision model on OpenRouter's Decisions API, through the caller's client,
        awaited.
        """
        # Imported here because the SDK is the optional extra hyoshigi[openrouter].
        from hyoshigi.models import openrouter as provider

        return provider.build_async_decision_model(
            client,
            model=model,
            perceives=perceives,
        )

    @classmethod
    def from_typesafe(
        cls,
        client: AsyncTypeSafeClient,
        *,
        model: str,
        perceives: MediaTypes = TEXT_ONLY,
    ) -> AsyncDecisionModel:
        """A decision model on TypeSafe's API, through the caller's async client."""
        # Imported here because the SDK is the optional extra hyoshigi[typesafe].
        from hyoshigi.models import typesafe as provider

        return provider.build_async_decision_model(
            client,
            model=model,
            perceives=perceives,
        )
