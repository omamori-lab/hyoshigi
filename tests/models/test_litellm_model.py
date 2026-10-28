"""LiteLLM's chat completions as a model: the request the caller's function gets and the outcome of
each reply and each error it raises.

LiteLLM is never installed. The caller's function is a stand-in that records the keyword arguments
it gets and returns an object shaped as LiteLLM's ModelResponse (litellm/types/utils.py at
v1.106.5), whose model_dump() gives the response as a mapping, or raises. The errors are OpenAI's
own types, which LiteLLM's exceptions subclass (litellm/exceptions.py at v1.106.5), and stand-ins
shaped as the LiteLLM classes the adapter reads by class.
"""

import asyncio
import functools
import math
import numbers
import re
import sys
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import httpx2
import openai
import pytest

from hyoshigi import (
    AlternativeToken,
    AsyncGenerativeModel,
    ConfigurationError,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    MediaPart,
    MediaTypes,
    Prompt,
    PromptPart,
    Refused,
    Rejected,
    TextPart,
    TokenLogProbability,
    Truncated,
    Unavailable,
)

MODEL = "anthropic/claude-haiku-5-5"
SYSTEM = "You are a helpful assistant."
TEXT_ONLY = MediaTypes()
ADDRESS = httpx2.Request(
    "POST",
    "https://litellm.test/chat/completions",
)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Reply:
    """Stands in for LiteLLM's ModelResponse, which gives itself as a mapping by model_dump()."""

    body: object

    def dump(self) -> object:
        return self.body

    # LiteLLM's ModelResponse names its dump model_dump.
    model_dump = dump


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Exchange:
    """What a model returned for a prompt, and the keyword arguments the function got."""

    model: GenerativeModel | AsyncGenerativeModel
    outcome: GenerationOutcome
    requests: list[dict[str, object]]


class Ask(Protocol):
    """Gives a prompt to a model whose function serves one reply, or raises one error."""

    def __call__(
        self,
        reply: object,
        prompt: Prompt = ...,
        *,
        model: str = ...,
        perceives: MediaTypes = ...,
        top_log_probabilities: object = ...,
        bound: Mapping[str, object] | None = ...,
    ) -> Exchange: ...


PROMPT = Prompt(
    system=SYSTEM,
    content=(TextPart(text="Hello!"),),
)


def generate_synchronously(
    reply: object,
    prompt: Prompt = PROMPT,
    *,
    model: str = MODEL,
    perceives: MediaTypes = TEXT_ONLY,
    top_log_probabilities: object = 0,
    bound: Mapping[str, object] | None = None,
) -> Exchange:
    requests: list[dict[str, object]] = []

    def completion(**arguments: object) -> object:
        requests.append(arguments)
        if isinstance(
            reply,
            BaseException,
        ):
            raise reply
        return reply

    built = GenerativeModel.from_litellm(
        completion
        if bound is None
        else functools.partial(
            completion,
            **bound,
        ),
        model=model,
        perceives=perceives,
        # Some tests pass a count of the wrong type, which the constructor refuses.
        top_log_probabilities=cast(
            int,
            top_log_probabilities,
        ),
    )
    return Exchange(
        model=built,
        outcome=built.generate(prompt),
        requests=requests,
    )


def generate_asynchronously(
    reply: object,
    prompt: Prompt = PROMPT,
    *,
    model: str = MODEL,
    perceives: MediaTypes = TEXT_ONLY,
    top_log_probabilities: object = 0,
    bound: Mapping[str, object] | None = None,
) -> Exchange:
    requests: list[dict[str, object]] = []

    async def acompletion(**arguments: object) -> object:
        requests.append(arguments)
        if isinstance(
            reply,
            BaseException,
        ):
            raise reply
        return reply

    built = AsyncGenerativeModel.from_litellm(
        acompletion
        if bound is None
        else functools.partial(
            acompletion,
            **bound,
        ),
        model=model,
        perceives=perceives,
        # Some tests pass a count of the wrong type, which the constructor refuses.
        top_log_probabilities=cast(
            int,
            top_log_probabilities,
        ),
    )

    async def generate() -> GenerationOutcome:
        return await built.generate(prompt)

    return Exchange(
        model=built,
        outcome=asyncio.run(generate()),
        requests=requests,
    )


@pytest.fixture(
    params=[
        generate_synchronously,
        generate_asynchronously,
    ],
    ids=["sync", "async"],
)
def ask(request: pytest.FixtureRequest) -> Ask:
    return request.param


# docs.litellm.ai/docs/completion/output, "Output Format" and its example response: the fields of
# a reply, and its content.
ANSWER = " I'm doing well, thank you for asking. I am Claude, an AI assistant created by Anthropic."
DOCUMENTED_REPLY = Reply(
    body={
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ANSWER,
                },
            },
        ],
        "created": 1691429984.3852863,
        "model": "claude-sonnet-5",
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 23,
            "total_tokens": 41,
        },
    },
)


def build_reply(
    finish_reason: object = "stop",
    content: object = ANSWER,
    **message_fields: object,
) -> Reply:
    """The documented reply with its finish reason, content and further message fields changed."""
    return Reply(
        body={
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        **message_fields,
                    },
                },
            ],
        },
    )


def build_logprobs_reply(
    logprobs: object,
    *,
    content: object = "VALID",
) -> Reply:
    """A stop reply with this content, and this value for its choice's logprobs."""
    return Reply(
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "logprobs": logprobs,
                },
            ],
        },
    )


SYSTEM_MESSAGE = {
    "role": "system",
    "content": SYSTEM,
}
HELLO = {
    "type": "text",
    "text": "Hello!",
}
REQUEST: dict[str, object] = {
    "model": MODEL,
    "messages": [
        SYSTEM_MESSAGE,
        {
            "role": "user",
            "content": [HELLO],
        },
    ],
}


def test_a_model_is_named_for_litellm_and_its_model_id(ask: Ask) -> None:
    exchange = ask(DOCUMENTED_REPLY)

    assert exchange.model.name == "litellm:anthropic/claude-haiku-5-5"


@pytest.mark.parametrize(
    "perceives",
    [
        MediaTypes("image/*", "application/pdf", "audio/wav"),
        MediaTypes("image/png", "image/jpeg"),
    ],
    ids=["all-it-carries", "types-within-a-range"],
)
def test_a_model_perceives_the_media_it_was_given(
    ask: Ask,
    perceives: MediaTypes,
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        perceives=perceives,
    )

    assert exchange.model.perceives == perceives


@pytest.mark.parametrize(
    ("patterns", "named"),
    [
        (
            ("video/*",),
            "video/*",
        ),
        (
            ("video/mp4",),
            "video/mp4",
        ),
        (
            ("application/*",),
            "application/*",
        ),
        (
            ("audio/mpeg",),
            "audio/mpeg",
        ),
        (
            ("text/*",),
            "text/*",
        ),
        (
            ("image/png", "audio/wav", "text/plain", "video/mp4"),
            "text/plain",
        ),
    ],
    ids=[
        "range-of-another-type",
        "another-type",
        "range-where-only-the-pdf-is-carried",
        "audio-other-than-wav",
        "text-range",
        "the-first-of-those-it-cannot-carry",
    ],
)
def test_a_model_that_perceives_media_the_api_cannot_carry_raises_naming_it_when_built(
    ask: Ask,
    patterns: tuple[str, ...],
    named: str,
) -> None:
    with pytest.raises(
        ConfigurationError,
        match=re.escape(named),
    ):
        ask(
            DOCUMENTED_REPLY,
            perceives=MediaTypes(*patterns),
        )


@pytest.mark.parametrize(
    "top_log_probabilities",
    [
        0,
        1,
        5,
        20,
    ],
)
def test_a_model_declares_the_log_probabilities_it_asks_for(
    ask: Ask,
    top_log_probabilities: int,
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        top_log_probabilities=top_log_probabilities,
    )

    assert exchange.model.top_log_probabilities == top_log_probabilities


@pytest.mark.parametrize(
    ("top_log_probabilities", "message"),
    [
        (
            -1,
            "top_log_probabilities must be at least 0, got -1",
        ),
        (
            1.5,
            "top_log_probabilities must be a whole number of at least 0, got 1.5",
        ),
        (
            True,
            "top_log_probabilities must be a whole number of at least 0, got True",
        ),
    ],
    ids=["negative", "float", "bool"],
)
def test_a_count_of_alternatives_that_is_not_a_whole_number_from_zero_raises_when_built(
    ask: Ask,
    top_log_probabilities: object,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask(
            DOCUMENTED_REPLY,
            top_log_probabilities=top_log_probabilities,
        )

    assert str(raised.value) == message


def test_a_model_without_a_model_id_raises_when_it_is_built(ask: Ask) -> None:
    with pytest.raises(
        ConfigurationError,
        match="a model needs a name",
    ):
        ask(
            DOCUMENTED_REPLY,
            model="",
        )


async def complete_later(**arguments: object) -> object:
    return DOCUMENTED_REPLY


def complete_now(**arguments: object) -> object:
    return DOCUMENTED_REPLY


@pytest.mark.parametrize(
    ("build", "function", "message"),
    [
        (
            GenerativeModel.from_litellm,
            complete_later,
            "completion must not be a coroutine function. "
            "Use AsyncGenerativeModel.from_litellm for litellm.acompletion.",
        ),
        (
            AsyncGenerativeModel.from_litellm,
            complete_now,
            "acompletion must be a coroutine function. "
            "Use GenerativeModel.from_litellm for litellm.completion.",
        ),
        (
            GenerativeModel.from_litellm,
            functools.partial(
                complete_later,
                drop_params=True,
            ),
            "completion must not be a coroutine function. "
            "Use AsyncGenerativeModel.from_litellm for litellm.acompletion.",
        ),
        (
            AsyncGenerativeModel.from_litellm,
            functools.partial(
                complete_now,
                drop_params=True,
            ),
            "acompletion must be a coroutine function. "
            "Use GenerativeModel.from_litellm for litellm.completion.",
        ),
    ],
    ids=[
        "async-function-to-the-sync-constructor",
        "sync-function-to-the-async-constructor",
        "partial-of-an-async-function-to-the-sync-constructor",
        "partial-of-a-sync-function-to-the-async-constructor",
    ],
)
def test_a_function_of_the_other_family_raises_naming_the_constructor_that_takes_it(
    build: Callable[..., GenerativeModel | AsyncGenerativeModel],
    function: Callable[..., object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build(
            function,
            model=MODEL,
        )

    assert str(raised.value) == message


@pytest.mark.parametrize(
    ("build", "named"),
    [
        (
            GenerativeModel.from_litellm,
            "completion must be callable, got an instance of str",
        ),
        (
            AsyncGenerativeModel.from_litellm,
            "acompletion must be callable, got an instance of str",
        ),
    ],
    ids=["sync", "async"],
)
def test_a_function_that_is_not_callable_raises_naming_its_type_when_the_model_is_built(
    build: Callable[..., GenerativeModel | AsyncGenerativeModel],
    named: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        build(
            "litellm.completion",
            model=MODEL,
        )

    assert str(raised.value) == named


def test_a_prompt_is_sent_as_a_system_message_and_one_user_message(ask: Ask) -> None:
    exchange = ask(DOCUMENTED_REPLY)

    assert exchange.requests == [REQUEST]


def test_options_bound_to_the_function_with_functools_partial_reach_the_call_with_the_request(
    ask: Ask,
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        bound={
            "max_completion_tokens": 64,
            "drop_params": True,
        },
    )

    assert exchange.requests == [
        {
            **REQUEST,
            "max_completion_tokens": 64,
            "drop_params": True,
        },
    ]


@pytest.mark.parametrize(
    "top_log_probabilities",
    [
        1,
        5,
        20,
    ],
)
def test_log_probabilities_are_asked_for_when_the_model_declares_them(
    ask: Ask,
    top_log_probabilities: int,
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        top_log_probabilities=top_log_probabilities,
    )

    assert exchange.requests == [
        {
            **REQUEST,
            "logprobs": True,
            "top_logprobs": top_log_probabilities,
        },
    ]


class ArrayInteger:
    """Stands in for numpy's integers: registered as integral, but not a subclass of int."""

    def __init__(
        self,
        value: int,
    ) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


numbers.Integral.register(ArrayInteger)


def test_a_whole_number_of_alternatives_that_is_not_an_int_is_sent_as_an_int(ask: Ask) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        top_log_probabilities=ArrayInteger(3),
    )

    assert exchange.requests == [
        {
            **REQUEST,
            "logprobs": True,
            "top_logprobs": 3,
        },
    ]


# The eight bytes every PNG file starts with (RFC 2083, section 3.1), "%PDF-1.7", the first
# bytes of a PDF (ISO 32000-1, section 7.5.2), and "RIFF", the first bytes of a WAVE file. Their
# base64 forms are those of RFC 4648, section 4.
PNG = b"\x89PNG\r\n\x1a\n"
PDF = b"%PDF-1.7"
WAVE = b"RIFF"
# docs.litellm.ai/docs/completion/vision and document_understanding: the URL of each example.
LOGO = (
    "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
)
DUMMY_PDF = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


def prompt_of(*parts: PromptPart) -> Prompt:
    return Prompt(
        system=SYSTEM,
        content=parts,
    )


# docs.litellm.ai/docs/completion/vision (image_url is an object with a url),
# document_understanding (a file by file_id for a URL and by file_data, a data URL, for bytes,
# with format) and audio (input_audio with base64 data and format "wav"). The filename of file
# data is the one the OpenAI adapters send, which OpenAI's API requires with file data.
@pytest.mark.parametrize(
    ("part", "block"),
    [
        (
            MediaPart(
                media_type="image/png",
                url=LOGO,
            ),
            {
                "type": "image_url",
                "image_url": {"url": LOGO},
            },
        ),
        (
            MediaPart(
                media_type="image/png",
                data=PNG,
            ),
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ),
        (
            MediaPart(
                media_type="image/*",
                url=LOGO,
            ),
            {
                "type": "image_url",
                "image_url": {"url": LOGO},
            },
        ),
        (
            MediaPart(
                media_type="application/pdf",
                url=DUMMY_PDF,
            ),
            {
                "type": "file",
                "file": {
                    "file_id": DUMMY_PDF,
                    "format": "application/pdf",
                },
            },
        ),
        (
            MediaPart(
                media_type="application/pdf",
                data=PDF,
            ),
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                    "filename": "attachment.pdf",
                    "format": "application/pdf",
                },
            },
        ),
        (
            MediaPart(
                media_type="audio/wav",
                data=WAVE,
            ),
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "UklGRg==",
                    "format": "wav",
                },
            },
        ),
    ],
    ids=[
        "image-at-a-url",
        "image-bytes-as-a-data-url",
        "image-range-at-a-url",
        "pdf-at-a-url",
        "pdf-bytes-as-a-data-url",
        "wav-bytes-as-base64",
    ],
)
def test_media_is_sent_as_the_content_block_of_its_kind(
    ask: Ask,
    part: MediaPart,
    block: dict[str, object],
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        prompt_of(
            TextPart(text="Hello!"),
            part,
        ),
    )

    assert exchange.requests == [
        {
            "model": MODEL,
            "messages": [
                SYSTEM_MESSAGE,
                {
                    "role": "user",
                    "content": [
                        HELLO,
                        block,
                    ],
                },
            ],
        },
    ]


def test_blank_text_is_sent_as_a_text_block(ask: Ask) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        prompt_of(TextPart(text=" ")),
    )

    assert exchange.requests == [
        {
            "model": MODEL,
            "messages": [
                SYSTEM_MESSAGE,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": " ",
                        },
                    ],
                },
            ],
        },
    ]


@pytest.mark.parametrize(
    "prompt",
    [
        Prompt(
            system="\ud800",
            content=(TextPart(text="Hello!"),),
        ),
        prompt_of(
            TextPart(text="Hello!"),
            TextPart(text="\ud800"),
        ),
        prompt_of(
            MediaPart(
                media_type="image/*",
                data=PNG,
            ),
        ),
        prompt_of(
            MediaPart(
                media_type="audio/wav",
                url="https://example.com/recording.wav",
            ),
        ),
    ],
    ids=[
        "system-with-no-utf-8-form",
        "text-with-no-utf-8-form",
        "image-range-bytes",
        "wav-at-a-url",
    ],
)
def test_a_prompt_in_a_form_the_api_cannot_take_is_unavailable_without_a_call(
    ask: Ask,
    prompt: Prompt,
) -> None:
    exchange = ask(
        DOCUMENTED_REPLY,
        prompt,
    )

    assert (
        exchange.outcome,
        exchange.requests,
    ) == (
        Unavailable(status=None),
        [],
    )


@pytest.mark.parametrize(
    "media_type",
    [
        "video/mp4",
        "application/zip",
        "application/*",
        "text/plain",
    ],
)
def test_media_the_api_cannot_carry_raises_naming_its_media_type(
    ask: Ask,
    media_type: str,
) -> None:
    with pytest.raises(
        ConfigurationError,
        match=re.escape(media_type),
    ):
        ask(
            DOCUMENTED_REPLY,
            prompt_of(
                MediaPart(
                    media_type=media_type,
                    url="https://example.com/file",
                ),
            ),
        )


@pytest.mark.parametrize(
    "parts",
    [
        (
            MediaPart(
                media_type="image/*",
                data=PNG,
            ),
            MediaPart(
                media_type="video/mp4",
                url="https://example.com/clip.mp4",
            ),
        ),
        (
            MediaPart(
                media_type="video/mp4",
                url="https://example.com/clip.mp4",
            ),
            MediaPart(
                media_type="image/*",
                data=PNG,
            ),
        ),
    ],
    ids=["after-bytes-it-cannot-name", "before-bytes-it-cannot-name"],
)
def test_media_the_api_can_never_carry_raises_whatever_the_order_of_the_parts(
    ask: Ask,
    parts: tuple[MediaPart, ...],
) -> None:
    with pytest.raises(
        ConfigurationError,
        match="video/mp4",
    ):
        ask(
            DOCUMENTED_REPLY,
            prompt_of(*parts),
        )


def test_a_stop_reply_is_a_generation_of_its_content(ask: Ask) -> None:
    exchange = ask(DOCUMENTED_REPLY)

    assert exchange.outcome == Generation(text=ANSWER)


def test_a_stop_reply_without_content_is_a_generation_of_empty_text(ask: Ask) -> None:
    exchange = ask(build_reply(content=None))

    assert exchange.outcome == Generation(text="")


def test_only_the_first_choice_is_read(ask: Ask) -> None:
    reply = Reply(
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "first",
                    },
                },
                {
                    "finish_reason": "length",
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "second",
                    },
                },
            ],
        },
    )

    exchange = ask(reply)

    assert exchange.outcome == Generation(text="first")


@pytest.mark.parametrize(
    ("content", "text"),
    [
        (
            "The cheapest flight is",
            "The cheapest flight is",
        ),
        (
            None,
            "",
        ),
    ],
    ids=["text-so-far", "no-text"],
)
def test_a_reply_cut_off_at_the_output_limit_is_truncated_with_its_text(
    ask: Ask,
    content: str | None,
    text: str,
) -> None:
    exchange = ask(
        build_reply(
            "length",
            content,
        ),
    )

    assert exchange.outcome == Truncated(text=text)


def test_a_reply_cut_off_at_the_output_limit_is_truncated_even_when_it_holds_a_refusal(
    ask: Ask,
) -> None:
    exchange = ask(
        build_reply(
            "length",
            "I can't",
            refusal="I can't help with that.",
        ),
    )

    assert exchange.outcome == Truncated(text="I can't")


# docs.litellm.ai/docs/completion/output, "Native Finish Reason": LiteLLM gives stop, length,
# tool_calls, function_call and content_filter. A reason added later is read as the others.
@pytest.mark.parametrize(
    "finish_reason",
    [
        "tool_calls",
        "function_call",
        "a_reason_added_later",
    ],
)
def test_a_reply_that_stopped_for_any_other_reason_is_truncated_with_its_text(
    ask: Ask,
    finish_reason: str,
) -> None:
    exchange = ask(
        build_reply(
            finish_reason,
            "I will look that up.",
        ),
    )

    assert exchange.outcome == Truncated(text="I will look that up.")


def test_a_reply_stopped_by_the_content_filter_is_a_refusal_without_an_explanation(
    ask: Ask,
) -> None:
    exchange = ask(
        build_reply(
            "content_filter",
            None,
        ),
    )

    assert exchange.outcome == Refused(explanation=None)


# LiteLLM's Message has no refusal field at v1.106.5, and convert_to_model_response_object
# (litellm_core_utils/llm_response_utils/convert_dict_to_response.py) moves a message key that is
# no field of Message into provider_specific_fields. A refusal is read in both places.
@pytest.mark.parametrize(
    "message_fields",
    [
        {"refusal": "I can't help with that."},
        {"provider_specific_fields": {"refusal": "I can't help with that."}},
    ],
    ids=["message-refusal", "provider-specific-refusal"],
)
@pytest.mark.parametrize(
    "finish_reason",
    [
        "stop",
        "content_filter",
    ],
)
def test_a_reply_with_a_refusal_is_a_refusal_with_its_text(
    ask: Ask,
    finish_reason: str,
    message_fields: dict[str, object],
) -> None:
    exchange = ask(
        build_reply(
            finish_reason,
            None,
            **message_fields,
        ),
    )

    assert exchange.outcome == Refused(explanation="I can't help with that.")


@pytest.mark.parametrize(
    "message_fields",
    [
        {"refusal": None},
        {"refusal": ""},
        {"refusal": ["I can't help with that."]},
        {"provider_specific_fields": None},
        {"provider_specific_fields": {"refusal": ""}},
        {"provider_specific_fields": {"native_finish_reason": "end_turn"}},
    ],
    ids=[
        "no-refusal",
        "empty-refusal",
        "refusal-that-is-not-text",
        "no-provider-specific-fields",
        "empty-provider-specific-refusal",
        "provider-specific-field-of-another-name",
    ],
)
def test_a_refusal_with_no_text_is_not_a_refusal(
    ask: Ask,
    message_fields: dict[str, object],
) -> None:
    exchange = ask(
        build_reply(
            "stop",
            ANSWER,
            **message_fields,
        ),
    )

    assert exchange.outcome == Generation(text=ANSWER)


@pytest.mark.parametrize(
    "reply",
    [
        Reply(body={}),
        Reply(body=[]),
        Reply(body={"choices": []}),
        Reply(body={"choices": "stop"}),
        Reply(body={"choices": ["stop"]}),
        Reply(body={"choices": [{"message": {"content": ANSWER}}]}),
        build_reply(finish_reason=None),
        build_reply(finish_reason=1),
        Reply(body={"choices": [{"finish_reason": "stop"}]}),
        Reply(
            body={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": ANSWER,
                    },
                ],
            },
        ),
        Reply(
            body={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant"},
                    },
                ],
            },
        ),
        build_reply(content=7),
        build_reply(
            content=[
                {
                    "type": "text",
                    "text": ANSWER,
                },
            ],
        ),
        object(),
    ],
    ids=[
        "no-choices-field",
        "dump-that-is-not-a-mapping",
        "no-choice",
        "choices-that-is-not-a-list",
        "choice-that-is-not-a-mapping",
        "choice-without-a-finish-reason",
        "finish-reason-none",
        "finish-reason-that-is-not-text",
        "choice-without-a-message",
        "message-that-is-not-a-mapping",
        "message-without-content",
        "content-that-is-a-number",
        "content-that-is-a-list",
        "reply-that-gives-no-mapping",
    ],
)
def test_a_reply_that_is_not_what_litellm_documents_is_unavailable_without_a_status(
    ask: Ask,
    reply: object,
) -> None:
    exchange = ask(reply)

    assert exchange.outcome == Unavailable(status=None)


def build_alternative(
    token: str,
    log_probability: float,
) -> dict[str, object]:
    """A top alternative of a position: the token and its log-probability."""
    return {
        "token": token,
        "logprob": log_probability,
    }


def build_position(
    token: str,
    log_probability: float,
    *alternatives: dict[str, object],
) -> dict[str, object]:
    """A position of a reply's logprobs.content: a token, its log-probability and the top ones."""
    return {
        "token": token,
        "logprob": log_probability,
        "top_logprobs": list(alternatives),
    }


# The rule: the positions of choices[0].logprobs.content, in order, are the generation's
# log-probabilities, each with its top alternatives. The fields are those of
# ChatCompletionTokenLogprob and TopLogprob in litellm/types/utils.py at v1.106.5, whose
# bytes field is not read.
def test_the_log_probabilities_of_a_reply_are_those_of_its_tokens_with_their_alternatives(
    ask: Ask,
) -> None:
    reply = build_logprobs_reply(
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "bytes": [86, 65, 76, 73, 68],
                    "top_logprobs": [
                        {
                            "token": "VALID",
                            "logprob": -0.25,
                            "bytes": [86, 65, 76, 73, 68],
                        },
                        {
                            "token": "INVALID",
                            "logprob": -1.5,
                            "bytes": [73, 78, 86, 65, 76, 73, 68],
                        },
                    ],
                },
            ],
        },
    )

    exchange = ask(
        reply,
        top_log_probabilities=2,
    )

    assert exchange.outcome == Generation(
        text="VALID",
        log_probabilities=(
            TokenLogProbability(
                token="VALID",
                log_probability=-0.25,
                alternatives=(
                    AlternativeToken(
                        token="VALID",
                        log_probability=-0.25,
                    ),
                    AlternativeToken(
                        token="INVALID",
                        log_probability=-1.5,
                    ),
                ),
            ),
        ),
    )


def test_the_positions_of_a_reply_keep_their_order_and_an_impossible_token_is_minus_infinity(
    ask: Ask,
) -> None:
    reply = build_logprobs_reply(
        {
            "content": [
                build_position(
                    "VAL",
                    0,
                ),
                build_position(
                    "ID",
                    -math.inf,
                ),
            ],
        },
    )

    exchange = ask(
        reply,
        top_log_probabilities=1,
    )

    assert exchange.outcome == Generation(
        text="VALID",
        log_probabilities=(
            TokenLogProbability(
                token="VAL",
                log_probability=0.0,
            ),
            TokenLogProbability(
                token="ID",
                log_probability=-math.inf,
            ),
        ),
    )


@pytest.mark.parametrize(
    "logprobs",
    [
        None,
        {"content": None},
    ],
    ids=["no-logprobs", "no-content"],
)
def test_a_model_that_declares_log_probabilities_and_gets_none_returns_a_generation_without_them(
    ask: Ask,
    logprobs: object,
) -> None:
    exchange = ask(
        build_logprobs_reply(logprobs),
        top_log_probabilities=3,
    )

    assert exchange.outcome == Generation(
        text="VALID",
        log_probabilities=(),
    )


def test_a_position_whose_alternatives_are_null_has_none(ask: Ask) -> None:
    reply = build_logprobs_reply(
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "bytes": None,
                    "top_logprobs": None,
                },
            ],
        },
    )

    exchange = ask(
        reply,
        top_log_probabilities=3,
    )

    assert exchange.outcome == Generation(
        text="VALID",
        log_probabilities=(
            TokenLogProbability(
                token="VALID",
                log_probability=-0.25,
            ),
        ),
    )


def test_log_probabilities_in_a_reply_are_ignored_when_the_model_declares_none(ask: Ask) -> None:
    reply = build_logprobs_reply(
        {
            "content": [
                build_position(
                    "VALID",
                    -0.25,
                ),
            ],
        },
    )

    exchange = ask(
        reply,
        top_log_probabilities=0,
    )

    assert exchange.outcome == Generation(text="VALID")


@pytest.mark.parametrize(
    "logprobs",
    [
        "VALID",
        {},
        {"content": "VALID"},
        {"content": 7},
        {"content": ["VALID"]},
        {
            "content": [
                {
                    "logprob": -0.25,
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": 7,
                    "logprob": -0.25,
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": "-0.25",
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": True,
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": 0.25,
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": math.nan,
                    "top_logprobs": [],
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "top_logprobs": "INVALID",
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "top_logprobs": 7,
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                },
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "top_logprobs": ["INVALID"],
                },
            ],
        },
        {
            "content": [
                build_position(
                    "VALID",
                    -0.25,
                    build_alternative(
                        "INVALID",
                        1.5,
                    ),
                ),
            ],
        },
        {
            "content": [
                {
                    "token": "VALID",
                    "logprob": -0.25,
                    "top_logprobs": [
                        {
                            "token": 3,
                            "logprob": -1.5,
                        },
                    ],
                },
            ],
        },
    ],
    ids=[
        "logprobs-that-is-not-a-mapping",
        "no-content",
        "content-that-is-not-a-list",
        "content-that-is-a-number",
        "position-that-is-not-a-mapping",
        "position-without-a-token",
        "token-that-is-not-text",
        "position-without-a-log-probability",
        "log-probability-that-is-text",
        "log-probability-that-is-a-boolean",
        "log-probability-above-zero",
        "log-probability-that-is-nan",
        "alternatives-that-are-not-a-list",
        "alternatives-that-are-a-number",
        "position-without-alternatives",
        "alternative-that-is-not-a-mapping",
        "alternative-with-a-log-probability-above-zero",
        "alternative-with-a-token-that-is-not-text",
    ],
)
def test_log_probabilities_that_are_not_what_litellm_documents_make_the_reply_unavailable(
    ask: Ask,
    logprobs: object,
) -> None:
    exchange = ask(
        build_logprobs_reply(logprobs),
        top_log_probabilities=3,
    )

    assert exchange.outcome == Unavailable(status=None)


class ContextWindowExceededError(openai.BadRequestError):
    """Shaped as litellm.ContextWindowExceededError (exceptions.py, line 535): status 400, no
    code.
    """


class ContentPolicyViolationError(openai.BadRequestError):
    """Shaped as litellm.ContentPolicyViolationError (exceptions.py, line 667): status 400, and
    no code unless the provider's body gave one.
    """


class ImageFetchError(openai.BadRequestError):
    """Shaped as litellm.ImageFetchError (exceptions.py, line 270): status 400."""


class GuardrailRaisedException(Exception):
    """Shaped as litellm.GuardrailRaisedException (exceptions.py, line 1128): not an OpenAI
    error.
    """


class BlockedPiiEntityError(Exception):
    """Shaped as litellm.BlockedPiiEntityError (exceptions.py, line 1156): not an OpenAI error."""


class BudgetExceededError(Exception):
    """Shaped as litellm.BudgetExceededError (exceptions.py, line 1047): not an OpenAI error."""


class APIError(openai.APIError):
    """Shaped as litellm.APIError (exceptions.py, line 859): an openai.APIError that is not a
    status error, with its status in the attribute status_code.
    """

    def __init__(
        self,
        status_code: int,
    ) -> None:
        self.status_code = status_code
        super().__init__(
            "stand-in",
            ADDRESS,
            body=None,
        )


class Timeout(openai.APITimeoutError):
    """Shaped as litellm.Timeout (exceptions.py, line 356): the SDK's timeout, with the status
    408 in the attribute status_code.
    """

    def __init__(self) -> None:
        super().__init__(request=ADDRESS)
        self.status_code = 408


class APIConnectionError(openai.APIConnectionError):
    """Shaped as litellm.APIConnectionError (exceptions.py, line 901): the SDK's connection
    error, with the status 500 in the attribute status_code.
    """

    def __init__(self) -> None:
        super().__init__(request=ADDRESS)
        self.status_code = 500


class UnsupportedParamsError(openai.BadRequestError):
    """Shaped as litellm.UnsupportedParamsError (exceptions.py, line 993): it sets its status
    and never calls the SDK's constructor, so it has no response or body.
    """

    def __init__(self) -> None:
        self.status_code = 400


@pytest.fixture
def litellm_exceptions(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Registers a module litellm.exceptions of the classes the adapter reads by class, as an
    imported LiteLLM would hold them.
    """
    module = types.ModuleType("litellm.exceptions")
    module.__dict__.update(
        {
            "ContextWindowExceededError": ContextWindowExceededError,
            "ContentPolicyViolationError": ContentPolicyViolationError,
            "ImageFetchError": ImageFetchError,
            "GuardrailRaisedException": GuardrailRaisedException,
            "BlockedPiiEntityError": BlockedPiiEntityError,
            "BudgetExceededError": BudgetExceededError,
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.exceptions",
        module,
    )
    return module


def fail(
    error: type[openai.APIStatusError],
    status: int,
    *,
    body: object = None,
) -> openai.APIStatusError:
    """An OpenAI status error as the SDK builds it."""
    return error(
        "stand-in",
        response=httpx2.Response(
            status,
            request=ADDRESS,
        ),
        body=body,
    )


# The rules, for the errors litellm/exceptions.py at v1.106.5 raises (exception_mapping_utils.py,
# _map_openai_exception and _map_exception_by_status): a 408, 409 or 429 and a 5xx are no answer
# now, with the status; a connection error, a timeout and an invalid body are no answer with no
# status, whatever status attribute LiteLLM gives them.
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            fail(
                openai.RateLimitError,
                429,
            ),
            Unavailable(status=429),
        ),
        (
            fail(
                openai.InternalServerError,
                500,
            ),
            Unavailable(status=500),
        ),
        (
            fail(
                openai.APIStatusError,
                502,
            ),
            Unavailable(status=502),
        ),
        (
            fail(
                openai.APIStatusError,
                503,
            ),
            Unavailable(status=503),
        ),
        (
            fail(
                openai.APIStatusError,
                408,
            ),
            Unavailable(status=408),
        ),
        (
            fail(
                openai.ConflictError,
                409,
            ),
            Unavailable(status=409),
        ),
        (
            APIError(529),
            Unavailable(status=529),
        ),
        (
            APIError(409),
            Unavailable(status=409),
        ),
        (
            Timeout(),
            Unavailable(status=None),
        ),
        (
            APIConnectionError(),
            Unavailable(status=None),
        ),
        (
            openai.APIResponseValidationError(
                httpx2.Response(
                    500,
                    request=ADDRESS,
                ),
                None,
            ),
            Unavailable(status=None),
        ),
        (
            openai.APIError(
                "stand-in",
                ADDRESS,
                body=None,
            ),
            Unavailable(status=None),
        ),
        (
            UnicodeEncodeError(
                "utf-8",
                "\ud800",
                0,
                1,
                "surrogates not allowed",
            ),
            Unavailable(status=None),
        ),
    ],
    ids=[
        "429",
        "500",
        "502",
        "503",
        "408",
        "409",
        "litellm-api-error-529",
        "litellm-api-error-409",
        "timeout",
        "connection-error",
        "response-validation-error",
        "api-error-with-no-status",
        "text-with-no-utf-8-form",
    ],
)
def test_an_error_that_may_pass_is_unavailable_with_its_status_if_it_has_one(
    ask: Ask,
    error: Exception,
    expected: Unavailable,
) -> None:
    exchange = ask(error)

    assert exchange.outcome == expected


# The rules of OpenAI's Responses adapter: a 413, and a code that says the prompt cannot be used,
# are rejected with the status. An error that LiteLLM raises as ContextWindowExceededError or
# ImageFetchError, with status 400 and no code, is rejected with 400.
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            fail(
                openai.BadRequestError,
                413,
            ),
            413,
        ),
        (
            APIError(413),
            413,
        ),
        (
            fail(
                openai.BadRequestError,
                400,
                body={"code": "context_length_exceeded"},
            ),
            400,
        ),
        (
            fail(
                openai.BadRequestError,
                400,
                body={"code": "invalid_prompt"},
            ),
            400,
        ),
        (
            fail(
                openai.BadRequestError,
                400,
                body={"code": "invalid_image_url"},
            ),
            400,
        ),
        (
            fail(
                ContextWindowExceededError,
                400,
            ),
            400,
        ),
        (
            fail(
                ImageFetchError,
                400,
            ),
            400,
        ),
    ],
    ids=[
        "413",
        "litellm-api-error-413",
        "context-length-code",
        "invalid-prompt-code",
        "image-code",
        "context-window-exceeded-error",
        "image-fetch-error",
    ],
)
def test_a_prompt_the_api_cannot_use_is_rejected_with_its_status(
    ask: Ask,
    litellm_exceptions: types.ModuleType,
    error: Exception,
    status: int,
) -> None:
    exchange = ask(error)

    assert exchange.outcome == Rejected(status=status)


@pytest.mark.parametrize(
    "error",
    [
        fail(
            openai.BadRequestError,
            400,
            body={"code": "content_policy_violation"},
        ),
        fail(
            openai.BadRequestError,
            400,
            body={"code": "refusal"},
        ),
        fail(
            ContentPolicyViolationError,
            400,
        ),
        GuardrailRaisedException("blocked"),
        BlockedPiiEntityError("blocked"),
    ],
    ids=[
        "policy-code",
        "refusal-code",
        "content-policy-violation-error",
        "guardrail-raised-exception",
        "blocked-pii-entity-error",
    ],
)
def test_content_that_a_policy_declined_is_a_refusal_without_an_explanation(
    ask: Ask,
    litellm_exceptions: types.ModuleType,
    error: Exception,
) -> None:
    exchange = ask(error)

    assert exchange.outcome == Refused(explanation=None)


def test_a_spent_budget_is_unavailable_without_a_status(
    ask: Ask,
    litellm_exceptions: types.ModuleType,
) -> None:
    exchange = ask(BudgetExceededError("Budget has been exceeded!"))

    assert exchange.outcome == Unavailable(status=None)


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            fail(
                openai.BadRequestError,
                400,
            ),
            400,
        ),
        (
            fail(
                openai.AuthenticationError,
                401,
            ),
            401,
        ),
        (
            fail(
                openai.BadRequestError,
                402,
            ),
            402,
        ),
        (
            fail(
                openai.PermissionDeniedError,
                403,
            ),
            403,
        ),
        (
            fail(
                openai.NotFoundError,
                404,
            ),
            404,
        ),
        (
            fail(
                openai.UnprocessableEntityError,
                422,
            ),
            422,
        ),
        (
            APIError(403),
            403,
        ),
        (
            UnsupportedParamsError(),
            400,
        ),
    ],
    ids=[
        "400",
        "401",
        "402",
        "403",
        "404",
        "422",
        "litellm-api-error-403",
        "unsupported-params-error",
    ],
)
def test_a_rejected_request_raises_naming_only_its_status(
    ask: Ask,
    error: Exception,
    status: int,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask(error)

    assert str(raised.value) == f"LiteLLM rejected the request with HTTP status {status}"


def test_a_rejected_request_does_not_carry_the_providers_error(ask: Ask) -> None:
    with pytest.raises(ConfigurationError) as raised:
        ask(
            fail(
                openai.AuthenticationError,
                401,
            ),
        )

    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.parametrize(
    "error",
    [
        ValueError("model param not passed in."),
        Exception("plain"),
    ],
    ids=["value-error", "plain-exception"],
)
def test_an_error_that_is_not_an_api_error_leaves_the_function_as_it_is(
    ask: Ask,
    litellm_exceptions: types.ModuleType,
    error: Exception,
) -> None:
    with pytest.raises(type(error)) as raised:
        ask(error)

    assert raised.value is error


@pytest.mark.parametrize(
    "error",
    [
        GuardrailRaisedException("blocked"),
        BlockedPiiEntityError("blocked"),
        BudgetExceededError("Budget has been exceeded!"),
    ],
    ids=[
        "guardrail-raised-exception",
        "blocked-pii-entity-error",
        "budget-exceeded-error",
    ],
)
def test_an_error_is_read_as_a_litellm_class_only_when_litellm_is_imported(
    ask: Ask,
    error: Exception,
) -> None:
    with pytest.raises(type(error)) as raised:
        ask(error)

    assert raised.value is error
