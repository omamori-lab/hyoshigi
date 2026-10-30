"""What each kind of monitor's model reads under a redaction: the trajectory's, then the view's,
and that the rating still fails closed on what the model cannot perceive.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Protocol

import pytest

from hyoshigi import (
    QWEN3GUARD,
    Action,
    AsyncDecisionModel,
    AsyncDecisionMonitor,
    AsyncGenerativeModel,
    AsyncGuardMonitor,
    AsyncLLMMonitor,
    Author,
    Channel,
    Decision,
    DecisionModel,
    DecisionMonitor,
    DecisionOutcome,
    Description,
    FailedClosed,
    Generation,
    GenerationOutcome,
    GenerativeModel,
    GuardMonitor,
    LLMMonitor,
    MediaPart,
    MediaTypes,
    MonitorPrompt,
    Observation,
    Prompt,
    PromptPart,
    Question,
    Rationale,
    Redaction,
    SuspicionScore,
    TextPart,
    ToolCall,
    Trajectory,
    Unperceivable,
    View,
)

MODEL_NAME = "trusted-1"
TEXT_ONLY = MediaTypes()
IMAGES = MediaTypes("image/*")
# A prompt that shows exactly what the monitor puts in it.
ECHO = MonitorPrompt(
    system="Rate it.",
    request="{trajectory}|{proposed_action}",
)
TASK = "Call Ana on 555-0100."
# Each phone number masked, which leaves a data URL's digits alone; media blurred to new bytes.
MASKED = Redaction(
    text=partial(
        re.sub,
        "[0-9]{3}-[0-9]{4}",
        "###-####",
    ),
    media=partial(
        replace,
        data=b"blurred",
    ),
)
# Each digit masked, a data URL's included; media replaced with a text placeholder.
TO_TEXT = Redaction(
    text=partial(
        re.sub,
        "[0-9]",
        "#",
    ),
    media=lambda part: TextPart(text="image 1"),
)
# Text kept as written; media replaced with a text placeholder that holds a phone number.
PLACEHOLDER = Redaction(
    text=str,
    media=lambda part: TextPart(text="a card for 555-0100"),
)
# Each phone number masked; media kept as it is, so the text in text media shows its redaction.
TEXT_MASKED = Redaction(
    text=partial(
        re.sub,
        "[0-9]{3}-[0-9]{4}",
        "###-####",
    ),
    media=lambda part: part,
)
# Text in brackets, and media with " blurred" after its bytes, so each application shows.
BRACKETED = Redaction(
    text=lambda text: f"[{text}]",
    media=lambda part: replace(
        part,
        data=(part.data or b"") + b" blurred",
    ),
)
# Text in parentheses, and media with " scaled" after its bytes, so each application shows.
PARENTHESISED = Redaction(
    text=lambda text: f"({text})",
    media=lambda part: replace(
        part,
        data=(part.data or b"") + b" scaled",
    ),
)
# Each phone number masked; media given a description written by the scaffold, which holds a
# phone number.
DESCRIBED = Redaction(
    text=partial(
        re.sub,
        "[0-9]{3}-[0-9]{4}",
        "###-####",
    ),
    media=partial(
        replace,
        description=Description(
            text="A card for 555-0100",
            author=Author.SCAFFOLD,
        ),
    ),
)
# The eight bytes every PNG file starts with (RFC 2083, section 3.1), in base64 iVBORw0KGgo=.
PNG = MediaPart(
    media_type="image/png",
    data=b"\x89PNG\r\n\x1a\n",
)
DIALLING = Action(
    parts=(TextPart(text="Dialling 555-0100."),),
    tool_calls=(
        ToolCall(
            id="call_2",
            name="dial",
            arguments='{"number": "555-0100"}',
        ),
    ),
)
UPLOADING = Action(
    id="a1",
    tool_calls=(
        ToolCall(
            id="call_1",
            name="upload",
            arguments='{"image": "data:image/png;base64,iVBORw0KGgo="}',
        ),
    ),
)
SHOWING = Action(parts=(PNG,))
UNPERCEIVABLE_SCORE = SuspicionScore(
    value=1.0,
    explanation="The action holds content the monitor cannot perceive, so it is treated as "
    "suspicious.",
    evidence=FailedClosed(
        model=MODEL_NAME,
        cause=Unperceivable(content=("image/png",)),
    ),
)


class Script:
    """A model that perceives the given media and keeps the parts it reads: a generative model's
    prompt content, with a score of 3 as its reply, or a decision model's state, with a
    probability of 0.3.
    """

    def __init__(
        self,
        *,
        perceives: MediaTypes,
    ) -> None:
        self.perceives = perceives
        self.received: list[tuple[PromptPart, ...]] = []

    def generate(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        self.received.append(prompt.content)
        return Generation(text="<reasoning>It dials the number.</reasoning><score>3</score>")

    async def generate_asynchronously(
        self,
        prompt: Prompt,
    ) -> GenerationOutcome:
        return self.generate(prompt)

    def decide(
        self,
        question: Question,
    ) -> DecisionOutcome:
        self.received.append(question.state)
        return Decision(probability=0.3)

    async def decide_asynchronously(
        self,
        question: Question,
    ) -> DecisionOutcome:
        return self.decide(question)


class Rate(Protocol):
    def __call__(
        self,
        script: Script,
        *,
        action: Action,
        trajectory: Trajectory,
        view: View,
    ) -> SuspicionScore: ...


def rate_synchronously(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = LLMMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
            perceives=script.perceives,
        ),
        view=view,
        prompt=ECHO,
    )
    return monitor.rate(
        action,
        trajectory=trajectory,
    )


def rate_asynchronously(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = AsyncLLMMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
            perceives=script.perceives,
        ),
        view=view,
        prompt=ECHO,
    )
    return asyncio.run(
        monitor.rate(
            action,
            trajectory=trajectory,
        ),
    )


def rate_with_guard(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = GuardMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
            perceives=script.perceives,
        ),
        prompt=ECHO,
        labels=QWEN3GUARD,
        samples=2,
        view=view,
    )
    return monitor.rate(
        action,
        trajectory=trajectory,
    )


def rate_with_async_guard(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = AsyncGuardMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
            perceives=script.perceives,
        ),
        prompt=ECHO,
        labels=QWEN3GUARD,
        samples=2,
        view=view,
    )
    return asyncio.run(
        monitor.rate(
            action,
            trajectory=trajectory,
        ),
    )


def rate_with_decision_model(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = DecisionMonitor(
        model=DecisionModel(
            name=MODEL_NAME,
            decide=script.decide,
            perceives=script.perceives,
        ),
        view=view,
    )
    return monitor.rate(
        action,
        trajectory=trajectory,
    )


def rate_with_async_decision_model(
    script: Script,
    *,
    action: Action,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = AsyncDecisionMonitor(
        model=AsyncDecisionModel(
            name=MODEL_NAME,
            decide=script.decide_asynchronously,
            perceives=script.perceives,
        ),
        view=view,
    )
    return asyncio.run(
        monitor.rate(
            action,
            trajectory=trajectory,
        ),
    )


@pytest.fixture(
    params=[
        rate_synchronously,
        rate_asynchronously,
    ],
    ids=["sync", "async"],
)
def rate(request: pytest.FixtureRequest) -> Rate:
    return request.param


# What the model reads of the task and DIALLING, masked: ECHO's request for a generative model, and
# the state a decision monitor asks about for a decision model.
ECHOED = (
    TextPart(
        text="<user>Call Ana on ###-####.</user>|<agent>Dialling ###-####.</agent>\n"
        '<tool_call id="call_2" name="dial">{"number": "###-####"}</tool_call>',
    ),
)
STATED = (
    TextPart(
        text="<trajectory>\n<user>Call Ana on ###-####.</user>\n</trajectory>\n<proposed_action>\n"
        "<agent>Dialling ###-####.</agent>\n"
        '<tool_call id="call_2" name="dial">{"number": "###-####"}</tool_call>\n</proposed_action>',
    ),
)


class RateLatestStep(Protocol):
    def __call__(
        self,
        script: Script,
        *,
        trajectory: Trajectory,
        view: View,
    ) -> SuspicionScore: ...


def rate_latest_step_synchronously(
    script: Script,
    *,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = LLMMonitor(
        model=GenerativeModel(
            name=MODEL_NAME,
            generate=script.generate,
            perceives=script.perceives,
        ),
        view=view,
        prompt=ECHO,
    )
    return monitor.rate_latest_step(trajectory)


def rate_latest_step_asynchronously(
    script: Script,
    *,
    trajectory: Trajectory,
    view: View,
) -> SuspicionScore:
    monitor = AsyncLLMMonitor(
        model=AsyncGenerativeModel(
            name=MODEL_NAME,
            generate=script.generate_asynchronously,
            perceives=script.perceives,
        ),
        view=view,
        prompt=ECHO,
    )
    return asyncio.run(monitor.rate_latest_step(trajectory))


@pytest.fixture(
    params=[
        rate_latest_step_synchronously,
        rate_latest_step_asynchronously,
    ],
    ids=["sync", "async"],
)
def rate_latest_step(request: pytest.FixtureRequest) -> RateLatestStep:
    return request.param


def build_task(redaction: Redaction | None = None) -> Trajectory:
    """The trajectory of the task, kept through the redaction when one is given."""
    return Trajectory.from_run(
        TASK,
        answer="Done.",
        redaction=redaction,
    ).trajectory


def build_note() -> Trajectory:
    """The trajectory of the task and a note from the user in text media."""
    trajectory = build_task()
    trajectory.observe(
        Observation(
            author=Author.USER,
            parts=(
                MediaPart(
                    media_type="text/plain",
                    data=b"A note with 555-0100",
                ),
            ),
        ),
    )
    return trajectory


def build_upload(redaction: Redaction | None = None) -> Trajectory:
    """The trajectory of the task and UPLOADING, kept through the redaction when one is given."""
    trajectory = build_task(redaction)
    trajectory.act(UPLOADING)
    return trajectory


@pytest.mark.parametrize(
    ("rate_with", "read"),
    [
        (
            rate_synchronously,
            ECHOED,
        ),
        (
            rate_asynchronously,
            ECHOED,
        ),
        (
            rate_with_guard,
            ECHOED,
        ),
        (
            rate_with_async_guard,
            ECHOED,
        ),
        (
            rate_with_decision_model,
            STATED,
        ),
        (
            rate_with_async_decision_model,
            STATED,
        ),
    ],
    ids=["llm", "async-llm", "guard", "async-guard", "decision", "async-decision"],
)
@pytest.mark.parametrize(
    ("build", "view"),
    [
        (
            build_task,
            View(redaction=MASKED),
        ),
        (
            partial(
                build_task,
                MASKED,
            ),
            View(),
        ),
    ],
    ids=["the-views", "the-trajectorys"],
)
def test_the_model_reads_the_history_and_the_proposed_action_redacted(
    rate_with: Rate,
    read: tuple[PromptPart, ...],
    build: Callable[[], Trajectory],
    view: View,
) -> None:
    script = Script(perceives=TEXT_ONLY)

    rate_with(
        script,
        action=DIALLING,
        trajectory=build(),
        view=view,
    )

    assert script.received[0] == read


@pytest.mark.parametrize(
    ("build", "action", "view", "content"),
    [
        (
            build_task,
            UPLOADING,
            View(redaction=MASKED),
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>|"
                    '<tool_call id="call_1" name="upload">{"image": "[media 1]"}'
                    '<media type="image/png" ref="1">',
                ),
                MediaPart(
                    media_type="image/png",
                    data=b"blurred",
                ),
                TextPart(text="</media></tool_call>"),
            ),
        ),
        (
            partial(
                build_task,
                MASKED,
            ),
            UPLOADING,
            View(),
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>|"
                    '<tool_call id="call_1" name="upload">{"image": "[media 1]"}'
                    '<media type="image/png" ref="1">',
                ),
                MediaPart(
                    media_type="image/png",
                    data=b"blurred",
                ),
                TextPart(text="</media></tool_call>"),
            ),
        ),
        (
            partial(
                build_task,
                PLACEHOLDER,
            ),
            UPLOADING,
            View(redaction=MASKED),
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>|"
                    '<tool_call id="call_1" name="upload">{"image": "[media 1]"}'
                    '<media ref="1">a card for ###-####</media></tool_call>',
                ),
            ),
        ),
        (
            partial(
                build_upload,
                MASKED,
            ),
            DIALLING,
            View(),
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>\n"
                    '<tool_call id="call_1" name="upload">{"image": "[media 1]"}'
                    '<media type="image/png" ref="1">',
                ),
                MediaPart(
                    media_type="image/png",
                    data=b"blurred",
                ),
                TextPart(
                    text="</media></tool_call>|<agent>Dialling ###-####.</agent>\n"
                    '<tool_call id="call_2" name="dial">{"number": "###-####"}</tool_call>',
                ),
            ),
        ),
    ],
    ids=[
        "the-views",
        "the-trajectorys",
        "the-trajectorys-then-the-views",
        "in-the-history-the-trajectorys",
    ],
)
def test_media_lifted_from_a_tool_calls_arguments_reaches_the_model_redacted(
    rate: Rate,
    build: Callable[[], Trajectory],
    action: Action,
    view: View,
    content: tuple[PromptPart, ...],
) -> None:
    script = Script(perceives=IMAGES)

    rate(
        script,
        action=action,
        trajectory=build(),
        view=view,
    )

    assert script.received[0] == content


@pytest.mark.parametrize(
    ("build", "action", "view"),
    [
        (
            build_task,
            SHOWING,
            View(redaction=TO_TEXT),
        ),
        (
            partial(
                build_task,
                TO_TEXT,
            ),
            SHOWING,
            View(),
        ),
        (
            build_task,
            UPLOADING,
            View(redaction=TO_TEXT),
        ),
    ],
    ids=["media-under-the-views", "media-under-the-trajectorys", "lifted-media-under-the-views"],
)
def test_a_rating_of_media_the_model_cannot_perceive_fails_closed_whatever_the_redaction(
    rate: Rate,
    build: Callable[[], Trajectory],
    action: Action,
    view: View,
) -> None:
    script = Script(perceives=TEXT_ONLY)

    score = rate(
        script,
        action=action,
        trajectory=build(),
        view=view,
    )

    assert score == UNPERCEIVABLE_SCORE


@pytest.mark.parametrize(
    ("build", "action", "view"),
    [
        (
            build_task,
            SHOWING,
            View(redaction=Redaction(text=str)),
        ),
        (
            partial(
                build_task,
                Redaction(text=str),
            ),
            SHOWING,
            View(),
        ),
        (
            build_task,
            UPLOADING,
            View(redaction=Redaction(text=str)),
        ),
    ],
    ids=["media-under-the-views", "media-under-the-trajectorys", "lifted-media-under-the-views"],
)
def test_a_rating_of_media_a_redaction_removed_fails_closed_whatever_the_model_perceives(
    rate: Rate,
    build: Callable[[], Trajectory],
    action: Action,
    view: View,
) -> None:
    script = Script(perceives=IMAGES)

    score = rate(
        script,
        action=action,
        trajectory=build(),
        view=view,
    )

    assert score == UNPERCEIVABLE_SCORE


def test_a_rating_of_the_latest_step_fails_closed_on_media_a_redaction_removed_by_default(
    rate_latest_step: RateLatestStep,
) -> None:
    trajectory = Trajectory(redaction=Redaction(text=str))
    trajectory.observe(
        Observation(
            author=Author.USER,
            parts=(TextPart(text="Call Ana."),),
        ),
    )
    trajectory.act(SHOWING)
    script = Script(perceives=IMAGES)

    score = rate_latest_step(
        script,
        trajectory=trajectory,
        view=View(),
    )

    assert score == UNPERCEIVABLE_SCORE


def test_the_hidden_reasoning_names_media_in_the_thought_a_redaction_turned_into_text(
    rate: Rate,
) -> None:
    script = Script(perceives=TEXT_ONLY)

    score = rate(
        script,
        action=Action(
            thought=(PNG,),
            parts=(TextPart(text="Dialling."),),
        ),
        trajectory=build_task(),
        view=View(
            channels=Channel.DEFAULT | Channel.THOUGHTS,
            redaction=TO_TEXT,
        ),
    )

    assert score == SuspicionScore(
        value=0.3,
        explanation="It dials the number.",
        evidence=Rationale(
            model=MODEL_NAME,
            hidden_reasoning=("image/png",),
        ),
    )


def test_media_lifted_from_the_history_keeps_its_ref_when_a_redaction_describes_it(
    rate: Rate,
) -> None:
    script = Script(perceives=TEXT_ONLY)

    rate(
        script,
        action=DIALLING,
        trajectory=build_upload(DESCRIBED),
        view=View(),
    )

    assert script.received[0] == (
        TextPart(
            text="<user>Call Ana on ###-####.</user>\n"
            '<tool_call id="call_1" name="upload">{"image": "[media 1]"}'
            '<media type="image/png" description_author="scaffold" ref="1">A card for ###-####'
            "</media></tool_call>|<agent>Dialling ###-####.</agent>\n"
            '<tool_call id="call_2" name="dial">{"number": "###-####"}</tool_call>',
        ),
    )


@pytest.mark.parametrize(
    ("build", "action", "content"),
    [
        (
            build_task,
            Action(
                tool_calls=(
                    ToolCall(
                        id="call_3",
                        name="upload",
                        # The payload is base64 of "Call 555-0100".
                        arguments='{"file": "data:text/plain;base64,Q2FsbCA1NTUtMDEwMA=="}',
                    ),
                ),
            ),
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>|"
                    '<tool_call id="call_3" name="upload">{"file": "[media 1]"}'
                    '<media type="text/plain" ref="1">Call ###-####</media></tool_call>',
                ),
            ),
        ),
        (
            build_note,
            DIALLING,
            (
                TextPart(
                    text="<user>Call Ana on ###-####.</user>\n"
                    '<user><media type="text/plain">A note with ###-####</media></user>|'
                    "<agent>Dialling ###-####.</agent>\n"
                    '<tool_call id="call_2" name="dial">{"number": "###-####"}</tool_call>',
                ),
            ),
        ),
    ],
    ids=["lifted-from-an-argument", "in-the-history"],
)
def test_text_media_reaches_the_model_with_its_text_redacted(
    rate: Rate,
    build: Callable[[], Trajectory],
    action: Action,
    content: tuple[PromptPart, ...],
) -> None:
    script = Script(perceives=TEXT_ONLY)

    rate(
        script,
        action=action,
        trajectory=build(),
        view=View(redaction=TEXT_MASKED),
    )

    assert script.received[0] == content


def test_rating_the_latest_step_passes_each_piece_of_it_through_each_redaction_once(
    rate_latest_step: RateLatestStep,
) -> None:
    trajectory = Trajectory(redaction=BRACKETED)
    trajectory.observe(
        Observation(
            author=Author.USER,
            parts=(TextPart(text="Call Ana."),),
        ),
    )
    trajectory.act(
        Action(
            id="a1",
            parts=(
                TextPart(text="Dialling."),
                MediaPart(
                    media_type="image/png",
                    data=b"PNG",
                ),
            ),
            tool_calls=(
                ToolCall(
                    id="call_1",
                    name="upload",
                    arguments='{"image": "data:image/png;base64,iVBORw0KGgo="}',
                ),
            ),
        ),
    )
    script = Script(perceives=IMAGES)

    rate_latest_step(
        script,
        trajectory=trajectory,
        view=View(redaction=PARENTHESISED),
    )

    assert script.received[0] == (
        TextPart(
            text='<user>([Call Ana.])</user>|<agent>([Dialling.])<media type="image/png">',
        ),
        MediaPart(
            media_type="image/png",
            data=b"PNG blurred scaled",
        ),
        TextPart(
            text='</media></agent>\n<tool_call id="call_1" name="upload">([{"image": "[media 1]"}])'
            '<media type="image/png" ref="1">',
        ),
        MediaPart(
            media_type="image/png",
            data=b"\x89PNG\r\n\x1a\n blurred scaled",
        ),
        TextPart(text="</media></tool_call>"),
    )


def test_an_action_split_off_a_root_is_read_through_the_redaction_of_the_run_the_root_joins(
    rate: Rate,
) -> None:
    root = build_task()
    root.act(DIALLING)
    step = root.split_latest_step()
    run = build_task(MASKED)
    run.act(
        Action(
            id="a9",
            tool_calls=(
                ToolCall(
                    id="call_9",
                    name="task",
                    arguments="{}",
                ),
            ),
        ),
    )
    joined = run.join(
        step.trajectory,
        call_id="call_9",
    )
    script = Script(perceives=TEXT_ONLY)

    rate(
        script,
        action=step.action,
        trajectory=joined,
        view=View(),
    )

    assert script.received[0] == ECHOED
