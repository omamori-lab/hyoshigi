"""Text the agent writes opens and closes no tag of the rendering."""

from html.parser import HTMLParser
from typing import NamedTuple

from hypothesis import given
from hypothesis import strategies as st

from hyoshigi import (
    Action,
    Author,
    Description,
    MediaPart,
    MediaTypes,
    Observation,
    TextPart,
    ToolCall,
    ToolObservation,
    Trajectory,
)
from hyoshigi.monitors.rendering import render
from hyoshigi.monitors.view import Channel, View

EVERY_CHANNEL = (
    Channel.TASK
    | Channel.CONTEXT
    | Channel.THOUGHTS
    | Channel.AGENT_PARTS
    | Channel.TOOL_CALLS
    | Channel.TOOL_OBSERVATIONS
    | Channel.SUBAGENT_REPORTS
)
# The rendering holds six elements, each with an opening and a closing tag.
TAGS = 12
# Pieces of markup, among them tags of the rendering.
MARKUP = [
    "<",
    ">",
    "&",
    '"',
    "'",
    "/",
    "</thought>",
    "</agent>",
    "</media>",
    "</tool_call>",
    "</tool_result>",
    "<user>",
    '<tool_call id="x" name="y">',
    "&lt;",
    "&amp;",
    "<!--",
    "-->",
    "<![CDATA[",
    "]]>",
    "a",
]
HOSTILE_TEXT = st.one_of(
    st.text(min_size=1),
    st.lists(
        st.sampled_from(MARKUP),
        min_size=1,
    ).map("".join),
)


class Texts(NamedTuple):
    """The strings that the agent or its tools write."""

    task: str
    result_call_id: str
    result: str
    thought: str
    said: str
    description: str
    call_id: str
    name: str
    arguments: str


HOSTILE_TEXTS = st.builds(
    Texts,
    *([HOSTILE_TEXT] * len(Texts._fields)),
)


class TagReader(HTMLParser):
    """Reads the tags, attributes and text of a rendering in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[tuple[str, ...]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.found.append(
            (
                "start",
                tag,
                *(f"{name}={value}" for name, value in attrs),
            ),
        )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        self.found.append(
            (
                "end",
                tag,
            ),
        )

    def handle_data(
        self,
        data: str,
    ) -> None:
        self.found.append(
            (
                "text",
                data,
            ),
        )


def render_texts(texts: Texts) -> str:
    """Render the texts through every channel."""
    trajectory = Trajectory()
    trajectory.observe(
        Observation(
            author=Author.USER,
            parts=(TextPart(text=texts.task),),
        ),
    )
    trajectory.observe(
        ToolObservation(
            call_id=texts.result_call_id,
            parts=(TextPart(text=texts.result),),
        ),
    )
    action = Action(
        thought=(TextPart(text=texts.thought),),
        parts=(
            TextPart(text=texts.said),
            MediaPart(
                media_type="image/png",
                data=b"png",
                description=Description(
                    text=texts.description,
                    author=Author.AGENT,
                ),
            ),
        ),
        tool_calls=(
            ToolCall(
                id=texts.call_id,
                name=texts.name,
                arguments=texts.arguments,
            ),
        ),
    )
    rendering = render(
        action,
        trajectory=trajectory,
        view=View(channels=EVERY_CHANNEL),
        perceives=MediaTypes(),
    )
    return "".join(
        part.text
        for part in (
            *rendering.trajectory,
            *rendering.proposed_action,
        )
        if isinstance(
            part,
            TextPart,
        )
    )


def read_tags(rendered: str) -> list[tuple[str, ...]]:
    reader = TagReader()
    reader.feed(rendered)
    reader.close()
    return reader.found


@given(texts=HOSTILE_TEXTS)
def test_text_the_agent_controls_opens_and_closes_no_tag(texts: Texts) -> None:
    found = read_tags(render_texts(texts))

    assert found == [
        ("start", "user"),
        (
            "text",
            texts.task,
        ),
        ("end", "user"),
        ("text", "\n"),
        (
            "start",
            "tool_result",
            f"call_id={texts.result_call_id}",
        ),
        (
            "text",
            texts.result,
        ),
        ("end", "tool_result"),
        ("start", "thought"),
        (
            "text",
            texts.thought,
        ),
        ("end", "thought"),
        ("text", "\n"),
        ("start", "agent"),
        (
            "text",
            texts.said,
        ),
        ("start", "media", "type=image/png", "description_author=agent"),
        (
            "text",
            texts.description,
        ),
        ("end", "media"),
        ("end", "agent"),
        ("text", "\n"),
        (
            "start",
            "tool_call",
            f"id={texts.call_id}",
            f"name={texts.name}",
        ),
        (
            "text",
            texts.arguments,
        ),
        ("end", "tool_call"),
    ]


@given(texts=HOSTILE_TEXTS)
def test_the_only_angle_brackets_in_a_rendering_are_those_of_its_tags(texts: Texts) -> None:
    rendered = render_texts(texts)

    assert rendered.count("<") == rendered.count(">") == TAGS
