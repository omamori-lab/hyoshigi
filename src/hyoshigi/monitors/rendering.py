"""A trajectory and the action a monitor rates, rendered as the parts its model reads.

Each entry is one element: an observation, or an action's thought, its parts or one of
its tool calls. Text inside an element is HTML-escaped, so a tool result cannot close a
tag and pose as another entry. Text media is its escaped text, and other media the model
perceives sits in place among the text; the rest is its description, tagged with who wrote
it, or a placeholder. The view's redaction applies to the history, which the trajectory keeps
through its own; the trajectory's redaction, then the view's, applies to the rated action and to
media lifted from a tool call's arguments. An action split off the trajectory, which kept it
redacted, is redacted by the view's alone, and its lifted media by both.

<https://omamori-lab.github.io/hyoshigi/how-to/redacting-what-leaves-the-process/>
"""

import html
import re
from base64 import b64decode
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import assert_never

from hyoshigi.models.model import MediaTypes, PromptPart
from hyoshigi.monitors.view import Channel, View
from hyoshigi.trajectory.history import Action, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.parts import (
    MEDIA_TYPE,
    Author,
    MediaPart,
    OpaquePart,
    Part,
    TextPart,
    read_text,
)
from hyoshigi.trajectory.redaction import Redactor
from hyoshigi.trajectory.trajectory import Trajectory, is_split_off

# RFC 2045, section 5.1: a parameter is an attribute and a value joined by =, each one or
# more US-ASCII characters except SPACE, controls and the tspecials ( ) < > @ , ; : \ " / [ ] ? =
_PARAMETER_CHARACTER = r"[!#$%&'*+.^_`{|}~0-9A-Za-z-]"
_PARAMETER = f";{_PARAMETER_CHARACTER}+={_PARAMETER_CHARACTER}+"
# A JSON string that is a whole data URL of base64 data (RFC 2397, section 3): a media
# type, parameters, then ;base64. The scheme is case-insensitive (RFC 3986, section 3.1),
# as are the parameter names (RFC 2045, section 5.1), and so ;base64. The URL is matched
# in the raw arguments, so arguments that are not valid JSON are read too.
_DATA_URL = re.compile(
    f'"data:(?P<media_type>{MEDIA_TYPE.pattern})(?:{_PARAMETER})*;base64,(?P<data>[^"]*)"',
    re.ASCII | re.IGNORECASE,
)


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class Rendering:
    """The history as a view shows it, the rated action, and what of the action is unseen.

    `hidden_reasoning` names the kinds of the rated thought's opaque parts and the media
    types of its media not shown, or shown only as the agent's own description, and is empty
    when the view does not show the thought. `unperceivable` names the same of the action's
    parts and tool calls. Both are found in the action before any redaction, and
    `unperceivable`, when that holds none, in the action as redacted.
    """

    trajectory: tuple[PromptPart, ...]
    proposed_action: tuple[PromptPart, ...]
    hidden_reasoning: tuple[str, ...]
    unperceivable: tuple[str, ...]


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class _Content:
    """Rendered parts, and the opaque kinds and media types among them left unseen."""

    parts: tuple[PromptPart, ...]
    unseen: tuple[str, ...] = ()


@dataclass(
    frozen=True,
    slots=True,
    kw_only=True,
)
class _Entry:
    """One rendered element of a trajectory, and the channel it belongs to."""

    channel: Channel
    content: _Content


def render(
    action: Action,
    *,
    trajectory: Trajectory,
    view: View,
    perceives: MediaTypes,
) -> Rendering:
    """Render the trajectory through the view, and the action rated against it.

    The action's parts and tool calls are shown whatever the view, and its thought when
    the view shows thoughts; `most_recent` limits only the history.
    """
    redactor = Redactor(
        trajectory.redaction,
        view.redaction,
    )
    history = _select(
        list(
            _render_history(
                trajectory,
                view=view,
                perceives=perceives,
                redactor=redactor,
            ),
        ),
        view=view,
    )
    shown = view.channels | Channel.AGENT_PARTS | Channel.TOOL_CALLS
    # The trajectory kept an action split off it through its redaction already.
    entry_redaction = (
        None
        if is_split_off(
            action,
            trajectory=trajectory,
        )
        else trajectory.redaction
    )
    proposed = [
        entry
        for entry in _render_action(
            Redactor(
                entry_redaction,
                view.redaction,
            ).redact_action(action),
            perceives=perceives,
            redactor=redactor,
        )
        if entry.channel in shown
    ]
    unredacted = [
        entry
        for entry in _render_action(
            action,
            perceives=perceives,
            redactor=Redactor(),
        )
        if entry.channel in shown
    ]
    return Rendering(
        trajectory=_join(history),
        proposed_action=_join(proposed),
        hidden_reasoning=_collect_unseen(
            unredacted,
            channels=Channel.THOUGHTS,
        ),
        unperceivable=_collect_unseen(
            unredacted,
            channels=Channel.AGENT_PARTS | Channel.TOOL_CALLS,
        )
        or _collect_unseen(
            proposed,
            channels=Channel.AGENT_PARTS | Channel.TOOL_CALLS,
        ),
    )


def merge_text(parts: Iterable[PromptPart]) -> tuple[PromptPart, ...]:
    """Return the parts with each run of adjacent text joined into one text part."""
    merged: list[PromptPart] = []
    run: list[str] = []
    for part in parts:
        match part:
            case TextPart(text=text):
                run.append(text)
            case MediaPart():
                if run:
                    merged.append(TextPart(text="".join(run)))
                    run.clear()
                merged.append(part)
            case unreachable:
                assert_never(unreachable)
    if run:
        merged.append(TextPart(text="".join(run)))
    return tuple(merged)


def _select(
    entries: list[_Entry],
    *,
    view: View,
) -> list[_Entry]:
    """Keep the entries the view shows, only the most recent ones when it limits them.

    When the limit drops older entries, every task entry is kept anyway, in order: the
    first states the task, and a later one can narrow or revoke it. A context note is
    never kept in their place.
    """
    shown = [entry for entry in entries if entry.channel in view.channels]
    if view.most_recent is None:
        return shown
    earlier = [entry for entry in shown[: -view.most_recent] if entry.channel is Channel.TASK]
    return [
        *earlier,
        *shown[-view.most_recent :],
    ]


def _join(entries: list[_Entry]) -> tuple[PromptPart, ...]:
    """Join the entries' parts, one entry per line."""
    parts: list[PromptPart] = []
    for entry in entries:
        if parts:
            parts.append(TextPart(text="\n"))
        parts.extend(entry.content.parts)
    return merge_text(parts)


def _collect_unseen(
    entries: list[_Entry],
    *,
    channels: Channel,
) -> tuple[str, ...]:
    """Return what is unseen in the entries on these channels, in order."""
    return tuple(
        unseen for entry in entries if entry.channel in channels for unseen in entry.content.unseen
    )


def _render_history(
    trajectory: Trajectory,
    *,
    view: View,
    perceives: MediaTypes,
    redactor: Redactor,
) -> Iterator[_Entry]:
    """Yield the entries of the trajectory's history, through the view's redaction, in order;
    its subagents' are not. `redactor` redacts the media lifted from tool calls.
    """
    delegations = set(trajectory.subagents)
    view_redactor = Redactor(view.redaction)
    for event in map(
        view_redactor.redact_event,
        trajectory.history,
    ):
        match event:
            case Action():
                delegations.update(
                    call.id for call in event.tool_calls if call.name in view.delegation_tools
                )
                yield from _render_action(
                    event,
                    perceives=perceives,
                    redactor=redactor,
                )
            case Observation():
                yield _render_observation(
                    event,
                    delegations=delegations,
                    perceives=perceives,
                    channels=view.channels,
                )
            case unreachable:
                assert_never(unreachable)


def _render_action(
    action: Action,
    *,
    perceives: MediaTypes,
    redactor: Redactor,
) -> Iterator[_Entry]:
    """Yield an action's thought, its parts, then each tool call; nothing for what is empty.

    `redactor` redacts the media lifted from the tool calls' arguments.
    """
    if action.thought:
        yield _render_entry(
            action.thought,
            channel=Channel.THOUGHTS,
            tag="thought",
            perceives=perceives,
        )
    if action.parts:
        yield _render_entry(
            action.parts,
            channel=Channel.AGENT_PARTS,
            tag="agent",
            perceives=perceives,
        )
    for call in action.tool_calls:
        yield _Entry(
            channel=Channel.TOOL_CALLS,
            content=_render_tool_call(
                call,
                perceives=perceives,
                redactor=redactor,
            ),
        )


def _render_observation(
    observation: Observation,
    *,
    delegations: set[str],
    perceives: MediaTypes,
    channels: Channel,
) -> _Entry:
    """Render an observation by who wrote it; only the user or the delegator sets the task.

    Without context notes, an unknown author's observation shows only its opaque parts, with
    the tool calls: it may be a tool the agent's provider ran, and the monitor sees its kind.
    """
    opaque = [
        part
        for part in observation.parts
        if isinstance(
            part,
            OpaquePart,
        )
    ]
    match observation:
        case ToolObservation():
            return _render_tool_observation(
                observation,
                delegations=delegations,
                perceives=perceives,
            )
        case Observation(author=Author.USER | Author.DELEGATOR as author):
            return _render_entry(
                observation.parts,
                channel=Channel.TASK,
                tag=author.value,
                perceives=perceives,
            )
        case Observation(author=Author.UNKNOWN) if opaque and Channel.CONTEXT not in channels:
            return _Entry(
                channel=Channel.TOOL_CALLS,
                content=_concatenate(
                    [
                        _render_part(
                            part,
                            channel=Channel.TOOL_CALLS,
                            perceives=perceives,
                        )
                        for part in opaque
                    ],
                ),
            )
        case _:
            return _render_entry(
                observation.parts,
                channel=Channel.CONTEXT,
                tag="context_note",
                perceives=perceives,
                author=observation.author.value,
                source=observation.source,
            )


def _render_tool_observation(
    observation: ToolObservation,
    *,
    delegations: set[str],
    perceives: MediaTypes,
) -> _Entry:
    """Render a tool's result as an error, a subagent's report or a plain result, with an author
    attribute when its author is not the tool.

    A tool error is on the tool calls' channel, so a view that shows a call shows that it
    failed or did not run, even when the call delegated to a subagent.
    """
    if observation.is_error:
        channel, tag = Channel.TOOL_CALLS, "tool_error"
    elif observation.call_id in delegations:
        channel, tag = Channel.SUBAGENT_REPORTS, "subagent_report"
    else:
        channel, tag = Channel.TOOL_OBSERVATIONS, "tool_result"
    return _render_entry(
        observation.parts,
        channel=channel,
        tag=tag,
        perceives=perceives,
        call_id=observation.call_id,
        author=None if observation.author is Author.TOOL else observation.author.value,
    )


def _render_entry(
    parts: tuple[Part, ...],
    *,
    channel: Channel,
    tag: str,
    perceives: MediaTypes,
    **attributes: str | None,
) -> _Entry:
    """Render parts as one element on a channel."""
    content = _concatenate(
        [
            _render_part(
                part,
                channel=channel,
                perceives=perceives,
            )
            for part in parts
        ],
    )
    return _Entry(
        channel=channel,
        content=_wrap(
            tag,
            content=content,
            **attributes,
        ),
    )


def _render_part(
    part: Part,
    *,
    channel: Channel,
    perceives: MediaTypes,
) -> _Content:
    """Render a part; an opaque part in a thought is reasoning the provider hid."""
    match part:
        case TextPart(text=text):
            return _escape(text)
        case MediaPart():
            return _render_media(
                part,
                perceives=perceives,
            )
        case OpaquePart(kind=kind):
            tag = "hidden_reasoning" if channel is Channel.THOUGHTS else "opaque"
            return _mark_unseen(
                tag,
                unseen=kind,
                kind=kind,
            )
        case unreachable:
            assert_never(unreachable)


def _render_media(
    part: MediaPart,
    *,
    perceives: MediaTypes,
    ref: str | None = None,
) -> _Content:
    """Render text media as its escaped text, and other media as itself if the model
    perceives it; otherwise media is its description or a mark.

    Text media is never passed on, so text by URL or in bytes that are not UTF-8 falls to its
    description or a mark. `ref` numbers media lifted from a tool call's arguments.
    """
    if part.media_type.lower().startswith("text/"):
        text = read_text(part)
        if text is None:
            return _describe_media(
                part,
                ref=ref,
            )
        return _wrap(
            "media",
            content=_escape(text),
            type=part.media_type,
            ref=ref,
        )
    if perceives.covers(part.media_type):
        return _wrap(
            "media",
            content=_Content(parts=(part,)),
            type=part.media_type,
            ref=ref,
        )
    return _describe_media(
        part,
        ref=ref,
    )


def _describe_media(
    part: MediaPart,
    *,
    ref: str | None,
) -> _Content:
    """Render media the model is not given as its description, or else as a mark.

    The agent's own description is its claim about the media, so the media stays unseen.
    Media lifted from a tool call's arguments has a description only when a redaction gave it
    one, and keeps its ref either way.
    """
    if part.description is not None:
        description = _escape(part.description.text)
        if part.description.author is Author.AGENT:
            description = replace(
                description,
                unseen=(part.media_type,),
            )
        return _wrap(
            "media",
            content=description,
            type=part.media_type,
            description_author=part.description.author.value,
            ref=ref,
        )
    return _mark_unseen(
        "omitted_media",
        unseen=part.media_type,
        type=part.media_type,
        ref=ref,
    )


def _render_tool_call(
    call: ToolCall,
    *,
    perceives: MediaTypes,
    redactor: Redactor,
) -> _Content:
    """Render a tool call, each data URL string in its arguments lifted out after them.

    The n-th lifted string is replaced by "[media n]", and its media, through `redactor`,
    carries ref="n".
    """
    lifted: list[MediaPart] = []

    def lift(found: re.Match[str]) -> str:
        try:
            data = b64decode(
                found["data"],
                validate=True,
            )
        # binascii.Error for data that is not base64, ValueError for text that is not ASCII.
        except ValueError:
            return found[0]
        lifted.append(
            MediaPart(
                media_type=found["media_type"],
                data=data,
            ),
        )
        return f'"[media {len(lifted)}]"'

    arguments = _escape(
        _DATA_URL.sub(
            lift,
            call.arguments,
        ),
    )
    media = [
        _render_lifted(
            redactor.redact_part(part),
            perceives=perceives,
            ref=str(number),
        )
        for number, part in enumerate(
            lifted,
            start=1,
        )
    ]
    return _wrap(
        "tool_call",
        content=_concatenate(
            [
                arguments,
                *media,
            ],
        ),
        id=call.id,
        name=call.name,
    )


def _render_lifted(
    part: Part,
    *,
    perceives: MediaTypes,
    ref: str,
) -> _Content:
    """Render media lifted from a tool call's arguments, or the part a redaction put in its
    place, in an element that carries its ref.
    """
    if isinstance(
        part,
        MediaPart,
    ):
        return _render_media(
            part,
            perceives=perceives,
            ref=ref,
        )
    return _wrap(
        "media",
        content=_render_part(
            part,
            channel=Channel.TOOL_CALLS,
            perceives=perceives,
        ),
        ref=ref,
    )


def _escape(text: str) -> _Content:
    """Return text escaped to sit inside an element."""
    return _Content(
        parts=(
            TextPart(
                text=html.escape(
                    text,
                    quote=False,
                ),
            ),
        ),
    )


def _wrap(
    tag: str,
    *,
    content: _Content,
    **attributes: str | None,
) -> _Content:
    """Wrap content in an element, leaving out each attribute that is None."""
    opening = TextPart(text=f"<{tag}{_format_attributes(attributes)}>")
    closing = TextPart(text=f"</{tag}>")
    return _Content(
        parts=(
            opening,
            *content.parts,
            closing,
        ),
        unseen=content.unseen,
    )


def _mark_unseen(
    tag: str,
    *,
    unseen: str,
    **attributes: str | None,
) -> _Content:
    """Return an empty element that stands for content the model does not see."""
    return _Content(
        parts=(TextPart(text=f"<{tag}{_format_attributes(attributes)}/>"),),
        unseen=(unseen,),
    )


def _format_attributes(attributes: dict[str, str | None]) -> str:
    """Return the attributes that are not None, each value escaped to sit inside quotes."""
    return "".join(
        f' {name}="{html.escape(value, quote=True)}"'
        for name, value in attributes.items()
        if value is not None
    )


def _concatenate(contents: list[_Content]) -> _Content:
    """Join rendered contents in order."""
    return _Content(
        parts=tuple(part for content in contents for part in content.parts),
        unseen=tuple(unseen for content in contents for unseen in content.unseen),
    )
