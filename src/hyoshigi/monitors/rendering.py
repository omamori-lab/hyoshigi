"""A trajectory and the action a monitor rates, rendered as the parts its model reads.

Each entry is one element: an observation, or an action's thought, its parts or one of
its tool calls. Text inside an element is HTML-escaped, so a tool result cannot close a
tag and pose as another entry. Media the model perceives sits in place among the text;
other media is its description, tagged with who wrote it, or a placeholder.
"""

import html
import re
from base64 import b64decode
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import assert_never

from hyoshigi.models.model import MediaTypes, PromptPart
from hyoshigi.monitors.view import Channel, View
from hyoshigi.trajectory.history import Action, Observation, ToolCall, ToolObservation
from hyoshigi.trajectory.parts import MEDIA_TYPE, Author, MediaPart, OpaquePart, Part, TextPart
from hyoshigi.trajectory.trajectory import Trajectory

# A JSON string that is a whole data URL of base64 data (RFC 2397, section 3). It is
# matched in the raw arguments, so arguments that are not valid JSON are read too.
_DATA_URL = re.compile(f'"data:(?P<media_type>{MEDIA_TYPE.pattern});base64,(?P<data>[^"]*)"')


@dataclass(frozen=True, slots=True, kw_only=True)
class Rendering:
    """The history as a view shows it, the rated action, and what of the action is unseen.

    `hidden_reasoning` names the kinds of the rated thought's opaque parts and the media
    types of its media not shown, and is empty when the view does not show the thought.
    `unperceivable` names the same of the action's parts and tool calls.
    """

    trajectory: tuple[PromptPart, ...]
    proposed_action: tuple[PromptPart, ...]
    hidden_reasoning: tuple[str, ...]
    unperceivable: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Content:
    """Rendered parts, and the opaque kinds and media types among them left unseen."""

    parts: tuple[PromptPart, ...]
    unseen: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Entry:
    """One rendered element of a trajectory, and the channel it belongs to."""

    channel: Channel
    content: _Content


def render(
    action: Action, *, trajectory: Trajectory, view: View, perceives: MediaTypes
) -> Rendering:
    """Render the trajectory through the view, and the action rated against it.

    The action's parts and tool calls are shown whatever the view, and its thought when
    the view shows thoughts; `most_recent` limits only the history.
    """
    history = _select(list(_render_history(trajectory, view=view, perceives=perceives)), view=view)
    shown = view.channels | Channel.AGENT_PARTS | Channel.TOOL_CALLS
    proposed = [
        entry for entry in _render_action(action, perceives=perceives) if entry.channel in shown
    ]
    return Rendering(
        trajectory=_join(history),
        proposed_action=_join(proposed),
        hidden_reasoning=_collect_unseen(proposed, channels=Channel.THOUGHTS),
        unperceivable=_collect_unseen(proposed, channels=Channel.AGENT_PARTS | Channel.TOOL_CALLS),
    )


def merge_text(parts: Iterable[PromptPart]) -> tuple[PromptPart, ...]:
    """Return the parts with each run of adjacent text joined into one text part."""
    merged: list[PromptPart] = []
    for part in parts:
        match merged[-1:], part:
            case [TextPart(text=before)], TextPart(text=after):
                merged[-1] = TextPart(text=before + after)
            case _:
                merged.append(part)
    return tuple(merged)


def _select(entries: list[_Entry], *, view: View) -> list[_Entry]:
    """Keep the entries the view shows, only the most recent ones when it limits them.

    When the limit drops older entries, every task entry is kept anyway, in order: the
    first states the task, and a later one can narrow or revoke it. A context note is
    never kept in their place.
    """
    shown = [entry for entry in entries if entry.channel in view.channels]
    if view.most_recent is None:
        return shown
    earlier = [entry for entry in shown[: -view.most_recent] if entry.channel is Channel.TASK]
    return [*earlier, *shown[-view.most_recent :]]


def _join(entries: list[_Entry]) -> tuple[PromptPart, ...]:
    """Join the entries' parts, one entry per line."""
    parts: list[PromptPart] = []
    for entry in entries:
        if parts:
            parts.append(TextPart(text="\n"))
        parts.extend(entry.content.parts)
    return merge_text(parts)


def _collect_unseen(entries: list[_Entry], *, channels: Channel) -> tuple[str, ...]:
    """Return what is unseen in the entries on these channels, in order."""
    return tuple(
        unseen for entry in entries if entry.channel in channels for unseen in entry.content.unseen
    )


def _render_history(
    trajectory: Trajectory, *, view: View, perceives: MediaTypes
) -> Iterator[_Entry]:
    """Yield the entries of the trajectory's history in order; its subagents' are not."""
    delegations = set(trajectory.subagents)
    for event in trajectory.history:
        match event:
            case Action():
                delegations.update(
                    call.id for call in event.tool_calls if call.name in view.delegation_tools
                )
                yield from _render_action(event, perceives=perceives)
            case Observation():
                yield _render_observation(event, delegations=delegations, perceives=perceives)
            case unreachable:
                assert_never(unreachable)


def _render_action(action: Action, *, perceives: MediaTypes) -> Iterator[_Entry]:
    """Yield an action's thought, its parts, then each tool call; nothing for what is empty."""
    if action.thought:
        yield _render_entry(
            action.thought, channel=Channel.THOUGHTS, tag="thought", perceives=perceives
        )
    if action.parts:
        yield _render_entry(
            action.parts, channel=Channel.AGENT_PARTS, tag="agent", perceives=perceives
        )
    for call in action.tool_calls:
        yield _Entry(
            channel=Channel.TOOL_CALLS, content=_render_tool_call(call, perceives=perceives)
        )


def _render_observation(
    observation: Observation, *, delegations: set[str], perceives: MediaTypes
) -> _Entry:
    """Render an observation by who wrote it; only the user or the delegator sets the task."""
    match observation:
        case ToolObservation():
            return _render_tool_observation(
                observation, delegations=delegations, perceives=perceives
            )
        case Observation(author=Author.USER | Author.DELEGATOR as author):
            return _render_entry(
                observation.parts, channel=Channel.TASK, tag=author.value, perceives=perceives
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
    observation: ToolObservation, *, delegations: set[str], perceives: MediaTypes
) -> _Entry:
    """Render a tool's result as an error, a subagent's report or a plain result.

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
        [_render_part(part, channel=channel, perceives=perceives) for part in parts]
    )
    return _Entry(channel=channel, content=_wrap(tag, content=content, **attributes))


def _render_part(part: Part, *, channel: Channel, perceives: MediaTypes) -> _Content:
    """Render a part; an opaque part in a thought is reasoning the provider hid."""
    match part:
        case TextPart(text=text):
            return _escape(text)
        case MediaPart():
            return _render_media(part, perceives=perceives)
        case OpaquePart(kind=kind):
            tag = "hidden_reasoning" if channel is Channel.THOUGHTS else "opaque"
            return _mark_unseen(tag, unseen=kind, kind=kind)
        case unreachable:
            assert_never(unreachable)


def _render_media(part: MediaPart, *, perceives: MediaTypes, ref: str | None = None) -> _Content:
    """Render media as itself if the model perceives it, else its description or a mark.

    `ref` numbers media lifted from a tool call's arguments, which has no description.
    """
    if perceives.covers(part.media_type):
        return _wrap("media", content=_Content(parts=(part,)), type=part.media_type, ref=ref)
    if part.description is not None:
        return _wrap(
            "media",
            content=_escape(part.description.text),
            type=part.media_type,
            description_author=part.description.author.value,
        )
    return _mark_unseen("omitted_media", unseen=part.media_type, type=part.media_type, ref=ref)


def _render_tool_call(call: ToolCall, *, perceives: MediaTypes) -> _Content:
    """Render a tool call, each data URL string in its arguments lifted out after them.

    The n-th lifted string is replaced by "[media n]", and its media carries ref="n".
    """
    lifted: list[MediaPart] = []

    def lift(found: re.Match[str]) -> str:
        try:
            data = b64decode(found["data"], validate=True)
        # binascii.Error for data that is not base64, ValueError for text that is not ASCII.
        except ValueError:
            return found[0]
        lifted.append(MediaPart(media_type=found["media_type"], data=data))
        return f'"[media {len(lifted)}]"'

    arguments = _escape(_DATA_URL.sub(lift, call.arguments))
    media = [
        _render_media(part, perceives=perceives, ref=str(number))
        for number, part in enumerate(lifted, start=1)
    ]
    return _wrap("tool_call", content=_concatenate([arguments, *media]), id=call.id, name=call.name)


def _escape(text: str) -> _Content:
    """Return text escaped to sit inside an element."""
    return _Content(parts=(TextPart(text=html.escape(text, quote=False)),))


def _wrap(tag: str, *, content: _Content, **attributes: str | None) -> _Content:
    """Wrap content in an element, leaving out each attribute that is None."""
    opening = TextPart(text=f"<{tag}{_format_attributes(attributes)}>")
    closing = TextPart(text=f"</{tag}>")
    return _Content(parts=(opening, *content.parts, closing), unseen=content.unseen)


def _mark_unseen(tag: str, *, unseen: str, **attributes: str | None) -> _Content:
    """Return an empty element that stands for content the model does not see."""
    return _Content(
        parts=(TextPart(text=f"<{tag}{_format_attributes(attributes)}/>"),), unseen=(unseen,)
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
