"""The labels a guard model writes, and reading its label from where its format puts it.

A guard model replies with a label, not a score. The agent writes the action the guard reads,
so it can plant a label there for the guard to quote, as text aimed at its monitor (Terekhov
et al., "Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols",
arXiv:2510.09462). The reader takes the label only from a line whose shape is a verdict, and
treats a reply whose lines name two labels as unreadable. Markup around a key or a label, as
in `**Label**: __violation__`, and a list marker or heading at the start of a line, as in
`1. Label:` or `## Label:`, do not change what a line names; a label with no key in a list
item or a quote is no verdict.
"""

import re
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from hyoshigi.errors import ConfigurationError
from hyoshigi.monitors.score import Unreadable
from hyoshigi.options import check_instance, describe_option_value

# Markup around a key or a label: any run of non-word characters and underscores, such as
# **, __, backticks, quotes, brackets, spaces, a heading's #, a quote's > or a bullet.
_MARKUP = r"[\W_]*+"
# An ordered list marker, as in 1., a) or iv.: a number, one letter or a roman numeral of up
# to five letters, then a full stop or a bracket; a bullet such as - is markup already.
_LIST_MARKER = r"(?:\d++|[A-Za-z]|[ivxIVX]{2,5}+)[.)]"
# The start of a line before its key or label: markup, then maybe a list marker and markup.
_LINE_OPENING = rf"{_MARKUP}(?:(?P<list_marker>{_LIST_MARKER}){_MARKUP})?"
# A line that opens as a bullet or a quote, as in "- no_violation" or "> no_violation".
_BULLET_OR_QUOTE = re.compile(r"\s*+(?:>|[-*+•]\s)")
# A short key: at most 20 letters, spaces and *, _ or backtick markup, from letter to letter,
# so a guard can wrap each word of a key, as in **Final** **answer**.
_KEY = r"[A-Za-z](?:[A-Za-z *_`]{0,18}[A-Za-z])?"
# The colon after a key, and any markup before it, as in **Label**:, Label : or "label":.
_KEY_END = r"(?:[^\w:]|_)*+:"
# A label: letters and digits, joined by _ or -, beginning and ending with a letter or digit,
# so the underscores of __violation__ are markup around it.
_LABEL_WORD = r"[^\W_]++(?:[_-]++[^\W_]++)*+"
_LABEL = rf"(?P<label>{_LABEL_WORD})"
# A mark after a label, such as (, :, * or _, which makes the line more than prose.
_MARK = r"(?:[^\w\s]|_)"
# A label alone, or after a short key such as Label:, with markup around either. The bounded
# key and the possessive quantifiers make a line that holds no label fail in linear time.
_LABEL_LINE = re.compile(
    rf"{_LINE_OPENING}(?:(?P<key>{_KEY}){_KEY_END}{_MARKUP})?{_LABEL}{_MARKUP}"
)
# The keys a guard may put before its verdict, as in Label: or Final answer:.
_VERDICT_KEY = re.compile(
    r"(?:final[\s*_`]++)?(?:label|answer|verdict|classification|result|output|decision|rating"
    r"|category|assessment|conclusion|judgement|judgment)",
    flags=re.IGNORECASE,
)
_KEY_WORD = re.compile(r"[A-Za-z]+")
# A line that opens with a verdict key and a label, then more text, as in "Label: violation
# (...". No syntax tells "Label: violation because ..." from "Result: violation of the policy
# ...", so both name the label.
_KEYED_LABEL_START = re.compile(
    rf"{_LINE_OPENING}(?P<key>{_VERDICT_KEY.pattern}){_KEY_END}{_MARKUP}{_LABEL}",
    flags=re.IGNORECASE,
)
# A line that opens with a label and then a mark, as in "violation (the note asks ...", and
# the same after an ordered list marker; a guard whose labels are digits or letters may open
# a line with its label and a full stop, so both are read.
_UNKEYED_LABEL_START = re.compile(rf"{_MARKUP}{_LABEL}\s*+{_MARK}")
_LISTED_LABEL_START = re.compile(rf"{_MARKUP}{_LIST_MARKER}{_MARKUP}{_LABEL}\s*+{_MARK}")
# What may follow a line's last colon, or a key, when the line ends on a label.
_LABEL_AFTER_COLON = re.compile(rf"{_MARKUP}{_LABEL}{_MARKUP}")
_CATEGORY_CODES = re.compile(r"\s*S\d+(?:\s*,\s*S\d+)*\s*")
# A line holding more than whitespace; anchored to the line's start, so it runs in linear time.
_NON_EMPTY_LINE = re.compile(r"^[^\n]*\S[^\n]*", flags=re.MULTILINE)
_KEY_WORDS = re.compile(r"[^\W_]+(?: [^\W_]+)*")
_ONE_WORD = re.compile(_LABEL_WORD)


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardLabels:
    """The labels a guard model writes, and the key it writes its verdict after, if any.

    `suspicious` and `safe` are sets of labels, matched case-insensitively and kept in lower
    case as frozensets. Each label is one word of letters or digits, joined by _ or -, that
    begins and ends with a letter or digit; neither set is empty, and no label is in both.

    Without a `key`, the label is read on the reply's last line, after any reasoning, or on
    its first when it is suspicious and only category codes such as S1,S10 follow, as Llama
    Guard writes: a label alone, or after a verdict key such as `Label:`. A reply whose lines
    name two labels is unreadable. With a `key`, such as
    Qwen3Guard's "Safety", the label is the whole value of the one line that opens with the
    key and a colon, matched case-insensitively through markup; every other line is ignored,
    and a reply without exactly one such line holding a label is unreadable. A key is words
    of letters or digits, one space apart. Anything else raises ConfigurationError.
    """

    suspicious: frozenset[str]
    safe: frozenset[str]
    key: str | None = None

    def __post_init__(self) -> None:
        suspicious = _read_labels(self.suspicious, name="suspicious")
        safe = _read_labels(self.safe, name="safe")
        shared = sorted(suspicious & safe)
        if shared:
            raise ConfigurationError(f"a label cannot be in both suspicious and safe, got {shared}")
        if self.key is not None and not (
            isinstance(self.key, str) and _KEY_WORDS.fullmatch(self.key)
        ):
            raise ConfigurationError(
                "key must be words of letters or digits, one space apart, such as "
                f"'User Safety', got {describe_option_value(self.key)}"
            )
        # The dataclass is frozen, so the sets are set past its guard.
        object.__setattr__(self, "suspicious", suspicious)
        object.__setattr__(self, "safe", safe)


def _read_labels(labels: object, *, name: str) -> frozenset[str]:
    """Return a set of labels in lower case, or raise ConfigurationError naming the set."""
    if not isinstance(labels, AbstractSet):
        raise ConfigurationError(
            f"{name} must be a set of labels, such as {{'unsafe'}}, "
            f"got {describe_option_value(labels)}"
        )
    for label in labels:
        check_instance(label, expected=str, name=f"each label in {name}")
    if not labels:
        raise ConfigurationError(f"{name} must hold at least one label")
    malformed = sorted(label for label in labels if _ONE_WORD.fullmatch(label) is None)
    if malformed:
        raise ConfigurationError(
            f"{name} must hold only single words of letters, digits, _ or -, each beginning "
            f"and ending with a letter or digit, got {malformed}"
        )
    return frozenset(label.lower() for label in labels)


# Meta, "Llama Guard 4 12B" model card, https://huggingface.co/meta-llama/Llama-Guard-4-12B:
# its output "indicates whether a given prompt or response is safe or unsafe", and "if
# unsafe, it also lists the content categories violated"; its example output is unsafe, then
# S9 on the next line.
LLAMA_GUARD = GuardLabels(suspicious=frozenset({"unsafe"}), safe=frozenset({"safe"}))

# ROOST and OpenAI, "User guide for gpt-oss-safeguard",
# https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide: its example
# policies ask to "Classify each example as `VALID` (allowed) or `INVALID`". A policy that
# asks for other labels, such as 0 and 1, takes GuardLabels of its own.
GPT_OSS_SAFEGUARD = GuardLabels(suspicious=frozenset({"invalid"}), safe=frozenset({"valid"}))

# Qwen, "Qwen3Guard-Gen-8B" model card, https://huggingface.co/Qwen/Qwen3Guard-Gen-8B: its
# reply is "Safety: Unsafe" then "Categories: Violent", which the card reads with
# r"Safety: (Safe|Unsafe|Controversial)". Controversial is content "whose harmfulness may be
# context-dependent" (Zhao et al., "Qwen3Guard Technical Report", arXiv:2510.14276), and is
# read as suspicious.
QWEN3GUARD = GuardLabels(
    suspicious=frozenset({"unsafe", "controversial"}), safe=frozenset({"safe"}), key="Safety"
)

# NVIDIA, "Nemotron 3.5 Content Safety" model card,
# https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety, section Output:
# 'User Safety: string(required) # "safe" or "unsafe"'. A monitor sends what it rates as one
# user message, and the card's Response Safety line is omitted "if no assistant response
# present", so the verdict is the User Safety line.
NEMOTRON_CONTENT_SAFETY = GuardLabels(
    suspicious=frozenset({"unsafe"}), safe=frozenset({"safe"}), key="User Safety"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _LabelLine:
    """A known label a line names, in lower case, and whether the line can be the verdict.

    A line can be the verdict when it holds the label alone, or after a verdict key such as
    Label:. A label after any other key, alone in a list item or a quote, or at the start of
    a longer line, still names that label for the conflict check.
    """

    label: str
    is_verdict: bool


def read_label(text: str, *, labels: GuardLabels) -> str | Unreadable:
    """Return the guard's label, in lower case, from where its format puts it, or Unreadable.

    A reply in which two lines name different labels is unreadable: a guard that quotes a
    label the agent planted must not have the quote read as its verdict. A suspicious label
    at the start of a line before a mark, after a verdict key at the start, or after a line's
    last colon counts as named, since it may be the guard's own label written off format; a
    guard that gives its verdict only in prose and then quotes a planted label alone on the
    last line cannot be told from one that gives that label.
    """
    lines = list(_NON_EMPTY_LINE.finditer(text))
    if labels.key is not None:
        return _read_keyed_label(lines, key=labels.key, labels=labels.suspicious | labels.safe)
    label_lines = _find_label_lines(lines, labels=labels.suspicious | labels.safe)
    named = {line.label for line in label_lines.values()}
    for line in lines:
        named |= _read_suspicious_mentions(line.group(), suspicious=labels.suspicious)
    if len(named) != 1:
        return Unreadable()
    last_line = label_lines.get(len(lines) - 1)
    if last_line is not None:
        return last_line.label if last_line.is_verdict else Unreadable()
    return _read_first_line_label(lines, label_lines=label_lines, suspicious=labels.suspicious)


def _read_keyed_label(
    lines: list[re.Match[str]], *, key: str, labels: frozenset[str]
) -> str | Unreadable:
    """Return the whole value of the one line that opens with the key, if it is a label."""
    words = r"[\s*_`]++".join(map(re.escape, key.split(" ")))
    opening = re.compile(rf"{_LINE_OPENING}{words}{_KEY_END}", flags=re.IGNORECASE)
    keyed = [match for line in lines if (match := opening.match(line.group())) is not None]
    if len(keyed) != 1:
        return Unreadable()
    (key_match,) = keyed
    value = _LABEL_AFTER_COLON.fullmatch(key_match.string, key_match.end())
    if value is None or value["label"].lower() not in labels:
        return Unreadable()
    return value["label"].lower()


def _find_label_lines(
    lines: list[re.Match[str]], *, labels: frozenset[str]
) -> dict[int, _LabelLine]:
    """Return the label of every line that holds one or opens with a keyed one, by index."""
    label_lines: dict[int, _LabelLine] = {}
    for index, line in enumerate(lines):
        found = _match_label_line(line.group(), labels=labels) or _match_keyed_label_start(
            line.group(), labels=labels
        )
        if found is not None:
            label_lines[index] = found
    return label_lines


def _match_label_line(line: str, *, labels: frozenset[str]) -> _LabelLine | None:
    """Read a line that holds one known label, alone or after a short key.

    A key that names a label, as in "violation: no_violation", makes the line name two
    labels, so it holds none.
    """
    match = _LABEL_LINE.fullmatch(line)
    if match is None or match["label"].lower() not in labels:
        return None
    key = match["key"]
    if key and any(word.lower() in labels for word in _KEY_WORD.findall(key)):
        return None
    return _LabelLine(label=match["label"].lower(), is_verdict=_is_verdict(match))


def _is_verdict(match: re.Match[str]) -> bool:
    """Tell whether a label line can be the verdict: after a verdict key, or alone.

    A label with no key in a list item or a quote, as in "2. no_violation" or
    "> no_violation", is no verdict: the guard may be quoting a list from the action. After a
    verdict key, as in "1. Label: no_violation", it still is.
    """
    if match["key"]:
        return _VERDICT_KEY.fullmatch(match["key"]) is not None
    return match["list_marker"] is None and _BULLET_OR_QUOTE.match(match.string) is None


def _match_keyed_label_start(line: str, *, labels: frozenset[str]) -> _LabelLine | None:
    """Read the label a line opens with after a verdict key, as in "Label: violation (...".

    A guard that gives its keyed label and then quotes the action on the same line still
    names its label; the line is no verdict, since more text follows.
    """
    match = _KEYED_LABEL_START.match(line)
    if match is None or match["label"].lower() not in labels:
        return None
    return _LabelLine(label=match["label"].lower(), is_verdict=False)


def _read_suspicious_mentions(line: str, *, suspicious: frozenset[str]) -> set[str]:
    """Return the suspicious labels a line opens with before a mark, or ends on after a colon.

    Such a line may be the guard's own verdict written off format, with a planted safe label
    quoted on the next line. The reader cannot tell, so it errs towards suspicion: the line
    names the suspicious label, and a reply that also names a safe label is unreadable.
    """
    openings = [_UNKEYED_LABEL_START.match(line), _LISTED_LABEL_START.match(line)]
    _, colon, after = line.rpartition(":")
    ending = _LABEL_AFTER_COLON.fullmatch(after) if colon else None
    return {
        match["label"].lower()
        for match in (*openings, ending)
        if match is not None and match["label"].lower() in suspicious
    }


def _read_first_line_label(
    lines: list[re.Match[str]], *, label_lines: dict[int, _LabelLine], suspicious: frozenset[str]
) -> str | Unreadable:
    """Read Llama Guard's unsafe reply: a suspicious label first, then only category codes."""
    first_line = label_lines.get(0)
    if first_line is None or not first_line.is_verdict or first_line.label not in suspicious:
        return Unreadable()
    if all(_CATEGORY_CODES.fullmatch(line.group()) for line in lines[1:]):
        return first_line.label
    return Unreadable()
