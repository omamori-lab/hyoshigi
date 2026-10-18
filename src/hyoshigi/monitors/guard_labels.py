"""The labels a guard model writes, and how its label is read from its reply."""

from dataclasses import dataclass

from hyoshigi.monitors.score import Unreadable


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardLabels:
    """The labels a guard model writes, and the key it writes its verdict after, if any."""

    suspicious: frozenset[str]
    safe: frozenset[str]
    key: str | None = None


LLAMA_GUARD = GuardLabels(suspicious=frozenset(), safe=frozenset())
GPT_OSS_SAFEGUARD = GuardLabels(suspicious=frozenset(), safe=frozenset())
QWEN3GUARD = GuardLabels(suspicious=frozenset(), safe=frozenset())
NEMOTRON_CONTENT_SAFETY = GuardLabels(suspicious=frozenset(), safe=frozenset())


def read_label(text: str, *, labels: GuardLabels) -> str | Unreadable:
    """Read the guard's label from its reply, or return Unreadable."""
    return ""
