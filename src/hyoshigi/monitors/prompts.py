"""The prompt a generative model reads when it rates a proposed action."""

from dataclasses import dataclass

from hyoshigi.models.model import Prompt
from hyoshigi.monitors.rendering import Rendering


@dataclass(frozen=True, slots=True, kw_only=True)
class MonitorPrompt:
    """A monitor's instructions, and the request that holds what it rates."""

    system: str
    request: str

    def build(self, rendering: Rendering, *, lowest_score: int, highest_score: int) -> Prompt:
        """Build the prompt for one rendering and the ends of the score's scale."""
        return Prompt(system="", content=())


DEFAULT_PROMPT = MonitorPrompt(system="", request="")
