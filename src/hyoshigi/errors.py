"""The errors hyoshigi raises when it is used in a way that cannot work."""


class ConfigurationError(ValueError):
    """An object was built, or a method called, with arguments that cannot work."""


class FormatError(ValueError):
    """A payload is not in the format its reader expects.

    The message names where in the payload the problem is, as a JSONPath query,
    and what was expected there. It quotes no value from the payload, which can
    hold an agent's private content; the path can hold a member name, such as a
    subagent's tool call id.
    """

    def __init__(
        self,
        *,
        path: str,
        expected: str,
    ) -> None:
        super().__init__(f"{path}: expected {expected}")
        self.path = path
