"""The errors hyoshigi raises when it is used in a way that cannot work."""


class ConfigurationError(ValueError):
    """An object was built, or a method called, with arguments that cannot work."""


class FormatError(ValueError):
    """A payload is not in the format its reader expects.

    The message names where in the payload the problem is, as a JSONPath query,
    and what was expected there; it never quotes the payload, which can hold an
    agent's private content.
    """

    def __init__(self, *, path: str, expected: str) -> None:
        super().__init__(f"{path}: expected {expected}")
        self.path = path
