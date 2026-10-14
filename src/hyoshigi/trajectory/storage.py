"""How a trajectory is written to and read from JSON-compatible data."""

type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None
