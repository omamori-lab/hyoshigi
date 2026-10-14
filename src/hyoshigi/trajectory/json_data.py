"""JSON-compatible data: what json.loads returns and json.dumps accepts."""

type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None
