"""The JSONPath that names where in a stored trajectory a problem is."""

_NAME_ESCAPES = {
    **{code: f"\\u{code:04x}" for code in range(0x20)},
    **{code: f"\\u{code:04x}" for code in range(0xD800, 0xE000)},
    ord("\b"): "\\b",
    ord("\t"): "\\t",
    ord("\n"): "\\n",
    ord("\f"): "\\f",
    ord("\r"): "\\r",
    ord("'"): "\\'",
    ord("\\"): "\\\\",
}


def select_member(
    name: str,
    *,
    path: str,
) -> str:
    r"""Return the path to the member `name` of the object at `path`.

    The member is named in single quotes and escaped as in a Normalized Path (RFC 9535,
    sec. 2.7). An unpaired surrogate, which no JSONPath can name, is written as an
    escape, such as `\ud800`.
    """
    return f"{path}['{name.translate(_NAME_ESCAPES)}']"
