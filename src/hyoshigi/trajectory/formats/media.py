"""Media read from a provider's payload: base64 data, data URLs and their media types."""

import re
from urllib.parse import unquote_to_bytes

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.parts import MediaPart
from hyoshigi.trajectory.storage.events import decode_data

# RFC 2397, section 3: data:[<mediatype>][;base64],<data>, where the media type, with its
# parameters, defaults to text/plain. Parameter names are case-insensitive (RFC 2045, section
# 5.1), and so is ;base64.
_DATA_URL = re.compile(
    r"data:(?P<media_type>[^;,]*)(?:;[^;,]*?)*?(?P<base64>;(?i:base64))?,(?P<data>.*)",
    re.DOTALL,
)


def build_media(
    media_type: str,
    *,
    data: bytes | None,
    path: str,
) -> MediaPart:
    """Build a media part from data, or raise FormatError at the path if it cannot be one."""
    try:
        return MediaPart(
            media_type=media_type,
            data=data,
        )
    except ConfigurationError:
        raise FormatError(
            path=path,
            expected="an IANA media type",
        ) from None


def read_data_url(
    url: str,
    *,
    path: str,
) -> MediaPart | None:
    """Read a data URL into media with its own type and bytes; return None for another URL."""
    found = _DATA_URL.fullmatch(url)
    if found is None:
        return None
    encoded = found["data"]
    data = (
        decode_data(
            encoded,
            path=path,
        )
        if found["base64"]
        else unquote_to_bytes(encoded)
    )
    return build_media(
        found["media_type"] or "text/plain",
        data=data,
        path=path,
    )
