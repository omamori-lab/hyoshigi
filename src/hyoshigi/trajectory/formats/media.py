"""Media read from a provider's payload: base64 data and its media type."""

from hyoshigi.errors import ConfigurationError, FormatError
from hyoshigi.trajectory.parts import MediaPart


def build_media(media_type: str, *, data: bytes | None, path: str) -> MediaPart:
    """Build a media part from base64 data, or raise FormatError at the path if it cannot be one."""
    try:
        return MediaPart(media_type=media_type, data=data)
    except ConfigurationError:
        raise FormatError(path=path, expected="an IANA media type") from None
