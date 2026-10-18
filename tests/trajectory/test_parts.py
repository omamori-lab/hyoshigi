"""Media parts: the media type and the source each one must give."""

import pytest

from hyoshigi import ConfigurationError, MediaPart

CHART_URL = "https://example.com/chart.png"
PNG_SIGNATURE = b"\x89PNG"


@pytest.mark.parametrize(
    ("url", "data"),
    [
        (None, None),
        (
            CHART_URL,
            PNG_SIGNATURE,
        ),
    ],
    ids=["neither", "both"],
)
def test_a_media_part_takes_exactly_one_of_url_and_data(
    url: str | None,
    data: bytes | None,
) -> None:
    with pytest.raises(ConfigurationError):
        MediaPart(
            media_type="image/png",
            url=url,
            data=data,
        )


# RFC 6838, section 4.2: a type and a subtype, each a letter or digit followed by up
# to 126 letters, digits or any of ! # $ & - ^ _ . +
@pytest.mark.parametrize(
    "media_type",
    [
        "image",
        "image/",
        "/png",
        "image/png/x",
        "ima ge/png",
        "image/png; charset=utf-8",
        ".image/png",
        "*/*",
        "image/" + "x" * 128,
    ],
)
def test_a_media_type_outside_the_iana_grammar_is_refused(media_type: str) -> None:
    with pytest.raises(ConfigurationError):
        MediaPart(
            media_type=media_type,
            url=CHART_URL,
        )


@pytest.mark.parametrize(
    "media_type",
    [
        "image/png",
        "audio/wav",
        "application/vnd.api+json",
        "image/svg+xml",
        "IMAGE/PNG",
        # RFC 9110, section 12.5.1: a media range names a top-level type whose subtype is unknown.
        "image/*",
    ],
)
def test_a_media_type_in_the_iana_grammar_is_accepted(media_type: str) -> None:
    part = MediaPart(
        media_type=media_type,
        url=CHART_URL,
    )

    assert part.media_type == media_type
