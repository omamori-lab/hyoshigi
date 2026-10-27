"""Parts: the media type and the source a media part must give, and the fields that cannot
work.
"""

import pytest

from hyoshigi import Author, ConfigurationError, Description, MediaPart, OpaquePart, TextPart

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


@pytest.mark.parametrize(
    ("kind", "fields", "message"),
    [
        pytest.param(
            TextPart,
            {"text": b"Revenue: 4.2M"},
            "text must be a str, got an instance of bytes",
            id="text-bytes",
        ),
        pytest.param(
            Description,
            {
                "text": None,
                "author": Author.AGENT,
            },
            "text must be a str, got None",
            id="description-text-none",
        ),
        pytest.param(
            Description,
            {
                "text": "A bar chart of revenue by quarter",
                "author": "agent",
            },
            "author must be an Author, got an instance of str",
            id="description-author-a-string",
        ),
        pytest.param(
            MediaPart,
            {
                "media_type": 1,
                "url": CHART_URL,
            },
            "media_type must be a str, got an instance of int",
            id="media-type-a-number",
        ),
        pytest.param(
            MediaPart,
            {
                "media_type": "image/png",
                "url": 1,
            },
            "url must be a str, got an instance of int",
            id="url-a-number",
        ),
        pytest.param(
            MediaPart,
            {
                "media_type": "image/png",
                "data": "iVBORw0KGgo=",
            },
            "data must be a bytes, got an instance of str",
            id="data-base64-text",
        ),
        pytest.param(
            MediaPart,
            {
                "media_type": "image/png",
                "url": CHART_URL,
                "description": "A bar chart of revenue by quarter",
            },
            "description must be a Description, got an instance of str",
            id="description-a-string",
        ),
        pytest.param(
            OpaquePart,
            {"kind": None},
            "kind must be a str, got None",
            id="kind-none",
        ),
    ],
)
def test_a_field_of_the_wrong_type_raises_naming_it(
    kind: type,
    fields: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError) as raised:
        kind(**fields)

    assert str(raised.value) == message
